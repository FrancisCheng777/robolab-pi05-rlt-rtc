"""Actor-critic updates from labelled physical rollouts; never opens hardware."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.deployment.runtime import config_from_args, make_backend, write_json
from pi05_rlt_rtc.learning.actor import QueueAwareRLTActor
from pi05_rlt_rtc.learning.checkpoint import PolicyBundle, state_dict_sha256
from pi05_rlt_rtc.learning.critic import TwinQWithTargets
from pi05_rlt_rtc.learning.defaults import stage2_settings
from pi05_rlt_rtc.learning.losses import actor_loss, bc_warmup_loss, build_transition_batch, critic_loss
from pi05_rlt_rtc.learning.serialization import atomic_torch_save
from .reward_override import reward_definition, relabel_terminal_transition


class QueueLearner:
    def __init__(self, cfg, spec, *, device, seed):
        self.cfg, self.spec, self.device = cfg, spec, device
        self.settings = stage2_settings(cfg.raw.get("training", {}))
        torch.manual_seed(seed)
        self.actor = QueueAwareRLTActor(spec, fixed_std=self.settings.get("exploration_std", .002)).to(device)
        self.critic = TwinQWithTargets(sum(d for _, d in spec.critic_state_layout()), spec.chunk * 8).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.settings["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critic.online.parameters(), lr=self.settings["critic_lr"])
        self.rng = np.random.default_rng(seed)
        self.torch_rng = torch.Generator(device=device).manual_seed(seed)
        self.counts = {"warmup_updates": 0, "updates": 0, "actor_updates": 0, "rounds": 0}
        self.metrics = []

    def _batch(self, replay):
        return build_transition_batch(replay.sample(self.settings["batch_size"], generator=self.rng),
                                      self.spec, contract=self.cfg.contract, device=self.device)

    @staticmethod
    def _step(loss, optimizer, params, *, clip_grad):
        if not torch.isfinite(loss):
            raise ContractError("nonfinite learner loss; refusing to save/deploy an invalid actor")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, clip_grad, error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach()), float(norm)

    def warmup(self, replay, count):
        for _ in range(count):
            loss, _ = bc_warmup_loss(self.actor, self._batch(replay))
            value, norm = self._step(loss, self.actor_opt, self.actor.parameters(),
                                     clip_grad=self.settings["actor_clip_grad"])
            self.metrics.append({"bc_loss": value, "actor_grad_norm": norm})
            self.counts["warmup_updates"] += 1

    def update(self, replay, count):
        for _ in range(count):
            batch = self._batch(replay)
            qloss, _ = critic_loss(self.critic, self.actor, batch, generator=self.torch_rng)
            qvalue, _ = self._step(qloss, self.critic_opt, self.critic.online.parameters(),
                                   clip_grad=self.settings["critic_clip_grad"])
            metrics = {"critic_loss": qvalue, "update_step": self.counts["updates"]}
            # Match RLinf: zero-based global step, preserved across calls/resume.
            if self.counts["updates"] % self.settings["critic_actor_ratio"] == 0:
                loss, _ = actor_loss(self.actor, self.critic, batch,
                    lambda_q=self.settings["lambda_q"], lambda_bc=self.settings["lambda_bc"],
                    reference_dropout_prob=self.settings["reference_dropout"], generator=self.torch_rng)
                avalue, _ = self._step(loss, self.actor_opt, self.actor.parameters(),
                                      clip_grad=self.settings["actor_clip_grad"])
                metrics["actor_loss"] = avalue
                self.counts["actor_updates"] += 1
            self.critic.soft_update(self.settings["polyak_tau"])
            self.metrics.append(metrics)
            self.counts["updates"] += 1

    def save(self, directory, metadata, replay):
        directory = Path(directory)
        policy_id = "queue-rlt:" + state_dict_sha256(self.actor.state_dict())
        bundle = PolicyBundle(spec=self.spec, contract=self.cfg.contract,
            actor_state=self.actor.state_dict(), critic_state=self.critic.state_dict(),
            feature_version_id=metadata["feature_version_id"], norm_stats_sha256=metadata["norm_stats_sha256"],
            policy_version_id=policy_id, training_counts=dict(self.counts),
            actor_hidden_dims=self.actor.hidden_dims, critic_hidden_dims=self.critic.online.hidden_dims,
            codec_name="rlinf_openpi_joint_codec", extra={"mock": metadata.get("mock", False), "runtime_contract": self.cfg.wire_contract(),
                "rtc_mode": self.cfg.rtc_mode, "num_denoising_steps": self.cfg.num_steps,
                "learner_reward_definition": reward_definition(self.settings),
                "exploration_std": self.actor.fixed_std, "control_frequency_hz": self.cfg.control_hz, "fingerprint": metadata["fingerprint"]})
        bundle.save(directory / "policy.pt")
        replay.save(directory / "replay.pt")
        atomic_torch_save({"policy_version_id": policy_id, "actor_optimizer": self.actor_opt.state_dict(),
            "critic_optimizer": self.critic_opt.state_dict(), "numpy_generator": self.rng.bit_generator.state,
            "torch_generator": self.torch_rng.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "counts": self.counts, "training_settings": self.settings,
            "learner_reward_definition": reward_definition(self.settings)}, directory / "learner_state.pt")
        write_json(directory / "metrics.json", {"mock": metadata.get("mock", False), "counts": self.counts, "updates": self.metrics})
        return bundle


def train(args, *, warmup_only=False):
    """Train on human-labelled cached features without loading the VLA or robot."""
    import json
    from pi05_rlt_rtc.learning.replay import ReplayBuffer
    from pi05_rlt_rtc.learning.planning_state import PlanningStateSpec
    from pi05_rlt_rtc.deployment.model_loader import file_sha256
    cfg = config_from_args(args)
    directories = getattr(args, "replay", None)
    if not directories:
        raise ContractError("supply --replay labelled_trial_dir [...]; training never collects or resets hardware")
    if args.warmup_updates < 0 or (not warmup_only and args.updates < 1):
        raise ContractError("invalid update counts")
    rewards = reward_definition(cfg.raw.get("training", {}))
    override = cfg.raw.get("training", {}).get("terminal_rewards") is not None
    reward_audit = []
    success_bc_audit = []
    replay, metadata, seen = None, None, set()
    for value in directories:
        directory = Path(value)
        annotation = json.loads((directory / "annotation.json").read_text())
        current = json.loads((directory / "runtime_manifest.json").read_text())
        if annotation.get("outcome") not in ("success", "failure") or not annotation.get("eligible"):
            raise ContractError("aborted/intervened/unlabelled trial is not eligible for learning")
        if annotation["trial_id"] in seen:
            raise ContractError("duplicate trial in replay input")
        seen.add(annotation["trial_id"])
        if file_sha256(annotation["rollout"]) != annotation["rollout_sha256"]:
            raise ContractError("labelled rollout fingerprint mismatch")
        if file_sha256(directory / "replay.pt") != annotation.get("replay_sha256"):
            raise ContractError("labelled replay fingerprint mismatch")
        if (current.get("mock") is not False and not (getattr(args, "allow_mock", False) and current.get("mock") is True)) or current.get("runtime_contract") != cfg.wire_contract():
            raise ContractError("physical replay runtime contract mismatch")
        if metadata is not None and current.get("mock") != metadata.get("mock"):
            raise ContractError("cannot mix mock and real replay")
        part = ReplayBuffer.load(directory / "replay.pt")
        if part.feature_version_id != current["feature_version_id"] or part.norm_stats_sha256 != current["norm_stats_sha256"]:
            raise ContractError("replay metadata mismatch")
        if replay is None:
            metadata = current
            replay = ReplayBuffer(cfg.raw.get("training", {}).get("replay_capacity", 10000),
                feature_version_id=part.feature_version_id, norm_stats_sha256=part.norm_stats_sha256)
        terminal_count = 0
        transitions = part.all_transitions()
        if getattr(args, 'success_bc', False):
            from .success_bc import success_execution_targets
            transitions, bc_audit = success_execution_targets(transitions, annotation, cfg.contract)
            success_bc_audit.append(bc_audit)
        for transition in transitions:
            if override:
                transition = relabel_terminal_transition(transition, annotation, rewards)
            terminal_count += int(transition.terminated and transition.is_learnable())
            replay.push(transition)
        if override and terminal_count != 1:
            raise ContractError("reward override requires exactly one learnable terminal transition per episode")
        reward_audit.append({"trial_id": annotation["trial_id"], "outcome": annotation["outcome"],
            "terminal_tick": annotation["terminal_tick"], "learnable_terminal_count": terminal_count,
            "terminal_reward": rewards[annotation["outcome"]],
            "source_replay_sha256": annotation["replay_sha256"]})
    if replay is None or len(replay.learnable_transitions()) < 2:
        raise ContractError("need at least two learnable, labelled physical transitions")
    spec = PlanningStateSpec.from_contract(cfg.contract, z_dim=int(cfg.raw["model"].get("z_dim", 2048)))
    learner = QueueLearner(cfg, spec, device=getattr(args, "learner_device", "cpu"), seed=args.seed)
    resume = getattr(args, "resume", None)
    reward_migration = None
    if resume:
        bundle = PolicyBundle.load(Path(resume) / "policy.pt")
        if (bundle.spec != spec or bundle.contract != cfg.contract or
            bundle.feature_version_id != metadata["feature_version_id"] or
            bundle.norm_stats_sha256 != metadata["norm_stats_sha256"] or
            bundle.extra.get("runtime_contract") != cfg.wire_contract() or bundle.extra.get("mock") != metadata.get("mock")):
            raise ContractError("resume checkpoint and replay contracts differ")
        learner.actor.load_state_dict(bundle.actor_state)
        learner.critic.load_state_dict(bundle.critic_state)
        state = torch.load(Path(resume) / "learner_state.pt", map_location=learner.device, weights_only=True)
        old_settings = state.get("training_settings", {})
        if old_settings != learner.settings:
            reward_only = ({k:v for k,v in old_settings.items() if k != "terminal_rewards"}
                           == {k:v for k,v in learner.settings.items() if k != "terminal_rewards"})
            if not (reward_only and getattr(args, "allow_reward_change", False)):
                raise ContractError("resume training settings changed; start a new run or explicitly authorize a reward-only comparison")
            reward_migration = {"from": reward_definition(old_settings), "to": rewards,
                "resume_checkpoint": str(Path(resume).resolve()),
                "actor_critic_and_optimizers": "preserved for controlled comparison"}
        if state["policy_version_id"] != bundle.policy_version_id:
            raise ContractError("resume optimizer/policy mismatch")
        learner.actor_opt.load_state_dict(state["actor_optimizer"])
        learner.critic_opt.load_state_dict(state["critic_optimizer"])
        learner.counts = dict(state["counts"])
        learner.rng.bit_generator.state = state["numpy_generator"]
        learner.torch_rng.set_state(state["torch_generator"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    elif args.warmup_updates < 1:
        raise ContractError("new actor requires BC warmup before critic-guided updates")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ContractError("use a new training output directory")
    learner.warmup(replay, args.warmup_updates)
    if not warmup_only:
        learner.update(replay, args.updates)
        learner.counts["rounds"] += 1
    learner.save(out / "latest", metadata, replay)
    write_json(out / "training_sources.json", {"trial_ids": sorted(seen), "replay_directories": directories,
        "learner_reward_definition": rewards, "reward_migration": reward_migration})
    write_json(out / "reward_audit.json", {"terminal_override": override,
        "learner_reward_definition": rewards, "migration": reward_migration, "episodes": reward_audit})
    if getattr(args, 'success_bc', False):
        write_json(out / 'success_bc_audit.json', {
            'scheme': 'successful_executed_policy_rtc_v1',
            'failure_bc_target': 'unchanged_frozen_teacher',
            'reference_inputs_and_rewards': 'unchanged', 'episodes': success_bc_audit})
    print(f"physical replay training saved {out / 'latest' / 'policy.pt'}; counts={learner.counts}")
    return 0
