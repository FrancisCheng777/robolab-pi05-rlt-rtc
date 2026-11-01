"""Toy end-to-end pipeline (CPU) used by the ``--toy`` CLI commands and tests.

The pipeline composes the real project components (contracts, queue, mock
feature service, full_vjp sampler, planning state, actor/critic, replay,
assembler) on a deterministic toy environment.  Everything it writes is
labelled ``toy``/``mock``; it validates wiring, not task performance.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np
import torch

from pi05_rlt_rtc import PLANNING_STATE_VERSION, SCHEMA_VERSION
from pi05_rlt_rtc.action_codec.codec import ActionCodec, NormStats
from pi05_rlt_rtc.adapters.feature_reference import FeatureReferenceResult
from pi05_rlt_rtc.adapters.mock_feature import MockFeatureReferenceService
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
from pi05_rlt_rtc.contracts.observation import EnvIDs
from pi05_rlt_rtc.contracts.plan import PlanRequest, PrevPlanContext
from pi05_rlt_rtc.contracts.spaces import ENV_ACTION_DIM, HorizonContract
from pi05_rlt_rtc.evaluation.toy_env import ToyDelayedControlEnv, ToyEnvConfig
from pi05_rlt_rtc.learning.actor import QueueAwareRLTActor
from pi05_rlt_rtc.learning.checkpoint import PolicyBundle, state_dict_sha256
from pi05_rlt_rtc.learning.critic import TwinQWithTargets
from pi05_rlt_rtc.learning.losses import (
    actor_loss,
    bc_warmup_loss,
    build_transition_batch,
    critic_loss,
)
from pi05_rlt_rtc.learning.planning_state import (
    PlanningState,
    PlanningStateSpec,
    build_planning_state,
    make_time_features,
)
from pi05_rlt_rtc.learning.replay import ReplayBuffer
from pi05_rlt_rtc.learning.transition_assembler import DelayedTransitionAssembler
from pi05_rlt_rtc.runtime.queue import ActionQueue
from pi05_rlt_rtc.runtime.scheduler import PlanResult, VirtualDelayScheduler
from pi05_rlt_rtc.rtc.canonical import canonical_velocity_from_native
from pi05_rlt_rtc.rtc.dispatcher import RTCDispatcher
from pi05_rlt_rtc.rtc.prefix_weights import guidance_mask_from_previous_plan

TOY_FEATURE_VERSION_ID = "MOCK-feature-v1"
TOY_NORM_STATS_SHA = "mock-symmetric-norm-stats-v1"
TOY_POLICY_VERSION_ID = "toy-untrained-v1"


class ToyFlowVelocity(torch.nn.Module):
    """Differentiable toy velocity field for exercising the real full_vjp path."""

    def __init__(self, *, scale: float = 0.5) -> None:
        super().__init__()
        self.scale = float(scale)

    def velocity_native(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return -self.scale * x


class MockRTCReferenceBackend:
    """Reference teacher + optional real full_vjp guidance on the toy flow."""

    is_mock = True

    def __init__(
        self,
        *,
        feature_service: MockFeatureReferenceService,
        codec: ActionCodec,
        contract: HorizonContract,
        mode: str = "none",
        num_denoising_steps: int = 5,
    ) -> None:
        self.feature_service = feature_service
        self.codec = codec
        self.contract = contract
        self.mode = mode
        self.dispatcher = RTCDispatcher(num_steps=num_denoising_steps)
        self.velocity = ToyFlowVelocity()
        self._features: dict[int, FeatureReferenceResult] = {}

    def feature_for(self, request_id: int) -> FeatureReferenceResult:
        if request_id not in self._features:
            raise ContractError(f"no cached feature result for request {request_id}")
        return self._features[request_id]

    def remember(self, request_id: int, result: FeatureReferenceResult) -> None:
        """Register an externally computed feature result (evaluation path)."""
        self._features[int(request_id)] = result

    def plan(
        self,
        request: PlanRequest,
        observation: ObservationSnapshot,
        pending_prefix: np.ndarray,
        prev_context: PrevPlanContext,
    ) -> PlanResult:
        result = self.feature_service.extract(request, observation)
        reference = result.reference_env_full
        if reference.ndim == 3:
            reference = reference[0]
        if self.mode in ("full_vjp", "approx") and request.request_id >= 0:
            reference, model_actions = self._guided_reference(request, result, prev_context)
            result = dataclasses.replace(result, reference_env_full=reference.copy(), vla_model_actions=model_actions)
            if reference.ndim == 3:
                reference = reference[0]
        self._features[result.request_id] = result
        submitted = reference[self.contract.suffix_start : self.contract.suffix_end, :].copy()
        return PlanResult(
            submitted=submitted,
            reference_env_full=reference.copy(),
            source="reference_teacher",
            bc_target_plan=submitted.copy(),
            bc_target_valid_mask=np.ones_like(submitted, dtype=np.float32),
            teacher_source="mock_rtc_reference_teacher",
        )

    def _guided_reference(
        self,
        request: PlanRequest,
        result: FeatureReferenceResult,
        prev_context: PrevPlanContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(request.rng_seed)
        x_init = torch.as_tensor(
            rng.standard_normal((1, self.contract.horizon, 32)), dtype=torch.float32
        )
        target = None
        mask = None
        prev_model_actions = None
        if prev_context.prev_env_plan is not None and prev_context.prev_env_plan.shape[0] > 0:
            overlap = min(prev_context.prev_env_plan.shape[0], self.contract.horizon)
            target = torch.zeros_like(x_init)
            encoded = self.codec.encode_env_for_guidance(prev_context.prev_env_plan[:overlap])
            prev_model_actions = torch.as_tensor(encoded, dtype=torch.float32)[None, ...]
            target[:, :overlap] = torch.as_tensor(encoded, dtype=torch.float32)
            mask_np = guidance_mask_from_previous_plan(
                horizon=self.contract.horizon,
                chunk=self.contract.chunk,
                delay=self.contract.delay,
                previous_valid_len=overlap,
                action_valid_mask=self.codec.valid_model_mask,
            )
            mask = torch.as_tensor(mask_np, dtype=torch.float32)[None, ...]
        dispatch = self.dispatcher.sample(
            self.mode,
            x_init=x_init,
            velocity_native=self.velocity.velocity_native,
            velocity_canonical=canonical_velocity_from_native(self.velocity.velocity_native),
            target_model=target,
            guidance_mask=mask,
            prev_model_actions=prev_model_actions,
            delay_steps=self.contract.delay,
        )
        env = self.codec.decode_full_horizon(dispatch.actions.numpy()[0]).actions
        return np.asarray(env, dtype=np.float32)[None, ...], dispatch.actions.detach().cpu().numpy()

    def provenance(self) -> dict[str, Any]:
        return {"mock": True, "rtc_mode": self.mode, "feature": self.feature_service.provenance()}


class MockStateEncoder:
    def __init__(
        self,
        *,
        backend: MockRTCReferenceBackend,
        spec: PlanningStateSpec,
        codec: ActionCodec,
        contract: HorizonContract,
    ) -> None:
        self.backend = backend
        self.spec = spec
        self.codec = codec
        self.contract = contract

    def __call__(
        self,
        *,
        request: PlanRequest,
        observation: ObservationSnapshot,
        pending_prefix: np.ndarray,
        pending_mask: np.ndarray,
        prev_context: PrevPlanContext,
        plan_event: dict[str, Any] | None = None,
    ) -> PlanningState:
        fr = self.backend.feature_for(request.request_id)
        spec = self.spec
        pending = np.zeros((1, spec.max_delay, ENV_ACTION_DIM), dtype=np.float32)
        p_mask = np.zeros((1, spec.max_delay, ENV_ACTION_DIM), dtype=bool)
        active = min(pending_prefix.shape[0], spec.max_delay)
        pending[0, :active] = np.asarray(pending_prefix, dtype=np.float32)[:active]
        p_mask[0, :active] = np.asarray(pending_mask, dtype=bool)[:active]
        prev = np.zeros((1, spec.horizon, ENV_ACTION_DIM), dtype=np.float32)
        prev_mask = np.zeros((1, spec.horizon), dtype=bool)
        last_command = np.zeros((1, ENV_ACTION_DIM), dtype=np.float32)
        if request.last_command is not None:
            last_command[0] = request.last_command
        if prev_context.prev_env_plan is not None and prev_context.prev_env_plan.shape[0] > 0:
            n = min(prev_context.prev_env_plan.shape[0], spec.horizon)
            prev[0, :n] = np.asarray(prev_context.prev_env_plan, dtype=np.float32)[:n]
            prev_mask[0, :n] = True
        ids = EnvIDs(
            env_id=request.env_id,
            episode_id=request.episode_id,
            episode_generation=request.episode_generation,
            policy_version_id=request.policy_version_id,
            feature_version_id=TOY_FEATURE_VERSION_ID,
            schema_version=SCHEMA_VERSION,
            planning_state_version=PLANNING_STATE_VERSION,
        )
        time_features = make_time_features(
            request.capture_tick,
            control_frequency_hz=10,
            delay_ticks=self.contract.delay,
        )
        return build_planning_state(
            spec=spec,
            ids=ids,
            capture_tick=request.capture_tick,
            plan_id=None if plan_event is None or plan_event.get("plan") is None else plan_event["plan"].plan_id,
            z_rl=fr.z_rl,
            proprio=self._proprio(fr.proprio_model, spec),
            pending_prefix=pending,
            pending_mask=p_mask,
            prev_plan_tail=prev,
            prev_plan_tail_mask=prev_mask,
            last_command=last_command,
            reference_full=fr.reference_env_full,
            time_features=time_features,
            norm_stats_sha256=TOY_NORM_STATS_SHA,
        )

    @staticmethod
    def _proprio(proprio: np.ndarray, spec: PlanningStateSpec) -> np.ndarray:
        out = np.zeros((1, spec.proprio_dim), dtype=np.float32)
        n = min(proprio.shape[-1], spec.proprio_dim)
        out[0, :n] = proprio[0, :n]
        return out


class ActorPolicyBackend:
    """Closed-loop policy backend: the queue-aware actor submits the plan.

    Evaluation uses the same fixed-delay queue protocol as collection; only the
    role of the backend changes (actor instead of reference teacher).
    """

    is_mock = True

    def __init__(
        self,
        *,
        backend: MockRTCReferenceBackend,
        encoder: "MockStateEncoder",
        actor: QueueAwareRLTActor,
        contract: HorizonContract,
    ) -> None:
        self.backend = backend
        self.encoder = encoder
        self.actor = actor
        self.contract = contract

    def plan(self, request, observation, pending_prefix, prev_context) -> PlanResult:
        reference_plan = self.backend.plan(request, observation, pending_prefix, prev_context)
        if request.request_id < 0:
            return reference_plan
        feature = self.backend.feature_for(request.request_id)
        state = self.encoder(
            request=request,
            observation=observation,
            pending_prefix=pending_prefix,
            pending_mask=np.ones_like(pending_prefix, dtype=bool),
            prev_context=prev_context,
            plan_event=None,
        )
        ref_suffix = state.reference_suffix_vector(self.contract).reshape(1, -1)
        with torch.no_grad():
            out = self.actor.act(
                state=torch.as_tensor(state.actor_state_vector(), dtype=torch.float32),
                reference_suffix=torch.as_tensor(ref_suffix, dtype=torch.float32),
                deterministic=True,
            )
        submitted = out.actions[0].numpy().astype(np.float32)
        reference = np.asarray(feature.reference_env_full, dtype=np.float32)
        if reference.ndim == 3:
            reference = reference[0]
        return PlanResult(
            submitted=submitted,
            reference_env_full=reference,
            source="actor",
            bc_target_plan=ref_suffix.reshape(self.contract.chunk, ENV_ACTION_DIM),
            teacher_source="rtc_reference_suffix",
        )


@dataclasses.dataclass
class ToyBuild:
    contract: HorizonContract
    spec: PlanningStateSpec
    codec: ActionCodec
    feature_service: MockFeatureReferenceService
    backend: MockRTCReferenceBackend
    encoder: MockStateEncoder
    env: ToyDelayedControlEnv
    queue: ActionQueue


def build_toy(
    *,
    horizon: int = 10,
    chunk: int = 5,
    delay: int = 2,
    z_dim: int = 32,
    rtc_mode: str = "none",
    gamma: float = 0.9,
    seed: int = 0,
) -> ToyBuild:
    contract = HorizonContract(horizon=horizon, chunk=chunk, delay=delay)
    spec = PlanningStateSpec(
        z_dim=z_dim,
        proprio_dim=9,
        env_action_dim=ENV_ACTION_DIM,
        horizon=horizon,
        chunk=chunk,
        max_delay=chunk,
    )
    codec = ActionCodec(NormStats.symmetric(), name="toy_codec")
    feature_service = MockFeatureReferenceService(
        contract=contract, codec=codec, spec_z_dim=z_dim, norm_stats_sha256=TOY_NORM_STATS_SHA
    )
    backend = MockRTCReferenceBackend(
        feature_service=feature_service, codec=codec, contract=contract, mode=rtc_mode
    )
    encoder = MockStateEncoder(backend=backend, spec=spec, codec=codec, contract=contract)
    env = ToyDelayedControlEnv(
        ToyEnvConfig(initial_state=1.5, episode_ticks=max(60, horizon * 6)), env_id="toy0"
    )
    queue = ActionQueue(contract, fallback_policy="stop")
    return ToyBuild(
        contract=contract,
        spec=spec,
        codec=codec,
        feature_service=feature_service,
        backend=backend,
        encoder=encoder,
        env=env,
        queue=queue,
    )


@dataclasses.dataclass
class ToyCollected:
    log: Any
    transitions: list
    replay: ReplayBuffer
    backend: MockRTCReferenceBackend
    encoder: MockStateEncoder
    build: ToyBuild


def collect_toy(
    build: ToyBuild,
    *,
    num_ticks: int = 120,
    planner_latency_ticks: int = 0,
    seed: int = 0,
    gamma: float = 0.9,
    player_capacity: int = 4096,
) -> ToyCollected:
    scheduler = VirtualDelayScheduler(
        contract=build.contract,
        queue=build.queue,
        backend=build.backend,
        env=build.env,
        planner_latency_ticks=planner_latency_ticks,
        seed=seed,
        feature_version_id=TOY_FEATURE_VERSION_ID,
        policy_version_id=TOY_POLICY_VERSION_ID,
        num_denoising_steps=5,
        rtc_mode=build.backend.mode,
    )
    log = scheduler.run(num_ticks)
    assembler = DelayedTransitionAssembler(
        contract=build.contract,
        state_encoder=build.encoder,
        gamma_per_tick=gamma,
    )
    assembler.ingest_run(log.events)
    transitions = assembler.finalize()
    replay = ReplayBuffer(
        capacity=player_capacity,
        feature_version_id=TOY_FEATURE_VERSION_ID,
        norm_stats_sha256=TOY_NORM_STATS_SHA,
    )
    for transition in transitions:
        replay.push(transition)
    return ToyCollected(
        log=log, transitions=transitions, replay=replay, backend=build.backend, encoder=build.encoder, build=build
    )


@dataclasses.dataclass
class ToyTrainingReport:
    warmup_losses: list[float]
    update_losses: list[float]
    actor_losses: list[float]
    critic_losses: list[float]
    metrics: dict[str, float]


def train_toy_actor_critic(
    collected: ToyCollected,
    *,
    updates: int = 200,
    batch_size: int = 32,
    warmup_updates: int = 50,
    actor_lr: float = 1e-3,
    critic_lr: float = 1e-3,
    lambda_q: float = 0.2,
    lambda_bc: float = 1.0,
    reference_dropout: float = 0.5,
    gamma: float = 0.9,
    seed: int = 0,
) -> tuple[QueueAwareRLTActor, TwinQWithTargets, ToyTrainingReport]:
    torch.manual_seed(seed)
    spec = collected.encoder.spec
    contract = collected.encoder.contract
    actor = QueueAwareRLTActor(spec, hidden_dims=(64, 64), reference_hidden_dims=(32,))
    critic = TwinQWithTargets(
        sum(d for _, d in spec.critic_state_layout()), spec.chunk * spec.env_action_dim, hidden_dims=(64, 64)
    )
    actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = torch.optim.Adam(critic.online.parameters(), lr=critic_lr)
    generator = np.random.default_rng(seed)
    torch_gen = torch.Generator().manual_seed(seed)

    learnable = collected.replay.learnable_transitions()
    if len(learnable) < 2:
        raise ContractError(
            f"toy collection produced only {len(learnable)} learnable transitions; "
            "the queue/assembler wiring is wrong, not the optimizer"
        )

    warmup_losses: list[float] = []
    for _ in range(warmup_updates):
        batch = build_transition_batch(
            collected.replay.sample(batch_size, generator=generator), spec, contract=contract
        )
        loss, _ = bc_warmup_loss(actor, batch)
        actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        actor_opt.step()
        warmup_losses.append(float(loss.detach().item()))

    critic_losses: list[float] = []
    actor_losses: list[float] = []
    for i in range(updates):
        batch = build_transition_batch(
            collected.replay.sample(batch_size, generator=generator), spec, contract=contract
        )
        loss, _ = critic_loss(
            critic, actor, batch, generator=torch_gen if i % 2 == 0 else None
        )
        critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        critic_opt.step()
        critic.soft_update(0.05)
        critic_losses.append(float(loss.detach().item()))

        a_loss, _ = actor_loss(
            actor,
            critic,
            batch,
            lambda_q=lambda_q,
            lambda_bc=lambda_bc,
            reference_dropout_prob=reference_dropout,
            generator=torch_gen,
        )
        actor_opt.zero_grad(set_to_none=True)
        a_loss.backward()
        actor_opt.step()
        actor_losses.append(float(a_loss.detach().item()))

    metrics = {
        "warmup_loss_start": warmup_losses[0] if warmup_losses else float("nan"),
        "warmup_loss_end": warmup_losses[-1] if warmup_losses else float("nan"),
        "critic_loss_start": critic_losses[0] if critic_losses else float("nan"),
        "critic_loss_end": critic_losses[-1] if critic_losses else float("nan"),
        "actor_loss_start": actor_losses[0] if actor_losses else float("nan"),
        "actor_loss_end": actor_losses[-1] if actor_losses else float("nan"),
        "learnable_transitions": len(learnable),
    }
    return actor, critic, ToyTrainingReport(
        warmup_losses=warmup_losses,
        update_losses=critic_losses,
        actor_losses=actor_losses,
        critic_losses=critic_losses,
        metrics=metrics,
    )


@dataclasses.dataclass
class ToyEvalReport:
    episodes: int
    successes: int
    mean_final_abs_state: float
    mean_reward: float
    random_mean_final_abs_state: float
    failures: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def evaluate_toy_actor(
    *,
    build: ToyBuild,
    actor: QueueAwareRLTActor,
    encoder: MockStateEncoder | None = None,
    episodes: int = 5,
    seed: int = 0,
    contract: HorizonContract | None = None,
    ticks_per_episode: int = 60,
) -> ToyEvalReport:
    """Closed-loop deterministic evaluation under the same fixed-D queue protocol."""
    from pi05_rlt_rtc.runtime.queue import ActionQueue

    contract = contract or build.contract
    encoder = encoder or build.encoder
    successes = 0
    finals: list[float] = []
    rewards: list[float] = []
    failures: list[dict[str, Any]] = []
    for episode in range(episodes):
        eval_feature_service = MockFeatureReferenceService(
            contract=contract,
            codec=build.codec,
            spec_z_dim=build.spec.z_dim,
            norm_stats_sha256=TOY_NORM_STATS_SHA,
        )
        eval_backend = MockRTCReferenceBackend(
            feature_service=eval_feature_service, codec=build.codec, contract=contract, mode=build.backend.mode
        )
        eval_encoder = MockStateEncoder(
            backend=eval_backend, spec=build.spec, codec=build.codec, contract=contract
        )
        policy_backend = ActorPolicyBackend(
            backend=eval_backend,
            encoder=eval_encoder,
            actor=actor,
            contract=contract,
        )
        env = ToyDelayedControlEnv(
            ToyEnvConfig(initial_state=1.5, episode_ticks=400),
            env_id=f"toy-eval-{seed}-{episode}",
        )
        queue = ActionQueue(contract, fallback_policy="stop")
        scheduler = VirtualDelayScheduler(
            contract=contract,
            queue=queue,
            backend=policy_backend,
            env=env,
            planner_latency_ticks=0,
            seed=seed + episode,
            env_id=f"toy-eval-{seed}-{episode}",
            feature_version_id=TOY_FEATURE_VERSION_ID,
            policy_version_id="toy-trained-policy",
            rtc_mode=build.backend.mode,
        )
        log = scheduler.run(ticks_per_episode)
        tick_events = [e for e in log.events if e["kind"] == "tick"]
        terminal_index = next(
            (i for i, e in enumerate(tick_events) if e.get("terminated")), None
        )
        if terminal_index is not None:
            window = tick_events[: terminal_index + 1]
            reached = True
        else:
            window = tick_events
            reached = False
        final_event = window[-1] if window else None
        final_state = (
            abs(float(final_event["observation_after"].states[0])) if final_event else float("nan")
        )
        total_reward = float(sum(e.get("reward", 0.0) for e in window))
        finals.append(final_state)
        rewards.append(total_reward)
        successes += int(reached)
        if not reached:
            failures.append(
                {"episode": episode, "final_abs_state": final_state, "ticks": len(window)}
            )

    from pi05_rlt_rtc.evaluation.toy_env import RandomPolicy

    random_finals: list[float] = []
    for episode in range(episodes):
        env = ToyDelayedControlEnv(
            ToyEnvConfig(initial_state=1.5, episode_ticks=400), env_id=f"toy-random-{seed}-{episode}"
        )
        obs = env.reset(seed=seed + episode)
        policy = RandomPolicy(seed=seed + episode)
        for _ in range(ticks_per_episode):
            outcome = env.step(policy.act(obs.states))
            obs = outcome.observation
            if outcome.terminated:
                break
        random_finals.append(abs(float(obs.states[0])))

    return ToyEvalReport(
        episodes=episodes,
        successes=successes,
        mean_final_abs_state=float(np.mean(finals)),
        mean_reward=float(np.mean(rewards)),
        random_mean_final_abs_state=float(np.mean(random_finals)),
        failures=failures,
    )


def export_toy_bundle(
    *,
    build: ToyBuild,
    actor: QueueAwareRLTActor,
    critic: TwinQWithTargets,
    out_dir: str,
) -> PolicyBundle:
    from pathlib import Path

    bundle = PolicyBundle(
        spec=build.spec,
        contract=build.contract,
        actor_state={k: v.detach() for k, v in actor.state_dict().items()},
        critic_state={k: v.detach() for k, v in critic.state_dict().items()},
        feature_version_id=TOY_FEATURE_VERSION_ID,
        norm_stats_sha256=TOY_NORM_STATS_SHA,
        policy_version_id=TOY_POLICY_VERSION_ID,
        training_counts={"toy": 1},
        actor_hidden_dims=tuple(actor.hidden_dims),
        actor_reference_hidden_dims=tuple(actor.reference_hidden_dims),
        codec_name=build.codec.name,
        extra={"mock": True, "warning": "toy artifact; not a trained robot policy",
               "rtc_mode": build.backend.mode, "num_denoising_steps": build.backend.dispatcher.num_steps,
               "control_frequency_hz": 10},
    )
    bundle.save(Path(out_dir) / "toy_actor.pt")
    import json

    (Path(out_dir) / "deployment_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "planning_state_version": PLANNING_STATE_VERSION,
                "mock": True,
                "actor_state_sha256": bundle.actor_state_sha256,
                "feature_version_id": TOY_FEATURE_VERSION_ID,
                "norm_stats_sha256": TOY_NORM_STATS_SHA,
                "contract": dataclasses.asdict(build.contract),
                "spec": dataclasses.asdict(build.spec),
                "codec": build.codec.provenance(),
                "generated_unix_seconds": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return bundle
