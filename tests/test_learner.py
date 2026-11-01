"""Delayed replay and learner tests T47-T60 (CPU; toy/mock where unavoidable)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pi05_rlt_rtc.contracts.errors import ContractError, SchemaMismatchError
from pi05_rlt_rtc.contracts.spaces import HorizonContract
from pi05_rlt_rtc.contracts.transition import compute_td_target, discounted_return
from pi05_rlt_rtc.learning.planning_state import PlanningStateSpec
from pi05_rlt_rtc.learning.replay import ReplayBuffer
from pi05_rlt_rtc.learning.transition_assembler import DelayedTransitionAssembler
from pi05_rlt_rtc.runtime.scheduler import NumberedReferencePlanner, VirtualDelayScheduler
from tests.helpers import (
    RecordingEnv,
    make_queue,
    make_zero_state_encoder,
)


def _spec(contract):
    return PlanningStateSpec.from_contract(contract, z_dim=16)


def _collect(contract, *, latency=0, num_ticks=10, env=None, feature_id="feature-v1"):
    queue = make_queue(contract)
    env = env or RecordingEnv()
    scheduler = VirtualDelayScheduler(
        contract=contract,
        queue=queue,
        backend=NumberedReferencePlanner(contract),
        env=env,
        planner_latency_ticks=latency,
        seed=0,
        feature_version_id=feature_id,
        stop_on_safety=False,
    )
    log = scheduler.run(num_ticks)
    spec = _spec(contract)
    assembler = DelayedTransitionAssembler(
        contract=contract,
        state_encoder=make_zero_state_encoder(spec, contract, feature_id=feature_id),
        gamma_per_tick=0.9,
    )
    assembler.ingest_run(log.events)
    transitions = assembler.finalize()
    return queue, log, spec, transitions


def test_t47_submitted_plan_differs_from_executed_trace():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    queue, log, spec, transitions = _collect(contract)
    tr = [t for t in transitions if t.request_id == 0][0]
    assert tr.submitted_plan.shape == (5, 8)
    assert tr.executed_trace.shape == (5, 8)
    assert tr.duration_ticks == 5
    # executed = P0 || U0[:3]; it must not equal the submitted plan
    assert not np.allclose(tr.executed_trace.reshape(-1), tr.submitted_plan.reshape(-1))
    assert np.allclose(tr.executed_trace[0], 0.9, atol=1e-6)
    assert np.allclose(tr.executed_trace[1], 0.9001, atol=1e-6)
    assert np.allclose(tr.executed_trace[2], 0.1, atol=1e-6)
    assert np.allclose(tr.submitted_plan[0], 0.1, atol=1e-6)


def test_t48_next_pending_is_last_d_of_submitted_plan():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    queue, log, spec, transitions = _collect(contract)
    tr = [t for t in transitions if t.request_id == 0][0]
    expected = tr.submitted_plan[3:5]
    next_state = tr.next_planning_state
    assert next_state is not None
    assert next_state.capture_tick == 5
    assert np.allclose(next_state.pending_prefix[0, :2], expected, atol=1e-6)
    assert next_state.pending_mask[0, :2].all()
    assert not next_state.pending_mask[0, 2:].any()


def test_t49_discounted_return_exact_example():
    value = discounted_return(np.asarray([1.0, 2.0, 3.0]), 0.9, duration_ticks=3)
    assert value == pytest.approx(5.23, abs=1e-9)
    # container padding beyond K must not change the result
    padded = discounted_return(
        np.asarray([1.0, 2.0, 3.0, 0.0, 0.0]), 0.9, duration_ticks=3
    )
    assert padded == pytest.approx(5.23, abs=1e-9)


def test_t50_td_target_exact_example():
    target = compute_td_target(5.23, 0.9, 3, 4.0, bootstrap_allowed=True)
    assert target == pytest.approx(8.146, abs=1e-9)
    terminal = compute_td_target(5.23, 0.9, 3, 4.0, bootstrap_allowed=False)
    assert terminal == pytest.approx(5.23, abs=1e-9)
    with pytest.raises(ContractError):
        compute_td_target(5.23, 0.9, 3, float("nan"), bootstrap_allowed=True)


def test_t51_gamma_exponent_uses_real_k_not_container_width():
    rewards = np.asarray([1.0, 2.0, 3.0, 5.0, 5.0])
    value = discounted_return(rewards, 0.9, duration_ticks=3)
    assert value == pytest.approx(5.23, abs=1e-9)
    # trailing padding excluded via an explicit valid mask as well
    value2 = discounted_return(
        rewards, 0.9, duration_ticks=5, valid_mask=np.asarray([True, True, True, False, False])
    )
    assert value2 == pytest.approx(5.23, abs=1e-9)


class _TerminatingEnv(RecordingEnv):
    def __init__(self, *, terminate_at_step: int) -> None:
        super().__init__()
        self._limit = terminate_at_step
        self._step = 0

    def reset(self, *, seed: int = 0):
        self._step = 0
        return super().reset(seed=seed)

    def step(self, action):
        import dataclasses

        outcome = super().step(action)
        self._step += 1
        return dataclasses.replace(outcome, terminated=self._step >= self._limit)


def test_t52_terminal_observation_does_not_bootstrap_from_reset_state():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    env = _TerminatingEnv(terminate_at_step=3)
    queue, log, spec, transitions = _collect(contract, num_ticks=12, env=env)
    tr = [t for t in transitions if t.request_id == 0][0]
    assert tr.terminated is True
    assert tr.next_planning_state is None
    assert tr.bootstrap_allowed is False
    assert tr.duration_ticks == 3
    assert tr.td_target(100.0) == pytest.approx(tr.reward_sum, abs=1e-9)
    assert tr.is_learnable()


def test_t53_reference_dropout_touches_only_actor_reference_branch():
    from pi05_rlt_rtc.learning.actor import QueueAwareRLTActor
    from pi05_rlt_rtc.learning.losses import TransitionBatch

    spec = PlanningStateSpec.from_contract(HorizonContract(10, 5, 2), z_dim=16)
    torch.manual_seed(0)
    actor = QueueAwareRLTActor(spec, hidden_dims=(32,), reference_hidden_dims=(16,))
    state = torch.randn(4, sum(d for _, d in spec.actor_state_layout()))
    ref = torch.randn(4, spec.chunk * spec.env_action_dim)
    torch.manual_seed(1)
    with_dropout = actor.act(
        state=state, reference_suffix=ref, deterministic=True, apply_reference_dropout=True,
        reference_dropout_prob=1.0,
    )
    no_dropout = actor.act(
        state=state, reference_suffix=ref, deterministic=True, apply_reference_dropout=False,
    )
    assert with_dropout.reference_dropped.all()
    assert not no_dropout.reference_dropped.any()
    assert not torch.allclose(with_dropout.actions, no_dropout.actions)

    # BC target and critic state are untouched by the actor's dropout flag.
    target = torch.randint(0, 2, (4, spec.chunk * spec.env_action_dim)).float()
    batch = TransitionBatch(
        actor_state=state,
        critic_state=torch.randn(4, sum(d for _, d in spec.critic_state_layout())),
        curr_reference_suffix=ref,
        next_actor_state=state,
        next_critic_state=torch.randn(4, sum(d for _, d in spec.critic_state_layout())),
        next_reference_suffix=ref,
        actions=target,
        bc_target=target.clone(),
        bc_valid_mask=torch.ones_like(target),
        reward_sum=torch.zeros(4, 1),
        bootstrap_discount=torch.zeros(4, 1),
        learnable=torch.ones(4, 1),
    )
    before = batch.bc_target.clone()
    from pi05_rlt_rtc.learning.critic import TwinQWithTargets
    from pi05_rlt_rtc.learning.losses import actor_loss, critic_loss

    critic = TwinQWithTargets(
        sum(d for _, d in spec.critic_state_layout()), spec.chunk * spec.env_action_dim, hidden_dims=(32,)
    )
    _, _ = actor_loss(
        actor, critic, batch, lambda_q=0.0, lambda_bc=1.0, reference_dropout_prob=1.0
    )
    assert torch.equal(batch.bc_target, before)
    with torch.no_grad():
        before_q = critic.online(batch.critic_state, batch.actions).clone()
    _, _ = critic_loss(critic, actor, batch)
    with torch.no_grad():
        after_q = critic.online(batch.critic_state, batch.actions).clone()
    assert torch.equal(before_q, after_q)


def test_t54_bc_warmup_compares_bounded_deployed_actions():
    from pi05_rlt_rtc.learning.actor import QueueAwareRLTActor
    from pi05_rlt_rtc.learning.losses import TransitionBatch, bc_warmup_loss

    spec = PlanningStateSpec.from_contract(HorizonContract(10, 5, 2), z_dim=16)
    torch.manual_seed(2)
    actor = QueueAwareRLTActor(spec, hidden_dims=(32,), reference_hidden_dims=(16,))
    with torch.no_grad():
        actor.head.bias.fill_(8.0)  # raw logits far outside tanh bounds
    state = torch.randn(2, sum(d for _, d in spec.actor_state_layout()))
    ref = torch.zeros(2, spec.chunk * spec.env_action_dim)
    target = torch.full((2, spec.chunk * spec.env_action_dim), 0.5)
    batch = TransitionBatch(
        actor_state=state,
        critic_state=torch.randn(2, sum(d for _, d in spec.critic_state_layout())),
        curr_reference_suffix=ref,
        next_actor_state=state,
        next_critic_state=torch.randn(2, sum(d for _, d in spec.critic_state_layout())),
        next_reference_suffix=ref,
        actions=target,
        bc_target=target,
        bc_valid_mask=torch.ones_like(target),
        reward_sum=torch.zeros(2, 1),
        bootstrap_discount=torch.zeros(2, 1),
        learnable=torch.ones(2, 1),
    )
    loss, _ = bc_warmup_loss(actor, batch)
    out = actor.act(state=state, reference_suffix=ref, deterministic=True)
    assert torch.allclose(out.actions, actor.bounded_actions(out.raw))
    assert (out.actions.abs() <= 1.0 + 1e-6).all()
    # the loss must equal MSE against the tanh action, not the raw logits
    expected = ((out.actions.reshape(target.shape[0], -1) - target) ** 2).mean()
    assert float(loss.detach()) == pytest.approx(float(expected.detach()), abs=1e-6)
    raw_mse = ((out.raw.reshape(target.shape[0], -1) - target) ** 2).mean()
    assert not np.isclose(float(loss.detach()), float(raw_mse.detach()))


def test_t55_actor_gets_q_gradients_while_q_params_stay_frozen():
    from pi05_rlt_rtc.learning.actor import QueueAwareRLTActor
    from pi05_rlt_rtc.learning.critic import TwinQWithTargets
    from pi05_rlt_rtc.learning.losses import TransitionBatch, actor_loss

    spec = PlanningStateSpec.from_contract(HorizonContract(10, 5, 2), z_dim=16)
    torch.manual_seed(3)
    actor = QueueAwareRLTActor(spec, hidden_dims=(32,), reference_hidden_dims=(16,))
    critic = TwinQWithTargets(
        sum(d for _, d in spec.critic_state_layout()), spec.chunk * spec.env_action_dim, hidden_dims=(32,)
    )
    q_before = [p.detach().clone() for p in critic.online.parameters()]
    batch = TransitionBatch(
        actor_state=torch.randn(4, sum(d for _, d in spec.actor_state_layout())),
        critic_state=torch.randn(4, sum(d for _, d in spec.critic_state_layout())),
        curr_reference_suffix=torch.randn(4, spec.chunk * spec.env_action_dim),
        next_actor_state=torch.randn(4, sum(d for _, d in spec.actor_state_layout())),
        next_critic_state=torch.randn(4, sum(d for _, d in spec.critic_state_layout())),
        next_reference_suffix=torch.randn(4, spec.chunk * spec.env_action_dim),
        actions=torch.randn(4, spec.chunk * spec.env_action_dim),
        bc_target=torch.randn(4, spec.chunk * spec.env_action_dim),
        bc_valid_mask=torch.ones(4, spec.chunk * spec.env_action_dim),
        reward_sum=torch.ones(4, 1),
        bootstrap_discount=0.9 * torch.ones(4, 1),
        learnable=torch.ones(4, 1),
    )
    loss, _ = actor_loss(actor, critic, batch, lambda_q=1.0, lambda_bc=0.5)
    for p in critic.parameters():
        p.grad = None
    for p in actor.parameters():
        p.grad = None
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
    assert all(p.grad is None for p in critic.parameters())
    for old, new in zip(q_before, critic.online.parameters()):
        assert torch.equal(old, new.detach())


def test_t56_next_state_must_come_from_the_next_capture():
    import dataclasses

    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    spec = _spec(contract)
    queue = make_queue(contract)
    scheduler = VirtualDelayScheduler(
        contract=contract,
        queue=queue,
        backend=NumberedReferencePlanner(contract),
        env=RecordingEnv(),
        planner_latency_ticks=0,
        seed=0,
        feature_version_id="feature-v1",
    )
    log = scheduler.run(10)
    # normal alignment
    assembler = DelayedTransitionAssembler(
        contract=contract,
        state_encoder=make_zero_state_encoder(spec, contract, feature_id="feature-v1"),
        gamma_per_tick=0.9,
    )
    assembler.ingest_run(log.events)
    tr = [t for t in assembler.finalize() if t.request_id == 0][0]
    assert tr.next_planning_state is not None
    assert tr.next_planning_state.capture_tick == 5
    assert tr.next_planning_state.ids.feature_version_id == "feature-v1"

    # mutated next feature version => censored, no fabricated bootstrap
    events = list(log.events)
    for event in events:
        if event["kind"] == "request" and event["request"].request_id == 1:
            event["request"] = dataclasses.replace(event["request"], feature_version_id="feature-v2")
    assembler2 = DelayedTransitionAssembler(
        contract=contract,
        state_encoder=make_zero_state_encoder(spec, contract, feature_id="feature-v1"),
        gamma_per_tick=0.9,
    )
    assembler2.ingest_run(events)
    tr2 = [t for t in assembler2.finalize() if t.request_id == 0][0]
    assert tr2.censored and tr2.censor_reason == "next_feature_version_mismatch"
    assert tr2.next_planning_state is None
    assert not tr2.is_learnable()


def test_t57_deadline_miss_and_intervention_are_filtered_but_logged():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    queue, log, spec, transitions = _collect(contract, latency=3, num_ticks=6)
    tr = [t for t in transitions if t.request_id == 0][0]
    assert tr.deadline_miss
    assert not tr.is_learnable()
    assert tr.censored and tr.censor_reason
    replay = ReplayBuffer(
        capacity=16, feature_version_id="feature-v1", norm_stats_sha256="norm-v1"
    )
    for transition in transitions:
        replay.push(transition)
    assert replay.counts()["censored_seen"] >= 1
    assert replay.learnable_transitions() == []
    with pytest.raises(ContractError):
        replay.sample(2, generator=np.random.default_rng(0))
    assert [e["event"] for e in queue.ledger.events if e["event"] == "deadline_miss"]


def test_t58_replay_save_restore_counts_and_sampling(tmp_path):
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    queue, log, spec, transitions = _collect(contract)
    learnable = [t for t in transitions if t.is_learnable()]
    assert learnable
    replay = ReplayBuffer(
        capacity=64, feature_version_id="feature-v1", norm_stats_sha256="norm-v1"
    )
    for transition in learnable:
        replay.push(transition)
    path = replay.save(tmp_path / "replay.pt")
    restored = ReplayBuffer.load(path)
    assert len(restored) == len(replay)
    assert restored.counts()["size"] == replay.counts()["size"]
    assert restored.counts()["learnable"] == replay.counts()["learnable"]
    assert restored.stats.pushed == replay.stats.pushed
    assert restored.schema_version == replay.schema_version
    sample = restored.sample(2, generator=np.random.default_rng(0))
    assert len(sample) == 2 and all(t.is_learnable() for t in sample)


def test_t59_old_synchronous_replay_is_rejected(tmp_path):
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    queue, log, spec, transitions = _collect(contract)
    replay = ReplayBuffer(
        capacity=16, feature_version_id="feature-v1", norm_stats_sha256="norm-v1"
    )
    learnable = [t for t in transitions if t.is_learnable()]
    assert learnable
    replay.push(learnable[0])
    path = replay.save(tmp_path / "replay.pt")

    restored = ReplayBuffer.load(path)
    with pytest.raises(SchemaMismatchError):
        restored.require_compatible(
            schema_version="pi05-rlt-rtc.schema.v0",
            feature_version_id="feature-v1",
            norm_stats_sha256="norm-v1",
        )
    restored.feature_version_id = "feature-v2"
    with pytest.raises(SchemaMismatchError):
        restored.push(learnable[0])


def test_t60_toy_delayed_control_learning_direction():
    from pi05_rlt_rtc.evaluation.toy_pipeline import (
        build_toy,
        collect_toy,
        evaluate_toy_actor,
        train_toy_actor_critic,
    )

    build = build_toy(horizon=10, chunk=5, delay=2, z_dim=24, seed=0)
    collected = collect_toy(build, num_ticks=200, seed=0, gamma=0.9)
    assert len(collected.replay.learnable_transitions()) >= 20

    actor, critic, report = train_toy_actor_critic(
        collected,
        updates=150,
        batch_size=32,
        warmup_updates=80,
        seed=0,
        gamma=0.9,
    )
    assert report.metrics["warmup_loss_end"] < report.metrics["warmup_loss_start"] * 0.7
    assert report.metrics["critic_loss_end"] < report.metrics["critic_loss_start"]
    assert report.metrics["critic_loss_end"] < 1.0

    evaluation = evaluate_toy_actor(build=build, actor=actor, episodes=5, seed=0)
    assert evaluation.successes >= 3
    assert evaluation.mean_final_abs_state < evaluation.random_mean_final_abs_state
    assert evaluation.mean_final_abs_state < 0.2

    # analytic directional Q check on the trained critic: for states where the
    # teacher action is not clipped, Q(x, a*) must exceed Q(x, random a).
    from pi05_rlt_rtc.learning.losses import build_transition_batch

    transitions = collected.replay.learnable_transitions()
    batch = build_transition_batch(transitions[:8], collected.encoder.spec, contract=build.contract)
    state = batch.next_critic_state
    best = -torch.clamp(state[:, 0:1], -1, 1).expand(-1, collected.encoder.spec.chunk * 8)
    random_actions = torch.randn_like(best) * 0.5
    with torch.no_grad():
        q_best = critic.online.q_min(state, best)
        q_random = critic.online.q_min(state, random_actions)
    assert float((q_best >= q_random).float().mean()) >= 0.6
