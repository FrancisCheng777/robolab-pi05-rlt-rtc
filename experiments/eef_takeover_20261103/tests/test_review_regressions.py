"""Regression cases found by the implementation-pack review (CPU only)."""

import dataclasses

import numpy as np
import pytest
import torch

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.spaces import HorizonContract
from pi05_rlt_rtc.contracts.transition import discounted_return
from pi05_rlt_rtc.learning.losses import build_transition_batch
from pi05_rlt_rtc.learning.replay import ReplayBuffer
from pi05_rlt_rtc.learning.transition_assembler import DelayedTransitionAssembler
from tests.helpers import make_final, make_queue, make_request, make_submitted, make_zero_state_encoder, warm_start
from tests.test_learner import _collect


@pytest.mark.parametrize("fallback", ["stop", "hold_last", "zero_delta"])
def test_intervention_stops_even_with_motion_fallback_configured(contract, fallback):
    queue = make_queue(contract, fallback=fallback)
    warm_start(queue, contract)
    queue.consume(0)
    queue.intervene(1, "operator_estop")
    result = queue.consume(1)
    assert result.safe_stop and result.action is None


def test_old_episode_duplicate_cannot_ack_a_new_episode_plan(contract):
    queue = make_queue(contract)
    warm_start(queue, contract)
    old_request = make_request(contract, request_id=0, capture_tick=0)
    old_submitted = make_submitted(contract, request=old_request, base_value=0.1)
    old_plan = make_final(contract, request=old_request, pending=queue.snapshot_pending(0)[0], submitted=old_submitted)
    assert queue.commit_plan(old_plan, old_submitted, now_tick=0).accepted
    queue.reset(env_id="env0", episode_id="env0-ep1", start_tick=0)
    warm_start(queue, contract)
    new_request = dataclasses.replace(old_request, episode_generation=queue.episode_generation, queue_generation=queue.queue_generation)
    new_submitted = make_submitted(contract, request=new_request, base_value=0.3)
    new_plan = make_final(contract, request=new_request, pending=queue.snapshot_pending(0)[0], submitted=new_submitted)
    assert queue.commit_plan(new_plan, new_submitted, now_tick=0).accepted
    assert not queue.commit_plan(old_plan, old_submitted, now_tick=0).accepted


def test_plan_composition_rejects_mismatched_request_identity(contract):
    request = make_request(contract, request_id=0, capture_tick=0)
    submitted = make_submitted(contract, request=request, base_value=0.1)
    with pytest.raises(ContractError):
        make_final(contract, request=request, pending=np.zeros((2, 8)), submitted=dataclasses.replace(submitted, request_id=1))


def test_queue_rejects_disagreement_between_final_and_submitted(contract):
    queue = make_queue(contract)
    warm_start(queue, contract)
    request = make_request(contract, request_id=0, capture_tick=0)
    submitted = make_submitted(contract, request=request, base_value=0.1)
    plan = make_final(contract, request=request, pending=queue.snapshot_pending(0)[0], submitted=submitted)
    altered = dataclasses.replace(submitted, actions=np.zeros_like(submitted.actions))
    assert not queue.commit_plan(plan, altered, now_tick=0).accepted


def test_consumed_tick_cannot_be_committed_again():
    contract = HorizonContract(horizon=10, chunk=5, delay=0)
    queue = make_queue(contract, fallback="zero_delta")
    queue.consume(0)
    request = make_request(contract, request_id=0, capture_tick=0)
    submitted = make_submitted(contract, request=request, base_value=0.1)
    plan = make_final(contract, request=request, pending=np.zeros((0, 8)), submitted=submitted)
    assert not queue.commit_plan(plan, submitted, now_tick=0).accepted


def test_bc_target_is_original_reference_not_actor_submission(contract):
    _, log, _, transitions = _collect(contract)
    tr = transitions[0]
    result = next(e["result"] for e in log.events if e["kind"] == "plan" and e["request_id"] == 0)
    expected = result.reference_env_full[2:7]
    assert not np.allclose(expected, tr.submitted_plan)
    np.testing.assert_array_equal(tr.bc_target_plan, expected)


def test_learner_requires_execution_contract_instead_of_using_padding_capacity(contract):
    _, _, spec, transitions = _collect(contract)
    with pytest.raises(ContractError, match="contract"):
        build_transition_batch([transitions[0]], spec)


def test_learner_rejects_delay_that_disagrees_with_saved_queue(contract):
    _, _, spec, transitions = _collect(contract)
    with pytest.raises(ContractError):
        build_transition_batch([transitions[0]], spec, contract=HorizonContract(horizon=10, chunk=5, delay=5))


@pytest.mark.parametrize("delay", [0, 2, 5])
def test_learner_uses_active_delay_for_both_reference_branches(delay):
    contract = HorizonContract(horizon=10, chunk=5, delay=delay)
    _, _, spec, transitions = _collect(contract)
    tr = transitions[0]
    for state in (tr.planning_state, tr.next_planning_state):
        state.reference_full[:] = np.arange(10, dtype=np.float32)[None, :, None]
    batch = build_transition_batch([tr], spec, contract=contract)
    expected = np.broadcast_to(np.arange(delay, delay + 5)[:, None], (5, 8)).reshape(1, -1)
    np.testing.assert_array_equal(batch.curr_reference_suffix.numpy(), expected)
    np.testing.assert_array_equal(batch.next_reference_suffix.numpy(), expected)


def test_missing_tick_is_censored_without_crashing_assembly(contract):
    _, log, spec, _ = _collect(contract)
    assembler = DelayedTransitionAssembler(contract=contract, state_encoder=make_zero_state_encoder(spec, contract), gamma_per_tick=0.9)
    assembler.ingest_run([e for e in log.events if not (e["kind"] == "tick" and e["tick"] == 2)])
    tr = assembler.finalize()[0]
    assert tr.censored and tr.censor_reason == "missing_tick_record:2"
    assert not tr.is_learnable()


def test_masked_nonfinite_padding_does_not_poison_reward():
    assert discounted_return(np.array([1., 2., np.nan, np.inf]), 0.9, duration_ticks=2) == pytest.approx(2.8)


def test_replay_retains_capacity_after_reload(contract, tmp_path):
    _, _, _, transitions = _collect(contract)
    buffer = ReplayBuffer(capacity=100, feature_version_id="feature-v1", norm_stats_sha256="norm-v1")
    buffer.push(transitions[0])
    restored = ReplayBuffer.load(buffer.save(tmp_path / "replay.pt"))
    assert restored.capacity == 100
    restored.push(transitions[0])
    assert len(restored) == 2


def test_snapshot_owns_camera_and_state_buffers():
    from tests.helpers import RecordingEnv
    original = RecordingEnv().reset()
    state = np.zeros(9, np.float32)
    pixels = np.zeros((4, 4, 3), np.uint8)
    obs = dataclasses.replace(original, states=state, images={"main": pixels, "wrist": pixels})
    state[:] = 42
    pixels[:] = 255
    assert np.count_nonzero(obs.states) == 0
    assert np.count_nonzero(obs.camera("main")) == 0


def test_environment_cannot_mutate_execution_history(contract):
    queue = make_queue(contract)
    warm_start(queue, contract)
    command = queue.consume(0).action
    command[:] = 0  # a driver reuses/mutates its input buffer
    np.testing.assert_allclose(queue.snapshot_pending(0)[0][0], 0.9)
    np.testing.assert_allclose(queue.ledger.entries[-1].command, 0.9)


def test_norm_stats_reject_nonfinite_zscore_parameters():
    from pi05_rlt_rtc.action_codec import NormStats, NormMethod
    with pytest.raises(ContractError):
        NormStats(NormMethod.ZSCORE, 32, mean=np.full(32, np.nan), std=np.ones(32))


def test_codec_can_decode_unbatched_full_horizon(codec):
    result = codec.decode_full_horizon(np.zeros((10, 32), np.float32))
    assert result.actions.shape == (10, 8)
    assert result.valid_mask.shape == (1, 32)


def test_episode_split_rejects_duplicate_ids():
    from pi05_rlt_rtc.data.auditor import episode_split
    with pytest.raises(ContractError):
        episode_split(["a", "b", "a", "c"])


def test_feature_result_rejects_mixed_batch_sizes(contract):
    from pi05_rlt_rtc.evaluation.toy_pipeline import build_toy
    build = build_toy(z_dim=16)
    request = make_request(contract, request_id=0, capture_tick=0)
    fr = build.backend.feature_service.extract(request, build.env.reset())
    fr.reference_env_full = np.repeat(fr.reference_env_full, 2, axis=0)
    with pytest.raises(ContractError):
        fr.validate(contract)


def test_guided_reference_in_state_matches_returned_plan():
    from pi05_rlt_rtc.evaluation.toy_pipeline import build_toy, collect_toy
    build = build_toy(z_dim=16, rtc_mode="full_vjp")
    collected = collect_toy(build, num_ticks=15)
    assert collected.replay.learnable_transitions()
    for tr in collected.transitions:
        if tr.planning_state is not None and tr.bc_target_plan is not None:
            expected = tr.planning_state.reference_suffix_vector(build.contract).reshape(5, 8)
            np.testing.assert_allclose(expected, tr.bc_target_plan)


def test_planning_state_retains_last_executed_command():
    from pi05_rlt_rtc.evaluation.toy_pipeline import build_toy, collect_toy
    collected = collect_toy(build_toy(z_dim=16), num_ticks=15)
    tr = next(t for t in collected.transitions if t.capture_tick > 0)
    command = next(e["action"] for e in collected.log.events if e["kind"] == "tick" and e["tick"] == tr.capture_tick - 1)
    np.testing.assert_array_equal(tr.planning_state.last_command[0], command)
