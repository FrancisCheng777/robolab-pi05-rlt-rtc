"""Queue, time and fault tests T33-T46 plus planner-server and config checks."""

from __future__ import annotations

import numpy as np
import pytest

from pi05_rlt_rtc.contracts.errors import ContractError, PlanValidationError
from pi05_rlt_rtc.contracts.plan import PrevPlanContext
from pi05_rlt_rtc.contracts.spaces import HorizonContract
from pi05_rlt_rtc.runtime.scheduler import (
    NumberedReferencePlanner,
    PlanResult,
    VirtualDelayScheduler,
)
from tests.helpers import (
    ENV_ID,
    EPISODE_ID,
    ledger_commands_by_tick,
    make_final,
    make_queue,
    make_request,
    make_submitted,
    warm_start,
)


class _ModifyingPlanner(NumberedReferencePlanner):
    """Submits values that differ sharply from the reference (T42/T43)."""

    def plan(self, request, observation, pending_prefix, prev_context) -> PlanResult:
        i = request.request_id
        if i < 0:
            return super().plan(request, observation, pending_prefix, prev_context)
        submitted = np.full(
            (self.contract.chunk, 8), 0.42 + 0.01 * i, dtype=np.float32
        )
        reference = np.full((self.contract.horizon, 8), 0.07, dtype=np.float32)
        return PlanResult(submitted=submitted, reference_env_full=reference)


def test_t33_c5_d2_numbered_execution_table():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    queue = make_queue(contract)
    from tests.helpers import RecordingEnv

    env = RecordingEnv()
    scheduler = VirtualDelayScheduler(
        contract=contract,
        queue=queue,
        backend=NumberedReferencePlanner(contract),
        env=env,
        planner_latency_ticks=0,
        seed=0,
    )
    log = scheduler.run(15)
    by_tick = ledger_commands_by_tick(queue)

    # warm start plan id 0, U_0..U_2 get plan ids 1..3
    plan_events = [e for e in log.events if e["kind"] == "plan" and e["accepted"]]
    plan_ids = {e["request_id"]: e["plan"].plan_id for e in plan_events}
    warm_ids = [e for e in log.events if e["kind"] == "warm_start"]
    assert warm_ids and warm_ids[0]["plan_id"] == 0
    assert plan_ids == {0: 1, 1: 2, 2: 3}

    expected = {
        0: (0.9, 0),  # P0[0]
        1: (0.9001, 0),  # P0[1]
        2: (0.1, 1),  # U0[0]
        3: (0.1001, 1),  # U0[1]
        4: (0.1002, 1),  # U0[2]
        5: (0.1003, 1),  # P1[0] = U0[3]
        6: (0.1004, 1),  # P1[1] = U0[4]
        7: (0.2, 2),  # U1[0]
        8: (0.2001, 2),
        9: (0.2002, 2),
        10: (0.2003, 2),  # P2[0] = U1[3]
        11: (0.2004, 2),
        12: (0.3, 3),  # U2[0]
        13: (0.3001, 3),
        14: (0.3002, 3),
    }
    for tick, (value, plan_id) in expected.items():
        assert tick in by_tick, f"missing execution ledger entry for tick {tick}"
        assert by_tick[tick]["command"] == pytest.approx(value, abs=1e-6), tick
        assert by_tick[tick]["plan_id"] == plan_id, tick
        assert by_tick[tick]["kind"] == "committed"


def test_t34_d_zero_synchronous_boundary():
    contract = HorizonContract(horizon=10, chunk=5, delay=0)
    queue = make_queue(contract)
    from tests.helpers import RecordingEnv

    scheduler = VirtualDelayScheduler(
        contract=contract,
        queue=queue,
        backend=NumberedReferencePlanner(contract),
        env=RecordingEnv(),
        planner_latency_ticks=0,
        seed=0,
    )
    scheduler.run(10)
    by_tick = ledger_commands_by_tick(queue)
    for tick in range(5):
        assert by_tick[tick]["command"] == pytest.approx(0.1 + 0.0001 * tick, abs=1e-6)
    for tick in range(5, 10):
        assert by_tick[tick]["command"] == pytest.approx(0.2 + 0.0001 * (tick - 5), abs=1e-6)


def test_t35_d_equals_c_all_credit_arrives_via_next_pending():
    contract = HorizonContract(horizon=10, chunk=5, delay=5)
    queue = make_queue(contract)
    from tests.helpers import RecordingEnv

    scheduler = VirtualDelayScheduler(
        contract=contract,
        queue=queue,
        backend=NumberedReferencePlanner(contract),
        env=RecordingEnv(),
        planner_latency_ticks=0,
        seed=0,
    )
    log = scheduler.run(10)
    by_tick = ledger_commands_by_tick(queue)
    for tick in range(5):
        assert by_tick[tick]["plan_id"] == 0  # P0 from warm start
    for tick in range(5, 10):
        assert by_tick[tick]["plan_id"] == 1  # U0 == P1
    request_1 = [e for e in log.events if e["kind"] == "request" and e["request"].request_id == 1][0]
    pending = request_1["pending_prefix"]
    assert pending.shape == (5, 8)
    assert np.allclose(pending, 0.1 + 0.0001 * np.arange(5)[:, None])


def test_t36_invalid_horizon_contracts_rejected():
    with pytest.raises(ContractError):
        HorizonContract(horizon=10, chunk=5, delay=6)
    with pytest.raises(ContractError):
        HorizonContract(horizon=6, chunk=5, delay=2)
    with pytest.raises(ContractError):
        HorizonContract(horizon=10, chunk=0, delay=0)
    with pytest.raises(ContractError):
        HorizonContract(horizon=10, chunk=5, delay=-1)


def test_t37_early_response_waits_for_apply_start():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    queue = make_queue(contract)
    warm_id = warm_start(queue, contract, value=0.5)
    request = make_request(contract, request_id=0, capture_tick=0)
    submitted = make_submitted(contract, request=request, base_value=0.2)
    pending, _ = queue.snapshot_pending(0)
    plan = make_final(contract, request=request, pending=pending, submitted=submitted)
    ack = queue.commit_plan(plan, submitted, now_tick=0)  # response arrives immediately
    assert ack.accepted
    r0 = queue.consume(0)
    r1 = queue.consume(1)
    r2 = queue.consume(2)
    assert r0.action[0] == pytest.approx(0.5)  # P[0]
    assert r1.action[0] == pytest.approx(0.5)  # P[1]
    assert r2.action[0] == pytest.approx(0.2)  # U[0], not earlier
    assert r2.source_plan_id != warm_id


def test_t38_late_response_rejected_not_pasted():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    queue = make_queue(contract)
    warm_start(queue, contract, value=0.5)
    request = make_request(contract, request_id=0, capture_tick=0)
    submitted = make_submitted(contract, request=request, base_value=0.2)
    pending, _ = queue.snapshot_pending(0)
    plan = make_final(contract, request=request, pending=pending, submitted=submitted)
    ack = queue.commit_plan(plan, submitted, now_tick=3)
    assert not ack.accepted and ack.reason == "late"
    assert queue.deadline_miss_count == 1
    # nothing was inserted for ticks 2/3
    assert 2 not in queue.entries and 3 not in queue.entries
    result = queue.consume(3)
    assert result.safe_stop and result.kind == "stop"


def test_t39_duplicate_out_of_order_and_cross_episode_responses():
    import dataclasses

    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    queue = make_queue(contract)
    warm_start(queue, contract, value=0.5)
    request0 = make_request(contract, request_id=0, capture_tick=0)
    submitted0 = make_submitted(contract, request=request0, base_value=0.2)
    pending, _ = queue.snapshot_pending(0)
    plan0 = make_final(contract, request=request0, pending=pending, submitted=submitted0)
    ack0 = queue.commit_plan(plan0, submitted0, now_tick=0)
    assert ack0.accepted
    plan_count = queue.next_plan_id

    # duplicate id is an idempotent ACK, even with a different payload
    dup = queue.commit_plan(plan0, submitted0, now_tick=0)
    assert dup.accepted and dup.duplicate and dup.reason == "duplicate"
    other_submitted = make_submitted(contract, request=request0, base_value=0.9)
    other_plan = make_final(contract, request=request0, pending=pending, submitted=other_submitted)
    dup2 = queue.commit_plan(other_plan, other_submitted, now_tick=0)
    assert dup2.duplicate
    assert queue.next_plan_id == plan_count, "duplicate must not double insert"
    assert queue.entries[2].action[0] == pytest.approx(0.2), "duplicate must not overwrite"

    # cross-episode/generation submission is rejected (fresh request id)
    request_cross = make_request(contract, request_id=1, capture_tick=5, episode_generation=1)
    submitted_cross = make_submitted(contract, request=request_cross, base_value=0.3)
    bad = dataclasses.replace(submitted_cross, episode_generation=999)
    rejected = queue.commit_plan(plan0, bad, now_tick=5)
    assert not rejected.accepted and "mismatch" in rejected.reason

    # out-of-order request ids are rejected once a later request was accepted
    contract0 = HorizonContract(horizon=10, chunk=5, delay=0)
    queue0 = make_queue(contract0)
    request_late = make_request(contract0, request_id=5, capture_tick=5)
    submitted_late = make_submitted(contract0, request=request_late, base_value=0.2)
    pending0, _ = queue0.snapshot_pending(5)
    plan_late = make_final(contract0, request=request_late, pending=pending0, submitted=submitted_late)
    assert queue0.commit_plan(plan_late, submitted_late, now_tick=5).accepted
    old_request = make_request(contract0, request_id=3, capture_tick=3)
    old_submitted = make_submitted(contract0, request=old_request, base_value=0.4)
    old_plan = make_final(contract0, request=old_request, pending=pending0, submitted=old_submitted)
    stale = queue0.commit_plan(old_plan, old_submitted, now_tick=5)
    assert not stale.accepted and stale.reason == "stale_request"


def test_t40_reset_invalidates_inflight_plan():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    queue = make_queue(contract)
    env = _TerminatingEnv(terminate_at_step=7)
    scheduler = VirtualDelayScheduler(
        contract=contract,
        queue=queue,
        backend=NumberedReferencePlanner(contract),
        env=env,
        planner_latency_ticks=2,
        seed=0,
        stop_on_safety=False,
    )
    log = scheduler.run(10)
    resets = [e for e in log.events if e["kind"] == "episode_start"]
    assert resets, "episode should have reset after termination"
    reset_tick = resets[0]["tick"]
    old_plan_ids = {
        entry.source_plan_id
        for entry in queue.ledger.entries
        if entry.tick < reset_tick and entry.source_plan_id is not None
    }
    assert old_plan_ids, "expected at least the warm-start plan id"
    after = [
        entry
        for entry in queue.ledger.entries
        if entry.tick >= reset_tick and entry.source_plan_id in old_plan_ids
    ]
    assert after == [], "no old plan may execute in the new episode"
    dropped = [e for e in log.events if e["kind"] == "inflight_dropped"]
    assert dropped, "in-flight request for the old episode must be dropped at reset"
    assert required_queue_generation(queue) >= 2


def test_t41_consumption_does_not_invalidate_inflight_requests():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    queue = make_queue(contract)
    warm_start(queue, contract, value=0.5)
    generation = queue.queue_generation
    request0 = make_request(contract, request_id=0, capture_tick=0)
    submitted0 = make_submitted(contract, request=request0, base_value=0.2)
    pending, _ = queue.snapshot_pending(0)
    plan0 = make_final(contract, request=request0, pending=pending, submitted=submitted0)
    assert queue.commit_plan(plan0, submitted0, now_tick=0).accepted

    queue.consume(0)
    queue.consume(1)
    assert queue.queue_generation == generation

    request1 = make_request(contract, request_id=1, capture_tick=5)
    pending1, _ = queue.snapshot_pending(5)
    submitted1 = make_submitted(contract, request=request1, base_value=0.3)
    plan1 = make_final(contract, request=request1, pending=pending1, submitted=submitted1)
    ack1 = queue.commit_plan(plan1, submitted1, now_tick=5)
    assert ack1.accepted
    assert queue.queue_generation == generation


def test_t42_next_rtc_target_uses_actor_modified_submitted_plan():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    queue = make_queue(contract)
    from tests.helpers import RecordingEnv

    scheduler = VirtualDelayScheduler(
        contract=contract,
        queue=queue,
        backend=_ModifyingPlanner(contract),
        env=RecordingEnv(),
        planner_latency_ticks=0,
        seed=0,
    )
    log = scheduler.run(10)
    request1 = [
        e for e in log.events if e["kind"] == "request" and e["request"].request_id == 1
    ][0]
    prev_plan = request1["prev_context"].prev_env_plan
    # M_0[5:10] = [U0[3], U0[4], ref[5], ref[6], ref[7]] = [0.42, 0.42, 0.07, 0.07, 0.07]
    assert np.allclose(prev_plan[0], 0.42, atol=1e-6)
    assert np.allclose(prev_plan[1], 0.42, atol=1e-6)
    assert np.allclose(prev_plan[2:], 0.07, atol=1e-6)
    deviations = [e for e in queue.ledger.events if e["event"] == "guidance_prefix_deviation"]
    assert deviations, "deviation between new plan and committed prefix must be recorded"


def test_t43_committed_prefix_survives_disagreeing_new_plan():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    queue = make_queue(contract)
    warm_start(queue, contract, value=0.5)
    request0 = make_request(contract, request_id=0, capture_tick=0)
    submitted0 = make_submitted(contract, request=request0, base_value=0.2)
    pending, _ = queue.snapshot_pending(0)
    plan0 = make_final(contract, request=request0, pending=pending, submitted=submitted0)
    queue.commit_plan(plan0, submitted0, now_tick=0)

    # A second plan at tick 5 with first D actions wildly different from P.
    request1 = make_request(contract, request_id=1, capture_tick=5)
    pending1, _ = queue.snapshot_pending(5)
    submitted1 = make_submitted(contract, request=request1, base_value=0.9)
    plan1 = make_final(contract, request=request1, pending=pending1, submitted=submitted1)
    ack1 = queue.commit_plan(plan1, submitted1, now_tick=5)
    assert ack1.accepted
    # ticks 5,6 must still execute the committed P (= U0[3], U0[4]), not the new 0.9
    assert queue.consume(5).action[0] == pytest.approx(0.23, abs=1e-6)
    assert queue.consume(6).action[0] == pytest.approx(0.24, abs=1e-6)
    assert queue.consume(7).action[0] == pytest.approx(0.9, abs=1e-6)


def test_t44_slow_planner_does_not_block_controller():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    from pi05_rlt_rtc.runtime.controller import WallClockController
    from pi05_rlt_rtc.runtime.queue import ActionQueue
    from tests.helpers import RecordingEnv

    queue = ActionQueue(contract, fallback_policy="hold_last")
    backend = _SleepyBackend(NumberedReferencePlanner(contract), seconds=0.06)
    controller = WallClockController(
        contract=contract,
        queue=queue,
        backend=backend,
        env=RecordingEnv(),
        tick_hz=100.0,
        seed=0,
    )
    report = controller.run(20)
    assert report.deadline_misses >= 1
    assert report.ticks_run == 20, "controller must keep ticking while the planner is busy"
    assert report.ticks_advanced_during_planning > 0
    assert report.planner_latencies_ms and max(report.planner_latencies_ms) > 30.0


def test_t45_nan_shape_and_stats_rejection():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    queue = make_queue(contract)
    warm_start(queue, contract)
    request = make_request(contract, request_id=0, capture_tick=0)

    # wrong shape
    bad_submitted = make_submitted(contract, request=request, base_value=0.2)
    with pytest.raises(PlanValidationError):
        import dataclasses

        dataclasses.replace(bad_submitted, actions=np.zeros((3, 8), dtype=np.float32)).validate(contract)

    # NaN plan content
    pending, _ = queue.snapshot_pending(0)
    submitted = make_submitted(contract, request=request, base_value=0.2)
    plan = make_final(contract, request=request, pending=pending, submitted=submitted)
    nan_plan = _with_nan(plan)
    ack = queue.commit_plan(nan_plan, submitted, now_tick=0)
    assert not ack.accepted and ack.reason == "nonfinite"

    # out-of-range plan
    submitted_oor = make_submitted(contract, request=request, base_value=2.5)
    plan_oor = make_final(contract, request=request, pending=pending, submitted=submitted_oor)
    ack = queue.commit_plan(plan_oor, submitted_oor, now_tick=0)
    assert not ack.accepted and ack.reason == "out_of_range"


def test_t46_safety_intervention_invalidates_generation():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    queue = make_queue(contract)
    warm_start(queue, contract)
    request = make_request(contract, request_id=0, capture_tick=0)
    submitted = make_submitted(contract, request=request, base_value=0.2)
    pending, _ = queue.snapshot_pending(0)
    plan = make_final(contract, request=request, pending=pending, submitted=submitted)

    queue.intervene(0, "operator_estop")
    assert queue.queue_generation == 2
    assert queue.force_fallback
    ack = queue.commit_plan(plan, submitted, now_tick=0)
    assert not ack.accepted and ack.reason == "generation_mismatch"
    result = queue.consume(0)
    assert result.safe_stop
    kinds = [e["event"] for e in queue.ledger.events]
    assert "intervention" in kinds


# ---------------------------------------------------------------------------
# supplementary (non T19 acceptance) checks


def test_planner_http_service_round_trip():
    import json
    import threading
    import urllib.request

    from pi05_rlt_rtc.evaluation.toy_pipeline import build_toy
    from pi05_rlt_rtc.tools.planner_server import PlannerService, make_handler
    from http.server import ThreadingHTTPServer

    build = build_toy(z_dim=16)
    service = PlannerService(build.backend, build.contract)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, shutdown_after=1))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = make_request(build.contract, request_id=0, capture_tick=0)
        obs = build.env.reset(seed=0)
        import dataclasses
        obs = dataclasses.replace(obs, env_id=request.env_id, episode_id=request.episode_id, tick=request.capture_tick)
        payload = {
            "request": {
                "request_id": request.request_id,
                "env_id": request.env_id,
                "episode_id": request.episode_id,
                "episode_generation": request.episode_generation,
                "capture_tick": request.capture_tick,
                "apply_start_tick": request.apply_start_tick,
                "queue_generation": request.queue_generation,
                "policy_version_id": request.policy_version_id,
                "feature_version_id": request.feature_version_id,
                "schema_version": request.schema_version,
                "planning_state_version": request.planning_state_version,
                "rng_seed": request.rng_seed,
                "sampler_mode": "none",
                "num_denoising_steps": request.num_denoising_steps,
                "rtc_context": {},
            },
            "observation": {
                "env_id": obs.env_id,
                "episode_id": obs.episode_id,
                "tick": obs.tick,
                "states": obs.states.tolist(),
                "images": {k: v.tolist() for k, v in obs.images.items()},
                "image_masks": {k: True for k in obs.images},
                "prompt": obs.prompt,
            },
            "pending_prefix": np.zeros((2, 8), dtype=np.float32).tolist(),
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/plan", data=body, method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
        assert len(result["submitted"]) == build.contract.chunk
        assert len(result["reference_env_full"]) == build.contract.horizon
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_config_rejects_unknown_keys_and_invalid_dimensions(tmp_path):
    import copy

    import yaml

    from pi05_rlt_rtc.config.design_config import load_design_config
    from pi05_rlt_rtc.contracts.errors import ConfigError

    example = None
    candidates = [
        "docs/spec/design_config.example.yaml",
    ]
    for candidate in candidates:
        path = __import__("pathlib").Path(candidate)
        if path.exists():
            example = yaml.safe_load(path.read_text(encoding="utf-8"))
    if example is None:
        pytest.skip("design_config.example.yaml not available")

    good = copy.deepcopy(example)
    good["assets"]["base_checkpoint_path"] = "/tmp/base"
    good["assets"]["stage1_checkpoint_path"] = "/tmp/stage1"
    good["assets"]["dataset_path"] = "/tmp/data"
    good["assets"]["norm_stats_path"] = "/tmp/stats"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(good), encoding="utf-8")
    cfg = load_design_config(path)
    assert cfg.horizon_contract().horizon == 10
    assert cfg.feature_profile_id()

    bad = copy.deepcopy(good)
    bad["integration"]["fixed_delay_ticks"] = 6
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_design_config(path)

    bad = copy.deepcopy(good)
    bad["unknown_section"] = {"a": 1}
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_design_config(path)

    bad = copy.deepcopy(good)
    bad["rtc"]["model_parameter_gradients_enabled"] = True
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_design_config(path)


# ---------------------------------------------------------------------------
# local helpers


class _TerminatingEnv:
    """Reports a true termination at a fixed step count."""

    def __init__(self, *, terminate_at_step: int) -> None:
        from tests.helpers import RecordingEnv

        self._inner = RecordingEnv()
        self._limit = terminate_at_step
        self._tick = 0

    def reset(self, *, seed: int = 0):
        self._tick = 0
        return self._inner.reset(seed=seed)

    def step(self, action):
        import dataclasses

        outcome = self._inner.step(action)
        self._tick += 1
        return dataclasses.replace(outcome, terminated=self._tick >= self._limit)


def required_queue_generation(queue) -> int:
    return queue.queue_generation


class _SleepyBackend:
    def __init__(self, inner, *, seconds: float) -> None:
        from pi05_rlt_rtc.runtime.controller import PlannerWorker  # noqa: F401

        self.inner = inner
        self.seconds = seconds

    def plan(self, request, observation, pending_prefix, prev_context):
        import time

        time.sleep(self.seconds)
        return self.inner.plan(request, observation, pending_prefix, prev_context)


def _with_nan(plan):
    import dataclasses

    final = plan.final.copy()
    final[0, 0] = np.nan
    return dataclasses.replace(plan, final=final)
