"""Deterministic controller timing tests; these are not robot timing results."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from pi05_rlt_rtc.contracts.clock import TickClock
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.runtime import controller as mod
from pi05_rlt_rtc.runtime.queue import ActionQueue
from pi05_rlt_rtc.runtime.scheduler import NumberedReferencePlanner
from tests.helpers import RecordingEnv


class ManualTime:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def manual_runtime(monkeypatch):
    clock = ManualTime()
    monkeypatch.setattr(mod, "time", clock)
    monkeypatch.setattr(mod, "TickClock", lambda **kw: TickClock(**({"start_monotonic": clock.now} | kw)))

    class ImmediateWorker:
        def __init__(self, backend, *, tick_probe, on_outcome):
            self.backend, self.tick_probe, self.on_outcome = backend, tick_probe, on_outcome
            self.jobs_failed = 0
            self.jobs_completed = 0
            self.alive = False

        def bind_contract(self, contract):
            self.contract = contract

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def stop(self, timeout=0):
            self.alive = False

        def submit(self, job):
            result = self.backend.plan(job.request, job.observation, job.pending_prefix, job.prev_context)
            self.jobs_completed += 1
            self.on_outcome(mod.PlannerOutcome(job, result, None, 0.0, clock.now, self.tick_probe()))

    monkeypatch.setattr(mod, "PlannerWorker", ImmediateWorker)
    return clock


def test_preparation_time_does_not_cause_burst_control_ticks(contract, manual_runtime):
    clock = manual_runtime
    stamps = []

    class SlowStartup(NumberedReferencePlanner):
        def plan(self, request, *args):
            if request.request_id == -1:
                clock.sleep(3.0)
            return super().plan(request, *args)

    class Env(RecordingEnv):
        def step(self, action):
            stamps.append(clock.now)
            return super().step(action)

    controller = mod.WallClockController(contract=contract, queue=ActionQueue(contract), backend=SlowStartup(contract), env=Env(), tick_hz=10)
    clock.sleep(5.0)  # construction-to-run delay also is not control time
    report = controller.run(4)
    assert report.ticks_run == 4
    np.testing.assert_allclose(np.diff(stamps), 0.1, atol=1e-9)


def test_episode_end_stops_without_automatic_robot_reset(contract, manual_runtime):
    class TerminalEnv(RecordingEnv):
        resets = 0

        def reset(self, **kwargs):
            self.resets += 1
            return super().reset(**kwargs)

        def step(self, action):
            return dataclasses.replace(super().step(action), terminated=True)

    env = TerminalEnv()
    queue = ActionQueue(contract)
    controller = mod.WallClockController(contract=contract, queue=queue, backend=NumberedReferencePlanner(contract), env=env, tick_hz=10)
    report = controller.run(10)
    assert report.ticks_run == 1 and env.resets == 1
    assert not queue.entries


def test_requests_and_snapshots_share_controller_identity(contract, manual_runtime):
    class CheckingPlanner(NumberedReferencePlanner):
        def plan(self, request, observation, *args):
            assert (observation.env_id, observation.episode_id, observation.tick) == (request.env_id, request.episode_id, request.capture_tick)
            return super().plan(request, observation, *args)

    controller = mod.WallClockController(contract=contract, queue=ActionQueue(contract), backend=CheckingPlanner(contract), env=RecordingEnv(), tick_hz=10)
    assert controller.run(5).ticks_run == 5


def test_request_period_cannot_disagree_with_chunk_protocol(contract, manual_runtime):
    controller = mod.WallClockController(contract=contract, queue=ActionQueue(contract), backend=NumberedReferencePlanner(contract), env=RecordingEnv(), tick_hz=10)
    with pytest.raises(ContractError):
        controller.run(3, request_every=1)


def test_completed_jobs_are_not_retained_forever(contract, manual_runtime):
    controller = mod.WallClockController(contract=contract, queue=ActionQueue(contract), backend=NumberedReferencePlanner(contract), env=RecordingEnv(), tick_hz=10)
    controller.run(20)
    assert len(controller._pending_jobs) <= 1


def test_invalid_tick_count_is_rejected_before_reset(contract, manual_runtime):
    controller = mod.WallClockController(contract=contract, queue=ActionQueue(contract), backend=NumberedReferencePlanner(contract), env=RecordingEnv(), tick_hz=10)
    with pytest.raises(ContractError):
        controller.run(0)


def test_thread_worker_preserves_thread_internal_stop_method():
    worker = mod.PlannerWorker(SimpleNamespace(), tick_probe=lambda: 0, on_outcome=lambda _: None)
    assert not hasattr(worker, "_stop") or callable(worker._stop)
    worker.start()
    worker.stop(timeout=1)
    assert not worker.is_alive()


@pytest.mark.parametrize("error", [None, "RuntimeError: planner failed"])
def test_response_after_monotonic_deadline_is_rejected_even_at_apply_tick(contract, manual_runtime, error):
    from tests.helpers import make_request, warm_start
    queue = ActionQueue(contract)
    queue.reset(env_id="env0", episode_id="env0-ep1", start_tick=0)
    warm_start(queue, contract)
    backend = NumberedReferencePlanner(contract)
    controller = mod.WallClockController(contract=contract, queue=queue, backend=backend, env=RecordingEnv(), tick_hz=10)
    request = make_request(contract, request_id=0, capture_tick=0)
    pending = queue.snapshot_pending(0)[0]
    job = mod.PlannerJob(request, RecordingEnv().reset(), pending, request.rtc_context, manual_runtime.now, 0, 0)
    result = backend.plan(request, job.observation, pending, request.rtc_context)
    controller._current_tick = 2
    outcome = mod.PlannerOutcome(job, result, error, 0.201, controller.clock.tick_to_monotonic(2) + 0.001, 2)
    controller._commit_outcome(outcome)
    assert queue.deadline_miss_count == 1
    assert not any(e["event"] == "plan_committed" for e in queue.ledger.events)


def test_hung_planner_counts_deadline_without_response_and_does_not_queue_more(contract):
    import threading
    released = threading.Event()

    class HungPlanner(NumberedReferencePlanner):
        def plan(self, request, *args):
            if request.request_id >= 0:
                released.wait(timeout=2)
            return super().plan(request, *args)

    queue = ActionQueue(contract, fallback_policy="hold_last")
    controller = mod.WallClockController(contract=contract, queue=queue, backend=HungPlanner(contract), env=RecordingEnv(), tick_hz=50)
    try:
        report = controller.run(8)
        assert report.deadline_misses == 1
        assert sum(e["event"] == "request_submitted" for e in queue.ledger.events) == 1
        assert not queue.entries
    finally:
        released.set()
        controller._worker.join(timeout=1)


def test_virtual_snapshots_match_authoritative_ticks_across_resets(contract):
    from pi05_rlt_rtc.runtime.scheduler import VirtualDelayScheduler
    class TerminalEnv(RecordingEnv):
        def step(self, action):
            return dataclasses.replace(super().step(action), terminated=self._tick == 3)

    class CheckingPlanner(NumberedReferencePlanner):
        def plan(self, request, observation, *args):
            assert (observation.env_id, observation.episode_id, observation.tick) == (request.env_id, request.episode_id, request.capture_tick)
            return super().plan(request, observation, *args)

    log = VirtualDelayScheduler(contract=contract, queue=ActionQueue(contract), backend=CheckingPlanner(contract), env=TerminalEnv()).run(8)
    assert log.episodes == 3
