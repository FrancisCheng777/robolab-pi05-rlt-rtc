"""Late planner work is diagnostic evidence, never an executable response."""
import threading
import time
from types import SimpleNamespace

from pi05_rlt_rtc.runtime.controller import PlannerWorker, PlannerJob


def test_inflight_completion_after_hold_is_recorded_but_not_published():
    entered, release = threading.Event(), threading.Event()
    delivered = []

    class Backend:
        def plan(self, *args):
            entered.set()
            assert release.wait(2)
            return SimpleNamespace(validate=lambda contract: None)

    worker = PlannerWorker(Backend(), tick_probe=lambda: 150, on_outcome=delivered.append)
    worker.bind_contract(SimpleNamespace())
    request = SimpleNamespace(request_id=29, capture_tick=145, apply_start_tick=150)
    worker.start()
    try:
        worker.submit(PlannerJob(request, None, None, None, time.monotonic(), 145, 29))
        assert entered.wait(2)
        worker.stop(timeout=0)
    finally:
        release.set()
        worker.stop(timeout=2)
    assert not worker.is_alive()
    assert delivered == []
    assert len(worker.timings) == 1
    timing = worker.timings[0]
    assert timing['request_id'] == 29
    assert timing['finished_after_stop'] is True
    assert timing['error'] is None
    assert timing['total_ms'] >= timing['backend_ms'] >= 0
    assert timing['dispatch_wait_ms'] >= 0
