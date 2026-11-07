"""Wall-clock controller and planner worker.

The controller is the only owner of the environment ``step`` and of queue
consumption.  It runs on an integer tick schedule and never waits for model
inference, logging or the learner (guide sections 12.1/15.1).  The planner
worker runs the (possibly slow) planner on a separate thread and hands
results back through a queue that the controller drains at tick boundaries,
so all queue mutations stay on the controller thread.

This module is also used by the wall-clock CPU test with an explicitly
labelled *mock* planner: the mock only replaces the model, never the timing
semantics being tested.
"""

from __future__ import annotations

import dataclasses
import contextlib
import gc
import queue as thread_queue
import threading
import sys
import time
from typing import Any, Callable

import numpy as np

from pi05_rlt_rtc.contracts.clock import TickClock
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
from pi05_rlt_rtc.contracts.plan import (
    FinalPlan,
    PlanRequest,
    PrevPlanContext,
    SubmittedPlan,
    make_submitted_mask,
)
from pi05_rlt_rtc.contracts.spaces import ENV_ACTION_DIM, HorizonContract
from pi05_rlt_rtc.runtime.queue import ActionQueue
from pi05_rlt_rtc.runtime.scheduler import EnvProtocol, PlanResult, PlannerProtocol


@contextlib.contextmanager
def _defer_cyclic_gc():
    """Avoid stop-the-world cycle scans during the one active control loop.

    Reference counting still releases ordinary objects. Restore the caller's GC
    policy after environment stop, including warmup/step failures. The hardware
    session lock prevents concurrent control loops in this process.
    """
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    arr = np.asarray(sorted(values), dtype=np.float64)
    idx = int(round((pct / 100.0) * (arr.size - 1)))
    return float(arr[idx])


@dataclasses.dataclass
class PlannerJob:
    request: PlanRequest
    observation: ObservationSnapshot
    pending_prefix: np.ndarray
    prev_context: Any
    submitted_at_monotonic: float
    tick_at_submit: int
    job_id: int
    deadline_missed: bool = False


@dataclasses.dataclass
class PlannerOutcome:
    job: PlannerJob
    result: PlanResult | None
    error: str | None
    latency_seconds: float
    finished_at_monotonic: float
    tick_at_finish: int


class PlannerWorker(threading.Thread):
    def __init__(
        self,
        backend: PlannerProtocol,
        *,
        tick_probe: Callable[[], int],
        on_outcome: Callable[[PlannerOutcome], None],
        name: str = "pi05-rlt-rtc-planner",
        daemon: bool = True,
    ) -> None:
        super().__init__(name=name, daemon=daemon)
        self.backend = backend
        self.tick_probe = tick_probe
        self.on_outcome = on_outcome
        self._jobs: thread_queue.Queue[PlannerJob | None] = thread_queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.busy = False
        self.last_latency_seconds: float | None = None
        # Include in-flight work that finishes after hold; it must not be
        # committed, but is often the very request needed to diagnose a miss.
        self.timings: list[dict] = []

    def submit(self, job: PlannerJob) -> None:
        if self._stop_event.is_set():
            raise ContractError("cannot submit to a stopped planner")
        self._jobs.put_nowait(job)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        try:
            self._jobs.put_nowait(None)
        except thread_queue.Full:
            pass
        if self.ident is not None:
            self.join(timeout=timeout)

    def run(self) -> None:  # pragma: no cover - exercised in wallclock test
        while not self._stop_event.is_set():
            job = self._jobs.get()
            if job is None or self._stop_event.is_set():
                break
            self.busy = True
            started = time.monotonic()
            result: PlanResult | None = None
            error: str | None = None
            try:
                result = self.backend.plan(
                    job.request, job.observation, job.pending_prefix, job.prev_context
                )
                result.validate(self._contract_for(job))
            except Exception as exc:  # noqa: BLE001 - record, never crash the controller
                error = f"{type(exc).__name__}: {exc}"
            finished = time.monotonic()
            self.busy = False
            self.last_latency_seconds = finished - started
            self.timings.append({
                "request_id": job.request.request_id,
                "capture_tick": job.request.capture_tick,
                "apply_start_tick": job.request.apply_start_tick,
                "dispatch_wait_ms": (started - job.submitted_at_monotonic) * 1000.,
                "backend_ms": (finished - started) * 1000.,
                "total_ms": (finished - job.submitted_at_monotonic) * 1000.,
                "finished_after_stop": self._stop_event.is_set(),
                "error": error,
            })
            if error is None:
                self.jobs_completed += 1
            else:
                self.jobs_failed += 1
            if self._stop_event.is_set():
                break
            self.on_outcome(
                PlannerOutcome(
                    job=job,
                    result=result,
                    error=error,
                    latency_seconds=finished - started,
                    finished_at_monotonic=finished,
                    tick_at_finish=self.tick_probe(),
                )
            )

    def _contract_for(self, job: PlannerJob) -> HorizonContract:
        # The contract is attached to the worker by the controller before start.
        contract = getattr(self, "_contract", None)
        if contract is None:
            raise ContractError("PlannerWorker started before bind_contract()")
        return contract

    def bind_contract(self, contract: HorizonContract) -> None:
        self._contract = contract


@dataclasses.dataclass
class WallClockReport:
    ticks_run: int
    deadline_misses: int
    rejected_plans: int
    safety_stops: int
    planner_errors: int
    tick_jitter_ms: list[float]
    planner_latencies_ms: list[float]
    ticks_advanced_during_planning: int
    underruns: int

    def summary(self) -> dict[str, Any]:
        return {
            "ticks_run": self.ticks_run,
            "deadline_misses": self.deadline_misses,
            "rejected_plans": self.rejected_plans,
            "safety_stops": self.safety_stops,
            "planner_errors": self.planner_errors,
            "underruns": self.underruns,
            "ticks_advanced_during_planning": self.ticks_advanced_during_planning,
            "planner_latency_ms_p50": _percentile(self.planner_latencies_ms, 50),
            "planner_latency_ms_p95": _percentile(self.planner_latencies_ms, 95),
            "planner_latency_ms_max": max(self.planner_latencies_ms) if self.planner_latencies_ms else None,
            "tick_jitter_ms_p50": _percentile(self.tick_jitter_ms, 50),
            "tick_jitter_ms_p95": _percentile(self.tick_jitter_ms, 95),
            "tick_jitter_ms_max": max(self.tick_jitter_ms) if self.tick_jitter_ms else None,
        }


class WallClockController:
    def __init__(
        self,
        *,
        contract: HorizonContract,
        queue: ActionQueue,
        backend: PlannerProtocol,
        env: EnvProtocol,
        tick_hz: float,
        env_id: str = "env0",
        episode_id: str = "env0-ep0",
        policy_version_id: str = "unset",
        feature_version_id: str = "unset",
        schema_version: str = "pi05-rlt-rtc.schema.v1",
        planning_state_version: str = "queue-planning-state.v1",
        rtc_mode: str = "full_vjp",
        num_denoising_steps: int = 5,
        seed: int = 0,
        record_events: bool = False,
        observation_recorder=None,
        tick_hook=None,
    ) -> None:
        if contract != queue.contract:
            raise ContractError("controller and queue must use the same horizon contract")
        if contract.delay == 0:
            raise ContractError("wall-clock asynchronous planning requires D > 0; use the virtual scheduler for D=0")
        self.contract = contract
        self.queue = queue
        self.backend = backend
        self.env = env
        self.clock = TickClock(hz=tick_hz)
        self.env_id = env_id
        self.episode_id = episode_id
        self.policy_version_id = policy_version_id
        self.feature_version_id = feature_version_id
        self.schema_version = schema_version
        self.planning_state_version = planning_state_version
        self.rtc_mode = rtc_mode
        self.num_denoising_steps = num_denoising_steps
        self.seed = seed
        self.record_events = record_events
        self.observation_recorder = observation_recorder
        self.tick_hook = tick_hook
        self.events: list[dict[str, Any]] = []

        self._current_tick = 0
        self._outcomes: thread_queue.Queue[PlannerOutcome] = thread_queue.Queue()
        self._worker = PlannerWorker(
            backend, tick_probe=lambda: self._current_tick, on_outcome=self._outcomes.put
        )
        self._worker.bind_contract(contract)
        self._pending_jobs: dict[int, PlannerJob] = {}
        self._job_counter = 0
        self._request_counter = 0
        self._has_run = False

    def _record(self, event):
        if self.record_events:
            self.events.append(event)

    def _stop_env(self):
        primary_error = sys.exc_info()[1]
        if callable(getattr(self.env, 'stop', None)):
            try:
                self.env.stop()
            except Exception as exc:
                self._record({'kind': 'stop_error', 'error': f'{type(exc).__name__}: {exc}'})
                if primary_error is None:
                    raise
                if hasattr(primary_error, 'add_note'):
                    primary_error.add_note(f'Hold also failed: {type(exc).__name__}: {exc}')

    @property
    def current_tick(self) -> int:
        return self._current_tick

    @_defer_cyclic_gc()
    def run(self, num_ticks: int, *, request_every: int | None = None) -> WallClockReport:
        """Run one trial. Episode termination stops control; reset is explicit.

        Warm-start inference is outside the control clock. A missed primitive
        tick stops the trial instead of sending a burst of overdue commands.
        """
        contract = self.contract
        if isinstance(num_ticks, bool) or not isinstance(num_ticks, int) or num_ticks <= 0:
            raise ContractError("num_ticks must be a positive integer")
        if request_every is not None and request_every != contract.chunk:
            raise ContractError("request_every must equal the submitted chunk length C")
        if self._has_run:
            raise ContractError("create a new controller for each trial; automatic reset is disabled")
        self._has_run = True
        self.queue.reset(env_id=self.env_id, episode_id=self.episode_id, start_tick=0)
        obs = self._capture(self.env.reset(seed=self.seed), 0)
        self._record({"kind": "episode_start", "tick": 0})
        try:
            self._warm_start(obs)
        except BaseException:
            self.queue.intervene(0, "warm_start_failed")
            self._stop_env()
            raise
        self.clock = TickClock(hz=self.clock.hz, start_monotonic=time.monotonic())

        jitters: list[float] = []
        latencies: list[float] = []
        ticks_advanced_during_planning = 0
        safety_stops = 0
        ticks_run = 0
        last_final: FinalPlan | None = None
        self._worker.start()
        try:
            for tick in range(num_ticks):
                self._current_tick = tick
                if self.tick_hook is not None:
                    obs, end = self.tick_hook(tick, obs)
                    if end:
                        # The previous tick is the last action actually submitted.
                        # Do not fabricate an extra executed action for a keypress.
                        for event in reversed(self.events):
                            if event["kind"] == "tick":
                                event["terminated"] = True
                                event["truncated"] = False
                                break
                        break

                if self.observation_recorder is not None:
                    self.observation_recorder.check()
                jitter = max(0.0, time.monotonic() - self.clock.tick_to_monotonic(tick))
                jitters.append(jitter * 1000.0)
                if jitter >= self.clock.period_seconds:
                    self.queue.intervene(tick, "control_tick_overrun")
                    safety_stops += 1
                    break

                # A physical robot continues evolving while this thread sleeps.
                # Capture fresh measured state/camera buffers at the actual tick,
                # rather than reusing the immediate post-command observation.
                if callable(getattr(self.env, "observe", None)):
                    obs = self._capture(self.env.observe(), tick)
                    if time.monotonic() >= self.clock.tick_to_monotonic(tick + 1):
                        self.queue.intervene(tick, "observation_tick_overrun")
                        safety_stops += 1
                        break

                # Only this thread mutates the execution queue.
                while True:
                    try:
                        outcome = self._outcomes.get_nowait()
                    except thread_queue.Empty:
                        break
                    self._pending_jobs.pop(outcome.job.job_id, None)
                    latencies.append((outcome.finished_at_monotonic - outcome.job.submitted_at_monotonic) * 1000.0)
                    ticks_advanced_during_planning += max(0, outcome.tick_at_finish - outcome.job.tick_at_submit)
                    accepted = self._commit_outcome(outcome)
                    if accepted is not None:
                        last_final = accepted

                # A hung/crashed service must count as a miss even without a response.
                for job in self._pending_jobs.values():
                    if tick >= job.request.apply_start_tick and not job.deadline_missed:
                        job.deadline_missed = True
                        self.queue.deadline_miss_count += 1
                        self.queue.ledger.event("deadline_miss", tick=tick, request_id=job.request.request_id, reason="response_unavailable")

                if tick % contract.chunk == 0:
                    pending, valid = self.queue.snapshot_pending(tick)
                    if self._pending_jobs or not valid.all() or self.queue.force_fallback:
                        self.queue.ledger.event("request_skipped", tick=tick, reason="planner_busy_or_prefix_unavailable")
                    else:
                        prev_context = self._prev_context(last_final, tick)
                        request = PlanRequest(
                            request_id=self._request_counter,
                            env_id=self.env_id, episode_id=self.episode_id,
                            episode_generation=self.queue.episode_generation,
                            capture_tick=tick, apply_start_tick=tick + contract.delay,
                            queue_generation=self.queue.queue_generation,
                            policy_version_id=self.policy_version_id,
                            feature_version_id=self.feature_version_id,
                            schema_version=self.schema_version,
                            planning_state_version=self.planning_state_version,
                            rtc_context=prev_context,
                            rng_seed=self.seed * 1000003 + self._request_counter,
                            sampler_mode=self.rtc_mode,
                            num_denoising_steps=self.num_denoising_steps,
                            deadline_tick=tick + contract.delay,
                            last_command=None if self.queue.last_command is None else self.queue.last_command.copy(),
                        )
                        job = PlannerJob(request, obs, pending, prev_context, time.monotonic(), tick, self._job_counter)
                        self._job_counter += 1
                        self._pending_jobs[job.job_id] = job
                        self._record({"kind": "request", "tick": tick, "request": request,
                            "observation": None, "pending_prefix": pending.copy(),
                            "pending_mask": valid.copy(), "prev_context": prev_context})
                        if self.observation_recorder is not None:
                            self.observation_recorder.submit(request, obs, pending)
                        self._worker.submit(job)
                        self.queue.ledger.event("request_submitted", tick=tick, request_id=request.request_id)
                        self._request_counter += 1

                if time.monotonic() >= self.clock.tick_to_monotonic(tick + 1):
                    self.queue.intervene(tick, "control_tick_overrun_before_command")
                    safety_stops += 1
                    break
                consumed = self.queue.consume(tick)
                if consumed.safe_stop or consumed.action is None:
                    safety_stops += 1
                    self.queue.ledger.event("trial_stopped_wallclock", tick=tick, reason=consumed.note)
                    break
                outcome = self.env.step(np.asarray(consumed.action, dtype=np.float32))
                self._record({"kind": "tick", "tick": tick,
                    "action": np.asarray(consumed.action, np.float32).copy(),
                    "consume_kind": consumed.kind, "source_plan_id": consumed.source_plan_id,
                    "request_id": consumed.request_id, "reward": float(outcome.reward),
                    "terminated": bool(outcome.terminated), "truncated": bool(outcome.truncated),
                    "info": dict(outcome.info)})
                ticks_run += 1
                obs = self._capture(outcome.observation, tick + 1)
                if outcome.terminated or outcome.truncated:
                    self.queue.ledger.event("episode_finished", tick=tick, terminated=outcome.terminated, truncated=outcome.truncated)
                    break
                target = self.clock.tick_to_monotonic(tick + 1)
                remaining = target - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            # A stopped trial cannot leave executable commands or accept stale jobs.
            self.queue.intervene(self._current_tick + 1, "controller_stopped")
            self._worker.stop(timeout=0.0)
            self._pending_jobs.clear()
            self._record({"kind": "run_end", "tick": ticks_run})
            self._stop_env()

        return WallClockReport(
            ticks_run=ticks_run,
            deadline_misses=self.queue.deadline_miss_count,
            rejected_plans=sum(e["event"] == "plan_rejected" for e in self.queue.ledger.events),
            safety_stops=safety_stops,
            planner_errors=self._worker.jobs_failed,
            tick_jitter_ms=jitters,
            planner_latencies_ms=latencies,
            ticks_advanced_during_planning=ticks_advanced_during_planning,
            underruns=self.queue.underrun_count,
        )

    def drain_planner(self, timeout: float = 30.0) -> None:
        """After hold, finish the last GPU job before learner updates or reload."""
        self._worker.stop(timeout=timeout)
        if self._worker.is_alive():
            raise ContractError("planner did not drain; refusing concurrent learner updates")

    def _capture(self, observation: ObservationSnapshot, tick: int) -> ObservationSnapshot:
        return dataclasses.replace(observation, env_id=self.env_id, episode_id=self.episode_id, tick=tick)

    def _warm_start(self, obs: ObservationSnapshot) -> None:
        """Same warm-start contract as the virtual scheduler (guide 12.6)."""
        contract = self.contract
        if contract.delay == 0:
            return
        tick = self.clock.start_tick
        request = PlanRequest(
            request_id=-1,
            env_id=self.env_id,
            episode_id=self.episode_id,
            episode_generation=self.queue.episode_generation,
            capture_tick=tick,
            apply_start_tick=tick + contract.delay,
            queue_generation=self.queue.queue_generation,
            policy_version_id=self.policy_version_id,
            feature_version_id=self.feature_version_id,
            schema_version=self.schema_version,
            planning_state_version=self.planning_state_version,
            rtc_context=PrevPlanContext.empty(),
            rng_seed=self.seed,
            sampler_mode=self.rtc_mode,
            num_denoising_steps=self.num_denoising_steps,
        )
        if self.observation_recorder is not None:
            self.observation_recorder.submit(request, obs, np.zeros((contract.delay, ENV_ACTION_DIM), np.float32))
        result = self.backend.plan(
            request,
            obs,
            np.zeros((contract.delay, ENV_ACTION_DIM), dtype=np.float32),
            PrevPlanContext.empty(),
        )
        result.validate(contract)
        self.queue.commit_initial_prefix(
            np.asarray(result.reference_env_full, dtype=np.float32)[: contract.delay],
            start_tick=tick,
            policy_version_id=self.policy_version_id,
            feature_version_id=self.feature_version_id,
        )

    def _commit_outcome(self, outcome: PlannerOutcome) -> FinalPlan | None:
        if outcome.error is not None:
            self.queue.ledger.event(
                "planner_error",
                tick=self._current_tick,
                request_id=outcome.job.request.request_id,
                error=outcome.error,
            )
        request = outcome.job.request
        deadline = self.clock.tick_to_monotonic(request.apply_start_tick)
        if outcome.job.deadline_missed or outcome.finished_at_monotonic > deadline:
            if not outcome.job.deadline_missed:
                self.queue.deadline_miss_count += 1
                self.queue.ledger.event("deadline_miss", tick=self._current_tick, request_id=request.request_id, reason="wall_clock_deadline")
            self.queue.ledger.event("plan_rejected", tick=self._current_tick, request_id=request.request_id, reason="late")
            self._record({"kind": "plan", "request_id": request.request_id,
                "accepted": False, "reason": "late", "result": outcome.result})
            return None
        if outcome.error is not None:
            self._record({"kind": "plan", "request_id": request.request_id,
                "accepted": False, "reason": "planner_error", "error": outcome.error, "result": None})
            return None
        result = outcome.result
        assert result is not None
        submitted = SubmittedPlan(
            request_id=request.request_id,
            env_id=request.env_id,
            episode_id=request.episode_id,
            episode_generation=request.episode_generation,
            capture_tick=request.capture_tick,
            apply_start_tick=request.apply_start_tick,
            actions=np.asarray(result.submitted, dtype=np.float32),
            valid_mask=make_submitted_mask(self.contract),
            source=result.source,
            policy_version_id=request.policy_version_id,
            feature_version_id=request.feature_version_id,
            schema_version=request.schema_version,
            planning_state_version=request.planning_state_version,
            queue_generation=request.queue_generation,
            raw_actor_plan=None
            if result.raw_actor_plan is None
            else np.asarray(result.raw_actor_plan, dtype=np.float32),
        )
        plan = FinalPlan.compose(
            request=request,
            pending_prefix=outcome.job.pending_prefix,
            submitted_plan=submitted,
            reference_env_full=result.reference_env_full,
            contract=self.contract,
            plan_id=self.queue.next_plan_id,
        )
        ack = self.queue.commit_plan(plan, submitted, now_tick=self._current_tick)
        if ack.accepted:
            if ack.plan_id is not None:
                plan = dataclasses.replace(plan, plan_id=ack.plan_id)
            self._record({"kind": "plan", "request_id": request.request_id,
                "accepted": True, "reason": ack.reason, "plan": plan, "submitted": submitted, "result": result})
            return plan
        self._record({"kind": "plan", "request_id": request.request_id,
            "accepted": False, "reason": ack.reason, "result": result})
        return None

    def _prev_context(self, last_final: FinalPlan | None, tick: int):
        if last_final is None:
            return PrevPlanContext.empty()
        shifted, mask = last_final.shifted_for_next_request(tick)
        return PrevPlanContext(
            prev_env_plan=shifted,
            prev_valid_mask=mask,
            prev_capture_tick=last_final.capture_tick,
            prev_plan_id=last_final.plan_id,
            prev_feature_version_id=last_final.feature_version_id,
        )
