"""Virtual-delay scheduler: the deterministic first-class execution model.

This mode implements the exact guide-section-12 protocol with an integer tick
clock: requests every C ticks, a fixed delay of D primitive ticks, committed
prefix preservation and an explicit planning latency in ticks.  Wall-clock
timing is *not* measured here; this mode is for training and for proving the
queue/reward semantics (guide section 15.2).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Protocol

import numpy as np

from pi05_rlt_rtc.contracts.errors import ContractError, ControlSafetyError
from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
from pi05_rlt_rtc.contracts.plan import (
    FinalPlan,
    PlanRequest,
    PrevPlanContext,
    SubmittedPlan,
    make_submitted_mask,
)
from pi05_rlt_rtc.contracts.spaces import ENV_ACTION_DIM, HorizonContract
from pi05_rlt_rtc.runtime.ledger import ExecutionLedger
from pi05_rlt_rtc.runtime.queue import ActionQueue

# ---------------------------------------------------------------------------
# protocols


@dataclasses.dataclass
class StepOutcome:
    observation: ObservationSnapshot
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = dataclasses.field(default_factory=dict)


class EnvProtocol(Protocol):
    def reset(self, *, seed: int = 0) -> ObservationSnapshot: ...

    def step(self, action: np.ndarray) -> StepOutcome: ...


class SnapshotFreeEvents(list):
    """Real collectors retain captured planning states, not every RGB frame."""
    def append(self, event):
        event = dict(event)
        for key in ("observation", "observation_before", "observation_after"):
            if key in event:
                event[key] = None
        super().append(event)


@dataclasses.dataclass
class PlanResult:
    """What a planner returns: the submitted chunk and the full-H reference."""

    submitted: np.ndarray  # [C, 8]
    reference_env_full: np.ndarray  # [H, 8]
    source: str = "reference_teacher"
    raw_actor_plan: np.ndarray | None = None
    bc_target_plan: np.ndarray | None = None
    bc_target_valid_mask: np.ndarray | None = None
    teacher_source: str | None = None
    planning_state: Any | None = None  # captured once by the real planner; not sent over inference RPC

    def validate(self, contract: HorizonContract) -> None:
        sub = np.asarray(self.submitted, dtype=np.float32)
        ref = np.asarray(self.reference_env_full, dtype=np.float32)
        if sub.shape != (contract.chunk, ENV_ACTION_DIM):
            raise ContractError(f"PlanResult.submitted must be [C, 8], got {sub.shape}")
        if ref.shape != (contract.horizon, ENV_ACTION_DIM):
            raise ContractError(
                f"PlanResult.reference_env_full must be full H [H, 8], got {ref.shape}; "
                "the reference is never truncated to C"
            )
        if not np.all(np.isfinite(sub)) or not np.all(np.isfinite(ref)):
            raise ContractError("PlanResult contains NaN/Inf; the whole plan is invalid")


class PlannerProtocol(Protocol):
    def plan(
        self,
        request: PlanRequest,
        observation: ObservationSnapshot,
        pending_prefix: np.ndarray,
        prev_context: PrevPlanContext,
    ) -> PlanResult: ...


# ---------------------------------------------------------------------------
# run log


@dataclasses.dataclass
class RunLog:
    events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    ticks: int = 0
    deadline_misses: int = 0
    rejected_plans: int = 0
    safety_stops: int = 0
    episodes: int = 0

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["kind"] == kind]


class VirtualDelayScheduler:
    def __init__(
        self,
        *,
        contract: HorizonContract,
        queue: ActionQueue,
        backend: PlannerProtocol,
        env: EnvProtocol,
        planner_latency_ticks: int = 0,
        seed: int = 0,
        env_id: str = "env0",
        stop_on_safety: bool = True,
        stop_on_episode_end: bool = False,
        retain_observations: bool = True,
        rtc_mode: str = "full_vjp",
        feature_version_id: str = "unset",
        policy_version_id: str = "unset",
        schema_version: str = "pi05-rlt-rtc.schema.v1",
        planning_state_version: str = "queue-planning-state.v1",
        num_denoising_steps: int = 5,
    ) -> None:
        if planner_latency_ticks < 0:
            raise ContractError("planner_latency_ticks must be >= 0")
        self.contract = contract
        self.queue = queue
        self.backend = backend
        self.env = env
        self.planner_latency_ticks = int(planner_latency_ticks)
        self.seed = int(seed)
        self.env_id = env_id
        self.stop_on_safety = stop_on_safety
        self.stop_on_episode_end = stop_on_episode_end
        self.retain_observations = retain_observations
        self.rtc_mode = rtc_mode
        self.feature_version_id = feature_version_id
        self.policy_version_id = policy_version_id
        self.schema_version = schema_version
        self.planning_state_version = planning_state_version
        self.num_denoising_steps = num_denoising_steps

    def run(self, num_ticks: int) -> RunLog:
        if not isinstance(num_ticks, int) or isinstance(num_ticks, bool) or num_ticks <= 0:
            raise ContractError("num_ticks must be a positive integer")
        contract = self.contract
        log = RunLog()
        if not self.retain_observations:
            log.events = SnapshotFreeEvents()
        episode_index = 0
        episode_id = f"{self.env_id}-ep{episode_index}"
        self.queue.reset(env_id=self.env_id, episode_id=episode_id, start_tick=0)
        obs = dataclasses.replace(self.env.reset(seed=self.seed), env_id=self.env_id, episode_id=episode_id, tick=0)
        log.episodes += 1

        request_id = 0
        episode_start_tick = 0
        next_request_tick = 0
        inflight: dict[int, dict[str, Any]] = {}
        last_final: FinalPlan | None = None
        trial_over = False

        # Synchronous warm start establishes P_0 before the first request.
        self._warm_start(obs, log)

        tick = 0
        while tick < num_ticks:
            # -- 1. deliver any planning responses whose simulated latency elapsed
            for rid in sorted(list(inflight)):
                item = inflight[rid]
                if item["ready_tick"] > tick:
                    continue
                self._complete_plan(item, log, tick, last_final)
                if item.get("plan") is not None:
                    last_final = item["plan"]
                del inflight[rid]
            # -- 2. issue a new request every C ticks (relative to episode start)
            if not trial_over and (tick - episode_start_tick) % contract.chunk == 0:
                pending, pending_mask = self.queue.snapshot_pending(tick)
                prev_context = self._prev_context(last_final, tick)
                request = PlanRequest(
                    request_id=request_id,
                    env_id=self.env_id,
                    episode_id=episode_id,
                    episode_generation=self.queue.episode_generation,
                    capture_tick=tick,
                    apply_start_tick=tick + contract.delay,
                    queue_generation=self.queue.queue_generation,
                    policy_version_id=self.policy_version_id,
                    feature_version_id=self.feature_version_id,
                    schema_version=self.schema_version,
                    planning_state_version=self.planning_state_version,
                    rtc_context=prev_context,
                    rng_seed=self.seed * 1000003 + request_id,
                    sampler_mode=self.rtc_mode,
                    num_denoising_steps=self.num_denoising_steps,
                    deadline_tick=tick + contract.delay,
                    last_command=None if self.queue.last_command is None else self.queue.last_command.copy(),
                )
                log.events.append(
                    {
                        "kind": "request",
                        "tick": tick,
                        "request": request,
                        "observation": obs,
                        "pending_prefix": pending,
                        "pending_mask": pending_mask,
                        "prev_context": prev_context,
                    }
                )
                inflight[request_id] = {
                    "request": request,
                    "observation": obs,
                    "pending_prefix": pending,
                    "pending_mask": pending_mask,
                    "prev_context": prev_context,
                    "ready_tick": tick + self.planner_latency_ticks,
                }
                request_id += 1

                # A zero-latency planner is synchronous: deliver its plan
                # before this tick's command is consumed (the D=0 regression
                # contract needs the plan to land before execution).
                if self.planner_latency_ticks == 0 and (request_id - 1) in inflight:
                    new_item = inflight.pop(request_id - 1)
                    self._complete_plan(new_item, log, tick, last_final)
                    if new_item.get("plan") is not None:
                        last_final = new_item["plan"]

            # -- 3. consume one command and step the environment
            consumed = self.queue.consume(tick)
            reward = 0.0
            terminated = False
            truncated = False
            obs_before = obs
            if consumed.safe_stop or consumed.action is None:
                log.safety_stops += 1
                log.events.append(
                    {
                        "kind": "tick",
                        "tick": tick,
                        "action": None,
                        "consume_kind": consumed.kind,
                        "source_plan_id": consumed.source_plan_id,
                        "request_id": consumed.request_id,
                        "reward": 0.0,
                        "terminated": False,
                        "truncated": False,
                        "observation_before": obs_before,
                        "observation_after": obs_before,
                        "note": consumed.note,
                    }
                )
                trial_over = True
                if self.stop_on_safety:
                    self.queue.ledger.event("trial_stopped", tick=tick, reason=consumed.note)
                    break
            else:
                outcome = self.env.step(np.asarray(consumed.action, dtype=np.float32))
                reward = float(outcome.reward)
                terminated = bool(outcome.terminated)
                truncated = bool(outcome.truncated)
                obs = dataclasses.replace(outcome.observation, env_id=self.env_id, episode_id=episode_id, tick=tick + 1)
                log.events.append(
                    {
                        "kind": "tick",
                        "tick": tick,
                        "action": np.asarray(consumed.action, dtype=np.float32).copy(),
                        "consume_kind": consumed.kind,
                        "source_plan_id": consumed.source_plan_id,
                        "request_id": consumed.request_id,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "observation_before": obs_before,
                        "observation_after": obs,
                        "info": outcome.info,
                    }
                )

            # -- 4. episode boundary
            if terminated or truncated:
                dropped = len(inflight)
                for rid in list(inflight):
                    log.events.append({"kind": "inflight_dropped", "tick": tick, "request_id": rid})
                    del inflight[rid]
                if dropped:
                    self.queue.ledger.event("inflight_dropped", tick=tick, count=dropped)
                if self.stop_on_episode_end:
                    tick += 1
                    log.ticks = tick
                    break
                episode_index += 1
                episode_id = f"{self.env_id}-ep{episode_index}"
                episode_start_tick = tick + 1
                last_final = None
                trial_over = False
                if episode_start_tick < num_ticks:
                    self.queue.reset(
                        env_id=self.env_id,
                        episode_id=episode_id,
                        start_tick=episode_start_tick,
                    )
                    obs = dataclasses.replace(self.env.reset(seed=self.seed + episode_index), env_id=self.env_id, episode_id=episode_id, tick=episode_start_tick)
                    log.episodes += 1
                    log.events.append({"kind": "episode_start", "tick": episode_start_tick, "episode_id": episode_id})
                    self._warm_start(obs, log)

            tick += 1
            log.ticks = tick

        log.deadline_misses = self.queue.deadline_miss_count
        log.rejected_plans = sum(
            1 for e in log.events if e["kind"] == "plan" and not e.get("accepted", False)
        )
        # The final plan of a still-running episode is speculative; do not
        # fabricate a next state for it.
        log.events.append({"kind": "run_end", "tick": tick})
        return log

    # -- helpers --------------------------------------------------------------
    def _warm_start(self, obs: ObservationSnapshot, log: RunLog) -> None:
        """Commit ``P_0`` from one synchronous initial reference pass (D > 0)."""
        contract = self.contract
        if contract.delay == 0:
            return
        warm_request = PlanRequest(
            request_id=-1,
            env_id=self.env_id,
            episode_id=self.queue.episode_id or f"{self.env_id}-ep0",
            episode_generation=self.queue.episode_generation,
            capture_tick=self.queue.cursor_tick if self.queue.cursor_tick is not None else 0,
            apply_start_tick=(self.queue.cursor_tick if self.queue.cursor_tick is not None else 0)
            + contract.delay,
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
        try:
            result = self.backend.plan(
                warm_request, obs, np.zeros((contract.delay, ENV_ACTION_DIM), dtype=np.float32), PrevPlanContext.empty()
            )
            result.validate(contract)
            start_tick = self.queue.cursor_tick if self.queue.cursor_tick is not None else 0
            plan_id = self.queue.commit_initial_prefix(
                np.asarray(result.reference_env_full, dtype=np.float32)[: contract.delay],
                start_tick=start_tick,
                policy_version_id=self.policy_version_id,
                feature_version_id=self.feature_version_id,
            )
            log.events.append(
                {
                    "kind": "warm_start",
                    "tick": start_tick,
                    "plan_id": plan_id,
                    "result": result,
                    "observation": obs,
                }
            )
        except Exception as exc:  # noqa: BLE001 - record and let the run fail loudly
            log.events.append(
                {"kind": "warm_start_failed", "tick": self.queue.cursor_tick or 0, "error": str(exc)}
            )
            self.queue.ledger.event("warm_start_failed", error=repr(exc))
            raise

    def _prev_context(self, last_final: FinalPlan | None, tick: int) -> PrevPlanContext:
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

    def _complete_plan(
        self,
        item: dict[str, Any],
        log: RunLog,
        now_tick: int,
        last_final: FinalPlan | None,
    ) -> None:
        request: PlanRequest = item["request"]
        try:
            result = self.backend.plan(
                request, item["observation"], item["pending_prefix"], item["prev_context"]
            )
            result.validate(self.contract)
        except Exception as exc:  # noqa: BLE001 - planner failures must be recorded
            log.events.append(
                {
                    "kind": "plan",
                    "tick": now_tick,
                    "request_id": request.request_id,
                    "accepted": False,
                    "reason": f"planner_error:{type(exc).__name__}",
                    "error": str(exc),
                    "plan": None,
                    "result": None,
                }
            )
            self.queue.ledger.event(
                "planner_error", tick=now_tick, request_id=request.request_id, error=repr(exc)
            )
            return

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
        try:
            plan = FinalPlan.compose(
                request=request,
                pending_prefix=item["pending_prefix"],
                submitted_plan=submitted,
                reference_env_full=result.reference_env_full,
                contract=self.contract,
                plan_id=self.queue.next_plan_id,
            )
        except Exception as exc:  # noqa: BLE001
            log.events.append(
                {
                    "kind": "plan",
                    "tick": now_tick,
                    "request_id": request.request_id,
                    "accepted": False,
                    "reason": f"compose_error:{type(exc).__name__}",
                    "error": str(exc),
                    "plan": None,
                    "result": result,
                }
            )
            return

        ack = self.queue.commit_plan(plan, submitted, now_tick=now_tick)
        if ack.accepted and ack.plan_id is not None:
            plan = dataclasses.replace(plan, plan_id=ack.plan_id)
        item["plan"] = plan if ack.accepted else None
        log.events.append(
            {
                "kind": "plan",
                "tick": now_tick,
                "request_id": request.request_id,
                "accepted": ack.accepted,
                "reason": ack.reason,
                "ack": ack,
                "plan": plan if ack.accepted else None,
                "result": result,
                "submitted": submitted,
            }
        )

    def _record_request_context(
        self,
        log: RunLog,
        *,
        request: PlanRequest | None,
        obs: ObservationSnapshot,
        pending: np.ndarray | None,
        prev_context: PrevPlanContext | None,
        event: str,
    ) -> None:
        log.events.append(
            {
                "kind": event,
                "observation": obs,
                "pending_prefix": pending,
                "prev_context": prev_context,
                "request": request,
            }
        )


class NumberedReferencePlanner:
    """Deterministic numbered-action planner for timing/queue tests.

    ``U_i[k] = ((i + 1) * 1000 + k) * unit`` so every index error is visible.
    ``A_ref[l] = ((i + 1) * 100 + l) * unit``.
    """

    def __init__(self, contract: HorizonContract, *, unit: float = 1e-4) -> None:
        self.contract = contract
        self.unit = float(unit)

    def plan(self, request, observation, pending_prefix, prev_context) -> PlanResult:
        i = request.request_id
        if i < 0:
            submitted = np.stack(
                [np.full(ENV_ACTION_DIM, (9000 + k) * self.unit) for k in range(self.contract.chunk)]
            ).astype(np.float32)
            reference = np.stack(
                [np.full(ENV_ACTION_DIM, (9000 + l) * self.unit) for l in range(self.contract.horizon)]
            ).astype(np.float32)
            return PlanResult(submitted=submitted, reference_env_full=reference, source="reference_teacher")
        submitted = np.stack(
            [np.full(ENV_ACTION_DIM, ((i + 1) * 1000 + k) * self.unit) for k in range(self.contract.chunk)]
        ).astype(np.float32)
        reference = np.stack(
            [np.full(ENV_ACTION_DIM, ((i + 1) * 100 + l) * self.unit) for l in range(self.contract.horizon)]
        ).astype(np.float32)
        return PlanResult(submitted=submitted, reference_env_full=reference)
