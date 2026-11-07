"""Assemble strictly time-aligned delayed transitions from scheduler events.

Responsibilities (guide sections 13.2/13.5):

- the RL action is the *submitted plan* ``U_i``; ``executed_trace`` is kept
  separately and never substituted for it;
- rewards come from the request window ``[r_i, r_i + K)`` with the real K;
- the next planning state comes from the real next capture snapshot, with
  matching feature/schema versions -- never recomputed at learner time;
- if the next request did not produce a legitimate planning state (deadline
  miss, intervention, run ended, reset), the transition is *censored with a
  reason* instead of using a fabricated ``next_ref`` bootstrap.  Censoring the
  current row is not enough: this assembler also refuses to pair it with an
  invalid successor.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from pi05_rlt_rtc import PLANNING_STATE_VERSION, SCHEMA_VERSION
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.plan import FinalPlan, PlanRequest, SubmittedPlan
from pi05_rlt_rtc.contracts.spaces import ENV_ACTION_DIM, HorizonContract, pending_from_submitted
from pi05_rlt_rtc.contracts.transition import DelayedTransition
from pi05_rlt_rtc.learning.planning_state import PlanningState

StateEncoder = Callable[..., PlanningState]


class DelayedTransitionAssembler:
    def __init__(
        self,
        *,
        contract: HorizonContract,
        state_encoder: StateEncoder,
        gamma_per_tick: float,
        schema_version: str = SCHEMA_VERSION,
        planning_state_version: str = PLANNING_STATE_VERSION,
    ) -> None:
        if not 0.0 < gamma_per_tick <= 1.0:
            raise ContractError("gamma_per_tick must be in (0, 1]")
        self.contract = contract
        self.state_encoder = state_encoder
        self.gamma_per_tick = float(gamma_per_tick)
        self.schema_version = str(schema_version)
        self.planning_state_version = str(planning_state_version)
        self.requests: dict[int, dict[str, Any]] = {}
        self.plans: dict[int, dict[str, Any]] = {}
        self.ticks: dict[int, dict[str, Any]] = {}
        self.resets: list[dict[str, Any]] = []
        self.interventions: list[dict[str, Any]] = []
        self.warm_starts: list[dict[str, Any]] = []
        self._finalized: list[DelayedTransition] = []
        self._finalized_once = False
        self._states: dict[int, PlanningState] = {}

    # -- ingestion --------------------------------------------------------------
    def ingest_run(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            self.ingest(event)

    def ingest(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        if kind == "request":
            request: PlanRequest = event["request"]
            self.requests[request.request_id] = event
        elif kind == "plan":
            self.plans[int(event["request_id"])] = event
        elif kind == "tick":
            self.ticks[int(event["tick"])] = event
        elif kind == "episode_start":
            self.resets.append(event)
        elif kind == "run_end":
            self._finalized_once = True
        elif kind in ("inflight_dropped",):
            self.interventions.append(event)
        elif kind == "warm_start":
            self.warm_starts.append(event)
        # Other event kinds are informational.

    # -- state construction ------------------------------------------------------
    def planning_state_for(self, request_id: int) -> PlanningState:
        if request_id in self._states:
            return self._states[request_id]
        if request_id not in self.requests:
            raise ContractError(f"no request event for request_id={request_id}")
        request_event = self.requests[request_id]
        plan_event = self.plans.get(request_id)
        state = self.state_encoder(
            request=request_event["request"],
            observation=request_event["observation"],
            pending_prefix=request_event["pending_prefix"],
            pending_mask=request_event["pending_mask"],
            prev_context=request_event["prev_context"],
            plan_event=plan_event,
        )
        if not isinstance(state, PlanningState):
            raise ContractError("state_encoder must return a PlanningState")
        self._states[request_id] = state
        return state

    # -- assembly ----------------------------------------------------------------
    def finalize(self) -> list[DelayedTransition]:
        if self._finalized:
            return list(self._finalized)
        request_ids = sorted(self.requests)
        transitions: list[DelayedTransition] = []
        for pos, rid in enumerate(request_ids):
            next_rid = request_ids[pos + 1] if pos + 1 < len(request_ids) else None
            transitions.append(self._assemble_one(rid, next_rid))
        self._finalized = [t for t in transitions if t is not None]
        return list(self._finalized)

    def _assemble_one(self, rid: int, next_rid: int | None) -> DelayedTransition:
        request_event = self.requests[rid]
        request: PlanRequest = request_event["request"]
        plan_event = self.plans.get(rid)
        capture = request.capture_tick

        base = dict(
            request_id=rid,
            env_id=request.env_id,
            episode_id=request.episode_id,
            episode_generation=request.episode_generation,
            capture_tick=capture,
            next_capture_tick=capture,
            plan_id=-1,
            next_plan_id=None,
            submitted_plan=np.zeros((self.contract.chunk, ENV_ACTION_DIM), dtype=np.float32),
            submitted_valid_mask=np.zeros((self.contract.chunk, ENV_ACTION_DIM), dtype=bool),
            raw_actor_plan=None,
            executed_trace=np.zeros((0, ENV_ACTION_DIM), dtype=np.float32),
            executed_source_plan_ids=np.zeros((0,), dtype=np.int64),
            primitive_rewards=np.zeros((self.contract.chunk,), dtype=np.float32),
            reward_valid_mask=np.zeros((self.contract.chunk,), dtype=bool),
            duration_ticks=0,
            planning_state=None,
            next_planning_state=None,
            bc_target_plan=None,
            bc_target_valid_mask=None,
            teacher_source=None,
            terminated=False,
            truncated=False,
            bootstrap_allowed=True,
            accepted=False,
            intervened=False,
            deadline_miss=False,
            censored=True,
            censor_reason="unassembled",
            schema_version=self.schema_version,
            planning_state_version=self.planning_state_version,
            feature_version_id=request.feature_version_id,
            norm_stats_sha256="",
            policy_version_id=request.policy_version_id,
            gamma_per_tick=self.gamma_per_tick,
        )

        if plan_event is None:
            base["censor_reason"] = "missing_plan_event"
            return DelayedTransition(**base)
        if not plan_event.get("accepted", False):
            base["deadline_miss"] = plan_event.get("reason") == "late"
            base["censor_reason"] = f"plan_not_accepted:{plan_event.get('reason')}"
            return self._with_state(base, rid)

        plan: FinalPlan = plan_event["plan"]
        submitted: SubmittedPlan = plan_event["submitted"]
        result = plan_event.get("result")
        base.update(
            submitted_plan=np.asarray(submitted.env_actions(), dtype=np.float32),
            submitted_valid_mask=np.asarray(submitted.valid_mask, dtype=bool),
            raw_actor_plan=(
                None
                if submitted.raw_actor_plan is None
                else np.asarray(submitted.raw_actor_plan, dtype=np.float32)
            ),
            accepted=True,
            censored=False,
            censor_reason=None,
            plan_id=int(plan.plan_id),
        )

        # ---- find the next capture / terminal boundary
        # The last committed action may still be executing when the operator
        # ends the episode and the next request has not produced a plan. Its
        # real terminal window can include D pending ticks + up to C own ticks.
        # Never invent a successor state or include a later accepted action.
        terminal_limit = (self.contract.delay + self.contract.chunk
                          if next_rid is None else self.contract.chunk)
        terminal_tick = self._terminal_tick_in_window(capture, terminal_limit)
        if terminal_tick is not None:
            terminal_record = self.ticks[terminal_tick]
            K = terminal_tick - capture + 1
            base["terminated"] = bool(terminal_record.get("terminated", False))
            base["truncated"] = bool(terminal_record.get("truncated", False))
            base["bootstrap_allowed"] = not base["terminated"]
            base["next_capture_tick"] = terminal_tick + 1
            base["next_planning_state"] = None
        else:
            if next_rid is None:
                base["censor_reason"] = "run_ended_without_next_request"
                return self._with_state(base, rid)
            next_request: PlanRequest = self.requests[next_rid]["request"]
            if (next_request.env_id, next_request.episode_id, next_request.episode_generation) != (request.env_id, request.episode_id, request.episode_generation):
                base["censor_reason"] = "next_episode_mismatch"
                return self._with_state(base, rid)
            next_plan_event = self.plans.get(next_rid)
            base["next_capture_tick"] = next_request.capture_tick
            K = next_request.capture_tick - capture
            if K != self.contract.chunk:
                base["censor_reason"] = f"request_period_mismatch:K={K}"
                return self._with_state(base, rid)
            if next_plan_event is None or not next_plan_event.get("accepted", False):
                base["deadline_miss"] = True
                base["censor_reason"] = (
                    "next_request_not_accepted:"
                    + str(None if next_plan_event is None else next_plan_event.get("reason"))
                )
                return self._with_state(base, rid)
            # The next state must be the real next capture with the same frozen
            # feature version; never recompute with learner-time randomness.
            if next_request.feature_version_id != request.feature_version_id:
                base["censor_reason"] = "next_feature_version_mismatch"
                return self._with_state(base, rid)
            if next_request.schema_version != request.schema_version:
                base["censor_reason"] = "next_schema_version_mismatch"
                return self._with_state(base, rid)
            next_plan: FinalPlan = next_plan_event["plan"]
            pending_next = np.asarray(next_plan.pending_prefix, dtype=np.float32)
            expected_pending = np.asarray(
                pending_from_submitted(base["submitted_plan"], self.contract), dtype=np.float32
            )
            if not np.allclose(pending_next, expected_pending, atol=1e-6, rtol=0.0):
                base["censor_reason"] = "next_pending_prefix_mismatch"
                return self._with_state(base, rid)
            base["next_plan_id"] = int(next_plan.plan_id)
            base["next_planning_state"] = self.planning_state_for(next_rid)

        max_duration = terminal_limit if terminal_tick is not None else self.contract.chunk
        if K <= 0 or K > max_duration:
            base["censor_reason"] = f"invalid_duration:K={K}"
            return self._with_state(base, rid)

        # ---- executed trace and reward window
        trace_rows: list[np.ndarray] = []
        trace_sources: list[int] = []
        rewards = np.zeros((max(self.contract.chunk, K),), dtype=np.float32)
        reward_valid = np.zeros_like(rewards, dtype=bool)
        for k in range(K):
            tick_record = self.ticks.get(capture + k)
            if tick_record is None:
                base["censor_reason"] = f"missing_tick_record:{capture + k}"
                return self._with_state(base, rid)
            action = tick_record.get("action")
            if tick_record.get("info", {}).get("reward_valid") is False:
                base["censor_reason"] = "unlabelled_physical_reward"
                return self._with_state(base, rid)
            if action is None:
                base["censor_reason"] = "fallback_in_execution_window"
                return self._with_state(base, rid)
            if tick_record.get("consume_kind") != "committed":
                base["censor_reason"] = f"non_committed_execution:{tick_record.get('consume_kind')}"
                return self._with_state(base, rid)
            trace_rows.append(np.asarray(action, dtype=np.float32))
            trace_sources.append(int(tick_record.get("source_plan_id", -1)))
            rewards[k] = float(tick_record.get("reward", 0.0))
            reward_valid[k] = True
            if tick_record.get("terminated") or tick_record.get("truncated"):
                reward_valid[k + 1 :] = False

        # Source-plan accounting: [0, D) belongs to the committed prefix, and
        # [D, K) must belong to this plan.  A mismatch means the committed
        # prefix was violated (T43) or the plan was pasted incorrectly.
        if self.contract.delay > 0:
            prefix_sources = trace_sources[: self.contract.delay]
            if any(src == base["plan_id"] for src in prefix_sources):
                base["censor_reason"] = "prefix_overwritten_by_new_plan"
                return self._with_state(base, rid)
        if K > self.contract.delay:
            for src in trace_sources[self.contract.delay :]:
                if src != base["plan_id"]:
                    base["censor_reason"] = "submitted_plan_not_executed_as_committed"
                    return self._with_state(base, rid)

        base["executed_trace"] = np.stack(trace_rows, axis=0)
        base["duration_ticks"] = K
        base["executed_source_plan_ids"] = np.asarray(trace_sources, dtype=np.int64)
        base["primitive_rewards"] = rewards
        base["reward_valid_mask"] = reward_valid

        # ---- BC target: teacher plan if provided, else this plan's reference suffix
        if result is not None and getattr(result, "bc_target_plan", None) is not None:
            base["bc_target_plan"] = np.asarray(result.bc_target_plan, dtype=np.float32)
            base["bc_target_valid_mask"] = (
                None
                if getattr(result, "bc_target_valid_mask", None) is None
                else np.asarray(result.bc_target_valid_mask, dtype=np.float32)
            )
            base["teacher_source"] = getattr(result, "teacher_source", "teacher")
        else:
            if result is None:
                base["censor_reason"] = "missing_reference_for_bc"
                return self._with_state(base, rid)
            # The composed plan contains the actor's U, not the teacher's reference.
            suffix = np.asarray(result.reference_env_full)[self.contract.suffix_start : self.contract.suffix_end, :]
            base["bc_target_plan"] = np.asarray(suffix, dtype=np.float32)
            base["bc_target_valid_mask"] = np.ones_like(base["bc_target_plan"], dtype=np.float32)
            base["teacher_source"] = "rtc_reference_suffix"
        if base["truncated"] and not base["terminated"] and base["next_planning_state"] is None:
            base["bootstrap_allowed"] = False
            base["censor_reason"] = "truncation_without_terminal_planning_state"
        return self._with_state(base, rid)

    def _with_state(self, base: dict[str, Any], rid: int) -> DelayedTransition:
        state = self.planning_state_for(rid)
        base["planning_state"] = state
        if base.get("norm_stats_sha256") == "":
            base["norm_stats_sha256"] = state.norm_stats_sha256
        base["feature_version_id"] = state.ids.feature_version_id
        if base.get("censor_reason"):
            # Any branch that recorded a reason must be marked censored so the
            # learner filters it; a reason alone is not enough.
            base["censored"] = True
            base["accepted"] = base.get("accepted", False)
        return DelayedTransition(**base)

    def _terminal_tick_in_window(self, capture: int, limit: int) -> int | None:
        for k in range(limit):
            record = self.ticks.get(capture + k)
            if record is not None and (record.get("terminated") or record.get("truncated")):
                return capture + k
        return None
