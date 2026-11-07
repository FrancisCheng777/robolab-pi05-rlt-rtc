"""Fixed-delay action queue with transactional commits.

Guide section 12: the controller owns the queue and consumes exactly one
primitive command per tick; commits are atomic and versioned; the waiting
planner never blocks a control tick.

Invariants enforced here:

- a response earlier than its ``apply_start_tick`` is simply held;
- a response later than ``apply_start_tick`` is rejected (never pasted at a
  random offset into the future);
- duplicate request ids are idempotent ACKs, never double insertions;
- out-of-order/episode-mismatched responses are rejected;
- the committed prefix ``P`` is checked against the queue before ``U`` lands;
- speculative reference tails are never executed without a fresh commit;
- normal consumption does not bump the queue generation, so in-flight
  requests stay valid (only reset/safety bumps it).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from pi05_rlt_rtc.contracts.errors import ContractError, PlanValidationError
from pi05_rlt_rtc.contracts.plan import FinalPlan, SubmittedPlan
from pi05_rlt_rtc.contracts.spaces import ENV_ACTION_DIM, HorizonContract
from pi05_rlt_rtc.runtime.ledger import ExecutionLedger, LedgerEntry

FALLBACK_POLICIES = ("stop", "hold_last", "zero_delta")


@dataclasses.dataclass
class PlanAck:
    accepted: bool
    reason: str
    request_id: int
    plan_id: int | None
    apply_start_tick: int
    queue_generation: int
    duplicate: bool = False
    committed_start_tick: int | None = None
    committed_end_tick: int | None = None

    def __bool__(self) -> bool:
        return self.accepted

    def to_record(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class QueueEntry:
    tick: int
    action: np.ndarray
    kind: str  # "committed" | "speculative" | "safety"
    plan_id: int
    request_id: int
    source: str
    policy_version_id: str
    feature_version_id: str
    episode_generation: int


@dataclasses.dataclass
class ConsumeResult:
    tick: int
    action: np.ndarray | None
    kind: str
    safe_stop: bool
    source_plan_id: int | None
    request_id: int | None
    note: str | None = None


class ActionQueue:
    def __init__(
        self,
        contract: HorizonContract,
        *,
        env_action_low: Any | None = None,
        env_action_high: Any | None = None,
        fallback_policy: str = "stop",
        ledger: ExecutionLedger | None = None,
        env_action_tolerance: float = 1e-5,
    ) -> None:
        self.contract = contract
        if fallback_policy not in FALLBACK_POLICIES:
            raise ContractError(f"unknown fallback_policy {fallback_policy!r}")
        self.fallback_policy = fallback_policy
        self.low = np.broadcast_to(
            np.full(ENV_ACTION_DIM, -1.0) if env_action_low is None else env_action_low,
            (ENV_ACTION_DIM,),
        ).astype(np.float64)
        self.high = np.broadcast_to(
            np.full(ENV_ACTION_DIM, 1.0) if env_action_high is None else env_action_high,
            (ENV_ACTION_DIM,),
        ).astype(np.float64)
        if not np.isfinite(self.low).all() or not np.isfinite(self.high).all() or np.any(self.high <= self.low):
            raise ContractError("env_action_high must be > env_action_low")
        self.ledger = ledger or ExecutionLedger()
        self.entries: dict[int, QueueEntry] = {}
        # Consumed entries are retained for a bounded window so that prefix
        # consistency can still be checked for responses that arrive up to D
        # ticks after capture (`now_tick <= apply_start`).  This is an audit
        # window, not an execution re-authorisation: consume() never replays it.
        self._recent_consumed: dict[int, QueueEntry] = {}
        self._recent_window = max(2 * int(contract.horizon), 4 * int(contract.chunk))
        self.next_plan_id = 0
        self.last_accepted_request_id = -1
        self._acks: dict[int, PlanAck] = {}
        self.cursor_tick: int | None = None
        self.last_command: np.ndarray | None = None
        self.force_fallback: bool = False
        self.env_id: str | None = None
        self.episode_id: str | None = None
        self.episode_generation: int = 0
        self.queue_generation: int = 0
        self.deadline_miss_count = 0
        self.underrun_count = 0

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        *,
        env_id: str,
        episode_id: str,
        episode_generation: int | None = None,
        start_tick: int = 0,
    ) -> None:
        """Clear all commitments and bump the generation (reset/e-stop/task switch)."""
        self.env_id = env_id
        self.episode_id = episode_id
        self.episode_generation = (
            self.episode_generation + 1 if episode_generation is None else int(episode_generation)
        )
        self.queue_generation += 1
        self.entries.clear()
        self._recent_consumed.clear()
        self._acks.clear()
        self.last_accepted_request_id = -1
        self.cursor_tick = int(start_tick)
        self.last_command = None
        self.force_fallback = False
        self.ledger.event(
            "reset",
            tick=start_tick,
            env_id=env_id,
            episode_id=episode_id,
            episode_generation=self.episode_generation,
            queue_generation=self.queue_generation,
        )

    def _check_identity(self, env_id: str, episode_id: str, episode_generation: int) -> str | None:
        if self.env_id is None:
            return "queue_not_reset"
        if env_id != self.env_id or episode_id != self.episode_id:
            return "episode_mismatch"
        if int(episode_generation) != self.episode_generation:
            return "generation_mismatch"
        return None

    # --------------------------------------------------------------- snapshot
    def snapshot_pending(self, tick: int, delay: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Committed prefix covering ``[tick, tick + D)`` with a validity mask.

        Looks at still-queued entries first and then at the bounded recent
        consumption window, so a response arriving after some prefix ticks have
        already executed can still be checked against what was executed.
        """
        delay = self.contract.delay if delay is None else int(delay)
        actions = np.zeros((delay, ENV_ACTION_DIM), dtype=np.float32)
        valid = np.zeros((delay, ENV_ACTION_DIM), dtype=bool)
        for offset in range(delay):
            entry = self.entries.get(tick + offset) or self._recent_consumed.get(tick + offset)
            if entry is not None and entry.kind == "committed":
                actions[offset] = entry.action
                valid[offset] = True
        return actions, valid

    def pending_plan_ids(self, tick: int, delay: int | None = None) -> list[int | None]:
        delay = self.contract.delay if delay is None else int(delay)
        ids: list[int | None] = []
        for offset in range(delay):
            entry = self.entries.get(tick + offset) or self._recent_consumed.get(tick + offset)
            ids.append(entry.plan_id if entry is not None and entry.kind == "committed" else None)
        return ids

    # ----------------------------------------------------------------- commit
    def commit_initial_prefix(
        self,
        actions: Any,
        *,
        start_tick: int,
        source: str = "reference_warm_start",
        policy_version_id: str = "unset",
        feature_version_id: str = "unset",
    ) -> int:
        """Insert the synchronous warm-start prefix ``P_0`` (guide section 12.6).

        The first ``D`` commands of an initial reference pass are committed
        before the first request; this is an explicit startup cost, not a
        silent free action.  Only allowed on a fresh (plan-free) queue.
        """
        arr = np.asarray(actions, dtype=np.float32)
        if arr.shape != (self.contract.delay, ENV_ACTION_DIM):
            raise ContractError(
                f"initial prefix must be [D, 8] = [{self.contract.delay}, 8], got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ContractError("initial prefix contains NaN/Inf")
        if np.any(arr < self.low - 1e-5) or np.any(arr > self.high + 1e-5):
            raise ContractError("initial prefix is outside the controller command range")
        if self._acks or self.entries:
            raise ContractError(
                "warm start must run before any plan commit; refusing to overwrite existing commitments"
            )
        plan_id = self.next_plan_id
        self.next_plan_id += 1
        for offset in range(self.contract.delay):
            self.entries[int(start_tick) + offset] = QueueEntry(
                tick=int(start_tick) + offset,
                action=np.asarray(arr[offset], dtype=np.float32).copy(),
                kind="committed",
                plan_id=plan_id,
                request_id=-1,
                source=source,
                policy_version_id=policy_version_id,
                feature_version_id=feature_version_id,
                episode_generation=self.episode_generation,
            )
        self.ledger.event(
            "warm_start_committed",
            tick=int(start_tick),
            plan_id=plan_id,
            prefix_len=self.contract.delay,
            source=source,
        )
        return plan_id

    def commit_plan(
        self,
        final_plan: FinalPlan,
        submitted: SubmittedPlan,
        *,
        now_tick: int,
        deadline_tick: int | None = None,
    ) -> PlanAck:
        """Atomically install ``U`` at ``[apply_start, apply_start + C)``.

        ``now_tick`` is the tick the controller is about to execute.  Early
        commits (``now_tick < apply_start``) are held by the timeline; late
        commits (``now_tick > apply_start``) are rejected as deadline misses.
        """
        contract = self.contract
        submitted.validate(contract)
        final_plan.submitted_valid_mask  # touched for contract clarity
        request_id = submitted.request_id
        apply_start = submitted.apply_start_tick

        identity_error = self._check_identity(
            submitted.env_id, submitted.episode_id, submitted.episode_generation
        )
        if identity_error is not None:
            return self._reject(request_id, identity_error, apply_start)

        if int(submitted.queue_generation) != self.queue_generation:
            return self._reject(request_id, "generation_mismatch", apply_start)

        if request_id in self._acks:
            existing = self._acks[request_id]
            return dataclasses.replace(existing, reason="duplicate", duplicate=True)

        if request_id <= self.last_accepted_request_id:
            return self._reject(request_id, "stale_request", apply_start)

        if deadline_tick is not None and now_tick > int(deadline_tick):
            self.deadline_miss_count += 1
            self.ledger.event("deadline_miss", tick=now_tick, request_id=request_id, reason="deadline")
            return self._reject(request_id, "late", apply_start)

        if now_tick > apply_start or (self.cursor_tick is not None and apply_start < self.cursor_tick):
            self.deadline_miss_count += 1
            self.ledger.event(
                "deadline_miss",
                tick=now_tick,
                request_id=request_id,
                apply_start_tick=apply_start,
                reason="later_than_apply_start",
            )
            return self._reject(request_id, "late", apply_start)

        if not np.all(np.isfinite(final_plan.final)):
            return self._reject(request_id, "nonfinite", apply_start)
        for name in ("request_id", "env_id", "episode_id", "episode_generation", "capture_tick", "source", "policy_version_id", "feature_version_id", "schema_version"):
            if getattr(final_plan, name) != getattr(submitted, name):
                return self._reject(request_id, "plan_mismatch", apply_start)
        expected_shapes = {
            "pending_prefix": (contract.delay, ENV_ACTION_DIM),
            "submitted": (contract.chunk, ENV_ACTION_DIM),
            "reference_tail": (contract.horizon - contract.suffix_end, ENV_ACTION_DIM),
            "final": (contract.horizon, ENV_ACTION_DIM),
        }
        if any(np.asarray(getattr(final_plan, name)).shape != shape for name, shape in expected_shapes.items()):
            return self._reject(request_id, "plan_mismatch", apply_start)
        composed = np.concatenate([final_plan.pending_prefix, submitted.actions, final_plan.reference_tail])
        if (not np.array_equal(final_plan.submitted, submitted.actions)
                or not np.array_equal(final_plan.final, composed)
                or not np.array_equal(final_plan.submitted_valid_mask, submitted.valid_mask)):
            return self._reject(request_id, "plan_mismatch", apply_start)
        submitted_actions = submitted.env_actions()
        if not np.all(np.isfinite(submitted_actions)):
            return self._reject(request_id, "nonfinite", apply_start)
        tol = 1e-5
        if np.any(submitted_actions < self.low - tol) or np.any(
            submitted_actions > self.high + tol
        ):
            self.ledger.event(
                "plan_rejected_out_of_range",
                tick=now_tick,
                request_id=request_id,
                min=float(np.min(submitted_actions)),
                max=float(np.max(submitted_actions)),
            )
            return self._reject(request_id, "out_of_range", apply_start)

        # Prefix commitment consistency: the passed P must equal the committed
        # queue contents for [capture_tick, capture_tick + D).
        committed_now, committed_valid = self.snapshot_pending(submitted.capture_tick)
        expected = np.asarray(final_plan.pending_prefix, dtype=np.float32)
        if contract.delay > 0 and not np.equal(committed_valid, np.ones_like(committed_valid)).all():
            return self._reject(request_id, "prefix_missing", apply_start)
        if contract.delay > 0 and not np.allclose(committed_now, expected, atol=1e-6, rtol=0.0):
            self.ledger.event(
                "prefix_mismatch",
                tick=now_tick,
                request_id=request_id,
                committed=committed_now.tolist(),
                expected=expected.tolist(),
            )
            return self._reject(request_id, "prefix_mismatch", apply_start)

        for offset in range(contract.chunk):
            existing = self.entries.get(apply_start + offset)
            if existing is not None and existing.kind == "committed":
                return self._reject(request_id, "committed_overwrite", apply_start)

        # Record how far the new plan deviates from the committed prefix inside
        # its own first D steps.  This is diagnostics only: P is never rewritten.
        if contract.delay > 0:
            deviation = float(
                np.max(
                    np.abs(
                        np.asarray(submitted_actions[: contract.delay], dtype=np.float32)
                        - expected[: contract.delay]
                    )
                )
            )
            if deviation > 1e-6:
                self.ledger.event(
                    "guidance_prefix_deviation",
                    tick=now_tick,
                    request_id=request_id,
                    max_abs_deviation=deviation,
                )

        plan_id = self.next_plan_id
        self.next_plan_id += 1
        for offset in range(contract.chunk):
            self.entries[apply_start + offset] = QueueEntry(
                tick=apply_start + offset,
                action=np.asarray(submitted_actions[offset], dtype=np.float32).copy(),
                kind="committed",
                plan_id=plan_id,
                request_id=request_id,
                source=submitted.source,
                policy_version_id=submitted.policy_version_id,
                feature_version_id=submitted.feature_version_id,
                episode_generation=submitted.episode_generation,
            )

        tail_start = apply_start + contract.chunk
        tail_end = submitted.capture_tick + contract.horizon
        for tick in range(tail_start, tail_end):
            offset = tick - tail_start
            if offset >= final_plan.reference_tail.shape[0]:
                break
            existing = self.entries.get(tick)
            if existing is not None and existing.kind == "committed":
                continue
            self.entries[tick] = QueueEntry(
                tick=tick,
                action=np.asarray(final_plan.reference_tail[offset], dtype=np.float32).copy(),
                kind="speculative",
                plan_id=plan_id,
                request_id=request_id,
                source="speculative_reference",
                policy_version_id=submitted.policy_version_id,
                feature_version_id=submitted.feature_version_id,
                episode_generation=submitted.episode_generation,
            )

        ack = PlanAck(
            accepted=True,
            reason="accepted",
            request_id=request_id,
            plan_id=plan_id,
            apply_start_tick=apply_start,
            queue_generation=self.queue_generation,
            duplicate=False,
            committed_start_tick=apply_start,
            committed_end_tick=apply_start + contract.chunk,
        )
        self._acks[request_id] = ack
        self.last_accepted_request_id = request_id
        self.ledger.event(
            "plan_committed",
            tick=now_tick,
            request_id=request_id,
            plan_id=plan_id,
            apply_start_tick=apply_start,
            source=submitted.source,
        )
        return ack

    def _reject(self, request_id: int, reason: str, apply_start: int) -> PlanAck:
        self.ledger.event("plan_rejected", request_id=request_id, reason=reason)
        return PlanAck(
            accepted=False,
            reason=reason,
            request_id=request_id,
            plan_id=None,
            apply_start_tick=apply_start,
            queue_generation=self.queue_generation,
        )

    # ---------------------------------------------------------------- consume
    def consume(self, tick: int) -> ConsumeResult:
        """Consume exactly one primitive command for ``tick`` (never blocks)."""
        if self.cursor_tick is not None and tick > self.cursor_tick:
            for skipped in range(self.cursor_tick, tick):
                self._record_underrun(skipped, "skipped_tick")
        if self.cursor_tick is not None and tick < self.cursor_tick:
            raise ContractError(f"tick {tick} already consumed (cursor={self.cursor_tick})")

        entry = self.entries.pop(tick, None)
        if entry is not None:
            self._recent_consumed[tick] = entry
            self._prune_recent(tick)

        if self.force_fallback or entry is None or entry.kind == "speculative":
            reason = (
                "safety_fallback"
                if self.force_fallback
                else ("speculative_not_authorized" if entry is not None else "queue_underrun")
            )
            result = self._fallback(tick, reason)
            if entry is not None and entry.kind == "speculative":
                self.ledger.event(
                    "speculative_blocked", tick=tick, plan_id=entry.plan_id, request_id=entry.request_id
                )
            self._record_ledger(tick, result, None)
            self.cursor_tick = tick + 1
            return result

        result = ConsumeResult(
            tick=tick,
            action=entry.action.copy(),
            kind=entry.kind,
            safe_stop=False,
            source_plan_id=entry.plan_id,
            request_id=entry.request_id,
        )
        self.last_command = entry.action.copy()
        self._record_ledger(tick, result, entry)
        self.cursor_tick = tick + 1
        return result

    def _fallback(self, tick: int, reason: str) -> ConsumeResult:
        self.underrun_count += 1
        if self.force_fallback or self.fallback_policy == "stop":
            return ConsumeResult(
                tick=tick,
                action=None,
                kind="stop",
                safe_stop=True,
                source_plan_id=None,
                request_id=None,
                note=reason,
            )
        if self.fallback_policy == "hold_last":
            if self.last_command is None:
                return ConsumeResult(
                    tick=tick,
                    action=None,
                    kind="stop",
                    safe_stop=True,
                    source_plan_id=None,
                    request_id=None,
                    note=f"{reason}:no_previous_command",
                )
            return ConsumeResult(
                tick=tick,
                action=self.last_command.copy(),
                kind="hold_last",
                safe_stop=False,
                source_plan_id=None,
                request_id=None,
                note=reason,
            )
        return ConsumeResult(
            tick=tick,
            action=np.zeros(ENV_ACTION_DIM, dtype=np.float32),
            kind="zero_delta",
            safe_stop=False,
            source_plan_id=None,
            request_id=None,
            note=f"{reason}:zero_delta_is_not_a_gripper_hold",
        )

    def _record_underrun(self, tick: int, reason: str) -> None:
        self.underrun_count += 1
        self.ledger.event("underrun", tick=tick, reason=reason)

    def _prune_recent(self, current_tick: int) -> None:
        cutoff = current_tick - self._recent_window
        stale = [t for t in self._recent_consumed if t < cutoff]
        for t in stale:
            del self._recent_consumed[t]

    def _record_ledger(self, tick: int, result: ConsumeResult, entry: QueueEntry | None) -> None:
        self.ledger.append(
            LedgerEntry(
                tick=tick,
                env_id=self.env_id or "",
                episode_id=self.episode_id or "",
                episode_generation=self.episode_generation,
                queue_generation=self.queue_generation,
                command=None if result.action is None else result.action.copy(),
                kind=result.kind,
                source_plan_id=result.source_plan_id,
                request_id=result.request_id,
                policy_version_id=None if entry is None else entry.policy_version_id,
                note=result.note,
            )
        )

    # ------------------------------------------------------------- safety ops
    def intervene(self, tick: int, reason: str) -> None:
        """Safety takeover/emergency stop: invalidate the generation and stop."""
        self.force_fallback = True
        self.queue_generation += 1
        self.entries.clear()
        self._recent_consumed.clear()
        self.ledger.event("intervention", tick=tick, reason=reason, queue_generation=self.queue_generation)
        self.ledger.append(
            LedgerEntry(
                tick=tick,
                env_id=self.env_id or "",
                episode_id=self.episode_id or "",
                episode_generation=self.episode_generation,
                queue_generation=self.queue_generation,
                command=None,
                kind="intervened",
                source_plan_id=None,
                request_id=None,
                policy_version_id=None,
                note=reason,
            )
        )

    def health(self) -> dict[str, Any]:
        return {
            "cursor_tick": self.cursor_tick,
            "next_plan_id": self.next_plan_id,
            "last_accepted_request_id": self.last_accepted_request_id,
            "queue_generation": self.queue_generation,
            "episode_generation": self.episode_generation,
            "deadline_miss_count": self.deadline_miss_count,
            "underrun_count": self.underrun_count,
            "force_fallback": self.force_fallback,
            "pending_entries": len(self.entries),
        }
