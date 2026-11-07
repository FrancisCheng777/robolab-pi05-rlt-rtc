"""Versioned replay buffer for delayed submitted-plan transitions.

Persisting replay is an explicit decision: the upstream example uses
``auto_save=False``; this buffer records what was saved and refuses to load a
replay produced with a different schema/feature version instead of silently
mixing meanings (T58/T59).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pi05_rlt_rtc import PLANNING_STATE_VERSION, SCHEMA_VERSION
from pi05_rlt_rtc.contracts.errors import ContractError, SchemaMismatchError
from pi05_rlt_rtc.contracts.observation import EnvIDs
from pi05_rlt_rtc.contracts.transition import DelayedTransition
from pi05_rlt_rtc.learning.planning_state import PlanningState, PlanningStateSpec
from pi05_rlt_rtc.learning.serialization import atomic_torch_save


@dataclasses.dataclass
class ReplayStats:
    pushed: int = 0
    rejected_version: int = 0
    censored_seen: int = 0
    evicted: int = 0
    loaded: int = 0
    saved: int = 0

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        schema_version: str = SCHEMA_VERSION,
        planning_state_version: str = PLANNING_STATE_VERSION,
        feature_version_id: str,
        norm_stats_sha256: str,
    ) -> None:
        if capacity <= 0:
            raise ContractError("replay capacity must be > 0")
        self.capacity = int(capacity)
        self.schema_version = str(schema_version)
        self.planning_state_version = str(planning_state_version)
        self.feature_version_id = str(feature_version_id)
        self.norm_stats_sha256 = str(norm_stats_sha256)
        self._items: list[DelayedTransition] = []
        self.stats = ReplayStats()

    def __len__(self) -> int:
        return len(self._items)

    def require_compatible(
        self,
        *,
        schema_version: str,
        feature_version_id: str,
        norm_stats_sha256: str,
    ) -> None:
        if schema_version != self.schema_version:
            raise SchemaMismatchError(
                f"replay schema {self.schema_version!r} != consumer {schema_version!r}; "
                "old synchronous replay cannot be fed to the delayed learner"
            )
        if feature_version_id != self.feature_version_id:
            raise SchemaMismatchError(
                f"replay feature version {self.feature_version_id!r} != {feature_version_id!r}"
            )
        if norm_stats_sha256 != self.norm_stats_sha256:
            raise SchemaMismatchError(
                "replay norm stats hash differs; cached z/reference meanings changed"
            )

    def push(self, transition: DelayedTransition) -> bool:
        if transition.schema_version != self.schema_version:
            self.stats.rejected_version += 1
            raise SchemaMismatchError(
                f"transition schema {transition.schema_version!r} != buffer {self.schema_version!r}"
            )
        if transition.planning_state_version != self.planning_state_version:
            self.stats.rejected_version += 1
            raise SchemaMismatchError(
                f"transition planning-state version {transition.planning_state_version!r} "
                f"!= buffer {self.planning_state_version!r}"
            )
        if transition.feature_version_id != self.feature_version_id:
            self.stats.rejected_version += 1
            raise SchemaMismatchError(
                "transition feature version differs from the buffer's pinned version"
            )
        if transition.norm_stats_sha256 != self.norm_stats_sha256:
            self.stats.rejected_version += 1
            raise SchemaMismatchError("transition norm-stats hash differs from the buffer's")
        if transition.censored:
            self.stats.censored_seen += 1
        self._items.append(transition)
        self.stats.pushed += 1
        if len(self._items) > self.capacity:
            self._items.pop(0)
            self.stats.evicted += 1
        return True

    def learnable_transitions(self) -> list[DelayedTransition]:
        return [t for t in self._items if t.is_learnable()]

    def sample(self, batch_size: int, *, generator: np.random.Generator) -> list[DelayedTransition]:
        pool = self.learnable_transitions()
        if not pool:
            raise ContractError("replay has no learnable transitions (check censoring/filters)")
        if batch_size <= 0:
            raise ContractError("batch_size must be > 0")
        idx = generator.integers(0, len(pool), size=int(batch_size))
        return [pool[int(i)] for i in idx]

    def all_transitions(self) -> list[DelayedTransition]:
        return list(self._items)

    # -- persistence ------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "capacity": self.capacity,
            "schema_version": self.schema_version,
            "planning_state_version": self.planning_state_version,
            "feature_version_id": self.feature_version_id,
            "norm_stats_sha256": self.norm_stats_sha256,
            "stats": self.stats.to_dict(),
            "items": [_transition_to_payload(t) for t in self._items],
        }
        atomic_torch_save(payload, path)
        self.stats.saved += 1
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"replay file not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        buffer = cls(
            capacity=payload.get("capacity", max(1, len(payload.get("items", [])))),
            schema_version=payload["schema_version"],
            planning_state_version=payload["planning_state_version"],
            feature_version_id=payload["feature_version_id"],
            norm_stats_sha256=payload["norm_stats_sha256"],
        )
        for item in payload.get("items", []):
            buffer.push(_payload_to_transition(item))
        saved_stats = payload.get("stats", {})
        for key, value in saved_stats.items():
            if hasattr(buffer.stats, key):
                setattr(buffer.stats, key, int(value))
        buffer.stats.loaded = len(buffer._items)
        return buffer

    def counts(self) -> dict[str, int]:
        learnable = len(self.learnable_transitions())
        return {
            "size": len(self._items),
            "learnable": learnable,
            "filtered": len(self._items) - learnable,
            **self.stats.to_dict(),
        }


# ---------------------------------------------------------------------------
# serialisation helpers


def _state_to_payload(state: PlanningState) -> dict[str, Any]:
    return {
        "spec": dataclasses.asdict(state.spec),
        "ids": dataclasses.asdict(state.ids),
        "capture_tick": state.capture_tick,
        "plan_id": state.plan_id,
        "z_rl": torch.as_tensor(state.z_rl),
        "proprio": torch.as_tensor(state.proprio),
        "pending_prefix": torch.as_tensor(state.pending_prefix),
        "pending_mask": torch.as_tensor(state.pending_mask),
        "prev_plan_tail": torch.as_tensor(state.prev_plan_tail),
        "prev_plan_tail_mask": torch.as_tensor(state.prev_plan_tail_mask),
        "last_command": torch.as_tensor(state.last_command),
        "reference_full": torch.as_tensor(state.reference_full),
        "time_features": torch.as_tensor(state.time_features),
        "feature_version_id": state.feature_version_id,
        "norm_stats_sha256": state.norm_stats_sha256,
    }


def _state_from_payload(payload: dict[str, Any]) -> PlanningState:
    spec = PlanningStateSpec(**payload["spec"])
    ids = EnvIDs(**payload["ids"])
    return PlanningState(
        spec=spec,
        ids=ids,
        capture_tick=int(payload["capture_tick"]),
        plan_id=payload["plan_id"],
        z_rl=payload["z_rl"].numpy(),
        proprio=payload["proprio"].numpy(),
        pending_prefix=payload["pending_prefix"].numpy(),
        pending_mask=payload["pending_mask"].numpy(),
        prev_plan_tail=payload["prev_plan_tail"].numpy(),
        prev_plan_tail_mask=payload["prev_plan_tail_mask"].numpy(),
        last_command=payload["last_command"].numpy(),
        reference_full=payload["reference_full"].numpy(),
        time_features=payload["time_features"].numpy(),
        feature_version_id=payload["feature_version_id"],
        norm_stats_sha256=payload["norm_stats_sha256"],
    )


def _t_or_none(value: Any) -> torch.Tensor | None:
    return None if value is None else torch.as_tensor(np.asarray(value))


def _transition_to_payload(tr: DelayedTransition) -> dict[str, Any]:
    return {
        "request_id": tr.request_id,
        "env_id": tr.env_id,
        "episode_id": tr.episode_id,
        "episode_generation": tr.episode_generation,
        "capture_tick": tr.capture_tick,
        "next_capture_tick": tr.next_capture_tick,
        "plan_id": tr.plan_id,
        "next_plan_id": tr.next_plan_id,
        "submitted_plan": torch.as_tensor(tr.submitted_plan),
        "submitted_valid_mask": torch.as_tensor(tr.submitted_valid_mask),
        "raw_actor_plan": _t_or_none(tr.raw_actor_plan),
        "executed_trace": torch.as_tensor(tr.executed_trace),
        "executed_source_plan_ids": torch.as_tensor(tr.executed_source_plan_ids),
        "primitive_rewards": torch.as_tensor(tr.primitive_rewards),
        "reward_valid_mask": torch.as_tensor(tr.reward_valid_mask),
        "duration_ticks": tr.duration_ticks,
        "planning_state": _state_to_payload(tr.planning_state),
        "next_planning_state": None
        if tr.next_planning_state is None
        else _state_to_payload(tr.next_planning_state),
        "bc_target_plan": _t_or_none(tr.bc_target_plan),
        "bc_target_valid_mask": _t_or_none(tr.bc_target_valid_mask),
        "teacher_source": tr.teacher_source,
        "terminated": tr.terminated,
        "truncated": tr.truncated,
        "bootstrap_allowed": tr.bootstrap_allowed,
        "accepted": tr.accepted,
        "intervened": tr.intervened,
        "deadline_miss": tr.deadline_miss,
        "censored": tr.censored,
        "censor_reason": tr.censor_reason,
        "schema_version": tr.schema_version,
        "planning_state_version": tr.planning_state_version,
        "feature_version_id": tr.feature_version_id,
        "norm_stats_sha256": tr.norm_stats_sha256,
        "policy_version_id": tr.policy_version_id,
        "gamma_per_tick": tr.gamma_per_tick,
    }


def _payload_to_transition(payload: dict[str, Any]) -> DelayedTransition:
    return DelayedTransition(
        request_id=int(payload["request_id"]),
        env_id=payload["env_id"],
        episode_id=payload["episode_id"],
        episode_generation=int(payload["episode_generation"]),
        capture_tick=int(payload["capture_tick"]),
        next_capture_tick=int(payload["next_capture_tick"]),
        plan_id=int(payload["plan_id"]),
        next_plan_id=payload["next_plan_id"],
        submitted_plan=payload["submitted_plan"].numpy(),
        submitted_valid_mask=payload["submitted_valid_mask"].numpy(),
        raw_actor_plan=None if payload["raw_actor_plan"] is None else payload["raw_actor_plan"].numpy(),
        executed_trace=payload["executed_trace"].numpy(),
        executed_source_plan_ids=payload["executed_source_plan_ids"].numpy(),
        primitive_rewards=payload["primitive_rewards"].numpy(),
        reward_valid_mask=payload["reward_valid_mask"].numpy(),
        duration_ticks=int(payload["duration_ticks"]),
        planning_state=_state_from_payload(payload["planning_state"]),
        next_planning_state=None
        if payload["next_planning_state"] is None
        else _state_from_payload(payload["next_planning_state"]),
        bc_target_plan=None if payload["bc_target_plan"] is None else payload["bc_target_plan"].numpy(),
        bc_target_valid_mask=None
        if payload["bc_target_valid_mask"] is None
        else payload["bc_target_valid_mask"].numpy(),
        teacher_source=payload["teacher_source"],
        terminated=bool(payload["terminated"]),
        truncated=bool(payload["truncated"]),
        bootstrap_allowed=bool(payload["bootstrap_allowed"]),
        accepted=bool(payload["accepted"]),
        intervened=bool(payload["intervened"]),
        deadline_miss=bool(payload["deadline_miss"]),
        censored=bool(payload["censored"]),
        censor_reason=payload["censor_reason"],
        schema_version=payload["schema_version"],
        planning_state_version=payload["planning_state_version"],
        feature_version_id=payload["feature_version_id"],
        norm_stats_sha256=payload["norm_stats_sha256"],
        policy_version_id=payload["policy_version_id"],
        gamma_per_tick=float(payload["gamma_per_tick"]),
    )
