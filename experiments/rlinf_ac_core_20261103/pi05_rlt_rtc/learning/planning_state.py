"""Queue-aware planning state ``X`` (guide section 11.2).

The state includes what the native RLT critic was missing: the committed
prefix ``P``, the previous final composed plan tail ``S`` (which still shapes
the RTC soft-overlap distribution), the last executed command, explicit time
features and the current full reference.  ``z_rl`` is a compressed
observation, **not** a claim of full observability; this state only fixes the
*observable scheduling memory* hole.

All action memory is in env-command space ``[*, 8]``.  Model-space padding
``[*, 32]`` never enters the actor or critic.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from pi05_rlt_rtc import PLANNING_STATE_VERSION, SCHEMA_VERSION
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.observation import EnvIDs
from pi05_rlt_rtc.contracts.spaces import ENV_ACTION_DIM, HorizonContract
from pi05_rlt_rtc.contracts.versioning import array_sha256, canonical_json, sha256_text


@dataclasses.dataclass(frozen=True)
class PlanningStateSpec:
    z_dim: int = 2048
    proprio_dim: int = 9
    env_action_dim: int = ENV_ACTION_DIM
    horizon: int = 10
    chunk: int = 5
    max_delay: int = 5
    time_feature_dim: int = 4

    def __post_init__(self) -> None:
        for name in ("z_dim", "proprio_dim", "env_action_dim", "horizon", "chunk", "max_delay"):
            value = getattr(self, name)
            if int(value) <= 0:
                raise ContractError(f"PlanningStateSpec.{name} must be > 0, got {value}")
        if self.max_delay < self.chunk:
            raise ContractError("max_delay must be >= chunk so that D can be padded to C")

    @classmethod
    def from_contract(
        cls,
        contract: HorizonContract,
        *,
        z_dim: int = 2048,
        proprio_dim: int = 9,
        time_feature_dim: int = 4,
    ) -> "PlanningStateSpec":
        return cls(
            z_dim=z_dim,
            proprio_dim=proprio_dim,
            env_action_dim=ENV_ACTION_DIM,
            horizon=contract.horizon,
            chunk=contract.chunk,
            max_delay=contract.chunk,
            time_feature_dim=time_feature_dim,
        )

    def actor_state_layout(self) -> list[tuple[str, int]]:
        e = self.env_action_dim
        return [
            ("z_rl", self.z_dim),
            ("proprio", self.proprio_dim),
            ("pending_prefix", self.max_delay * e),
            ("pending_prefix_mask", self.max_delay * e),
            ("prev_plan_tail", self.horizon * e),
            ("prev_plan_tail_mask", self.horizon),
            ("last_command", e),
            ("time_features", self.time_feature_dim),
        ]

    def critic_state_layout(self) -> list[tuple[str, int]]:
        return self.actor_state_layout() + [
            ("current_reference_full", self.horizon * self.env_action_dim),
            ("current_reference_mask", self.horizon * self.env_action_dim),
        ]

    def describe_layout(self, which: str = "actor") -> str:
        layout = self.actor_state_layout() if which == "actor" else self.critic_state_layout()
        rows = [f"  {name}: {dim}" for name, dim in layout]
        total = sum(dim for _, dim in layout)
        return f"{which} state layout (total {total}):\n" + "\n".join(rows)


@dataclasses.dataclass
class PlanningState:
    spec: PlanningStateSpec
    ids: EnvIDs
    capture_tick: int
    plan_id: int | None
    z_rl: np.ndarray  # [B, z_dim]
    proprio: np.ndarray  # [B, proprio_dim]
    pending_prefix: np.ndarray  # [B, max_delay, 8]
    pending_mask: np.ndarray  # [B, max_delay, 8] bool
    prev_plan_tail: np.ndarray  # [B, H, 8]
    prev_plan_tail_mask: np.ndarray  # [B, H] bool
    last_command: np.ndarray  # [B, 8]
    reference_full: np.ndarray  # [B, H, 8] env space
    time_features: np.ndarray  # [B, time_feature_dim]
    feature_version_id: str
    norm_stats_sha256: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        e = self.spec.env_action_dim
        expectations = {
            "z_rl": (self.spec.z_dim,),
            "proprio": (self.spec.proprio_dim,),
            "pending_prefix": (self.spec.max_delay, e),
            "pending_mask": (self.spec.max_delay, e),
            "prev_plan_tail": (self.spec.horizon, e),
            "prev_plan_tail_mask": (self.spec.horizon,),
            "last_command": (e,),
            "reference_full": (self.spec.horizon, e),
            "time_features": (self.spec.time_feature_dim,),
        }
        for name, tail in expectations.items():
            arr = getattr(self, name)
            if arr.ndim < 1 or tuple(arr.shape[1:]) != tail:
                raise ContractError(
                    f"PlanningState.{name}: expected [B, {', '.join(map(str, tail))}], got {arr.shape}"
                )
            if name.endswith("mask"):
                continue
            if not np.all(np.isfinite(arr.astype(np.float32))):
                raise ContractError(f"PlanningState.{name} contains NaN/Inf")
        if self.ids.schema_version != SCHEMA_VERSION:
            raise ContractError(
                f"PlanningState schema {self.ids.schema_version!r} != {SCHEMA_VERSION!r}"
            )
        if self.ids.planning_state_version != PLANNING_STATE_VERSION:
            raise ContractError(
                f"PlanningState contract {self.ids.planning_state_version!r} != "
                f"{PLANNING_STATE_VERSION!r}"
            )
        if self.feature_version_id != self.ids.feature_version_id:
            raise ContractError("feature_version_id must match ids.feature_version_id")

    @property
    def batch_size(self) -> int:
        return int(self.z_rl.shape[0])

    @property
    def z_sha256(self) -> str:
        return array_sha256(self.z_rl)

    # -- flattened views --------------------------------------------------------
    def actor_state_vector(self) -> np.ndarray:
        spec = self.spec
        parts = [
            self.z_rl,
            self.proprio,
            self.pending_prefix.reshape(self.batch_size, -1),
            self.pending_mask.reshape(self.batch_size, -1).astype(np.float32),
            self.prev_plan_tail.reshape(self.batch_size, -1),
            self.prev_plan_tail_mask.astype(np.float32),
            self.last_command,
            self.time_features,
        ]
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def reference_suffix_vector(self, contract: HorizonContract) -> np.ndarray:
        """``A_ref[D:D+C]`` in env space, flattened.  The only suffix selector."""
        suffix = self.reference_full[..., contract.suffix_start : contract.suffix_end, :]
        return suffix.reshape(self.batch_size, -1).astype(np.float32)

    def critic_state_vector(self) -> np.ndarray:
        base = self.actor_state_vector()
        ref = self.reference_full.reshape(self.batch_size, -1)
        ref_mask = np.ones_like(self.reference_full, dtype=np.float32).reshape(self.batch_size, -1)
        return np.concatenate([base, ref, ref_mask], axis=-1).astype(np.float32)

    def _infer_delay(self) -> int:
        """Active D inferred from the contiguous valid prefix rows of ``P``.

        The fixed protocol always writes a contiguous committed prefix; this
        helper is only used for provenance/diagnostics, never for slicing
        references (those always take an explicit :class:`HorizonContract`).
        """
        valid_rows = self.pending_mask.any(axis=-1).all(axis=0)
        if valid_rows.all():
            return self.spec.max_delay
        return int(np.argmin(valid_rows))

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": self.ids.schema_version,
            "planning_state_version": self.ids.planning_state_version,
            "feature_version_id": self.feature_version_id,
            "norm_stats_sha256": self.norm_stats_sha256,
            "z_rl_sha256": self.z_sha256,
            "capture_tick": self.capture_tick,
            "plan_id": self.plan_id,
        }


def make_time_features(
    capture_tick: int,
    *,
    control_frequency_hz: float,
    delay_ticks: int,
    observation_age_ticks: int = 0,
) -> np.ndarray:
    """Small deterministic time encoding: [time_s, delay_s, age_s, phase]."""
    if control_frequency_hz <= 0:
        raise ContractError("control_frequency_hz must be > 0")
    period = 1.0 / float(control_frequency_hz)
    time_s = capture_tick * period
    phase = float((capture_tick * period) % 1.0)
    return np.asarray(
        [time_s, delay_ticks * period, observation_age_ticks * period, phase], dtype=np.float32
    ).reshape(1, 4)


def build_planning_state(
    *,
    spec: PlanningStateSpec,
    ids: EnvIDs,
    capture_tick: int,
    plan_id: int | None,
    z_rl: np.ndarray,
    proprio: np.ndarray,
    pending_prefix: np.ndarray,
    pending_mask: np.ndarray,
    prev_plan_tail: np.ndarray | None,
    prev_plan_tail_mask: np.ndarray | None,
    last_command: np.ndarray,
    reference_full: np.ndarray,
    time_features: np.ndarray,
    norm_stats_sha256: str,
) -> PlanningState:
    """Build a validated, batch-first planning state (batch dimension added)."""
    B = _batch_size(z_rl)
    e = spec.env_action_dim
    ref = _batchify(reference_full, B, (spec.horizon, e), "reference_full")
    return PlanningState(
        spec=spec,
        ids=ids,
        capture_tick=int(capture_tick),
        plan_id=plan_id,
        z_rl=_batchify(z_rl, B, (spec.z_dim,), "z_rl").astype(np.float32),
        proprio=_batchify(proprio, B, (spec.proprio_dim,), "proprio").astype(np.float32),
        pending_prefix=_batchify(pending_prefix, B, (spec.max_delay, e), "pending_prefix").astype(np.float32),
        pending_mask=_batchify(pending_mask, B, (spec.max_delay, e), "pending_mask").astype(bool),
        prev_plan_tail=(
            np.zeros((B, spec.horizon, e), dtype=np.float32)
            if prev_plan_tail is None
            else _batchify(prev_plan_tail, B, (spec.horizon, e), "prev_plan_tail").astype(np.float32)
        ),
        prev_plan_tail_mask=(
            np.zeros((B, spec.horizon), dtype=bool)
            if prev_plan_tail_mask is None
            else _batchify(prev_plan_tail_mask, B, (spec.horizon,), "prev_plan_tail_mask").astype(bool)
        ),
        last_command=_batchify(last_command, B, (e,), "last_command").astype(np.float32),
        reference_full=ref.astype(np.float32),
        time_features=_batchify(time_features, B, (spec.time_feature_dim,), "time_features").astype(np.float32),
        feature_version_id=ids.feature_version_id,
        norm_stats_sha256=str(norm_stats_sha256),
    )


def _batch_size(array: Any) -> int:
    arr = np.asarray(array)
    if arr.ndim == 0:
        raise ContractError("arrays must have at least one dimension")
    return int(arr.shape[0])


def _batchify(array: Any, batch: int, tail: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(array)
    if tuple(arr.shape) == tail:
        arr = arr[None, ...]
    if tuple(arr.shape) != (batch, *tail):
        raise ContractError(f"{name}: expected [B, {tail}] (or {tail}), got {arr.shape}")
    return arr


def state_layout_fingerprint(spec: PlanningStateSpec) -> str:
    return sha256_text(canonical_json(spec.actor_state_layout() + spec.critic_state_layout()))
