"""Plan contracts: request, submitted plan, final composed plan.

Fixed-delay protocol (guide section 12), all indices relative to a request at
``capture_tick = r_i``:

- ``P_i``: committed prefix covering ``[r_i, r_i + D)``.  The executor holds
  it; a new plan may never overwrite it.
- ``U_i``: submitted plan covering ``[r_i + D, r_i + D + C)``.
- ``A_ref``: full reference of length H for this observation.
- Final composed plan ``M_i = P_i || U_i || A_ref[D+C:H]``; its tail is
  *speculative* (used only as the next RTC target and never executed without a
  fresh commit).
- Next request at ``r_{i+1} = r_i + C``; next target ``S_{i+1} = M_i[C:H]``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from pi05_rlt_rtc.contracts.errors import (
    ContractError,
    PlanValidationError,
    StaleEpisodeError,
)
from pi05_rlt_rtc.contracts.spaces import (
    ENV_ACTION_DIM,
    HorizonContract,
    pending_from_submitted,
    reference_env_suffix,
)

PLAN_SOURCES = ("reference_teacher", "actor", "safe_fallback")
RTC_MODES = ("none", "approx", "full_vjp")


def _as_finite_array(value: Any, shape: tuple[int, ...], name: str, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.shape != shape:
        raise PlanValidationError(f"{name}: expected shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise PlanValidationError(f"{name}: contains NaN/Inf")
    return arr.copy()


@dataclasses.dataclass(frozen=True)
class PrevPlanContext:
    """The previous final composed plan, shifted to the *current* request tick.

    ``prev_env_plan[s]`` is the command planned for current-relative tick
    ``s``; the first ``H - C`` entries are valid (that is all we keep of the
    previous H plan after shifting by the request period).  The target used by
    RTC is encoded from this in env space and converted to model space.
    """

    prev_env_plan: np.ndarray | None
    prev_valid_mask: np.ndarray | None
    prev_capture_tick: int | None
    prev_plan_id: int | None
    prev_feature_version_id: str | None

    @classmethod
    def empty(cls) -> "PrevPlanContext":
        return cls(None, None, None, None, None)

    def validate(self, contract: HorizonContract) -> None:
        if self.prev_env_plan is None:
            if self.prev_valid_mask is not None:
                raise PlanValidationError("prev_valid_mask without prev_env_plan")
            return
        plan = np.asarray(self.prev_env_plan, dtype=np.float32)
        if plan.ndim != 2 or plan.shape[0] > contract.horizon or plan.shape[1] != ENV_ACTION_DIM:
            raise PlanValidationError(
                "prev_env_plan must be [<=H, 8], got "
                f"{plan.shape}; full-H references are kept, never truncated to C early"
            )
        if self.prev_valid_mask is None:
            raise PlanValidationError("prev_valid_mask is required when prev_env_plan is set")
        mask = np.asarray(self.prev_valid_mask)
        if mask.shape not in ((plan.shape[0],), plan.shape) or not np.isin(mask, [0, 1]).all():
            raise PlanValidationError("prev_valid_mask must be a boolean [K] or [K, 8] mask")
        if not np.all(np.isfinite(plan)):
            raise PlanValidationError("prev_env_plan contains NaN/Inf")


@dataclasses.dataclass(frozen=True)
class PlanRequest:
    """A planning request captured before executing the action at ``capture_tick``."""

    request_id: int
    env_id: str
    episode_id: str
    episode_generation: int
    capture_tick: int
    apply_start_tick: int
    queue_generation: int
    policy_version_id: str
    feature_version_id: str
    schema_version: str
    planning_state_version: str
    rtc_context: PrevPlanContext
    rng_seed: int
    sampler_mode: str
    num_denoising_steps: int
    deadline_tick: int | None = None
    last_command: np.ndarray | None = None  # last command actually issued before capture

    def validate(self, contract: HorizonContract, *, allow_warm_start: bool = False) -> None:
        if self.request_id < ( -1 if allow_warm_start else 0):
            raise PlanValidationError("request_id must be >= 0")
        if self.apply_start_tick != self.capture_tick + contract.delay:
            raise PlanValidationError(
                f"apply_start_tick must equal capture_tick + D: "
                f"{self.apply_start_tick} != {self.capture_tick} + {contract.delay}"
            )
        if self.sampler_mode not in RTC_MODES:
            raise PlanValidationError(f"unknown sampler_mode {self.sampler_mode!r}")
        if self.num_denoising_steps <= 0:
            raise PlanValidationError("num_denoising_steps must be > 0")
        if self.last_command is not None:
            _as_finite_array(self.last_command, (ENV_ACTION_DIM,), "last_command")
        if self.deadline_tick is not None and self.deadline_tick < self.apply_start_tick:
            raise PlanValidationError(
                "deadline_tick earlier than apply_start_tick makes the request unschedulable"
            )
        self.rtc_context.validate(contract)


@dataclasses.dataclass(frozen=True)
class SubmittedPlan:
    """``U_i``: the C-step env-space chunk actually submitted by the policy."""

    request_id: int
    env_id: str
    episode_id: str
    episode_generation: int
    capture_tick: int
    apply_start_tick: int
    actions: np.ndarray  # [C, 8]
    valid_mask: np.ndarray  # [C, 8] bool
    source: str
    policy_version_id: str
    feature_version_id: str
    schema_version: str
    planning_state_version: str
    queue_generation: int
    raw_actor_plan: np.ndarray | None = None  # pre-safety actor output, diagnostics only

    def validate(self, contract: HorizonContract) -> None:
        if self.apply_start_tick != self.capture_tick + contract.delay:
            raise PlanValidationError("submitted apply_start_tick must equal capture_tick + D")
        if self.source not in PLAN_SOURCES:
            raise PlanValidationError(f"unknown plan source {self.source!r}")
        _as_finite_array(self.actions, (contract.chunk, ENV_ACTION_DIM), "submitted.actions")
        mask = np.asarray(self.valid_mask)
        if mask.shape != (contract.chunk, ENV_ACTION_DIM):
            raise PlanValidationError(
                f"submitted.valid_mask must be [{contract.chunk}, {ENV_ACTION_DIM}], got {mask.shape}"
            )
        if not np.isin(mask, [0, 1]).all() or not mask.astype(bool).all():
            raise PlanValidationError("all submitted command dimensions must be valid before execution")
        if self.raw_actor_plan is not None:
            _as_finite_array(
                self.raw_actor_plan, (contract.chunk, ENV_ACTION_DIM), "submitted.raw_actor_plan"
            )

    def env_actions(self) -> np.ndarray:
        return np.asarray(self.actions, dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class FinalPlan:
    """``M_i = P || U || reference_tail`` with provenance."""

    request_id: int
    env_id: str
    episode_id: str
    episode_generation: int
    capture_tick: int
    plan_id: int
    pending_prefix: np.ndarray  # [D, 8]
    submitted: np.ndarray  # [C, 8]
    reference_tail: np.ndarray  # [H-D-C, 8] (speculative)
    final: np.ndarray  # [H, 8]
    submitted_valid_mask: np.ndarray
    reference_valid_mask: np.ndarray
    source: str
    policy_version_id: str
    feature_version_id: str
    schema_version: str

    @classmethod
    def compose(
        cls,
        *,
        request: PlanRequest,
        pending_prefix: np.ndarray,
        submitted_plan: SubmittedPlan,
        reference_env_full: np.ndarray,
        contract: HorizonContract,
        plan_id: int,
    ) -> "FinalPlan":
        request.validate(contract)
        submitted_plan.validate(contract)
        for name in ("request_id", "env_id", "episode_id", "episode_generation", "capture_tick", "apply_start_tick", "queue_generation", "policy_version_id", "feature_version_id", "schema_version", "planning_state_version"):
            if getattr(request, name) != getattr(submitted_plan, name):
                raise PlanValidationError(f"request/submitted identity mismatch: {name}")
        pending = _as_finite_array(
            pending_prefix, (contract.delay, ENV_ACTION_DIM), "pending_prefix"
        )
        reference = _as_finite_array(
            reference_env_full, (contract.horizon, ENV_ACTION_DIM), "reference_env_full"
        )
        submitted = submitted_plan.env_actions().copy()
        tail = reference[contract.suffix_end : contract.horizon, :]
        final = np.concatenate([pending, submitted, tail], axis=0)
        if final.shape != (contract.horizon, ENV_ACTION_DIM):
            raise ContractError(f"composed final plan has wrong shape {final.shape}")
        ref_valid = np.ones((contract.horizon,), dtype=bool)
        # Speculative tail validity mirrors the submitted valid mask semantics.
        return cls(
            request_id=request.request_id,
            env_id=request.env_id,
            episode_id=request.episode_id,
            episode_generation=request.episode_generation,
            capture_tick=request.capture_tick,
            plan_id=plan_id,
            pending_prefix=pending,
            submitted=submitted,
            reference_tail=tail,
            final=final,
            submitted_valid_mask=np.array(submitted_plan.valid_mask, dtype=bool, copy=True),
            reference_valid_mask=ref_valid,
            source=submitted_plan.source,
            policy_version_id=submitted_plan.policy_version_id,
            feature_version_id=submitted_plan.feature_version_id,
            schema_version=submitted_plan.schema_version,
        )

    def shifted_for_next_request(self, next_capture_tick: int) -> tuple[np.ndarray, np.ndarray]:
        """``S_{i+1} = M_i[C:H]`` with a validity mask over the remaining ticks.

        The caller must check ``next_capture_tick == capture_tick + C`` so that
        the shift is by *actual request tick difference*, not by an estimate of
        how long the response took (guide section 9.6).
        """
        shift = int(next_capture_tick) - int(self.capture_tick)
        if shift <= 0:
            raise StaleEpisodeError(
                f"next_capture_tick {next_capture_tick} must be after {self.capture_tick}"
            )
        if shift >= self.final.shape[0]:
            return (
                np.zeros((0, self.final.shape[1]), dtype=np.float32),
                np.zeros((0,), dtype=bool),
            )
        return self.final[shift:, :].copy(), np.ones((self.final.shape[0] - shift,), dtype=bool)

    def pending_for_next_request(self, contract: HorizonContract) -> np.ndarray:
        return pending_from_submitted(self.submitted, contract)


def make_submitted_mask(contract: HorizonContract, valid: bool = True) -> np.ndarray:
    return np.full((contract.chunk, ENV_ACTION_DIM), valid, dtype=bool)
