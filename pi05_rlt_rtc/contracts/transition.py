"""Delayed transition contract, reward attribution and TD targets.

Guide section 13: the RL action is the *submitted plan* ``U_i``, not the mixed
executed block ``E_i = P_i[0:D] || U_i[0:C-D]``.  The reward window is
``[r_i, r_i + K)`` with the *actual* duration K, and the next state carries the
pending queue so that the last ``D`` actions of ``U_i`` receive credit later.

Exact worked example used by the tests (T49/T50):

    rewards = [1, 2, 3], gamma = 0.9 -> R = 1 + 0.9*2 + 0.9^2*3 = 5.23
    non-terminal next Q_min = 4   -> y = 5.23 + 0.9^3 * 4 = 8.146
    terminal                       -> y = 5.23
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from pi05_rlt_rtc import PLANNING_STATE_VERSION
from pi05_rlt_rtc.contracts.errors import ContractError, PlanValidationError
from pi05_rlt_rtc.contracts.spaces import ENV_ACTION_DIM, HorizonContract


def discounted_return(
    rewards: np.ndarray,
    gamma: float,
    *,
    duration_ticks: int | None = None,
    valid_mask: np.ndarray | None = None,
) -> float:
    """``R_i = sum_{k=0}^{K-1} gamma^k r_{r_i+k}`` with the real K.

    ``rewards`` is the ``[C]`` container; entries beyond ``duration_ticks`` or
    with ``valid_mask == False`` are excluded.  The gamma exponent therefore
    comes from the real duration, never from the padded tensor width (T51).
    """
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if rewards.size == 0:
        raise ContractError("rewards container is empty")
    k = rewards.size if duration_ticks is None else int(duration_ticks)
    if k < 0 or k > rewards.size:
        raise ContractError(f"duration_ticks {k} outside container size {rewards.size}")
    if valid_mask is None:
        valid = np.arange(rewards.size) < k
    else:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.size != rewards.size:
            raise ContractError("valid_mask size must match rewards container")
        valid = valid & (np.arange(rewards.size) < k)
    exponents = np.arange(rewards.size, dtype=np.float64)
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ContractError("gamma must be finite and in (0, 1]")
    if not np.all(np.isfinite(rewards[valid])):
        raise ContractError("reward window contains NaN/Inf")
    return float(np.sum(rewards[valid] * np.power(float(gamma), exponents[valid])))


def compute_td_target(
    reward_sum: float,
    gamma: float,
    duration_ticks: int,
    next_q_min: float | np.ndarray | None,
    bootstrap_allowed: bool,
) -> float | np.ndarray:
    """``y = R + 1[bootstrap] * gamma^K * min_j Q_target_j(X', a(X'))``."""
    if duration_ticks < 0:
        raise ContractError("duration_ticks must be >= 0")
    if not np.isfinite(reward_sum):
        raise ContractError("reward_sum is not finite")
    if not bootstrap_allowed or next_q_min is None:
        return reward_sum
    q = np.asarray(next_q_min, dtype=np.float64)
    if not np.all(np.isfinite(q)):
        raise ContractError("next_q_min contains NaN/Inf - refusing to fabricate a bootstrap")
    return reward_sum + float(gamma) ** int(duration_ticks) * q


@dataclasses.dataclass
class DelayedTransition:
    """One strictly time-aligned delayed transition (guide section 13.2).

    ``planning_state`` / ``next_planning_state`` are :class:`PlanningState`
    objects (numpy); they carry z/proprio/queue/history/reference and their
    version ids.  ``executed_trace`` records the raw primitive commands that
    were actually issued in ``[r_i, r_i + K)`` and their source plan ids.
    """

    # identity
    request_id: int
    env_id: str
    episode_id: str
    episode_generation: int
    capture_tick: int
    next_capture_tick: int
    plan_id: int
    next_plan_id: int | None

    # the RL action (submitted plan) and what was actually executed
    submitted_plan: np.ndarray  # [C, 8] = U_i
    submitted_valid_mask: np.ndarray  # [C, 8]
    raw_actor_plan: np.ndarray | None  # optional, pre-safety
    executed_trace: np.ndarray  # [K, 8] = E_i
    executed_source_plan_ids: np.ndarray  # [K] int

    # rewards and their true duration
    primitive_rewards: np.ndarray  # [max(C,K)]; terminal committed tail may have C < K <= D+C
    reward_valid_mask: np.ndarray  # same shape as primitive_rewards
    duration_ticks: int  # K

    # learning targets / flags
    planning_state: Any
    next_planning_state: Any | None
    bc_target_plan: np.ndarray | None
    bc_target_valid_mask: np.ndarray | None
    teacher_source: str | None
    terminated: bool
    truncated: bool
    bootstrap_allowed: bool
    accepted: bool
    intervened: bool
    deadline_miss: bool
    censored: bool
    censor_reason: str | None

    # provenance
    schema_version: str
    planning_state_version: str
    feature_version_id: str
    norm_stats_sha256: str
    policy_version_id: str
    gamma_per_tick: float

    def __post_init__(self) -> None:
        if self.planning_state_version != PLANNING_STATE_VERSION:
            raise ContractError(
                f"unknown planning_state_version {self.planning_state_version!r}"
            )
        if self.executed_trace.shape[-1] != ENV_ACTION_DIM:
            raise PlanValidationError("executed_trace last dim must be 8")
        if self.submitted_plan.shape[-1] != ENV_ACTION_DIM:
            raise PlanValidationError("submitted_plan last dim must be 8")
        if self.duration_ticks != self.executed_trace.shape[0]:
            raise PlanValidationError(
                "duration_ticks must equal executed_trace length; a padded tensor width "
                "is not the real K"
            )
        if self.reward_valid_mask.shape[0] != self.primitive_rewards.shape[0]:
            raise PlanValidationError("reward_valid_mask must align with the reward container")
        if self.censored and self.censor_reason is None:
            raise PlanValidationError("censored transitions must record a reason")

    # -- rewards / TD ----------------------------------------------------------
    @property
    def reward_sum(self) -> float:
        return discounted_return(
            self.primitive_rewards,
            self.gamma_per_tick,
            duration_ticks=self.duration_ticks,
            valid_mask=self.reward_valid_mask,
        )

    def td_target(self, next_q_min: float | np.ndarray | None) -> float | np.ndarray:
        return compute_td_target(
            self.reward_sum,
            self.gamma_per_tick,
            self.duration_ticks,
            next_q_min,
            self.bootstrap_allowed and not self.terminated and self.next_planning_state is not None,
        )

    def is_learnable(self) -> bool:
        """Only transitions with a real next planning state and an accepted plan.

        Deadline misses, interventions and censored rows are kept in the raw
        ledger but excluded from normal Bellman updates (guide section 13.4).
        """
        if self.censored or self.deadline_miss or self.intervened or not self.accepted:
            return False
        if self.terminated:
            return True  # terminal transitions are learnable with bootstrap off
        return self.next_planning_state is not None and self.bootstrap_allowed

    def to_record(self) -> dict[str, Any]:
        """JSON-safe audit record (arrays are summarised, not silently dropped)."""
        return {
            "request_id": self.request_id,
            "env_id": self.env_id,
            "episode_id": self.episode_id,
            "episode_generation": self.episode_generation,
            "capture_tick": self.capture_tick,
            "next_capture_tick": self.next_capture_tick,
            "duration_ticks": self.duration_ticks,
            "submitted_plan_shape": list(self.submitted_plan.shape),
            "executed_trace_shape": list(self.executed_trace.shape),
            "reward_container_len": int(self.primitive_rewards.shape[0]),
            "reward_sum": self.reward_sum,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "bootstrap_allowed": self.bootstrap_allowed,
            "accepted": self.accepted,
            "intervened": self.intervened,
            "deadline_miss": self.deadline_miss,
            "censored": self.censored,
            "censor_reason": self.censor_reason,
            "learnable": self.is_learnable(),
            "features": {
                "z_rl_sha256": getattr(self.planning_state, "z_sha256", None),
                "next_z_rl_sha256": getattr(self.next_planning_state, "z_sha256", None),
            },
            "schema_version": self.schema_version,
            "planning_state_version": self.planning_state_version,
            "feature_version_id": self.feature_version_id,
            "norm_stats_sha256": self.norm_stats_sha256,
            "policy_version_id": self.policy_version_id,
        }


@dataclasses.dataclass(frozen=True)
class ActionWindow:
    """H-step demo action window with a tail mask; never crosses an episode."""

    actions: np.ndarray  # [H, A]
    valid: np.ndarray  # [H] bool
    episode_id: str
    anchor_index: int


def build_action_windows(
    episode_actions: np.ndarray,
    episode_id: str,
    contract: HorizonContract,
    *,
    action_dim: int = ENV_ACTION_DIM,
    stride: int = 1,
    tail_padding: str = "mask",
) -> list[ActionWindow]:
    """Build H-step windows from a single episode (T03).

    ``tail_padding="mask"`` pads with zeros and marks the tail invalid; it never
    crosses an episode boundary and never fabricates future expert actions.
    """
    actions = np.asarray(episode_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != action_dim:
        raise ContractError(f"episode_actions must be [T, {action_dim}], got {actions.shape}")
    if stride <= 0:
        raise ContractError("stride must be > 0")
    if tail_padding != "mask":
        raise ContractError(
            f"tail_padding={tail_padding!r} is not supported; only explicit masking is allowed"
        )
    windows: list[ActionWindow] = []
    for start in range(0, actions.shape[0], stride):
        end = start + contract.horizon
        chunk = actions[start:end]
        valid = np.ones((chunk.shape[0],), dtype=bool)
        if chunk.shape[0] < contract.horizon:
            pad = np.zeros((contract.horizon - chunk.shape[0], action_dim), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
            valid = np.concatenate(
                [valid, np.zeros((contract.horizon - valid.shape[0],), dtype=bool)]
            )
        windows.append(
            ActionWindow(actions=chunk, valid=valid, episode_id=episode_id, anchor_index=start)
        )
    return windows
