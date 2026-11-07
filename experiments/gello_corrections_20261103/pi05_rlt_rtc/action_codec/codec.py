"""The single bidirectional action transform between env commands and VLA model space.

Objects (guide section 8.1):

- ``vla_model_actions`` ``[B, H, 32]``: normalised/padded model space.
- ``reference_env_full`` ``[B, H, 8]``: decoded controller commands, full H.
- ``submitted_env_plan`` ``[B, C, 8]``: what the actor/teacher actually submits.

This codec mirrors the openpi transform ordering used by the pinned RLinf
data config: ``PandaOutputs`` slices the first ``output_action_dim`` dims
*after* the model-space inverse normalisation is applied.  It is deliberately
independent from upstream so it can be unit-tested on CPU.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

import numpy as np

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.spaces import ENV_ACTION_DIM, MODEL_ACTION_DIM
from pi05_rlt_rtc.contracts.versioning import array_sha256, arrays_fingerprint


class ControlMode(str, enum.Enum):
    """Only one control mode may be active per codec (guide section 8.4)."""

    JOINT_DELTA = "pd_joint_delta_pos"


class NormMethod(str, enum.Enum):
    QUANTILE = "quantile"
    ZSCORE = "zscore"


@dataclasses.dataclass(frozen=True)
class NormStats:
    """Action normalisation definition shared by Stage 1, feature service and deployment."""

    method: NormMethod
    model_action_dim: int
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "method", NormMethod(self.method))
        except ValueError as exc:
            raise ContractError(f"unknown norm method {self.method!r}") from exc
        for name in ("q01", "q99", "mean", "std"):
            value = getattr(self, name)
            if value is not None:
                arr = np.array(value, dtype=np.float64, copy=True)
                if arr.shape != (self.model_action_dim,) or not np.isfinite(arr).all():
                    raise ContractError(f"{name} must be a finite vector of length {self.model_action_dim}")
                arr.setflags(write=False)
                object.__setattr__(self, name, arr)
        if self.model_action_dim <= 0:
            raise ContractError("model_action_dim must be > 0")
        if self.method is NormMethod.QUANTILE:
            if self.q01 is None or self.q99 is None:
                raise ContractError("quantile norm stats require q01 and q99")
            for name, arr in (("q01", self.q01), ("q99", self.q99)):
                a = np.asarray(arr, dtype=np.float64).reshape(-1)
                if a.shape[0] != self.model_action_dim:
                    raise ContractError(
                        f"{name} has {a.shape[0]} dims, expected {self.model_action_dim}"
                    )
                if not np.all(np.isfinite(a)):
                    raise ContractError(f"{name} contains NaN/Inf")
            scale = np.asarray(self.q99, dtype=np.float64) - np.asarray(self.q01, dtype=np.float64)
            if np.any(scale <= 0):
                raise ContractError(
                    "q99 - q01 must be > 0 on every dim; a degenerate scale cannot be "
                    "silently repaired"
                )
        else:
            if self.mean is None or self.std is None:
                raise ContractError("zscore norm stats require mean and std")
            std = np.asarray(self.std, dtype=np.float64).reshape(-1)
            if std.shape[0] != self.model_action_dim or np.any(std <= 0):
                raise ContractError("std must be positive and match model_action_dim")
            if np.asarray(self.mean).reshape(-1).shape[0] != self.model_action_dim:
                raise ContractError("mean must match model_action_dim")

    @classmethod
    def symmetric(
        cls,
        model_action_dim: int = MODEL_ACTION_DIM,
        *,
        method: NormMethod = NormMethod.QUANTILE,
        source: str | None = None,
    ) -> "NormStats":
        """Neutral identity stats: q01=-1, q99=1 (env commands live in [-1, 1])."""
        ones = np.ones(model_action_dim, dtype=np.float64)
        if method is NormMethod.QUANTILE:
            return cls(method, model_action_dim, q01=-ones, q99=ones, source=source)
        return cls(method, model_action_dim, mean=np.zeros(model_action_dim), std=ones, source=source)

    @property
    def sha256(self) -> str:
        arrays = {}
        if self.q01 is not None:
            arrays["q01"] = np.asarray(self.q01)
        if self.q99 is not None:
            arrays["q99"] = np.asarray(self.q99)
        if self.mean is not None:
            arrays["mean"] = np.asarray(self.mean)
        if self.std is not None:
            arrays["std"] = np.asarray(self.std)
        arrays["method"] = np.asarray([self.method.value])
        return arrays_fingerprint(arrays)

    def normalize(self, env_actions: np.ndarray) -> np.ndarray:
        x = np.asarray(env_actions, dtype=np.float64)
        if x.shape[-1] > self.model_action_dim:
            raise ContractError(
                f"normalize input last dim {x.shape[-1]} exceeds model dim {self.model_action_dim}"
            )
        d = x.shape[-1]
        if self.method is NormMethod.QUANTILE:
            q01 = np.asarray(self.q01, dtype=np.float64)[:d]
            q99 = np.asarray(self.q99, dtype=np.float64)[:d]
            return 2.0 * (x - q01) / (q99 - q01) - 1.0
        return (x - np.asarray(self.mean, dtype=np.float64)[:d]) / np.asarray(self.std, dtype=np.float64)[:d]

    def denormalize(self, model_actions: np.ndarray) -> np.ndarray:
        m = np.asarray(model_actions, dtype=np.float64)
        if m.shape[-1] > self.model_action_dim:
            raise ContractError(
                f"denormalize input last dim {m.shape[-1]} exceeds model dim {self.model_action_dim}"
            )
        d = m.shape[-1]
        if self.method is NormMethod.QUANTILE:
            q01 = np.asarray(self.q01, dtype=np.float64)[:d]
            q99 = np.asarray(self.q99, dtype=np.float64)[:d]
            return (m + 1.0) / 2.0 * (q99 - q01) + q01
        return m * np.asarray(self.std, dtype=np.float64)[:d] + np.asarray(self.mean, dtype=np.float64)[:d]


@dataclasses.dataclass(frozen=True)
class DecodeResult:
    actions: np.ndarray  # [..., H, 8]
    valid_mask: np.ndarray  # [..., 1, 32] bool
    clipped: np.ndarray  # bool, same shape as actions
    finite: bool


class ActionCodec:
    """Encode env commands into guidance space and decode full-H model actions.

    Parameters
    ----------
    norm_stats:
        Action normalisation; ``model_action_dim`` must be 32 by default.
    env_min / env_max:
        Controller command bounds in env space (``pd_joint_delta_pos`` uses
        per-dimension limits; defaults to the documented [-1, 1] contract).
    control_mode:
        Only ``pd_joint_delta_pos`` is allowed for the first closed loop.  The
        codec intentionally does not implement absolute/relative re-anchoring;
        extending it requires a new mode + tests (guide section 8.4).
    """

    def __init__(
        self,
        norm_stats: NormStats | None = None,
        *,
        env_action_dim: int = ENV_ACTION_DIM,
        model_action_dim: int = MODEL_ACTION_DIM,
        env_min: Any | None = None,
        env_max: Any | None = None,
        control_mode: ControlMode | str = ControlMode.JOINT_DELTA,
        name: str = "panda_robotiq_joint_delta_codec",
    ) -> None:
        self.control_mode = ControlMode(control_mode)
        if self.control_mode is not ControlMode.JOINT_DELTA:
            raise ContractError(f"unsupported control_mode {self.control_mode}")
        self.env_action_dim = int(env_action_dim)
        self.model_action_dim = int(model_action_dim)
        if self.env_action_dim <= 0 or self.env_action_dim > self.model_action_dim:
            raise ContractError("env_action_dim must be in (0, model_action_dim]")
        self.norm_stats = norm_stats or NormStats.symmetric(self.model_action_dim)
        if self.norm_stats.model_action_dim != self.model_action_dim:
            raise ContractError(
                f"norm stats dim {self.norm_stats.model_action_dim} != codec model dim "
                f"{self.model_action_dim}"
            )
        self.env_min = np.broadcast_to(
            np.asarray(-np.ones(self.env_action_dim) if env_min is None else env_min, dtype=np.float64),
            (self.env_action_dim,),
        ).copy()
        self.env_max = np.broadcast_to(
            np.asarray(np.ones(self.env_action_dim) if env_max is None else env_max, dtype=np.float64),
            (self.env_action_dim,),
        ).copy()
        if not np.isfinite(self.env_min).all() or not np.isfinite(self.env_max).all() or np.any(self.env_max <= self.env_min):
            raise ContractError("env_max must be > env_min on every dimension")
        self.name = name

    # -- shapes -----------------------------------------------------------------
    @property
    def valid_model_mask(self) -> np.ndarray:
        """``[1, 1, 32]`` bool: True on the env-controlled dimensions only."""
        mask = np.zeros((1, 1, self.model_action_dim), dtype=bool)
        mask[..., : self.env_action_dim] = True
        return mask

    def provenance(self) -> dict[str, Any]:
        return {
            "codec_name": self.name,
            "control_mode": self.control_mode.value,
            "env_action_dim": self.env_action_dim,
            "model_action_dim": self.model_action_dim,
            "norm_stats_sha256": self.norm_stats.sha256,
            "env_min_sha256": array_sha256(self.env_min),
            "env_max_sha256": array_sha256(self.env_max),
        }

    # -- encode -----------------------------------------------------------------
    def encode_env_for_guidance(
        self,
        env_actions: Any,
        state_anchor: Any | None = None,
    ) -> np.ndarray:
        """Env commands (``[..., K, 8]``) -> model space (``[..., K, 32]``).

        Padding dims are exactly zero.  Guidance is only ever applied on the
        valid mask; zeros are *not* fabricated observations of the real robot.
        """
        if state_anchor is not None:
            raise ContractError(
                "state_anchor is only meaningful for absolute/relative control modes; "
                "the joint-delta codec refuses re-anchoring instead of guessing"
            )
        x = self._to_numpy(env_actions)
        if x.shape[-1] != self.env_action_dim:
            raise ContractError(
                f"env action dim {x.shape[-1]} != {self.env_action_dim}"
            )
        if not np.all(np.isfinite(x)):
            raise ContractError("env actions contain NaN/Inf; refusing to encode")
        flat = x.reshape(-1, self.env_action_dim)
        model = np.zeros((flat.shape[0], self.model_action_dim), dtype=np.float64)
        model[:, : self.env_action_dim] = self.norm_stats.normalize(flat)
        return model.reshape(*x.shape[:-1], self.model_action_dim).astype(np.float32)

    # -- decode -----------------------------------------------------------------
    def decode_full_horizon(
        self,
        model_actions: Any,
        state_anchor: Any | None = None,
        *,
        clip: bool = True,
    ) -> DecodeResult:
        """Model space (``[..., H, 32]``) -> full-H env reference (``[..., H, 8]``).

        The full H is preserved here; the only suffix selection is
        :func:`pi05_rlt_rtc.contracts.spaces.reference_env_suffix`, called
        downstream.  ``clipped`` records where the controller bound was hit so
        that clipping is never confused with float noise (T05).
        """
        if state_anchor is not None:
            raise ContractError("state_anchor is unsupported for the joint-delta codec")
        m = self._to_numpy(model_actions)
        if m.shape[-1] != self.model_action_dim:
            raise ContractError(
                f"model action dim {m.shape[-1]} != {self.model_action_dim}"
            )
        if not np.all(np.isfinite(m)):
            raise ContractError("model actions contain NaN/Inf; the whole plan must be rejected")
        flat = m.reshape(-1, self.model_action_dim)
        env = self.norm_stats.denormalize(flat)[:, : self.env_action_dim]
        clipped = np.zeros_like(env, dtype=bool)
        if clip:
            clipped = (env < self.env_min) | (env > self.env_max)
            env = np.clip(env, self.env_min, self.env_max)
        out = env.reshape(*m.shape[:-1], self.env_action_dim).astype(np.float32)
        return DecodeResult(
            actions=out,
            valid_mask=np.broadcast_to(
                self.valid_model_mask.reshape(1, self.model_action_dim), (*m.shape[:-2], 1, self.model_action_dim)
            ).copy(),
            clipped=clipped.reshape(out.shape),
            finite=True,
        )

    def encode_for_reference_target(
        self,
        reference_env_full: Any,
        state_anchor: Any | None = None,
    ) -> np.ndarray:
        """Alias with an explicit name for the RTC target path."""
        return self.encode_env_for_guidance(reference_env_full, state_anchor)

    def roundtrip_error(self, env_actions: Any) -> float:
        """Max abs error of ``decode(encode(a))`` over the declared invertible domain."""
        x = self._to_numpy(env_actions)
        encoded = self.encode_env_for_guidance(x)
        decoded = self.decode_full_horizon(encoded, clip=False).actions
        return float(np.max(np.abs(decoded - x)))

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach"):  # torch tensor without importing torch
            return value.detach().cpu().numpy()
        return np.asarray(value)
