"""Unified feature/reference service contract (guide section 10).

One observation snapshot, one prefix pass, then z_rl and the full-H reference
come out of the *same* frozen model version.  The service must never run the
vision/language backbone twice per request and never substitute a newer camera
frame for an in-flight request.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol

import numpy as np

from pi05_rlt_rtc import SCHEMA_VERSION
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
from pi05_rlt_rtc.contracts.plan import PlanRequest
from pi05_rlt_rtc.contracts.spaces import ENV_ACTION_DIM, MODEL_ACTION_DIM, HorizonContract
from pi05_rlt_rtc.contracts.versioning import array_sha256


@dataclasses.dataclass
class FeatureReferenceResult:
    """Return contract of ``FeatureReferenceService.extract`` (section 10.2)."""

    request_id: int
    env_id: str
    episode_id: str
    capture_tick: int
    apply_start_tick: int

    z_rl: np.ndarray  # [B, 2048] fp32
    proprio_model: np.ndarray  # [B, 9] fp32, same preprocessing as Stage 1
    vla_model_actions: np.ndarray  # [B, H, 32] fp32 (diagnostics + guidance space)
    reference_env_full: np.ndarray  # [B, H, 8] fp32, never truncated early

    feature_version_id: str
    norm_stats_sha256: str
    schema_version: str
    planning_state_version: str

    rng_seed: int
    sampler_mode: str
    num_denoising_steps: int
    prefix_calls: int
    timings: dict[str, float]
    is_mock: bool = False
    codec_provenance: dict[str, Any] = dataclasses.field(default_factory=dict)

    def validate(self, contract: HorizonContract, *, expected_batch: int | None = None) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"feature result schema {self.schema_version!r} is not current")
        if self.z_rl.ndim != 2:
            raise ContractError(f"z_rl must be [B, 2048], got {self.z_rl.shape}")
        B = self.z_rl.shape[0]
        if B <= 0:
            raise ContractError("feature result batch must be non-empty")
        if expected_batch is not None and B != expected_batch:
            raise ContractError(f"feature batch {B} != expected {expected_batch}")
        if self.proprio_model.shape != (B, 9):
            raise ContractError(
                f"proprio_model must be [B, 9], got {self.proprio_model.shape}"
            )
        if self.vla_model_actions.shape != (B, contract.horizon, MODEL_ACTION_DIM):
            raise ContractError(
                "vla_model_actions must keep the full model horizon "
                f"[B, {contract.horizon}, {MODEL_ACTION_DIM}], got {self.vla_model_actions.shape}; "
                "never truncate to C before selecting the suffix"
            )
        if self.reference_env_full.shape != (B, contract.horizon, ENV_ACTION_DIM):
            raise ContractError(
                f"reference_env_full must be [B, {contract.horizon}, {ENV_ACTION_DIM}], "
                f"got {self.reference_env_full.shape}"
            )
        for name in ("z_rl", "proprio_model", "vla_model_actions", "reference_env_full"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ContractError(f"feature result {name} contains NaN/Inf")
        if self.apply_start_tick != self.capture_tick + contract.delay:
            raise ContractError("feature result apply_start_tick disagrees with capture_tick + D")
        if self.prefix_calls != 1:
            raise ContractError(
                f"per-request prefix passes must be exactly 1, got {self.prefix_calls}; "
                "running RLT extraction and RTC sampling on separate prefixes breaks the "
                "z/reference consistency and the latency budget"
            )

    def reference_suffix(self, contract: HorizonContract) -> np.ndarray:
        return self.reference_env_full[..., contract.suffix_start : contract.suffix_end, :]

    @property
    def z_sha256(self) -> str:
        return array_sha256(self.z_rl)

    def planning_metadata(self) -> dict[str, Any]:
        return {
            "feature_version_id": self.feature_version_id,
            "norm_stats_sha256": self.norm_stats_sha256,
            "sampler_mode": self.sampler_mode,
            "rng_seed": self.rng_seed,
            "num_denoising_steps": self.num_denoising_steps,
            "z_sha256": self.z_sha256,
            "is_mock": self.is_mock,
        }


class FeatureReferenceService(Protocol):
    def extract(
        self, request: PlanRequest, observation: ObservationSnapshot
    ) -> FeatureReferenceResult: ...


def check_prefix_length(sequence_length: int, capacity: int) -> None:
    """Reject over-long prefixes instead of silently truncating them (T14)."""
    if capacity <= 0:
        raise ContractError(f"prefix capacity must be > 0, got {capacity}")
    if sequence_length > capacity:
        raise ContractError(
            f"prefix length {sequence_length} exceeds configured capacity {capacity}; "
            "truncating extra tokens silently would change the feature semantics. "
            "Extend the position handling or adjust the configuration and retrain."
        )
