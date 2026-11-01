"""Observation snapshot contract.

An observation snapshot is immutable once captured: the planner must never
swap in a newer camera frame for an in-flight request (guide section 15.3).
"""

from __future__ import annotations

import dataclasses
import copy
from typing import Any, Mapping, Sequence

import numpy as np

from pi05_rlt_rtc.contracts.errors import ContractError

CAMERA_SEMANTIC_NAMES = ("wrist",)


@dataclasses.dataclass(frozen=True)
class ObservationSnapshot:
    """One versioned environment observation captured at an integer tick."""

    env_id: str
    episode_id: str
    tick: int
    states: np.ndarray
    images: Mapping[str, np.ndarray]
    image_masks: Mapping[str, bool]
    prompt: str
    captured_monotonic: float | None = None
    source_timestamps: Mapping[str, float] = dataclasses.field(default_factory=dict)

    sensor_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        # Robot/camera drivers commonly reuse their buffers on the next tick.
        # A frozen dataclass alone does not freeze the referenced arrays.
        def owned(value):
            arr = np.array(value, copy=True)
            arr.setflags(write=False)
            return arr

        object.__setattr__(self, "states", owned(self.states))
        object.__setattr__(self, "images", {key: owned(value) for key, value in self.images.items()})
        object.__setattr__(self, "image_masks", dict(self.image_masks))
        object.__setattr__(self, "source_timestamps", dict(self.source_timestamps))
        object.__setattr__(self, "sensor_metadata", copy.deepcopy(dict(self.sensor_metadata)))

    def validate(
        self,
        *,
        proprio_dim: int | None = None,
        required_cameras: Sequence[str] = CAMERA_SEMANTIC_NAMES,
    ) -> None:
        if not self.env_id or not self.episode_id:
            raise ContractError("ObservationSnapshot requires env_id and episode_id")
        states = np.asarray(self.states)
        if states.ndim == 0:
            raise ContractError("ObservationSnapshot.states must have at least one dimension")
        if proprio_dim is not None and states.shape[-1] != proprio_dim:
            raise ContractError(
                f"ObservationSnapshot.states last dim {states.shape[-1]} != proprio_dim {proprio_dim}"
            )
        if not np.all(np.isfinite(states)):
            raise ContractError("ObservationSnapshot.states contains NaN/Inf")
        for cam in required_cameras:
            if cam not in self.images:
                raise ContractError(
                    f"ObservationSnapshot missing camera {cam!r} "
                    f"(declared cameras: {sorted(self.images)})"
                )
            img = np.asarray(self.images[cam])
            if img.size == 0:
                raise ContractError(f"ObservationSnapshot camera {cam!r} is empty")
            if not np.issubdtype(img.dtype, np.uint8):
                raise ContractError(
                    f"camera {cam!r} must stay uint8 at capture time; got {img.dtype}. "
                    "Never pre-normalise images the upstream transform will normalise again."
                )

    def camera(self, name: str) -> np.ndarray:
        if name not in self.images:
            raise ContractError(
                f"camera {name!r} not in snapshot; the missing-view protocol must be "
                "handled explicitly, never by copying another camera"
            )
        return np.asarray(self.images[name])


@dataclasses.dataclass(frozen=True)
class EnvIDs:
    """Full version identity carried by every request/response/transition."""

    env_id: str
    episode_id: str
    episode_generation: int
    policy_version_id: str
    feature_version_id: str
    schema_version: str
    planning_state_version: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def validate_episode_alignment(a: EnvIDs, b: EnvIDs) -> None:
    """Refuse to attach a response/transition to a different episode identity."""
    if (a.env_id, a.episode_id, a.episode_generation) != (
        b.env_id,
        b.episode_id,
        b.episode_generation,
    ):
        raise ContractError(
            "episode mismatch: "
            f"{a.env_id}/{a.episode_id}#{a.episode_generation} vs "
            f"{b.env_id}/{b.episode_id}#{b.episode_generation}"
        )
