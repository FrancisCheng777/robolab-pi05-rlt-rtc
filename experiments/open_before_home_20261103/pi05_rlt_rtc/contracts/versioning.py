"""Versioning helpers: feature/norm-stats/schema provenance.

Stage 1 is frozen before Stage 2 runs.  Once frozen, the meaning of ``z_rl``
and of every reference action must not change: a replay buffer or deployment
package that mixes two ``feature_version`` values is invalid.  These helpers
make the provenance explicit and hash-based.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from pi05_rlt_rtc import SCHEMA_VERSION
from pi05_rlt_rtc.contracts.errors import ContractError, SchemaMismatchError


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing metadata dictionaries."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def array_sha256(array: np.ndarray) -> str:
    """Hash of an array's dtype + shape + bytes (byte-order preserved)."""
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(repr(arr.shape).encode("ascii"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def arrays_fingerprint(arrays: Mapping[str, np.ndarray]) -> str:
    """Fingerprint of a named collection of arrays (e.g. norm stats)."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode("utf-8"))
        digest.update(array_sha256(arrays[name]).encode("ascii"))
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class FeatureVersion:
    """Immutable identity of "what the frozen feature service produces".

    ``z_rl``, the decoded reference action space, the camera contract and the
    norm stats are all part of this identity.  Consumers must call
    :meth:`require_compatible` before accepting cached features/replay.
    """

    schema_version: str
    feature_checkpoint_sha256: str
    norm_stats_sha256: str
    camera_contract: str
    prompt: str
    codec_name: str
    extra: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["extra"] = dict(self.extra)
        return d

    @property
    def feature_version_id(self) -> str:
        return sha256_text(canonical_json(self.as_dict()))[:16]

    def require_compatible(self, other: "FeatureVersion", *, what: str = "consumer") -> None:
        if not isinstance(other, FeatureVersion):
            raise SchemaMismatchError(
                f"{what}: expected FeatureVersion, got {type(other).__name__}"
            )
        if self.as_dict() != other.as_dict():
            raise SchemaMismatchError(
                f"{what}: feature version mismatch "
                f"({self.feature_version_id} != {other.feature_version_id}); "
                "cached z/reference/replay must not be mixed across feature versions"
            )


@dataclasses.dataclass(frozen=True)
class PolicyVersion:
    """Identity of a Stage 2 actor bundle (weights + input layout + code)."""

    schema_version: str
    planning_state_version: str
    policy_checkpoint_sha256: str
    created_unix_seconds: float

    @property
    def policy_version_id(self) -> str:
        return sha256_text(canonical_json(dataclasses.asdict(self)))[:16]


def require_schema_version(found: str, *, expected: str | None = None, what: str = "artifact") -> None:
    expected = SCHEMA_VERSION if expected is None else expected
    if found != expected:
        raise SchemaMismatchError(
            f"{what}: schema_version {found!r} != expected {expected!r}; "
            "refusing to mix artifacts across schema versions"
        )


def trajectory_schema_fields() -> Sequence[str]:
    """Canonical field list of a persisted transition (used by replay checks)."""
    return (
        "planning_state",
        "submitted_plan",
        "executed_trace",
        "primitive_rewards",
        "duration_ticks",
        "next_planning_state",
        "terminated",
        "truncated",
        "bootstrap_allowed",
        "accepted",
        "intervened",
        "deadline_miss",
        "censored",
        "schema_version",
        "planning_state_version",
        "feature_version_id",
        "norm_stats_sha256",
        "policy_version_id",
    )


def check_schema_dict(data: Mapping[str, Any], *, what: str = "artifact") -> None:
    require_schema_version(str(data.get("schema_version", "<missing>")), what=what)
    if "planning_state_version" in data:
        found = str(data["planning_state_version"])
        if found != "queue-planning-state.v1":
            raise ContractError(
                f"{what}: unknown planning_state_version {found!r}"
            )
