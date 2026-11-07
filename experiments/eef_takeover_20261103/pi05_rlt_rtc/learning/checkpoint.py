"""Stage 2 policy bundle save/load with full provenance.

A Stage 2 checkpoint must let a consumer answer: which feature checkpoint,
which norm stats, which planning-state layout and which C/D/H contract was
this actor trained under.  Loading a C=10 actor into a C=5 queue layout is
refused unless explicitly converted (guide section 4.3-G).
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pi05_rlt_rtc import PLANNING_STATE_VERSION, SCHEMA_VERSION
from pi05_rlt_rtc.contracts.errors import ContractError, SchemaMismatchError
from pi05_rlt_rtc.contracts.spaces import HorizonContract
from pi05_rlt_rtc.learning.actor import QueueAwareRLTActor
from pi05_rlt_rtc.learning.critic import TwinQWithTargets
from pi05_rlt_rtc.learning.planning_state import PlanningStateSpec
from pi05_rlt_rtc.learning.serialization import atomic_torch_save


AC_ARCHITECTURE = "rlinf_mlp_queue_v2"

def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().to(torch.float32).cpu().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor).tobytes())
    return digest.hexdigest()


@dataclasses.dataclass
class PolicyBundle:
    spec: PlanningStateSpec
    contract: HorizonContract
    actor_state: dict[str, torch.Tensor]
    critic_state: dict[str, torch.Tensor] | None
    feature_version_id: str
    norm_stats_sha256: str
    policy_version_id: str
    training_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    actor_hidden_dims: tuple[int, ...] = (256, 256, 256)
    ac_architecture: str = AC_ARCHITECTURE
    critic_hidden_dims: tuple[int, ...] | None = None
    schema_version: str = SCHEMA_VERSION
    planning_state_version: str = PLANNING_STATE_VERSION
    codec_name: str = "panda_robotiq_joint_delta_codec"
    created_unix_seconds: float = dataclasses.field(default_factory=time.time)
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ac_architecture != AC_ARCHITECTURE:
            raise ContractError("unsupported AC architecture; start a new Stage-2 run")
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError("PolicyBundle schema version mismatch")
        if self.planning_state_version != PLANNING_STATE_VERSION:
            raise ContractError("PolicyBundle planning-state version mismatch")
        if self.spec.horizon != self.contract.horizon or self.spec.chunk != self.contract.chunk:
            raise ContractError("planning-state spec and horizon contract disagree")
        if self.contract.delay > self.spec.max_delay:
            raise ContractError("execution delay exceeds the saved planning-state capacity")
        if self.critic_state is not None and self.critic_hidden_dims is None:
            # Infer custom test widths from linear layers only; LayerNorm also has weights.
            weights = [(int(key.split(".")[-2]), value) for key, value in self.critic_state.items()
                       if key.startswith("online.heads.0.") and key.endswith(".weight") and value.ndim == 2]
            self.critic_hidden_dims = tuple(value.shape[0] for _, value in sorted(weights)[:-1])

    @property
    def actor_state_sha256(self) -> str:
        return state_dict_sha256(self.actor_state)

    def save(self, path: str | Path, *, include_critic: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "planning_state_version": self.planning_state_version,
            "spec": dataclasses.asdict(self.spec),
            "contract": dataclasses.asdict(self.contract),
            "actor_hidden_dims": list(self.actor_hidden_dims),
            "ac_architecture": self.ac_architecture,
            "critic_hidden_dims": None if self.critic_hidden_dims is None else list(self.critic_hidden_dims),
            "actor_state": {k: v.detach().cpu() for k, v in self.actor_state.items()},
            "critic_state": (
                None
                if self.critic_state is None or not include_critic
                else {k: v.detach().cpu() for k, v in self.critic_state.items()}
            ),
            "feature_version_id": self.feature_version_id,
            "norm_stats_sha256": self.norm_stats_sha256,
            "policy_version_id": self.policy_version_id,
            "training_counts": self.training_counts,
            "codec_name": self.codec_name,
            "created_unix_seconds": self.created_unix_seconds,
            "extra": self.extra,
            "actor_state_sha256": self.actor_state_sha256,
        }
        atomic_torch_save(payload, path)
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_norm_stats_sha256: str | None = None,
        expected_feature_version_id: str | None = None,
    ) -> "PolicyBundle":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"policy bundle not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("ac_architecture") != AC_ARCHITECTURE:
            raise ContractError("unsupported or legacy AC architecture; start a new Stage-2 run")
        if payload.get("actor_state_sha256") != state_dict_sha256(payload["actor_state"]):
            raise ContractError(
                "actor state hash mismatch inside checkpoint; refusing to load a "
                "possibly corrupted bundle"
            )
        if (
            expected_norm_stats_sha256 is not None
            and payload["norm_stats_sha256"] != expected_norm_stats_sha256
        ):
            raise SchemaMismatchError(
                "checkpoint norm-stats hash differs from the deployment/config norm stats"
            )
        if (
            expected_feature_version_id is not None
            and payload["feature_version_id"] != expected_feature_version_id
        ):
            raise SchemaMismatchError(
                "checkpoint feature version differs from the frozen feature service"
            )
        return cls(
            spec=PlanningStateSpec(**payload["spec"]),
            contract=HorizonContract(**payload["contract"]),
            actor_state=payload["actor_state"],
            critic_state=payload["critic_state"],
            feature_version_id=payload["feature_version_id"],
            norm_stats_sha256=payload["norm_stats_sha256"],
            policy_version_id=payload["policy_version_id"],
            training_counts=dict(payload.get("training_counts", {})),
            actor_hidden_dims=tuple(payload["actor_hidden_dims"]),
            ac_architecture=payload["ac_architecture"],
            critic_hidden_dims=None if payload.get("critic_hidden_dims") is None else tuple(payload["critic_hidden_dims"]),
            schema_version=payload["schema_version"],
            planning_state_version=payload["planning_state_version"],
            codec_name=payload.get("codec_name", "unknown"),
            created_unix_seconds=float(payload.get("created_unix_seconds", 0.0)),
            extra=dict(payload.get("extra", {})),
        )

    def build_actor(self, *, device: torch.device | str = "cpu") -> QueueAwareRLTActor:
        actor = QueueAwareRLTActor(
            self.spec,
            hidden_dims=self.actor_hidden_dims,
            fixed_std=self.extra.get("exploration_std", .002),
        )
        actor.load_state_dict(self.actor_state)
        actor.to(device)
        actor.eval()
        for param in actor.parameters():
            param.requires_grad_(False)
        return actor

    def build_critic(self, *, device: torch.device | str = "cpu") -> TwinQWithTargets:
        if self.critic_state is None:
            raise ContractError("bundle does not contain a critic state (deployment export)")
        critic = TwinQWithTargets(
            sum(d for _, d in self.spec.critic_state_layout()),
            self.spec.chunk * self.spec.env_action_dim,
            hidden_dims=self.critic_hidden_dims or (),
        )
        critic.load_state_dict(self.critic_state)
        critic.to(device)
        critic.eval()
        return critic
