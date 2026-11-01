"""Load the pinned RLinf Stage-1 model and the exact OpenPI normalization."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pi05_rlt_rtc.adapters.panda_policy import register_panda_profile
from pi05_rlt_rtc import PLANNING_STATE_VERSION
from pi05_rlt_rtc.adapters.rlinf_feature import PINNED_RLINF_COMMIT, RLinfFeatureReferenceService
from pi05_rlt_rtc.contracts.errors import ContractError, EnvironmentNotReadyError, MissingAssetError, MockPolicyError


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def activate_rlinf(root: Path):
    root = Path(root).resolve()
    from .assets import verify_runtime_source
    verify_runtime_source(root)
    loaded = sys.modules.get("rlinf")
    if loaded is not None and Path(loaded.__file__).resolve().parent != root / "rlinf":
        raise ContractError("a different RLinf installation is already imported")
    sys.path.insert(0, str(root))


class UpstreamActionCodec:
    """Use OpenPI Normalize itself, including its epsilon and quantile semantics.

    Guidance padding is always zero. Statistics for state AND actions are bound
    by the file hash; constant training channels remain valid in upstream code.
    """
    env_action_dim = 8
    model_action_dim = 32
    valid_model_mask = np.array([True] * 8 + [False] * 24)

    def __init__(self, normalize, stats_sha):
        self.normalize = normalize
        self.norm_stats = SimpleNamespace(sha256=stats_sha)

    def encode_env_for_guidance(self, actions):
        actions = np.asarray(actions, np.float32)
        if actions.shape[-1] != 8 or not np.isfinite(actions).all():
            raise ContractError("guidance must contain finite 8-dimensional env commands")
        normalized = self.normalize({"actions": actions.copy()})["actions"]
        out = np.zeros((*actions.shape[:-1], 32), np.float32)
        out[..., :8] = normalized[..., :8]
        return out

    @staticmethod
    def _to_numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    def provenance(self):
        return {"name": "rlinf_openpi_joint_codec", "norm_stats_sha256": self.norm_stats.sha256,
                "env_action_dim": 8, "model_action_dim": 32, "padding": "zero"}


def build_model_config(cfg, checkpoint, stats):
    architecture = {"paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m",
                    "max_token_len": 200, "discrete_state_input": True,
                    "rlt_input_dim": 2048, "rlt_embed_dim": 2048, "rlt_prefix_seq_len": 1024,
                    "rlt_num_layers": 2, "rlt_num_heads": 8, "rlt_mlp_ratio": 4.0,
                    "rlt_image_only": False, "rlt_use_mask": True}
    overrides = cfg.raw["model"].get("architecture", {})
    if set(overrides) - set(architecture):
        raise ContractError(f"unsupported model architecture keys: {set(overrides) - set(architecture)}")
    architecture.update(overrides)
    return {"model_type": "openpi", "pi05": True, "model_path": str(checkpoint),
            "precision": cfg.raw["model"].get("precision", "bf16"),
            "num_action_chunks": cfg.contract.horizon, "action_dim": 8, "num_steps": cfg.num_steps,
            "openpi_data": {"repo_id": cfg.raw["model"].get("dataset_id", "local/pi05_plug_train"),
                            "default_prompt": cfg.raw["prompt"], "norm_stats_path": str(stats)},
            "openpi": {**architecture, "task": "eval", "config_name": "pi05_rlt_panda_robotiq_joint",
                       "action_horizon": cfg.contract.horizon, "action_chunk": cfg.contract.horizon,
                       "action_env_dim": 8, "model_action_dim": 32, "num_steps": cfg.num_steps,
                       "num_images_in_input": 1, "use_rlt": True, "rlt_alpha": 1., "rtc_enabled": False, "torch_compile": False}}


def load_planner(cfg):
    from pi05_rlt_rtc.deployment.planner import FeaturePlanner
    from pi05_rlt_rtc.learning.planning_state import PlanningStateSpec
    from pi05_rlt_rtc.rtc.dispatcher import RTCDispatcher
    from .assets import validate_assets
    assets = validate_assets(cfg)
    root, checkpoint, stats = cfg.require_model_assets()
    if not stats.is_file():
        raise MissingAssetError("model.norm_stats must name the exact norm_stats.json file")
    activate_rlinf(root)
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        import torch
        import openpi.transforms as transforms
        from omegaconf import OmegaConf
        from rlinf.models.embodiment.openpi import get_model
        from rlinf.models.embodiment.openpi.checkpoint import resolve_full_weights
        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    except ImportError as exc:
        raise EnvironmentNotReadyError(f"install the pinned RLinf/OpenPI runtime: {exc}") from exc
    register_panda_profile(tokenizer_path=assets["tokenizer"],
        max_token_len=cfg.raw["model"].get("architecture", {}).get("max_token_len", 200),
        action_horizon=cfg.contract.horizon)
    weights = resolve_full_weights(checkpoint)
    if weights is None:
        raise MissingAssetError("Stage 2 requires Stage-1 full_weights.pt including trained rlt_module; base weights are insufficient")
    device = cfg.raw["model"].get("device", "cuda:0")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise EnvironmentNotReadyError("CUDA is unavailable on this model host")
    model_cfg = build_model_config(cfg, checkpoint, stats)
    fingerprint = {"single_camera_profile_sha256": file_sha256(Path(__file__).parents[1] / "data/profile.py"),
                   "feature_adapter_sha256": file_sha256(Path(__file__).parents[1] / "adapters/rlinf_feature.py"),
                   "tokenizer_implementation_sha256": file_sha256(Path(__file__).parents[1] / "adapters/tokenizer.py"),
                   "tokenizer_sha256": assets["tokenizer_sha256"],
                   "source": {key: assets["source"].get(key) for key in ("upstream_commit", "source_tree_sha256")},
                   "checkpoint_sha256": file_sha256(weights), "norm_stats_sha256": file_sha256(stats),
                   "rlinf_commit": PINNED_RLINF_COMMIT, "architecture": model_cfg["openpi"],
                   "openpi_transforms_sha256": file_sha256(transforms.__file__),
                   "action_contract_id": cfg.action_contract_id,
                   "physical_transforms_sha256": file_sha256(Path(__file__).parents[1] / "adapters/panda_policy.py")}
    feature_id = "pi05-rlt:" + hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    train_cfg = get_openpi_config("pi05_rlt_panda_robotiq_joint", model_path=str(checkpoint),
                                 data_kwargs=model_cfg["openpi_data"])
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_cfg.norm_stats is None or not {"state", "actions"}.issubset(data_cfg.norm_stats):
        raise ContractError("real inference requires both state and action normalization statistics")
    codec = UpstreamActionCodec(transforms.Normalize({"actions": data_cfg.norm_stats["actions"]},
                                use_quantiles=data_cfg.use_quantile_norm), fingerprint["norm_stats_sha256"])
    model = get_model(OmegaConf.create(model_cfg)).to(device).eval().requires_grad_(False)
    service = RLinfFeatureReferenceService(model=model, contract=cfg.contract, codec=codec,
        rtc_dispatcher=RTCDispatcher(num_steps=cfg.num_steps), feature_version_id=feature_id,
        norm_stats_sha256=codec.norm_stats.sha256, planning_state_version=PLANNING_STATE_VERSION)
    spec = PlanningStateSpec.from_contract(cfg.contract, z_dim=int(cfg.raw["model"].get("z_dim", 2048)))
    if model.rlt_module.z_dim != spec.z_dim:
        raise ContractError("model.z_dim does not match the loaded RLT token encoder")
    actor, policy_id = None, "teacher:" + feature_id
    if cfg.raw["policy"]["kind"] == "actor":
        from pi05_rlt_rtc.learning.checkpoint import PolicyBundle
        bundle = PolicyBundle.load(cfg.asset(cfg.raw["policy"].get("checkpoint"), "policy.checkpoint"))
        if bundle.extra.get("mock") is not False:
            raise MockPolicyError("deployment requires an explicitly non-mock policy bundle")
        if (bundle.contract != cfg.contract or bundle.spec != spec or bundle.feature_version_id != feature_id
                or bundle.norm_stats_sha256 != codec.norm_stats.sha256):
            raise ContractError("actor bundle does not match the frozen model, statistics or H/C/D layout")
        for key, value in cfg.wire_contract().items():
            if bundle.extra.get("runtime_contract", {}).get(key) != value:
                raise ContractError(f"actor runtime contract mismatch: {key}")
        actor, policy_id = bundle.build_actor(device=device), bundle.policy_version_id
    planner = FeaturePlanner(service=service, spec=spec, contract=cfg.contract,
                             control_hz=cfg.control_hz, policy_version_id=policy_id, actor=actor, rtc_mode=cfg.rtc_mode,
                             deterministic=not cfg.raw["policy"].get("exploration", False))
    metadata = {"protocol": 1, "mock": False, "policy_kind": cfg.raw["policy"]["kind"], "feature_version_id": feature_id,
                "policy_version_id": policy_id, "norm_stats_sha256": codec.norm_stats.sha256,
                "runtime_contract": cfg.wire_contract(), "fingerprint": fingerprint}
    return planner, metadata
