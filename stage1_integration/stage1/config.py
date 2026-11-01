"""Compose and validate the independent multi-GPU full-SFT contract."""

import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def load_config(overrides=()):
    os.environ.setdefault("PI05_PROJECT_ROOT", str(ROOT))
    if not OmegaConf.has_resolver("pi05_actor_placement"):
        OmegaConf.register_new_resolver("pi05_actor_placement", lambda n: f"0-{int(n) - 1}")
    searchpath = [f"file://{ROOT}/vendor/rlinf_stage1/examples/embodiment/config"]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.1"):
        return compose(config_name=os.environ.get("PI05_CONFIG_NAME", "stage1_realworld"),
                       overrides=["hydra.searchpath=" + json.dumps(searchpath), *overrides])


def validate_contract(cfg):
    a, f, m = cfg.actor, cfg.actor.fsdp_config, cfg.actor.model
    world_size = cfg.stage1.expected_world_size
    checks = {
        "at least two integer workers": type(world_size) is int and world_size >= 2,
        "placement": str(cfg.cluster.component_placement.actor) == f"0-{world_size - 1}",
        "full sharding": f.sharding_strategy == "full_shard"
        and f.allow_openpi_full_sft,
        "all parameters": not m.is_lora and not m.openpi.train_expert_only,
        "RTC disabled during SFT": not m.openpi.rtc_enabled,
        "single wrist image": m.openpi.num_images_in_input == 1,
        "RLT SFT": m.openpi.task == "sft" and m.openpi.use_rlt,
        "checkpointing": f.gradient_checkpointing
        and not f.gradient_checkpointing_use_reentrant,
        "sharded accumulated gradients": not f.enable_gradient_accumulation,
        "FP32 parameter storage": m.precision == "fp32",
        "BF16 autocast": f.amp_autocast.enabled and f.amp_autocast.precision == "bf16",
        "no conflicting mixed precision": all(
            f.mixed_precision[k] is None
            for k in ("param_dtype", "reduce_dtype", "buffer_dtype")
        ),
        "full export": f.save_full_model_weights,
        "action contract": m.action_dim == 8
        and m.openpi.action_env_dim == 8
        and m.openpi.model_action_dim == 32,
        "horizon": m.openpi.action_horizon == 10 and m.openpi.action_chunk == 10,
        "positive integer batch": all(type(n) is int and n > 0 for n in (a.micro_batch_size, a.global_batch_size)),
    }
    for label, valid in checks.items():
        if not valid:
            raise ValueError(f"Stage-1 contract failed: {label}")
    if a.global_batch_size % (world_size * a.micro_batch_size):
        raise ValueError("global_batch_size must be divisible by world_size * micro_batch_size")
    if cfg.stage1.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    return {
        "world_size": world_size,
        "micro_batch_size": a.micro_batch_size,
        "global_batch_size": a.global_batch_size,
        "accumulation_steps": a.global_batch_size // (world_size * a.micro_batch_size),
        "full_shard": True,
        "gradient_checkpointing": True,
        "synthetic": bool(cfg.stage1.synthetic),
    }


def validate_resume_config(current, saved):
    """Exact resume keeps topology, model, optimizer, batch and data semantics.

    RLinf restores optimizer/scheduler state, so changing YAML does not reset LR.
    Runner max_steps may extend the budget without changing the cosine horizon.
    Compare requested keys: RLinf may add derived keys to the saved configuration.
    """
    paths = ["actor", "cluster", "stage1.expected_world_size", "stage1.synthetic",
             "stage1.dataset_root", "stage1.tokenizer_path"]

    def compare(path, expected, actual):
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                raise ValueError(f"Resume configuration changed: {path}")
            for key, value in expected.items():
                compare(f"{path}.{key}", value, actual.get(key))
        elif expected != actual:
            raise ValueError(f"Resume configuration changed: {path}")

    now = OmegaConf.to_container(current, resolve=True)
    old = OmegaConf.to_container(saved, resolve=True)
    for path in paths:
        def select(root):
            for key in path.split("."):
                root = root.get(key) if isinstance(root, dict) else None
            return root
        compare(path, select(now), select(old))
