"""Compose and validate the independent two-GPU full-SFT contract."""

import os
from pathlib import Path

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]


def load_config(overrides=()):
    os.environ.setdefault("PI05_PROJECT_ROOT", str(ROOT))
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.1"):
        return compose(config_name="stage1_2xa800", overrides=list(overrides))


def validate_contract(cfg):
    a, f, m = cfg.actor, cfg.actor.fsdp_config, cfg.actor.model
    checks = {
        "two workers": cfg.stage1.expected_world_size == 2,
        "placement": str(cfg.cluster.component_placement.actor) == "0-1",
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
        "positive batch": a.micro_batch_size > 0 and a.global_batch_size > 0,
    }
    for label, valid in checks.items():
        if not valid:
            raise ValueError(f"Stage-1 contract failed: {label}")
    if a.global_batch_size % (2 * a.micro_batch_size):
        raise ValueError("global_batch_size must be divisible by 2 * micro_batch_size")
    if cfg.stage1.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    return {
        "world_size": 2,
        "micro_batch_size": a.micro_batch_size,
        "global_batch_size": a.global_batch_size,
        "accumulation_steps": a.global_batch_size // (2 * a.micro_batch_size),
        "full_shard": True,
        "gradient_checkpointing": True,
        "synthetic": bool(cfg.stage1.synthetic),
    }
