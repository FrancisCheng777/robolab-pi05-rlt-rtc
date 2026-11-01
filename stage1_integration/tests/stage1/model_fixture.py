"""Load real model leaf modules without starting RLinf's Ray scheduler."""

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor/rlinf_stage1"


def load_model_modules():
    # Tests isolate package initializers; all numerical implementations are real.
    for name in (
        "rlinf",
        "rlinf.models",
        "rlinf.models.embodiment",
        "rlinf.models.embodiment.openpi",
        "rlinf.models.embodiment.openpi.modules",
        "rlinf.models.embodiment.modules",
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(VENDOR / name.replace(".", "/"))]
            sys.modules[name] = package
    pi0 = importlib.import_module("rlinf.models.embodiment.openpi.pi0")
    config = importlib.import_module("rlinf.models.embodiment.openpi.pi0_config")
    rlt = importlib.import_module("rlinf.models.embodiment.openpi.rlt_config")
    utils = importlib.import_module("rlinf.models.embodiment.openpi.modules.utils")
    utils.set_torch_compile(False)
    return pi0, config, rlt


def tiny_model(eval_mode=False):
    pi0, config, rlt = load_model_modules()
    cfg = config.Pi0Config(
        pi05=True,
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=3,
        action_dim=32,
        max_token_len=8,
    )
    vision = {
        "width": 32,
        "depth": 2,
        "mlp_dim": 64,
        "num_heads": 4,
        "patch_size": (14, 14),
    }
    with (
        patch.object(pi0.gemma, "PALIGEMMA_VOCAB_SIZE", 256),
        patch.object(pi0.siglip, "_decode_variant", return_value=vision),
    ):
        if eval_mode:
            from rlinf.models.embodiment.openpi.tasks.eval import Pi0Eval
            factory = Pi0Eval
        else:
            factory = pi0.Pi0
        model = factory(
            cfg,
            action_env_dim=8,
            action_chunk=3,
            rlt_cfg=rlt.OpenPiPytorchRLTConfig(
                use_rlt=True,
                rlt_input_dim=64,
                rlt_embed_dim=64,
                rlt_prefix_seq_len=1024,
                rlt_num_layers=2,
                rlt_num_heads=4,
                rlt_mlp_ratio=2.0,
                rlt_use_mask=True,
            ),
        )
    torch.nn.init.normal_(model.img.head.weight, std=0.02)
    for name, param in model.named_parameters():
        if "ada_modulation" in name:
            torch.nn.init.normal_(param, std=0.02)
    return model


def batch():
    return {
        "observation": {
            "image": {
                k: torch.zeros(1, 224, 224, 3, dtype=torch.uint8)
                for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            },
            "image_mask": {
                k: torch.tensor([k != "right_wrist_0_rgb"], dtype=torch.bool)
                for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            },
            "state": torch.zeros(1, 32),
            "tokenized_prompt": torch.arange(8)[None],
            "tokenized_prompt_mask": torch.ones(1, 8, dtype=torch.bool),
        },
        "actions": torch.zeros(1, 3, 32),
    }
