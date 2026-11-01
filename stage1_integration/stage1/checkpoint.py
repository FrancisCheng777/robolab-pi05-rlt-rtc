"""Validate complete Stage-1 exports without allocating GPU memory."""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from pi05_rlt_rtc.contracts.physical import ACTION_CONTRACT_ID as CONTRACT_ID


def validate_checkpoint(actor_dir, allow_synthetic=False):
    from rlinf.models.embodiment.openpi.pi0 import Pi0
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi.rlt_config import build_rlt_config

    path = Path(actor_dir)
    metadata = json.loads((path / "stage1_metadata.json").read_text())
    if metadata["action_contract_id"] != CONTRACT_ID:
        raise ValueError("Checkpoint action contract mismatch")
    if metadata["synthetic"] and not allow_synthetic:
        raise ValueError("Synthetic smoke checkpoint is not a robot policy")
    cfg = OmegaConf.load(path / "stage1_resolved.yaml")
    m = cfg.actor.model.openpi
    weights = torch.load(
        path / "model_state_dict/full_weights.pt",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    with torch.device("meta"):
        model = Pi0(
            Pi0Config(
                pi05=True,
                dtype="float32",
                paligemma_variant=m.paligemma_variant,
                action_expert_variant=m.action_expert_variant,
                action_horizon=m.action_horizon,
                action_dim=m.model_action_dim,
                max_token_len=m.max_token_len,
                discrete_state_input=m.discrete_state_input,
            ),
            action_env_dim=m.action_env_dim,
            action_chunk=m.action_chunk,
            rlt_cfg=build_rlt_config(m),
        )
    model.load_state_dict(weights, strict=True, assign=True)
    count = sum(p.numel() for p in model.parameters())
    if count != metadata["parameters"]:
        raise ValueError("Parameter count does not match training metadata")
    if not metadata["synthetic"]:
        from .data import read_manifest

        read_manifest(path)
    if not (path / "paligemma_tokenizer.model").is_file():
        raise ValueError("Missing tokenizer")
    return {
        "strict_load": True,
        "parameters": count,
        "synthetic": metadata["synthetic"],
        "rlt_parameters": sum(p.numel() for p in model.rlt_module.parameters()),
        "gpu_forward_tested": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actor_dir")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(validate_checkpoint(args.actor_dir, args.allow_synthetic), indent=2)
    )


if __name__ == "__main__":
    main()
