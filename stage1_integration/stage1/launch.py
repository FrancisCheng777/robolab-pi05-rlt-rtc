"""Inspect, preflight, smoke-test or train without touching robot processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from .config import load_config, validate_contract, validate_resume_config
from pi05_rlt_rtc.data.windows import validate_resume_window_policy


def preflight(cfg):
    import torch

    from . import (
        worker,  # Verify the real worker imports before allocating Ray workers.
    )
    from .data import PlugDataset, SyntheticDataset

    report = validate_contract(cfg)
    world_size = cfg.stage1.expected_world_size
    if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
        raise RuntimeError(
            f"Expose exactly {world_size} CUDA GPUs matching stage1.expected_world_size"
        )
    report["gpus"] = []
    for i in range(world_size):
        p = torch.cuda.get_device_properties(i)
        if p.total_memory / 2**30 < cfg.stage1.minimum_gpu_gib:
            raise RuntimeError(
                f"GPU {i} has {p.total_memory / 2**30:.1f} GiB; expected >={cfg.stage1.minimum_gpu_gib} GiB"
            )
        if p.major < 8:
            raise RuntimeError("BF16 training requires Ampere or newer")
        report["gpus"].append({"name": p.name, "total_gib": p.total_memory / 2**30})
    base = Path(cfg.actor.model.model_path)
    weights = base if base.is_file() else base / "model.safetensors"
    full = [
        base / x
        for x in (
            "actor/model_state_dict/full_weights.pt",
            "model_state_dict/full_weights.pt",
            "full_weights.pt",
        )
    ]
    if not weights.is_file() and not any(x.is_file() for x in full):
        raise FileNotFoundError(
            f"Missing local Pi0.5 model.safetensors/full_weights.pt: {base}"
        )
    m, s = cfg.actor.model.openpi, cfg.stage1
    dataset = (
        SyntheticDataset(s.tokenizer_path, m.action_horizon, m.max_token_len)
        if s.synthetic
        else PlugDataset(
            s.dataset_root, s.tokenizer_path, m.action_horizon, m.max_token_len
        )
    )
    sample = dataset[0]
    report["dataset_frames"] = len(dataset.ds) if hasattr(dataset, "ds") else len(dataset)
    report["training_windows"] = len(dataset)
    report["action_shape"] = list(sample["actions"].shape)
    report["state_shape"] = list(sample["observation"]["state"].shape)
    report["tokenizer_sha256"] = hashlib.sha256(
        Path(s.tokenizer_path).read_bytes()
    ).hexdigest()
    report["torch"] = torch.__version__
    report["cuda"] = torch.version.cuda
    report["worker"] = worker.PlugSftWorker.__name__
    if cfg.runner.resume_dir:
        actor_dir = Path(cfg.runner.resume_dir) / "actor"
        validate_resume_config(cfg, OmegaConf.load(actor_dir / "stage1_resolved.yaml"))
        metadata = json.loads((actor_dir / "stage1_metadata.json").read_text())
        validate_resume_window_policy(metadata)
        from pi05_rlt_rtc.contracts.physical import ACTION_CONTRACT_ID as CONTRACT_ID

        if metadata["action_contract_id"] != CONTRACT_ID or metadata["world_size"] != world_size:
            raise ValueError("Resume checkpoint contract/world size mismatch")
        if bool(metadata["synthetic"]) != bool(s.synthetic):
            raise ValueError("Cannot mix synthetic and real-data runs on resume")
        if not s.synthetic:
            for name in ("manifest.json", "norm_stats.json", "action_codec.json"):
                if (actor_dir / name).read_bytes() != (
                    Path(s.dataset_root) / name
                ).read_bytes():
                    raise ValueError(f"Resume dataset changed: {name}")
        if (actor_dir / "paligemma_tokenizer.model").read_bytes() != Path(
            s.tokenizer_path
        ).read_bytes():
            raise ValueError("Resume tokenizer changed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "preflight", "smoke", "train"))
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    overrides = list(args.overrides)
    if args.mode == "smoke":
        defaults = [
            "stage1.synthetic=true",
            "runner.max_steps=5",
            "runner.save_interval=5",
            "actor.optim.total_training_steps=5",
            "actor.optim.lr_warmup_steps=1",
        ]
        overrides = defaults + overrides
    cfg = load_config(overrides)
    if args.mode == "smoke" and not any(x.startswith("actor.global_batch_size=") for x in args.overrides):
        cfg.actor.global_batch_size = math.lcm(8, cfg.stage1.expected_world_size * cfg.actor.micro_batch_size)
    info = validate_contract(cfg)
    if args.mode == "train" and cfg.stage1.synthetic:
        raise ValueError("Synthetic inputs are only permitted in explicit smoke mode")
    if args.mode == "smoke" and not cfg.stage1.synthetic:
        raise ValueError("Use train mode for real demonstrations")
    if args.mode == "inspect":
        print(json.dumps(info, indent=2))
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    info = preflight(cfg)
    print(json.dumps(info, indent=2))
    if args.mode == "preflight":
        return
    run_dir = Path(cfg.runner.logger.log_path) / cfg.runner.logger.experiment_name
    # Never silently combine a new run with existing checkpoints/logs.
    if run_dir.exists() and not cfg.runner.resume_dir:
        raise FileExistsError(
            f"Choose a fresh PI05_RUN_NAME or set runner.resume_dir: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "preflight.json").write_text(json.dumps(info, indent=2))
    OmegaConf.save(cfg, run_dir / "requested_config.yaml", resolve=True)

    from rlinf.config import validate_cfg
    from rlinf.runners.sft_runner import SFTRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    from .worker import PlugSftWorker

    cfg = validate_cfg(cfg)
    # Ray workers must receive concrete values, including custom topology resolvers.
    OmegaConf.resolve(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster).get_strategy("actor")
    actor = PlugSftWorker.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=placement
    )
    runner = SFTRunner(cfg=cfg, actor=actor)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
