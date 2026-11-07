"""Real LeRobot v3 SFT windows: no cross-episode or repeated-delta padding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.physical import ACTION_CONTRACT_ID, CONTROL_HZ, PROMPT


from .windows import complete_windows, table_window_starts, WINDOW_POLICY


class PhysicalChunkDataset:
    def __init__(self, root, horizon=10, *, norm_stats_path, allow_mock=False):
        root = Path(root).resolve()
        manifest = json.loads((root.parent / "manifest.json").read_text())
        from .profile import validate_dataset_profile
        validate_dataset_profile(manifest)
        info = json.loads((root / "meta/info.json").read_text())
        self.mock = manifest.get("mock")
        self.window_policy = dict(WINDOW_POLICY)
        if (self.mock is not False and not (allow_mock and self.mock is True)) or manifest.get("action_contract_id") != ACTION_CONTRACT_ID:
            raise ContractError("SFT requires real demonstrations with the Panda/Robotiq action contract")
        if info.get("fps") != CONTROL_HZ or not str(info.get("codebase_version", "")).startswith("v3"):
            raise ContractError("physical SFT requires the 15 Hz LeRobot v3 export")
        stats_hash = hashlib.sha256(Path(norm_stats_path).read_bytes()).hexdigest()
        if manifest.get("norm_stats_sha256") != stats_hash or manifest.get("normalization_source") != "train split only":
            raise ContractError("normalization must match the exported train-only statistics")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.ds = LeRobotDataset(f"local/pi05_plug_{root.name}", root=root, video_backend="pyav")
        self.horizon = horizon
        table = self.ds.hf_dataset
        self.episodes = np.asarray(table["episode_index"]).reshape(-1)
        self.indices = table_window_starts(table, horizon)
        # Read action labels directly: future labels do not require decoding future videos.
        self.actions = np.asarray(table["action"], np.float32)
        if self.actions.shape != (len(self.episodes), 8) or not np.isfinite(self.actions).all() or np.any(np.abs(self.actions) > 1):
            raise ContractError("physical action labels must be finite [N,8] in [-1,1]")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        row = dict(self.ds[i])
        if row.get("task", PROMPT) != PROMPT:
            raise ContractError("unexpected physical task prompt")
        row["action"] = self.actions[i:i+self.horizon].copy()
        return row


def build_sft_dataloader(cfg, world_size, rank, data_paths, eval_dataset=False):
    """RLinf's dedicated physical loader, using the official OpenPI transforms."""
    import torch
    from omegaconf import OmegaConf
    import openpi.training.data_loader as loaders
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from pi05_rlt_rtc.adapters.panda_policy import register_panda_profile
    register_panda_profile()
    paths = OmegaConf.to_container(data_paths, resolve=True) if OmegaConf.is_config(data_paths) else data_paths
    if len(paths) != 1:
        raise ContractError("provide one physical train or validation export per loader")
    root = Path(paths[0]["dataset_path"]).expanduser().resolve()
    if root.name != ("validation" if eval_dataset else "train"):
        raise ContractError("use the session-separated train/validation export")
    model = cfg.actor.model
    kwargs = OmegaConf.to_container(model.openpi_data, resolve=True)
    config = get_openpi_config(model.openpi.config_name, model_path=model.model_path, data_kwargs=kwargs)
    if int(model.openpi.action_horizon) != config.model.action_horizon:
        raise ContractError("SFT data and model horizons differ")
    if int(model.openpi.model_action_dim) != config.model.action_dim:
        raise ContractError("SFT data and model action dimensions differ")
    data = config.data.create(config.assets_dirs, config.model)
    ds = PhysicalChunkDataset(root, config.model.action_horizon, norm_stats_path=kwargs["norm_stats_path"])
    transformed = loaders.transform_dataset(ds, data)
    sampler = torch.utils.data.DistributedSampler(transformed, num_replicas=world_size, rank=rank,
        shuffle=not eval_dataset, seed=int(cfg.actor.seed), drop_last=True)
    size = int(cfg.actor.get("eval_batch_size", cfg.actor.micro_batch_size) if eval_dataset else cfg.actor.micro_batch_size)
    if len(sampler) < size:
        raise ContractError("not enough complete physical windows for one batch per rank")
    inner = loaders.TorchDataLoader(transformed, size, sampler=sampler, framework="pytorch",
        num_workers=int(cfg.data.get("num_workers", 0)), seed=int(cfg.actor.seed))
    return loaders.DataLoaderImpl(data, inner), data
