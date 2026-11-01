"""Project action contract, quantile normalization and Pi0.5 tokenization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from pi05_rlt_rtc.contracts.physical import ACTION_CONTRACT_ID as CONTRACT_ID
from pi05_demo.chunks import table_window_starts, WINDOW_POLICY

CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def read_manifest(root: Path, allow_mock: bool = False) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["action_contract_id"] != CONTRACT_ID:
        raise ValueError("Dataset action contract does not match Panda plug demo")
    if manifest["mock"] and not allow_mock:
        raise ValueError("Mock data cannot be used for production SFT")
    from pi05_rlt_rtc.data.profile import validate_dataset_profile
    validate_dataset_profile(manifest)
    norm = root / "norm_stats.json"
    if hashlib.sha256(norm.read_bytes()).hexdigest() != manifest["norm_stats_sha256"]:
        raise ValueError("Normalization statistics checksum mismatch")
    if manifest.get("normalization_source") != "train split only":
        raise ValueError("Normalization must be computed from training sessions only")
    return manifest


class Transform:
    """Same quantile and discrete-state ordering as OpenPI's Pi0.5 pipeline."""

    def __init__(self, norm_stats: dict, tokenizer_path: str, max_len: int = 200):
        import sentencepiece

        self.stats = norm_stats["norm_stats"]
        for key, dim in (("state", 9), ("actions", 8)):
            for field in ("q01", "q99"):
                arr = np.asarray(self.stats[key][field], dtype=np.float32)
                if arr.shape != (dim,) or not np.isfinite(arr).all():
                    raise ValueError(f"Invalid {key}.{field}")
            if np.any(np.asarray(self.stats[key]["q99"]) < self.stats[key]["q01"]):
                raise ValueError(f"Reversed quantiles for {key}")
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(tokenizer_path)
        )
        self.max_len = max_len

    def normalize(self, value: np.ndarray, key: str) -> np.ndarray:
        lo = np.asarray(self.stats[key]["q01"], np.float32)
        hi = np.asarray(self.stats[key]["q99"], np.float32)
        return (value - lo) / (hi - lo + 1e-6) * 2.0 - 1.0

    def __call__(self, raw: dict) -> dict:
        state = np.asarray(raw["state"], dtype=np.float32)
        actions = np.asarray(raw["actions"], dtype=np.float32)
        if state.shape != (9,) or actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError("Expected state[9] and actions[H,8]")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("Nonfinite state or actions")
        if np.any(np.abs(actions) > 1.00001):
            raise ValueError("Raw action outside normalized robot command bounds")
        state = self.normalize(state, "state")
        actions = self.normalize(actions, "actions")
        # Tokenize the nine real state dimensions BEFORE padding to model width 32.
        discrete = np.digitize(state, np.linspace(-1.0, 1.0, 257)[:-1]) - 1
        prompt = raw["prompt"].strip().replace("_", " ").replace("\n", " ")
        text = f"Task: {prompt}, State: {' '.join(map(str, discrete))};\nAction: "
        tokens = self.tokenizer.encode(text, add_bos=True)
        if len(tokens) > self.max_len:
            raise ValueError("Pi0.5 prompt would truncate; increase max_token_len")
        valid = len(tokens)
        observation = {
            "image": {k: np.asarray(raw["image"][k], dtype=np.uint8) for k in CAMERAS},
            "image_mask": {
                k: np.asarray(raw["image_mask"][k], dtype=bool) for k in CAMERAS
            },
            "state": np.pad(state, (0, 23)).astype(np.float32),
            "tokenized_prompt": np.asarray(
                tokens + [0] * (self.max_len - valid), np.int64
            ),
            "tokenized_prompt_mask": np.arange(self.max_len) < valid,
        }
        for image in observation["image"].values():
            if image.shape != (224, 224, 3):
                raise ValueError("Expected RGB224 image")
        return {
            "observation": observation,
            "actions": np.pad(actions, ((0, 0), (0, 24))).astype(np.float32),
        }


class PlugDataset(Dataset):
    def __init__(
        self,
        root: str,
        tokenizer_path: str,
        horizon: int = 10,
        max_len: int = 200,
        split: str = "train",
        allow_mock: bool = False,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root)
        read_manifest(self.root, allow_mock)
        self.ds = LeRobotDataset(
            f"local/pi05_plug_{split}", root=self.root / split, video_backend="pyav"
        )
        self.transform = Transform(
            json.loads((self.root / "norm_stats.json").read_text()),
            tokenizer_path,
            max_len,
        )
        self.horizon = horizon
        # Read action chunks from the table, not repeated video decodes.
        self.actions = np.stack(
            [np.asarray(x, np.float32) for x in self.ds.hf_dataset["action"]]
        )
        self.episodes = np.asarray(self.ds.hf_dataset["episode_index"], np.int64)
        self.starts = table_window_starts(self.ds.hf_dataset, horizon)
        self.window_policy = dict(WINDOW_POLICY)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        row = self.ds[start]
        from pi05_rlt_rtc.data.profile import model_images, lerobot_rgb
        images, masks = model_images(lerobot_rgb(row["observation.images.wrist"]))
        raw = {
            "state": np.asarray(row["observation.state"]),
            "actions": self.actions[start:start + self.horizon].copy(),
            "image": images,
            "image_mask": masks,
            "prompt": row["task"],
        }
        return self.transform(raw)


class SyntheticDataset(Dataset):
    """Explicitly synthetic full-model memory/gradient smoke inputs; no success metric."""

    def __init__(
        self, tokenizer_path: str, horizon: int, max_len: int, length: int = 4096
    ):
        norm = {
            "norm_stats": {
                key: {"q01": [-1.0] * dim, "q99": [1.0] * dim}
                for key, dim in (("state", 9), ("actions", 8))
            }
        }
        self.transform = Transform(norm, tokenizer_path, max_len)
        self.horizon, self.length = horizon, length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        return self.transform(
            {
                "state": rng.uniform(-0.5, 0.5, 9).astype(np.float32),
                "actions": rng.uniform(-0.5, 0.5, (self.horizon, 8)).astype(np.float32),
                "image": {
                    k: rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
                    if i < 2
                    else np.zeros((224, 224, 3), np.uint8)
                    for i, k in enumerate(CAMERAS)
                },
                "image_mask": dict(zip(CAMERAS, (True, False, False))),
                "prompt": "insert the plug into the power strip",
            }
        )
