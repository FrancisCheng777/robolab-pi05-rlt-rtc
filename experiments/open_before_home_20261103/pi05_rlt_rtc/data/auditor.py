"""Dataset audit for the LeRobot-format physical Panda demonstrations (guide 5.2).

The auditor reports facts and never repairs them: a wrong action dimension, a
NaN, a non-monotonic timestamp or an episode-crossing window are recorded as
errors, not silently clipped.  Reads of parquet files are optional; if the
reader is unavailable the affected checks are SKIPPED with a reason instead of
being marked passed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.spaces import ENV_ACTION_DIM, PROPRIO_DIM
from pi05_rlt_rtc.contracts.transition import build_action_windows
from pi05_rlt_rtc.contracts.spaces import HorizonContract

EXPECTED_STATE_DIM = PROPRIO_DIM
EXPECTED_ACTION_DIM = ENV_ACTION_DIM


@dataclasses.dataclass
class AuditCheck:
    name: str
    status: str  # PASSED | FAILED | SKIPPED
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class DataAuditReport:
    root: str
    info: dict[str, Any]
    checks: list[AuditCheck] = dataclasses.field(default_factory=list)
    errors: list[str] = dataclasses.field(default_factory=list)
    skipped: list[str] = dataclasses.field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        if status not in ("PASSED", "FAILED", "SKIPPED"):
            raise ContractError(f"unknown audit status {status}")
        self.checks.append(AuditCheck(name, status, detail))
        if status == "FAILED":
            self.errors.append(f"{name}: {detail}")
        elif status == "SKIPPED":
            self.skipped.append(f"{name}: {detail}")

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def complete(self) -> bool:
        return self.ok and not self.skipped

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "dataset_audit",
            "root": self.root,
            "ok": self.ok,
            "complete": self.complete,
            "info": self.info,
            "checks": [c.to_dict() for c in self.checks],
            "errors": self.errors,
            "skipped": self.skipped,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def summary_text(self) -> str:
        lines = [f"dataset audit root={self.root} ok={self.ok} complete={self.complete}"]
        for check in self.checks:
            lines.append(f"  [{check.status}] {check.name}: {check.detail}")
        return "\n".join(lines)


def _load_parquet(path: Path) -> Any:
    try:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pandas()
    except ImportError:
        return None


class DatasetAuditor:
    def __init__(
        self,
        root: str | Path,
        *,
        horizon: int = 10,
        expected_cameras: tuple[str, ...] = ("main", "wrist"),
        default_prompt: str = "insert the plug into the power strip",
    ) -> None:
        self.root = Path(root)
        self.horizon = int(horizon)
        self.expected_cameras = expected_cameras
        self.default_prompt = default_prompt

    def run(self) -> DataAuditReport:
        report = DataAuditReport(root=str(self.root), info={})
        if not self.root.exists():
            report.add("root_exists", "FAILED", f"{self.root} does not exist")
            return report
        meta_info = self.root / "meta" / "info.json"
        if not meta_info.exists():
            report.add("meta_info", "FAILED", f"missing {meta_info}; not a LeRobot root")
            return report
        try:
            info = json.loads(meta_info.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.add("meta_info_parse", "FAILED", f"invalid JSON: {exc}")
            return report
        report.info = {
            "codebase_version": info.get("codebase_version"),
            "fps": info.get("fps"),
            "repo_id": info.get("repo_id"),
            "features": sorted((info.get("features") or {}).keys()),
        }
        report.add("meta_info", "PASSED", f"codebase_version={info.get('codebase_version')} fps={info.get('fps')}")

        features = info.get("features") or {}
        self._check_features(report, features)
        data_files = sorted((self.root / "data").rglob("*.parquet"))
        if not data_files:
            report.add(
                "data_files",
                "FAILED",
                f"no parquet files under {self.root / 'data'}; a parent directory is not a dataset",
            )
            return report
        report.add("data_files", "PASSED", f"{len(data_files)} parquet file(s)")

        frames = None
        load_error: str | None = None
        for path in data_files:
            try:
                table = _load_parquet(path)
            except Exception as exc:  # noqa: BLE001
                load_error = f"{path.name}: {type(exc).__name__}: {exc}"
                report.add("read_frames", "FAILED", load_error)
                return report
            if table is None:
                report.add("read_frames", "SKIPPED", f"cannot read {path.name}: pyarrow/pandas not installed; no partial-data PASS")
                return report
            frames = table if frames is None else _concat(frames, table)
        if frames is None:
            report.add(
                "read_frames",
                "SKIPPED",
                f"cannot read parquet ({load_error}); install pyarrow to run frame-level checks",
            )
            return report
        report.add("read_frames", "PASSED", f"{len(frames)} frame rows")
        if len(frames) == 0:
            report.add("nonempty_frames", "FAILED", "dataset contains no frame rows")
            return report
        for name, check in (("frame_dims", self._check_dims_and_nans), ("episode_boundaries", self._check_episodes),
                            ("timestamps", self._check_timestamps), ("rewards", self._check_rewards),
                            ("language", self._check_language), ("action_windows", self._check_windows)):
            try:
                check(report, frames)
            except (ValueError, TypeError, KeyError, IndexError, ContractError) as exc:
                report.add(name, "FAILED", f"malformed frame data: {exc}")
        return report

    # -- checks -----------------------------------------------------------------
    def _check_features(self, report: DataAuditReport, features: dict[str, Any]) -> None:
        state = features.get("observation.state") or features.get("state")
        action = features.get("action") or features.get("actions")
        if state is None or action is None:
            report.add(
                "feature_dims",
                "FAILED",
                f"features must declare state and action; found {sorted(features)}",
            )
            return
        state_dim = _feature_dim(state)
        action_dim = _feature_dim(action)
        if state_dim == EXPECTED_STATE_DIM and action_dim == EXPECTED_ACTION_DIM:
            report.add("feature_dims", "PASSED", f"state={state_dim} action={action_dim}")
        else:
            report.add(
                "feature_dims",
                "FAILED",
                f"expected state={EXPECTED_STATE_DIM} action={EXPECTED_ACTION_DIM}, "
                f"found state={state_dim} action={action_dim}",
            )
        camera_keys = [k for k in features if "image" in k]
        if len(camera_keys) >= len(self.expected_cameras):
            report.add("cameras_declared", "PASSED", f"image keys: {camera_keys}")
        else:
            report.add("cameras_declared", "FAILED", f"expected {len(self.expected_cameras)} distinct camera declarations, found {camera_keys}")

    def _check_dims_and_nans(self, report: DataAuditReport, frames: Any) -> None:
        state_col = _pick(frames, ["observation.state", "state"])
        action_col = _pick(frames, ["action", "actions"])
        if state_col is None or action_col is None:
            report.add("frame_dims", "FAILED", "frame table lacks state/action columns")
            return
        states = np.stack([np.asarray(x, dtype=np.float32) for x in state_col.to_numpy()])
        actions = np.stack([np.asarray(x, dtype=np.float32) for x in action_col.to_numpy()])
        problems: list[str] = []
        if states.shape[1] != EXPECTED_STATE_DIM:
            problems.append(f"state dim {states.shape[1]} != {EXPECTED_STATE_DIM}")
        if actions.shape[1] != EXPECTED_ACTION_DIM:
            problems.append(f"action dim {actions.shape[1]} != {EXPECTED_ACTION_DIM}")
        if not np.all(np.isfinite(states)):
            problems.append("state contains NaN/Inf")
        if not np.all(np.isfinite(actions)):
            problems.append("action contains NaN/Inf")
        if problems:
            report.add("frame_dims", "FAILED", "; ".join(problems))
        else:
            report.add("frame_dims", "PASSED", f"state={states.shape[1]} action={actions.shape[1]} finite")

    def _check_episodes(self, report: DataAuditReport, frames: Any) -> None:
        ep_col = _pick(frames, ["episode_index", "episode_id", "episode"])
        if ep_col is None:
            report.add("episode_boundaries", "FAILED", "no episode index column; cannot verify windows")
            return
        episodes = ep_col.to_numpy()
        unique = np.unique(episodes)
        lengths = [int(np.sum(episodes == e)) for e in unique]
        report.add(
            "episode_boundaries",
            "PASSED",
            f"{len(unique)} episode(s), lengths min={min(lengths)} max={max(lengths)}",
        )

    def _check_timestamps(self, report: DataAuditReport, frames: Any) -> None:
        ts_col = _pick(frames, ["timestamp", "time"])
        ep_col = _pick(frames, ["episode_index", "episode_id", "episode"])
        if ts_col is None:
            report.add("timestamps", "SKIPPED", "no timestamp column")
            return
        timestamps = np.asarray(ts_col, dtype=np.float64)
        if not np.isfinite(timestamps).all():
            report.add("timestamps", "FAILED", "timestamps contain NaN/Inf")
            return
        if ep_col is None:
            report.add("timestamps", "SKIPPED", "no episode column to group timestamps")
            return
        episodes = ep_col.to_numpy()
        bad = 0
        for e in np.unique(episodes):
            ts = timestamps[episodes == e]
            if np.any(np.diff(ts) <= 0):
                bad += 1
        if bad:
            report.add("timestamps", "FAILED", f"{bad} episode(s) with non-monotonic timestamps")
        else:
            report.add("timestamps", "PASSED", "timestamps strictly increase within each episode")

    def _check_rewards(self, report: DataAuditReport, frames: Any) -> None:
        reward_col = _pick(frames, ["next.reward", "reward"])
        done_col = _pick(frames, ["next.done", "done"])
        if reward_col is None:
            report.add("rewards", "SKIPPED", "no reward column; success statistics unavailable")
            return
        rewards = np.asarray(reward_col, dtype=np.float32)
        if not np.isfinite(rewards).all():
            report.add("rewards", "FAILED", "rewards contain NaN/Inf")
            return
        info = {"nonzero_rewards": int(np.sum(rewards != 0))}
        if done_col is not None:
            info["terminal_frames"] = int(np.sum(np.asarray(done_col).astype(bool)))
        report.add("rewards", "PASSED", json.dumps(info))

    def _check_language(self, report: DataAuditReport, frames: Any) -> None:
        task_col = _pick(frames, ["task", "prompt"])
        if task_col is None:
            report.add(
                "language",
                "PASSED",
                f"no per-row prompt; the config default prompt {self.default_prompt!r} is used",
            )
        else:
            values = set(map(str, task_col.to_numpy()))
            if values and values != {self.default_prompt}:
                report.add("language", "FAILED", f"unexpected task values {sorted(values)}")
            else:
                report.add("language", "PASSED", f"task values {sorted(values)}")

    def _check_windows(self, report: DataAuditReport, frames: Any) -> None:
        action_col = _pick(frames, ["action", "actions"])
        ep_col = _pick(frames, ["episode_index", "episode_id", "episode"])
        if action_col is None or ep_col is None:
            report.add("action_windows", "SKIPPED", "action/episode columns missing")
            return
        actions = np.stack([np.asarray(x, dtype=np.float32) for x in action_col.to_numpy()])
        episodes = ep_col.to_numpy()
        contract = HorizonContract(horizon=self.horizon, chunk=max(1, self.horizon // 2), delay=0)
        total = 0
        bad = 0
        for e in np.unique(episodes):
            ep_actions = actions[episodes == e]
            windows = build_action_windows(ep_actions, str(e), contract)
            total += len(windows)
            for w in windows:
                if not np.all(np.isfinite(w.actions)):
                    bad += 1
        report.add(
            "action_windows",
            "PASSED" if bad == 0 else "FAILED",
            f"{total} H={self.horizon} window(s); {bad} invalid",
        )


# ---------------------------------------------------------------------------
# splits and norm stats


def episode_split(
    episode_ids: list[str], *, seed: int = 0, train_frac: float = 0.8, val_frac: float = 0.1
) -> dict[str, list[str]]:
    """Deterministic episode-level split (never a frame-level random leak)."""
    if not episode_ids:
        raise ContractError("no episodes to split")
    if len(set(episode_ids)) != len(episode_ids):
        raise ContractError("duplicate episode ids could leak one episode across splits")
    if not 0.0 < train_frac < 1.0 or not 0.0 <= val_frac < 1.0:
        raise ContractError("invalid split fractions")
    if train_frac + val_frac > 1.0:
        raise ContractError("train+val fractions exceed 1")
    ranked = sorted(
        episode_ids,
        key=lambda e: hashlib.sha256(f"{seed}:{e}".encode("utf-8")).hexdigest(),
    )
    n = len(ranked)
    n_train = max(1, int(round(n * train_frac)))
    n_val = int(round(n * val_frac))
    return {
        "train": ranked[:n_train],
        "val": ranked[n_train : n_train + n_val],
        "test": ranked[n_train + n_val :],
    }


@dataclasses.dataclass
class NormStatsArrays:
    method: str
    model_action_dim: int
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def to_arrays(self) -> dict[str, np.ndarray]:
        out = {}
        if self.q01 is not None:
            out["q01"] = self.q01
        if self.q99 is not None:
            out["q99"] = self.q99
        if self.mean is not None:
            out["mean"] = self.mean
        if self.std is not None:
            out["std"] = self.std
        return out


def compute_action_norm_stats(actions: np.ndarray, *, method: str = "quantile") -> NormStatsArrays:
    """Compute norm stats from the train split only (caller's responsibility)."""
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2:
        raise ContractError(f"actions must be [N, A], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ContractError("cannot compute norm stats on NaN/Inf actions")
    if method == "quantile":
        q01 = np.quantile(arr, 0.01, axis=0)
        q99 = np.quantile(arr, 0.99, axis=0)
        if np.any(q99 - q01 <= 0):
            raise ContractError("degenerate quantile scale; report and fix the data")
        return NormStatsArrays("quantile", arr.shape[1], q01=q01, q99=q99)
    if method == "zscore":
        std = arr.std(axis=0)
        if np.any(std <= 0):
            raise ContractError("degenerate std; report and fix the data")
        return NormStatsArrays("zscore", arr.shape[1], mean=arr.mean(axis=0), std=std)
    raise ContractError(f"unknown norm method {method!r}")


def _feature_dim(feature: Any) -> int | None:
    if isinstance(feature, dict):
        shape = feature.get("shape")
        if isinstance(shape, list) and shape:
            return int(shape[-1])
        if "dim" in feature:
            return int(feature["dim"])
    return None


def _pick(frames: Any, names: list[str]) -> Any:
    for name in names:
        if name in frames.columns:
            return frames[name]
    return None


def _concat(left: Any, right: Any) -> Any:
    import pandas as pd

    return pd.concat([left, right], ignore_index=True)
