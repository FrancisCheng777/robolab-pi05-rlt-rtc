"""Dataset auditor checks on synthetic fixtures (marks them as fixtures)."""

from __future__ import annotations

import json

from pi05_rlt_rtc.data.auditor import (
    DatasetAuditor,
    compute_action_norm_stats,
    episode_split,
)
from pi05_rlt_rtc.contracts.errors import ContractError

import pytest


def _write_meta(root, *, state_dim=9, action_dim=8):
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "fps": 10,
                "repo_id": "fixture/peg",
                "features": {
                    "observation.state": {"shape": [state_dim], "dtype": "float32"},
                    "action": {"shape": [action_dim], "dtype": "float32"},
                    "observation.image": {"shape": [3, 4, 4], "dtype": "image"},
                    "observation.wrist_image": {"shape": [3, 4, 4], "dtype": "image"},
                },
            }
        ),
        encoding="utf-8",
    )


def test_auditor_fails_on_missing_root(tmp_path):
    report = DatasetAuditor(tmp_path / "nope").run()
    assert not report.ok
    assert any("root_exists" in error for error in report.errors)


def test_auditor_fails_on_wrong_dims(tmp_path):
    root = tmp_path / "fixture"
    _write_meta(root, state_dim=7, action_dim=7)
    (root / "data").mkdir(parents=True)
    report = DatasetAuditor(root).run()
    assert not report.ok
    assert any("feature_dims" in error for error in report.errors)


def test_auditor_reports_missing_data_dir(tmp_path):
    root = tmp_path / "fixture"
    _write_meta(root)
    report = DatasetAuditor(root).run()
    assert not report.ok
    assert any("data_files" in error for error in report.errors)


def test_episode_split_is_deterministic_and_episode_level():
    episodes = [f"ep{i}" for i in range(10)]
    first = episode_split(episodes, seed=0, train_frac=0.8, val_frac=0.1)
    second = episode_split(episodes, seed=0, train_frac=0.8, val_frac=0.1)
    assert first == second
    assert len(first["train"]) == 8
    assert len(first["val"]) == 1
    assert len(first["test"]) == 1
    all_ids = first["train"] + first["val"] + first["test"]
    assert sorted(all_ids) == sorted(episodes)
    with pytest.raises(ContractError):
        episode_split([], seed=0)


def test_norm_stats_are_train_split_arrays_only():
    rng = __import__("numpy").random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(100, 8))
    quantile = compute_action_norm_stats(actions, method="quantile")
    assert quantile.q01.shape == (8,) and quantile.q99.shape == (8,)
    assert (quantile.q99 > quantile.q01).all()
    zscore = compute_action_norm_stats(actions, method="zscore")
    assert zscore.mean.shape == (8,) and zscore.std.shape == (8,)
    with pytest.raises(ContractError):
        compute_action_norm_stats(actions, method="bogus")
    constant = __import__("numpy").zeros((10, 8))
    with pytest.raises(ContractError):
        compute_action_norm_stats(constant, method="quantile")
