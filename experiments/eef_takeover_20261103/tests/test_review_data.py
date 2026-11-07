import json

import numpy as np

from pi05_rlt_rtc.data import auditor as mod
from tests.test_auditor import _write_meta


class Column:
    def __init__(self, values):
        self.values = np.asarray(values)

    def to_numpy(self):
        return self.values

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)


class Frames:
    def __init__(self, *, timestamps=(0.0, 0.1)):
        self.data = {"observation.state": np.zeros((2, 9)), "action": np.zeros((2, 8)),
                     "timestamp": timestamps, "episode_index": [0, 0], "reward": [0, 1]}
        self.columns = tuple(self.data)

    def __len__(self):
        return 2

    def __getitem__(self, key):
        return Column(self.data[key])


def test_second_unreadable_shard_cannot_pass_partial_dataset(tmp_path, monkeypatch):
    _write_meta(tmp_path)
    (tmp_path / "data").mkdir()
    for name in ("a.parquet", "b.parquet"):
        (tmp_path / "data" / name).touch()
    def load(path):
        if path.name == "b.parquet":
            raise OSError("corrupt second shard")
        return Frames()
    monkeypatch.setattr(mod, "_load_parquet", load)
    report = mod.DatasetAuditor(tmp_path).run()
    assert not report.ok
    assert not any(check.name == "read_frames" and check.status == "PASSED" for check in report.checks)


def test_nan_timestamps_fail_the_audit():
    report = mod.DataAuditReport("fixture", {})
    mod.DatasetAuditor("fixture")._check_timestamps(report, Frames(timestamps=(0.0, np.nan)))
    assert not report.ok


def test_one_camera_is_not_a_two_camera_dataset(tmp_path):
    _write_meta(tmp_path)
    features = json.loads((tmp_path / "meta/info.json").read_text())["features"]
    del features["observation.wrist_image"]
    report = mod.DataAuditReport("fixture", {})
    mod.DatasetAuditor(tmp_path)._check_features(report, features)
    assert not report.ok


def test_incomplete_audit_is_explicitly_incomplete(tmp_path, monkeypatch):
    _write_meta(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/a.parquet").touch()
    monkeypatch.setattr(mod, "_load_parquet", lambda _: None)
    report = mod.DatasetAuditor(tmp_path).run()
    path = report.write(tmp_path / "audit.json")
    assert json.loads(path.read_text())["complete"] is False
