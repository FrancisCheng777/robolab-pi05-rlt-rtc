"""Physical contract checks; no robot or camera is opened by these tests."""
import copy
import json

import numpy as np
import pytest

from pi05_rlt_rtc.config.runtime_config import load_runtime_config
from pi05_rlt_rtc.adapters.franka import load_bridge_config
from pi05_rlt_rtc.contracts.errors import ContractError


def test_default_runtime_matches_the_real_demonstration_contract():
    cfg = load_runtime_config("configs/runtime_franka.yaml")
    assert cfg.control_hz == 15
    assert cfg.action_contract_id == "panda_delta_q0.05_robotiq_width0.085_v2"
    assert cfg.raw["model"]["config_name"] == "pi05_rlt_panda_robotiq_joint"
    assert cfg.raw["franka"]["cameras"]["kwargs"]["wrist_model"] == "D405"


def test_wrong_named_hardware_contract_cannot_be_silently_accepted(tmp_path):
    cfg = load_runtime_config("configs/runtime_franka.yaml")
    raw = json.loads((cfg.path.parent / cfg.raw["franka"]["bridge_config"]).read_text())
    raw.update(action_contract_id="panda_delta_q0.03_robotiq_width0.085_v1",
               load_mass=0.9, load_center=[0, 0, .057], load_inertia=[0]*9,
               gripper_device="/dev/test-only", robotiq_open_position=0,
               robotiq_closed_position=255, hardware_verified=True)
    raw["joint_delta_scale"] = [.2]*7
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ContractError, match="contract"):
        load_bridge_config(path)


def test_sft_windows_never_repeat_a_delta_or_cross_an_episode():
    from pi05_rlt_rtc.data.physical_sft import complete_windows
    episodes = [0]*4 + [1]*4
    timestamps = [0, 1/15, 2/15, 3/15]*2
    assert complete_windows(episodes, timestamps, 3) == [0, 1, 4, 5]
    timestamps[2] += .04
    assert complete_windows(episodes, timestamps, 3) == [4, 5]
    assert complete_windows(episodes, timestamps, 3, interventions=[0,0,0,0,0,1,0,0]) == []


def test_quantile_precision_matches_collection_before_state_tokenization():
    from dataclasses import dataclass
    from pi05_rlt_rtc.adapters.panda_policy import physical_norm_stats
    @dataclass
    class Stats:
        mean: np.ndarray
        std: np.ndarray
        q01: np.ndarray
        q99: np.ndarray
    original = Stats(np.array([1.]), np.array([.001]),
                     np.array([1.00000001]), np.array([1.0001]))
    converted = physical_norm_stats({'state': original})['state']
    assert original.q01.dtype == np.float64
    assert converted.q01.dtype == np.float32
    state = np.array([1.000051], dtype=np.float32)
    normalized = (state-converted.q01)/(converted.q99-converted.q01+1e-6)*2-1
    expected = (state-original.q01.astype(np.float32))/(original.q99.astype(np.float32)-original.q01.astype(np.float32)+1e-6)*2-1
    np.testing.assert_array_equal(normalized, expected)


def test_physical_dataset_checks_export_and_reads_future_labels_without_images(tmp_path, monkeypatch):
    import hashlib
    import sys
    from types import ModuleType
    from pi05_rlt_rtc.data.physical_sft import PhysicalChunkDataset
    from pi05_rlt_rtc.contracts.physical import ACTION_CONTRACT_ID
    root = tmp_path / 'train'
    (root / 'meta').mkdir(parents=True)
    (root / 'meta/info.json').write_text(json.dumps({'fps': 15, 'codebase_version': 'v3.0'}))
    stats = tmp_path / 'norm_stats.json'
    stats.write_text('{}')
    manifest = {'mock': False, 'action_contract_id': ACTION_CONTRACT_ID,
                'norm_stats_sha256': hashlib.sha256(stats.read_bytes()).hexdigest(),
                'camera_views': ['wrist'], 'policy_image_mapping': {'base_0_rgb':'wrist'},
                'action_codec': {'joint_delta_scale':[.05]*7}, 'normalization_source': 'train split only'}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    actions = np.arange(48, dtype=np.float32).reshape(6, 8) / 100
    class Table(dict):
        @property
        def column_names(self): return list(self)
    class DatasetDouble:
        def __init__(self, repo_id, **kwargs):
            assert kwargs['root'] == root
            self.hf_dataset = Table(episode_index=[0]*3+[1]*3, source_time=[0,1/15,2/15]*2,
                                    intervention=[False]*6, action=actions)
            self.image_reads = []
        def __getitem__(self, i):
            self.image_reads.append(i)
            return {'action': actions[i]}
    module = ModuleType('lerobot.datasets.lerobot_dataset')
    module.LeRobotDataset = DatasetDouble
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ds = PhysicalChunkDataset(root, 3, norm_stats_path=stats)
    assert len(ds) == 2
    np.testing.assert_array_equal(ds[1]['action'], actions[3:6])
    assert ds.ds.image_reads == [3]
    stats.write_text('{"changed":true}')
    with pytest.raises(ContractError, match='train-only'):
        PhysicalChunkDataset(root, 3, norm_stats_path=stats)
