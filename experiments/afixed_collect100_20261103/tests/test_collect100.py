import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from pi05_rlt_rtc.deployment import session, managed_eval, bridge_preflight, training
from pi05_rlt_rtc.config import runtime_config
from pi05_rlt_rtc.learning.replay import ReplayBuffer
from tests.test_review_controller import manual_runtime
from tests.test_session_loop import episode_setup


@pytest.mark.parametrize('mode,settings,flag', [
    ('train', {'exploration': False}, '--no-exploration'),
    ('train', {}, '--exploration'),
    ('eval', {'exploration': True}, '--no-exploration'),
])
def test_launch_exploration_setting(tmp_path, monkeypatch, mode, settings, flag):
    path = tmp_path/'bridge.json'; path.write_text('{}')
    cfg = SimpleNamespace(raw={'franka': {'socket': 'mock', 'bridge_config': str(path)},
                               'session': settings}, asset=lambda *a: path)
    monkeypatch.setattr(runtime_config, 'load_runtime_config', lambda _: cfg)
    monkeypatch.setattr(bridge_preflight, 'prepare_socket', Mock())
    monkeypatch.setattr(bridge_preflight, 'probe_bridge', Mock())
    bridge = Mock(); bridge.poll.return_value = None
    child = Mock(); child.wait.return_value = 0
    popen = Mock(side_effect=[bridge, child])
    monkeypatch.setattr(managed_eval.subprocess, 'Popen', popen)
    monkeypatch.setattr(managed_eval, 'stop_child', Mock())
    assert managed_eval.run_eval('mock', tmp_path, 'mock', mode=mode) == 0
    command = popen.call_args_list[1].args[0]
    assert flag in command
    assert ('--exploration' if flag == '--no-exploration' else '--no-exploration') not in command


@pytest.mark.parametrize('count', [0, 1, 20, 99, 100])
def test_only_hundred_new_episodes_trigger_training(episode_setup, tmp_path, monkeypatch, count):
    cfg, _, _, _ = episode_setup
    cfg = dataclasses.replace(cfg, raw={**cfg.raw, 'session': {
        **cfg.raw['session'], 'update_every_episodes': 100, 'updates_per_batch': 1000}})
    previous = tmp_path/'rounds/round-000010'; previous.mkdir(parents=True)
    monkeypatch.setattr(session, 'training_batch', lambda *a: (previous, [], [str(i) for i in range(count)], 100))
    monkeypatch.setattr(ReplayBuffer, 'load', lambda *a: SimpleNamespace(learnable_transitions=lambda: [None, None]))
    calls = []
    def fake_train(args):
        from pathlib import Path
        calls.append(args); (Path(args.out)/'latest').mkdir(parents=True)
    monkeypatch.setattr(training, 'train', fake_train)
    checkpoint = session.update_round(cfg, tmp_path, seed=0, device='cpu', mock=True)
    if count < 100:
        assert checkpoint == previous/'latest'
        assert calls == []
    else:
        assert len(calls) == 1 and calls[0].updates == 1000
        assert calls[0].resume == str(previous/'latest')
        assert calls[0].warmup_updates == 0


def test_invalid_exploration_setting_rejected(tmp_path):
    cfg = runtime_config.load_runtime_config('configs/runtime_franka_stage2.yaml')
    cfg.raw['session']['exploration'] = 'false'
    path = tmp_path/'runtime.yaml'; path.write_text(yaml.safe_dump(cfg.raw))
    with pytest.raises(runtime_config.ConfigError, match='exploration'):
        runtime_config.load_runtime_config(path)
