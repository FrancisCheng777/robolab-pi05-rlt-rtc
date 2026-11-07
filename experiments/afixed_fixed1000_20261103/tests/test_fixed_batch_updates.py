import dataclasses
import json
from types import SimpleNamespace

import pytest
import yaml

from pi05_rlt_rtc.config.runtime_config import ConfigError, load_runtime_config
from pi05_rlt_rtc.deployment import session, training
from pi05_rlt_rtc.learning.checkpoint import PolicyBundle
from pi05_rlt_rtc.learning.replay import ReplayBuffer
from tests.test_review_controller import manual_runtime
from tests.test_session_loop import Keys, episode_setup


@pytest.mark.parametrize('pending_count', [20, 23])
def test_fixed_budget_does_not_scale_with_pending_data(episode_setup, tmp_path, monkeypatch, pending_count):
    cfg, _, _, _ = episode_setup
    cfg = dataclasses.replace(cfg, raw={**cfg.raw, 'session': {
        **cfg.raw['session'], 'update_every_episodes': 20,
        'updates_per_episode': 100, 'updates_per_batch': 1000}})
    previous = tmp_path/'rounds'/'round-000011'
    previous.mkdir(parents=True)
    labels = [f'label-{i}' for i in range(pending_count)]
    monkeypatch.setattr(session, 'training_batch', lambda *a: (previous, [], labels, 20))
    monkeypatch.setattr(ReplayBuffer, 'load', lambda *a: SimpleNamespace(learnable_transitions=lambda: [None, None]))
    calls = []
    def fake_train(args):
        calls.append(args)
        from pathlib import Path
        (Path(args.out)/'latest').mkdir(parents=True)
    monkeypatch.setattr(training, 'train', fake_train)
    session.update_round(cfg, tmp_path, seed=1, device='cpu', mock=True)
    assert calls[0].updates == 1000
    assert calls[0].warmup_updates == 0
    assert calls[0].resume == str(previous/'latest')
    status = json.loads((tmp_path/'batch_status.json').read_text())
    assert status['critic_updates'] == 1000


def test_fixed_budget_resume_preserves_actor_ratio_and_no_duplicate_update(episode_setup, tmp_path):
    cfg, _, backend, metadata = episode_setup
    cfg = dataclasses.replace(cfg, raw={**cfg.raw, 'session': {
        **cfg.raw['session'], 'update_every_episodes': 2,
        'updates_per_episode': 100, 'updates_per_batch': 4}})
    checkpoint = None
    for i in range(1, 5):
        session.run_episode(cfg, backend, metadata, None, None, Keys({5: ['b'], 30: ['c']}),
                            tmp_path/'episodes'/f'{i:06d}', seed=i)
        current = session.update_round(cfg, tmp_path, seed=1, device='cpu', mock=True)
        if i % 2:
            assert current == checkpoint
        else:
            checkpoint = current
            counts = PolicyBundle.load(checkpoint/'policy.pt').training_counts
            assert counts['updates'] == (i // 2) * 4
            assert counts['actor_updates'] == i // 2
            assert session.update_round(cfg, tmp_path, seed=1, device='cpu', mock=True) == checkpoint


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '1000'])
def test_invalid_fixed_update_budget_rejected(tmp_path, value):
    cfg = load_runtime_config('configs/runtime_franka_stage2.yaml')
    cfg.raw['session']['updates_per_batch'] = value
    path = tmp_path/'runtime.yaml'
    path.write_text(yaml.safe_dump(cfg.raw))
    with pytest.raises(ConfigError, match='updates_per_batch'):
        load_runtime_config(path)
