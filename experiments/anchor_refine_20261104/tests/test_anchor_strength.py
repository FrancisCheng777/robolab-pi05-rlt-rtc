import dataclasses
from types import SimpleNamespace

import pytest
import torch

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.deployment import session, training
from tests.test_review_controller import manual_runtime
from tests.test_session_loop import episode_setup
from tests.test_actor_anchor import sample


@pytest.mark.parametrize('strength', [2.5, 10.0])
def test_explicit_strength_override_preserves_other_settings(episode_setup, tmp_path, monkeypatch, strength):
    cfg, replay, spec = sample(episode_setup, tmp_path)
    checkpoint = session.update_round(cfg, tmp_path, seed=1, device='cpu', mock=True)
    changed = dataclasses.replace(cfg, raw={**cfg.raw, 'training':{**cfg.raw['training'], 'lambda_bc':strength}})
    monkeypatch.setattr(training, 'config_from_args', lambda _:changed)
    args = SimpleNamespace(runtime_config=str(cfg.path), replay=[str(tmp_path/'episodes/one/label')],
        warmup_updates=0, updates=4, seed=1, learner_device='cpu', resume=str(checkpoint),
        out=str(tmp_path/'variant'), allow_mock=True, anchor_checkpoint=str(checkpoint/'policy.pt'))
    with pytest.raises(ContractError, match='settings changed'):
        training.train(args)
    args.allow_anchor_strength_change = True
    assert training.train(args) == 0
    old = torch.load(checkpoint/'learner_state.pt', weights_only=True)
    new = torch.load(tmp_path/'variant/latest/learner_state.pt', weights_only=True)
    assert new['training_settings'] == {**old['training_settings'], 'lambda_bc':strength}
    assert new['counts']['updates']-old['counts']['updates']==4
    assert new['counts']['actor_updates']-old['counts']['actor_updates']==1
    changed = dataclasses.replace(changed, raw={**changed.raw, 'training':{**changed.raw['training'], 'actor_lr':1e-4}})
    args.out = str(tmp_path/'rejected')
    with pytest.raises(ContractError, match='settings changed'):
        training.train(args)
