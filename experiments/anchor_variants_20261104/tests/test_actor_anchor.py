import copy
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.deployment import session, training
from pi05_rlt_rtc.deployment.actor_anchor import anchor_targets
from pi05_rlt_rtc.learning.planning_state import PlanningStateSpec
from pi05_rlt_rtc.learning.replay import ReplayBuffer
from tests.test_review_controller import manual_runtime
from tests.test_session_loop import Keys, episode_setup


def sample(episode_setup, tmp_path):
    cfg, _, backend, metadata = episode_setup
    episode = tmp_path/'episodes/one'
    session.run_episode(cfg, backend, metadata, None, None, Keys({5: ['b'], 30: ['c']}), episode, seed=1)
    replay = ReplayBuffer.load(episode/'label/replay.pt')
    spec = PlanningStateSpec.from_contract(cfg.contract, z_dim=16)
    return cfg, replay, spec


def test_anchor_uses_full_reference_without_mutating_executions(episode_setup, tmp_path):
    cfg, replay, spec = sample(episode_setup, tmp_path)
    actor = training.QueueLearner(cfg, spec, device='cpu', seed=7).actor.eval()
    actor.requires_grad_(False)
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    rows = replay.all_transitions()
    refs = [tr.planning_state.reference_full.copy() for tr in rows]
    changed, count = anchor_targets(rows, actor, cfg.contract)
    assert count == len(replay.learnable_transitions())
    for old, new, ref in zip(rows, changed, refs):
        assert new.planning_state is old.planning_state
        assert new.submitted_plan is old.submitted_plan
        assert new.executed_trace is old.executed_trace
        assert new.primitive_rewards is old.primitive_rewards
        np.testing.assert_array_equal(new.planning_state.reference_full, ref)
        with torch.inference_mode():
            expected = actor.act(state=torch.as_tensor(old.planning_state.actor_state_vector()),
                reference_suffix=torch.as_tensor(old.planning_state.reference_suffix_vector(cfg.contract)),
                deterministic=True).actions[0].numpy()
        np.testing.assert_allclose(new.bc_target_plan, expected, atol=1e-6)
        assert new.bc_target_valid_mask.all()
    assert all(torch.equal(before[k], v) for k, v in actor.state_dict().items())
    actor.train()
    with pytest.raises(ContractError, match='frozen'): anchor_targets(rows, actor, cfg.contract)


def test_critic_only_phase_preserves_actor_and_optimizer(episode_setup, tmp_path):
    cfg, replay, spec = sample(episode_setup, tmp_path)
    learner = training.QueueLearner(cfg, spec, device='cpu', seed=1)
    learner.warmup(replay, 1)
    actor = {k:v.clone() for k,v in learner.actor.state_dict().items()}
    critic = {k:v.clone() for k,v in learner.critic.state_dict().items()}
    opt = copy.deepcopy(learner.actor_opt.state_dict())
    learner.update(replay, 8, critic_only_updates=8)
    assert all(torch.equal(actor[k], v) for k,v in learner.actor.state_dict().items())
    now = learner.actor_opt.state_dict()
    assert now['param_groups'] == opt['param_groups']
    for key, state in opt['state'].items():
        for name, value in state.items():
            assert torch.equal(value, now['state'][key][name])
    assert any(not torch.equal(critic[k], v) for k,v in learner.critic.state_dict().items())
    assert learner.counts['actor_updates'] == 0
    learner.update(replay, 4)
    assert learner.counts['updates'] == 12 and learner.counts['actor_updates'] == 1
    assert any(not torch.equal(actor[k], v) for k,v in learner.actor.state_dict().items())


def test_dropout_override_is_explicit_and_cannot_change_learning_rate(episode_setup, tmp_path, monkeypatch):
    cfg, replay, spec = sample(episode_setup, tmp_path)
    checkpoint = session.update_round(cfg, tmp_path, seed=1, device='cpu', mock=True)
    config = dataclasses.replace(cfg, raw={**cfg.raw, 'training':{**cfg.raw['training'], 'reference_dropout':0.0}})
    monkeypatch.setattr(training, 'config_from_args', lambda _:config)
    args = SimpleNamespace(runtime_config=str(cfg.path), replay=[str(tmp_path/'episodes/one/label')],
        warmup_updates=0, updates=4, seed=1, learner_device='cpu', resume=str(checkpoint),
        out=str(tmp_path/'changed'), allow_mock=True, anchor_checkpoint=str(checkpoint/'policy.pt'))
    with pytest.raises(ContractError, match='settings changed'): training.train(args)
    args.allow_reference_dropout_change = True
    assert training.train(args) == 0
    config = dataclasses.replace(config, raw={**config.raw, 'training':{**config.raw['training'], 'actor_lr':1e-4}})
    args.out = str(tmp_path/'forbidden')
    with pytest.raises(ContractError, match='settings changed'): training.train(args)
