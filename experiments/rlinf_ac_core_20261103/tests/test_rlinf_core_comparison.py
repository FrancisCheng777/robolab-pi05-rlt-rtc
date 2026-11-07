"""The comparison uses upstream AC inputs while retaining the RTC transport."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.spaces import HorizonContract
from pi05_rlt_rtc.deployment.training import QueueLearner
from pi05_rlt_rtc.deployment.reward_override import relabel_terminal_transition
from pi05_rlt_rtc.learning.checkpoint import PolicyBundle
from pi05_rlt_rtc.learning.planning_state import PlanningStateSpec
from pi05_rlt_rtc.learning.losses import build_transition_batch
from tests.test_terminal_reward_override import example


def learner():
    spec = PlanningStateSpec(z_dim=16)
    return QueueLearner(SimpleNamespace(raw={'training': {}}), spec, device='cpu', seed=1234)


def test_core_actor_ignores_queue_but_uses_observation_and_reference():
    model = learner().actor
    state = torch.randn(2, model.state_dim)
    ref = torch.randn(2, model.ref_dim)
    changed = state.clone()
    changed[:, 25:] += 10
    expected = model(state, ref).actions
    torch.testing.assert_close(model(changed, ref).actions, expected, rtol=0, atol=0)
    changed[:, :25] += 1
    assert not torch.equal(model(changed, ref).actions, expected)
    assert not torch.equal(model(state, ref + 1).actions, expected)


def test_core_critic_ignores_queue_and_teacher_reference():
    model = learner().critic.online
    state = torch.randn(2, model.state_dim)
    action = torch.randn(2, model.action_dim)
    changed = state.clone()
    changed[:, 25:] += 10
    expected = model(state, action)
    torch.testing.assert_close(model(changed, action), expected, rtol=0, atol=0)
    changed[:, :25] += 1
    assert not torch.equal(model(changed, action), expected)
    assert not torch.equal(model(state, action + 1), expected)


def test_reward_and_gamma_migration_preserves_source_and_recomputes_return():
    source = example()
    original = source.primitive_rewards.copy()
    label = {'eligible': True, 'outcome': 'failure', 'terminal_tick': 12,
             'reward_source': 'manual_terminal'}
    changed = relabel_terminal_transition(source, label, {'success': 1., 'failure': -1.},
                                         gamma_per_tick=.96)
    assert source.gamma_per_tick != .96
    np.testing.assert_array_equal(source.primitive_rewards, original)
    assert changed.gamma_per_tick == .96
    assert changed.reward_sum == pytest.approx(-.96**2)
    assert changed.terminated and not changed.bootstrap_allowed
    with pytest.raises(ContractError):
        relabel_terminal_transition(source, label, {'success': 1., 'failure': -1.},
                                    gamma_per_tick=float('nan'))


def test_core_bundle_roundtrip_and_old_architecture_rejection(tmp_path):
    model = learner()
    bundle = PolicyBundle(spec=model.spec, contract=HorizonContract(10, 5, 5),
        actor_state=model.actor.state_dict(), critic_state=model.critic.state_dict(),
        feature_version_id='test-feature', norm_stats_sha256='test-stats', policy_version_id='test')
    assert bundle.ac_architecture == 'rlinf_core_rtc_v1'
    path = bundle.save(tmp_path / 'policy.pt')
    restored = PolicyBundle.load(path)
    actor, critic = restored.build_actor(), restored.build_critic()
    state = torch.randn(1, model.actor.state_dim)
    ref = torch.randn(1, model.actor.ref_dim)
    torch.testing.assert_close(actor(state, ref).actions, model.actor(state, ref).actions)
    cstate = torch.randn(1, model.critic.online.state_dim)
    torch.testing.assert_close(critic.online(cstate, ref), model.critic.online(cstate, ref))
    with pytest.raises(ContractError):
        dataclasses.replace(bundle, ac_architecture='rlinf_mlp_queue_v2')


def test_nonterminal_discount_changes_in_the_actual_learning_batch():
    from tests.test_learner import _collect
    contract = HorizonContract(10, 5, 2)
    _, _, spec, transitions = _collect(contract, num_ticks=30)
    source = next(t for t in transitions if t.is_learnable() and t.next_planning_state is not None)
    label = {'eligible': True, 'outcome': 'success', 'terminal_tick': 100,
             'reward_source': 'manual_terminal'}
    changed = relabel_terminal_transition(source, label, {'success': 1., 'failure': -1.},
                                         gamma_per_tick=.96)
    batch = build_transition_batch([changed], spec, contract=contract)
    assert batch.bootstrap_discount.item() == pytest.approx(.96**changed.duration_ticks)
    assert batch.reward_sum.item() == 0
    assert source.gamma_per_tick == .9


def test_source_manifest_ids_need_not_share_directory_order(tmp_path):
    import json
    import os
    from scripts.train_rlinf_core_comparison import ordered_sources
    folders = []
    for trial, timestamp in [('b', 100), ('a', 200)]:
        folder = tmp_path / trial
        folder.mkdir()
        rollout = folder / 'rollout.pt'
        rollout.write_bytes(b'ordering test only')
        os.utime(rollout, ns=(timestamp, timestamp))
        (folder / 'annotation.json').write_text(json.dumps({'trial_id': trial, 'rollout': str(rollout)}))
        folders.append(str(folder))
    sources = {'trial_ids': ['a', 'b'], 'replay_directories': list(reversed(folders))}
    directories, ids = ordered_sources(sources)
    assert ids == ['b', 'a']
    assert directories == folders
    with pytest.raises(RuntimeError, match='disagree'):
        ordered_sources({**sources, 'trial_ids': ['a', 'c']})
