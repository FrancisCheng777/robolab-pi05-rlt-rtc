"""Reward comparisons preserve raw labels and the terminal tick discount."""
import dataclasses
import numpy as np
import pytest
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.deployment.reward_override import relabel_terminal_transition, reward_definition
from tests.test_learner import _collect
from pi05_rlt_rtc.contracts.spaces import HorizonContract


def example():
    _, _, _, transitions = _collect(HorizonContract(horizon=10, chunk=5, delay=2), num_ticks=30)
    return dataclasses.replace(transitions[0], capture_tick=10, next_capture_tick=13,
        duration_ticks=3, executed_trace=transitions[0].executed_trace[:3].copy(),
        primitive_rewards=np.array([0.,0.,1.,0.,0.],dtype=np.float32),
        reward_valid_mask=np.array([True,True,True,False,False]),
        terminated=True,bootstrap_allowed=False)


@pytest.mark.parametrize('outcome,value',[('success',2.),('failure',-.5)])
def test_relabels_only_terminal_tick_and_preserves_original(outcome,value):
    t=example();before=t.primitive_rewards.copy()
    label={'eligible':True,'outcome':outcome,'terminal_tick':12,'reward_source':'manual_terminal'}
    changed=relabel_terminal_transition(t,label,{'success':2.,'failure':-.5})
    np.testing.assert_array_equal(t.primitive_rewards,before)
    np.testing.assert_array_equal(changed.primitive_rewards,[0,0,value,0,0])
    assert changed.reward_sum == pytest.approx(value*t.gamma_per_tick**2)
    assert changed.terminated and not changed.bootstrap_allowed
    assert changed.planning_state is t.planning_state


def test_nonterminal_is_not_given_failure_penalty_each_tick():
    t=dataclasses.replace(example(),terminated=False,primitive_rewards=np.zeros(5,dtype=np.float32))
    label={'eligible':True,'outcome':'failure','terminal_tick':22,'reward_source':'manual_terminal'}
    result=relabel_terminal_transition(t,label,{'success':2.,'failure':-.5})
    assert result.reward_sum==0.


def test_invalid_terminal_tick_or_label_is_rejected():
    t=example()
    with pytest.raises(ContractError,match='terminal'):
        relabel_terminal_transition(t,{'eligible':True,'outcome':'failure','terminal_tick':20,'reward_source':'manual_terminal'}, {'success':2.,'failure':-.5})
    with pytest.raises(ContractError,match='eligible'):
        relabel_terminal_transition(t,{'eligible':False,'outcome':'aborted'}, {'success':2.,'failure':-.5})


def test_reward_definition_rejects_nonfinite():
    assert reward_definition({'reward_scheme':'rlt_binary'})=={'success':1.,'failure':0.}
    with pytest.raises(ContractError):reward_definition({'terminal_rewards':{'success':float('nan'),'failure':-.5}})


from tests.test_session_loop import episode_setup, Keys
from tests.test_review_controller import manual_runtime
from types import SimpleNamespace
import hashlib
import json
from pi05_rlt_rtc.deployment import session, training
from pi05_rlt_rtc.learning.checkpoint import PolicyBundle
from pi05_rlt_rtc.learning.replay import ReplayBuffer


def test_reward_comparison_is_explicit_and_preserves_labels_and_parent(episode_setup,tmp_path,monkeypatch):
    cfg,_,backend,metadata=episode_setup
    root=tmp_path/'base'
    for i,key in enumerate(('c','a'),1):
        session.run_episode(cfg,backend,metadata,None,None,Keys({5:['b'],30:[key]}),
            root/'episodes'/str(i),seed=i)
    parent=session.update_round(cfg,root,seed=1,device='cpu',mock=True)
    labels=[str(p) for p in sorted((root/'episodes').glob('*/label'))]
    paths=[parent/'policy.pt']+[__import__('pathlib').Path(p)/'replay.pt' for p in labels]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    new_cfg=dataclasses.replace(cfg,raw={**cfg.raw,'training':{**cfg.raw['training'],
        'terminal_rewards':{'success':2.,'failure':-.5}}})
    monkeypatch.setattr(training,'config_from_args',lambda args:new_cfg)
    args=SimpleNamespace(replay=labels,warmup_updates=0,updates=4,seed=1,
        learner_device='cpu',resume=str(parent),out=str(tmp_path/'comparison'),allow_mock=True)
    with pytest.raises(ContractError,match='resume training settings changed'):
        training.train(args)
    args.allow_reward_change=True
    training.train(args)
    result=PolicyBundle.load(tmp_path/'comparison/latest/policy.pt')
    assert result.extra['learner_reward_definition']=={'success':2.,'failure':-.5}
    replay=ReplayBuffer.load(tmp_path/'comparison/latest/replay.pt')
    assert sorted(float(t.primitive_rewards.sum()) for t in replay.learnable_transitions() if t.terminated)==[-.5,2.]
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==before[str(p)] for p in paths)
    audit=json.loads((tmp_path/'comparison/reward_audit.json').read_text())
    assert audit['migration']['from']=={'success':1.,'failure':0.}
    bad=dataclasses.replace(new_cfg,raw={**new_cfg.raw,'training':{**new_cfg.raw['training'],'actor_lr':.001}})
    monkeypatch.setattr(training,'config_from_args',lambda args:bad)
    with pytest.raises(ContractError,match='resume training settings changed'):
        training.train(args)
