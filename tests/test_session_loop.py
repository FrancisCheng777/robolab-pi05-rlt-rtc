"""Critical phase and persistent update transactions; all observations are mock."""
import dataclasses
import json
from types import SimpleNamespace
import numpy as np
import pytest
from tests.test_review_controller import manual_runtime
from tests.test_platform_integration import backend_double
from pi05_rlt_rtc.deployment import session, session_env, training
from pi05_rlt_rtc.deployment.rollout import load_rollout, label_rollout
from pi05_rlt_rtc.learning.checkpoint import PolicyBundle
from pi05_rlt_rtc.learning.replay import ReplayBuffer


@pytest.fixture
def episode_setup(monkeypatch, manual_runtime):
    from pi05_rlt_rtc.evaluation.toy_env import ToyDelayedControlEnv
    cfg,toy,backend,metadata=backend_double()
    cfg=dataclasses.replace(cfg, raw={**cfg.raw,
        'model':{**cfg.raw['model'],'z_dim':16,'device':'cpu'},
        'session':{**cfg.raw['session'],'warmup_episodes':1,'min_replay_transitions':2,
                   'warmup_updates':1,'updates_per_episode':1}})
    metadata.update(mock=True,annotator='mock protocol test')
    class Env(ToyDelayedControlEnv):
        def __init__(self,*args):
            super().__init__();self.rlt_active=False;self.command_log=[];self.stop_request_acknowledged=False
        def observe(self):
            return dataclasses.replace(self._snapshot(),sensor_metadata={'rlt_active':self.rlt_active})
        def step(self,action):
            result=super().step(action)
            return dataclasses.replace(result,observation=self.observe(),terminated=False,truncated=False)
        def stop(self):self.stop_request_acknowledged=True
    monkeypatch.setattr(session_env,'SessionEnv',Env)
    monkeypatch.setattr(training,'config_from_args',lambda args:cfg)
    backend.critical_only=True
    return cfg,toy,backend,metadata


class Keys:
    def __init__(self, schedule):self.tick=0;self.schedule=schedule
    def read(self):
        keys=self.schedule.get(self.tick,[]);self.tick+=1;return keys


def test_critical_phase_reward_terminal_partial_chunk_and_no_entry(episode_setup,tmp_path):
    cfg,toy,backend,metadata=episode_setup
    out=tmp_path/'trial'
    signals=session.run_episode(cfg,backend,metadata,None,None,Keys({7:['b'],28:['c']}),out,seed=1)
    assert signals.terminal_tick==27
    data=load_rollout(out/'rollout.pt')
    assert len([e for e in data['events'] if e['kind']=='tick'])==28
    replay=ReplayBuffer.load(out/'label/replay.pt')
    transitions=replay.learnable_transitions()
    assert len(transitions)>=2
    assert all(t.capture_tick>=7 for t in transitions)
    assert sum(t.primitive_rewards.sum() for t in transitions)==1
    assert transitions[-1].terminated and not transitions[-1].bootstrap_allowed
    data['metadata']['critical_start_tick']=None
    replay,annotation=label_rollout(data,outcome='success',annotator='test',allow_mock=True)
    assert not annotation['eligible'] and len(replay)==0


def test_episode_round_resume_is_once_and_keeps_history(episode_setup,tmp_path):
    cfg,toy,backend,metadata=episode_setup
    root=tmp_path/'session'
    for index,outcome in [(1,'c'),(2,'a')]:
        session.run_episode(cfg,backend,metadata,None,None,Keys({5:['b'],30:[outcome]}),
                            root/'episodes'/f'{index:06d}',seed=index)
        saved=session.update_round(cfg,root,seed=1,device='cpu',mock=True)
        bundle=PolicyBundle.load(saved/'policy.pt')
        assert bundle.training_counts['rounds']==index
        assert bundle.training_counts['warmup_updates']==1
        digest=bundle.actor_state_sha256
        assert session.update_round(cfg,root,seed=1,device='cpu',mock=True)==saved
        assert PolicyBundle.load(saved/'policy.pt').actor_state_sha256==digest
        session.bind_policy(backend,metadata,saved,cfg,mode='train')
        assert not backend.deterministic
    replay=ReplayBuffer.load(saved/'replay.pt')
    assert len({t.planning_state.ids.episode_id for t in replay.learnable_transitions()})==2
    sources=json.loads((saved.parent/'training_sources.json').read_text())
    assert len(sources['trial_ids'])==2
    session.bind_policy(backend,metadata,saved,cfg,mode='eval')
    assert backend.deterministic


def test_actor_switch_waits_for_new_request_and_never_mutates_pending():
    import torch
    from pi05_rlt_rtc.learning.actor import QueueAwareRLTActor
    from pi05_rlt_rtc.tools.cli import _toy_request
    cfg,toy,backend,metadata=backend_double()
    backend.actor=QueueAwareRLTActor(toy.spec)
    backend.critical_only=True
    obs=toy.env.reset()
    pending=np.full((2,8),.13,np.float32);original=pending.copy()
    req=_toy_request(toy)
    assert backend.plan(req,obs,pending,req.rtc_context).source=='reference_teacher'
    obs=dataclasses.replace(obs,sensor_metadata={'rlt_active':True})
    backend.deterministic=False
    result=backend.plan(req,obs,pending,req.rtc_context)
    assert result.source=='actor' and np.max(np.abs(result.submitted))<=1
    np.testing.assert_array_equal(pending,original)
    assert backend.plan(dataclasses.replace(req,request_id=-1),obs,pending,req.rtc_context).source=='reference_teacher'


def test_planner_must_drain_before_updates():
    from pi05_rlt_rtc.runtime.controller import WallClockController
    from pi05_rlt_rtc.contracts.errors import ContractError
    fake=SimpleNamespace(_worker=SimpleNamespace(stop=lambda **kw:None,is_alive=lambda:True))
    with pytest.raises(ContractError,match='drain'):
        WallClockController.drain_planner(fake,timeout=0)


def test_late_handoff_without_any_new_plan_is_not_learning_data(episode_setup,tmp_path):
    cfg,toy,backend,metadata=episode_setup
    out=tmp_path/'late'
    session.run_episode(cfg,backend,metadata,None,None,Keys({28:['b'],29:['c']}),out,seed=1)
    annotation=json.loads((out/'label/annotation.json').read_text())
    assert not annotation['eligible']
    assert annotation['reason']=='no_learnable_critical_transition'


def test_resume_rejects_changed_learning_settings(episode_setup, tmp_path, monkeypatch):
    from pi05_rlt_rtc.contracts.errors import ContractError
    cfg, toy, backend, metadata = episode_setup
    root = tmp_path / 'session'
    session.run_episode(cfg, backend, metadata, None, None, Keys({5:['b'],30:['c']}),
                        root/'episodes'/'000001', seed=1)
    saved = session.update_round(cfg, root, seed=1, device='cpu', mock=True)
    assert saved is not None
    session.run_episode(cfg, backend, metadata, None, None, Keys({5:['b'],30:['a']}),
                        root/'episodes'/'000002', seed=2)
    changed = dataclasses.replace(cfg, raw={**cfg.raw,
        'training': {**cfg.raw['training'], 'lambda_bc': 2.5}})
    monkeypatch.setattr(training, 'config_from_args', lambda args: changed)
    with pytest.raises(ContractError, match='resume training settings changed'):
        session.update_round(changed, root, seed=1, device='cpu', mock=True)
