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
        'session':{**cfg.raw['session'],'warmup_episodes':1,'update_every_episodes':1,'min_replay_transitions':2,
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


def test_episode_report_keeps_primary_fault_when_hold_also_fails(episode_setup, tmp_path, monkeypatch):
    cfg, toy, backend, metadata = episode_setup
    original = session_env.SessionEnv
    class BrokenEnv(original):
        def step(self, action):
            raise RuntimeError('first state transport fault')

        def stop(self):
            raise TimeoutError('secondary hold timeout')
    monkeypatch.setattr(session_env, 'SessionEnv', BrokenEnv)
    out = tmp_path/'fault'
    with pytest.raises(RuntimeError, match='first state transport fault'):
        session.run_episode(cfg, backend, metadata, None, None, Keys({}), out, seed=1)
    report = json.loads((out/'controller_report.json').read_text())
    assert report['error'] == 'RuntimeError: first state transport fault'
    assert report['cleanup_errors'][0] == 'hold: TimeoutError: secondary hold timeout'
    assert report['stop_request_acknowledged'] is False
    assert json.loads((out/'label/annotation.json').read_text())['eligible'] is False


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
    for index,outcome in [(1,'c'),(2,'a'),(3,'a'),(4,'c'),(5,'c')]:
        session.run_episode(cfg,backend,metadata,None,None,Keys({5:['b'],30:[outcome]}),
                            root/'episodes'/f'{index:06d}',seed=index)
        saved=session.update_round(cfg,root,seed=1,device='cpu',mock=True)
        bundle=PolicyBundle.load(saved/'policy.pt')
        assert bundle.training_counts['rounds']==index
        assert bundle.training_counts['warmup_updates']==1
        assert bundle.training_counts['updates']==index
        assert bundle.training_counts['actor_updates']==(index+3)//4
        import torch
        state=torch.load(saved/'learner_state.pt',weights_only=True)
        assert {int(s['step']) for s in state['actor_optimizer']['state'].values()}=={1+(index+3)//4}
        assert {int(s['step']) for s in state['critic_optimizer']['state'].values()}=={index}
        if index in (2,3,4):
            assert bundle.actor_state_sha256==previous_digest
        elif index==5:
            assert bundle.actor_state_sha256!=previous_digest
        previous_digest=bundle.actor_state_sha256
        digest=bundle.actor_state_sha256
        assert session.update_round(cfg,root,seed=1,device='cpu',mock=True)==saved
        assert PolicyBundle.load(saved/'policy.pt').actor_state_sha256==digest
        session.bind_policy(backend,metadata,saved,cfg,mode='train')
        assert not backend.deterministic
    replay=ReplayBuffer.load(saved/'replay.pt')
    assert len({t.planning_state.ids.episode_id for t in replay.learnable_transitions()})==5
    sources=json.loads((saved.parent/'training_sources.json').read_text())
    assert len(sources['trial_ids'])==5
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


def test_training_waits_for_each_ten_new_eligible_episodes(episode_setup, tmp_path):
    cfg, _, backend, metadata = episode_setup
    cfg = dataclasses.replace(cfg, raw={**cfg.raw, 'session': {
        **cfg.raw['session'], 'warmup_episodes': 10,
        'update_every_episodes': 10, 'updates_per_episode': 2}})
    root = tmp_path/'batches'
    # An aborted trial must not advance the batch counter.
    session.run_episode(cfg, backend, metadata, None, None, Keys({20: ['e']}),
                        root/'episodes'/'000000', seed=0)
    previous = None
    for index in range(1, 21):
        session.run_episode(cfg, backend, metadata, None, None,
                            Keys({5: ['b'], 30: ['c' if index % 3 else 'a']}),
                            root/'episodes'/f'{index:06d}', seed=index)
        checkpoint = session.update_round(cfg, root, seed=1, device='cpu', mock=True)
        if index % 10:
            assert checkpoint == previous
        else:
            assert checkpoint is not None and checkpoint != previous
            bundle = PolicyBundle.load(checkpoint/'policy.pt')
            assert bundle.training_counts['updates'] == index * 2
            assert bundle.training_counts['warmup_updates'] == 1
            previous = checkpoint
            assert session.update_round(cfg, root, seed=1, device='cpu', mock=True) == checkpoint
        assert len(list((root/'rounds').glob('round-*'))) == index // 10


def test_unchanged_checkpoint_keeps_actor_and_does_not_announce_new_weights(episode_setup, tmp_path, capsys):
    cfg, _, backend, metadata = episode_setup
    root = tmp_path/'binding'
    session.run_episode(cfg, backend, metadata, None, None, Keys({5: ['b'], 30: ['c']}),
                        root/'episodes'/'000001', seed=1)
    checkpoint = session.update_round(cfg, root, seed=1, device='cpu', mock=True)
    cfg = dataclasses.replace(cfg, raw={**cfg.raw, 'model': {
        **cfg.raw['model'], 'device': 'cuda:0', 'actor_inference_device': 'cpu'}})
    session.bind_policy(backend, metadata, checkpoint, cfg, mode='train')
    original_actor = backend.actor
    assert next(original_actor.parameters()).device.type == 'cpu'
    capsys.readouterr()
    # A failed or successful trial below the next update gate returns this same path.
    session.bind_policy(backend, metadata, checkpoint, cfg, mode='train')
    assert backend.actor is original_actor
    assert '新权重已就绪' not in capsys.readouterr().out
    assert metadata['policy_version_id'] == backend.policy_version_id
    # Evaluation/exploration settings still change without rebuilding the actor.
    session.bind_policy(backend, metadata, checkpoint, cfg, mode='eval')
    assert backend.actor is original_actor and backend.deterministic


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
