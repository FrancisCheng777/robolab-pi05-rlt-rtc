"""CPU-only teacher/actor boundary doubles: never connect to a robot or live model."""
import copy
import dataclasses
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.deployment.rpc import ResidentTeacherActor
from pi05_rlt_rtc.learning.actor import QueueAwareRLTActor
from tests.test_platform_integration import backend_double
from pi05_rlt_rtc.tools.cli import _toy_request


def setup_actor():
    cfg,toy,backend,meta=backend_double()
    cfg=dataclasses.replace(cfg,raw={**cfg.raw,'model':{**cfg.raw['model'],'z_dim':16}})
    seen=[]
    def plan(request,*args):
        seen.append(request.policy_version_id)
        return backend.plan(request,*args)
    remote=ResidentTeacherActor(SimpleNamespace(metadata=meta,plan=plan),cfg)
    remote.actor=QueueAwareRLTActor(toy.spec)
    remote.policy_version_id='actor-test'
    obs=dataclasses.replace(toy.env.reset(),env_id='toy0',episode_id='toy0-ep1')
    req=dataclasses.replace(_toy_request(toy),policy_version_id='actor-test')
    return cfg,toy,backend,remote,obs,req,seen


def test_actor_runs_only_after_critical_and_remote_teacher_stays_frozen():
    cfg,toy,backend,client,obs,req,seen=setup_actor()
    result=client.plan(req,obs,np.zeros((2,8)),req.rtc_context)
    assert result.source=='reference_teacher'
    np.testing.assert_array_equal(result.submitted,result.bc_target_plan)
    active=dataclasses.replace(obs,sensor_metadata={'rlt_active':True})
    actor=client.plan(req,active,np.zeros((2,8)),req.rtc_context)
    assert actor.source=='actor'
    assert actor.planning_state.ids.policy_version_id=='actor-test'
    assert backend.actor is None
    assert seen==[backend.policy_version_id]*2
    assert np.isfinite(actor.submitted).all() and (abs(actor.submitted)<=1).all()
    # Exact parity with the existing local planner and identical frozen features.
    backend.actor=copy.deepcopy(client.actor);backend.policy_version_id='actor-test';backend.critical_only=True
    local=backend.plan(req,active,np.zeros((2,8)),req.rtc_context)
    np.testing.assert_array_equal(actor.submitted,local.submitted)
    np.testing.assert_array_equal(actor.bc_target_plan,local.bc_target_plan)
    np.testing.assert_array_equal(actor.planning_state.actor_state_vector(),local.planning_state.actor_state_vector())


def test_exploration_seeded_warm_start_teacher_and_mismatched_version_rejected():
    cfg,toy,backend,client,obs,req,_=setup_actor()
    obs=dataclasses.replace(obs,sensor_metadata={'rlt_active':True})
    args=(req,obs,np.zeros((2,8)),req.rtc_context)
    fixed=client.plan(*args).submitted
    client.deterministic=False
    first=client.plan(*args).submitted
    assert not np.array_equal(first,fixed)
    np.testing.assert_array_equal(first,client.plan(*args).submitted)
    warm=dataclasses.replace(req,request_id=-1)
    assert client.plan(warm,obs,np.zeros((2,8)),warm.rtc_context).source=='reference_teacher'
    with pytest.raises(ContractError,match='policy version'):
        client.plan(dataclasses.replace(req,policy_version_id='wrong'),obs,np.zeros((2,8)),req.rtc_context)


def test_no_missing_features_or_wrong_layout_allowed():
    cfg,toy,backend,client,obs,req,_=setup_actor()
    original=client.teacher.plan
    def missing(*args):
        result=original(*args);result.planning_state=None;return result
    client.teacher.plan=missing
    with pytest.raises(ContractError,match='planning features'):
        client.plan(req,obs,np.zeros((2,8)),req.rtc_context)


def test_teacher_collect_then_train_share_session_but_collect_never_updates(tmp_path,monkeypatch):
    import threading
    from collections import deque
    from pi05_rlt_rtc.deployment import session,session_env,bridge_preflight
    from pi05_rlt_rtc.adapters import cameras
    cfg,toy,backend,meta=backend_double()
    # CPU boundary doubles only; no bridge, camera, VLA or GPU opened.
    monkeypatch.setattr(session,'config_from_args',lambda _:cfg)
    monkeypatch.setattr(session,'make_backend',lambda *a:(backend,dict(meta)))
    monkeypatch.setattr(bridge_preflight,'probe_bridge',lambda *a:None)
    class Owner:
        def __init__(self,cfg):pass
        def close(self):pass
        def interrupt(self):pass
    class Keys:
        def __init__(self,**kwargs):self.lock=threading.Lock();self.pending=deque();self.error=None
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self):return ['q']
    monkeypatch.setattr(session_env,'OwnerClient',Owner)
    monkeypatch.setattr(session,'TerminalKeys',Keys)
    monkeypatch.setattr(cameras,'create_cameras',lambda _:SimpleNamespace(close=lambda:None))
    calls=[]
    monkeypatch.setattr(session,'update_round',lambda *a,**kw:calls.append(kw))
    args=SimpleNamespace(execute=True,annotator='CPU test',mode='train',policy=None,
        planner_url='test-only',resident_teacher=True,collect_only=True,out=str(tmp_path/'session'),seed=42,exploration=True)
    assert session._run_session(args)==0
    assert calls==[]
    args.collect_only=False
    assert session._run_session(args)==0
    assert len(calls)==1
