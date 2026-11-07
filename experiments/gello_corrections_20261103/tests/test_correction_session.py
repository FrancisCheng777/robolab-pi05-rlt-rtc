"""Exercise the full handover state machine with explicit CPU/device doubles."""
import dataclasses
import json
import threading
from collections import deque
from types import SimpleNamespace
import numpy as np
import pytest
from tests.test_review_controller import manual_runtime
from tests.test_takeover import config,sample
from tests.test_correction_training import demo
from pi05_rlt_rtc.runtime.scheduler import StepOutcome
from pi05_rlt_rtc.deployment import corrections,session_env


@pytest.mark.parametrize('confirmed,release',[(True,False),(False,False),(True,True)])
def test_teacher_to_gello_handover_and_recording_admission(tmp_path,monkeypatch,manual_runtime,confirmed,release):
    cfg,backend,metadata,_=demo(tmp_path)
    cfg=dataclasses.replace(cfg,raw={**cfg.raw,'session':{**cfg.raw['session'],
        'correction_collection':True,'takeover_gate':config()}})
    monkeypatch.setattr(corrections,'time',manual_runtime)
    class Leader:
        calls=0
        def read(self):
            self.calls+=1
            return np.zeros(7),(.085 if release and self.calls>=15 else 0.)
    class Owner:
        q=np.zeros(7);interrupted=threading.Event();correction_leader=Leader();actions=[]
        def rpc(self,op,**kw):
            if op=='status':return {'ready':True,'state':{'q':self.q.tolist()}}
            if op=='action':
                a=kw['action'];self.q=self.q+.05*np.array(a[:7]);self.actions.append(a)
                return dict(q_target=self.q.tolist(),normalized_action=a,bridge_acknowledged=True)
        def arm_episode(self):self.interrupted.clear()
        def hold(self):return {'ready':True}
    owner=Owner()
    class Keys:
        count=0;lock=threading.RLock();pending=deque();error=None
        def read(self):
            i=self.count;self.count+=1
            return ['g'] if confirmed and i==2 else ['c'] if i==50 else []
    from pi05_rlt_rtc.evaluation.toy_env import ToyDelayedControlEnv
    template=ToyDelayedControlEnv().reset()
    class Env:
        def __init__(self,*args):self._tick=0;self.command_log=[];self.stop_request_acknowledged=False
        def observe(self):
            return dataclasses.replace(template,env_id='franka',episode_id='trial',tick=self._tick,
                captured_monotonic=manual_runtime.now,source_timestamps={'robot':manual_runtime.now},
                states=np.r_[owner.q,.013,.013].astype(np.float32),
                sensor_metadata={'rlt_active':False,'takeover_sample':sample(manual_runtime.now)})
        def reset(self,**kw):return self.observe()
        def step(self,a):
            self.command_log.append(owner.rpc('action',action=a.tolist()));self._tick+=1
            return StepOutcome(self.observe(),0.,False,False,{})
        def stop(self):owner.hold();self.stop_request_acknowledged=True
    monkeypatch.setattr(session_env,'SessionEnv',Env)
    out=tmp_path/'trial'
    sig=corrections.run_correction_episode(cfg,backend,metadata,owner,None,Keys(),out,seed=1)
    m=json.loads((out/'correction.json').read_text())
    assert m['complete'] and not m['rl_eligible']
    assert m['grasp_confirmed'] is confirmed
    assert m['eligible_bc'] is (confirmed and not release)
    if confirmed:
        frames=[json.loads(x) for x in (out/'human_frames.jsonl').read_text().splitlines()]
        assert frames
        np.testing.assert_array_equal(frames[0]['audit']['normalized_action'][:7],0)
        assert all(x['audit']['normalized_action'][7]==-1 for x in frames)
        assert m['handover']['observation_tick']>m['gate_trigger_tick']
    else:
        assert not (out/'human_frames.jsonl').exists()
    if release:assert sig.reason=='gello_trigger_released'
