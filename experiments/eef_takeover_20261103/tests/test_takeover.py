import numpy as np
from pi05_rlt_rtc.deployment.takeover import PoseGate

def config():
    return dict(mode='auto',frame='panda_base_to_desk_ee',center=np.eye(4).tolist(),
                flange_to_ee=np.eye(4).tolist(),position_tolerance_m=[.01]*3,
                rotation_tolerance_deg=5.,max_joint_speed=.05,dwell_s=.2,
                max_sample_gap_s=.15,max_state_age_s=.05,max_gripper_age_s=.5,
                max_gripper_width_m=.06)

def sample(t, **kw):
    return dict(pose=np.eye(4).tolist(),dq=[0.]*7,stamp=t,
                gripper_stamp=t,gripper_width=.02,**kw)

def test_requires_fresh_stable_armed_dwell():
    gate=PoseGate(config())
    assert not gate.update(sample(1),now=1,armed=True)['trigger']
    assert not gate.update(sample(1.1),now=1.1,armed=True)['trigger']
    assert gate.update(sample(1.21),now=1.21,armed=True)['trigger']

import json
from pathlib import Path
import pytest
from pi05_rlt_rtc.deployment.takeover import panda_ee, TakeoverProtocol, observation_sample
from pi05_rlt_rtc.deployment.episode import EpisodeSignals


def test_fk_matches_all_live_robot_transforms():
    rows=json.loads((Path(__file__).parents[1]/'calibration.json').read_text())
    for r in rows:
        tool=np.array(r['F_T_EE']).reshape(4,4,order='F')
        expected=np.array(r['O_T_EE']).reshape(4,4,order='F')
        np.testing.assert_allclose(panda_ee(r['q'],tool),expected,atol=1e-8)


@pytest.mark.parametrize('failure',['unarmed','open','stale','duplicate','far','rotated','fast','nan','gap'])
def test_invalid_conditions_reset_dwell(failure):
    g=PoseGate(config())
    g.update(sample(1),now=1,armed=True)
    s=sample(1.1);now=1.1;armed=True
    if failure=='unarmed':armed=False
    if failure=='open':s['gripper_width']=.082
    if failure=='stale':now=1.2
    if failure=='duplicate':s=sample(1)
    if failure=='far':s['pose'][0][3]=.011
    if failure=='rotated':s['pose']=np.diag([-1,-1,1,1]).tolist()
    if failure=='fast':s['dq'][0]=.051
    if failure=='nan':s['dq'][0]=float('nan')
    if failure=='gap':s=sample(1.3);now=1.3
    assert not g.update(s,now=now,armed=armed)['trigger']
    t=max(now,1.3)+.1
    assert not g.update(sample(t),now=t,armed=True)['trigger']


def test_latches_once_and_new_episode_resets():
    g=PoseGate(config())
    for t in (1,1.1,1.21):r=g.update(sample(t),now=t,armed=True)
    assert r['trigger']
    assert not g.update(sample(1.3),now=1.3,armed=True)['trigger']
    assert not PoseGate(config()).update(sample(1.4),now=1.4,armed=True)['trigger']


def test_shadow_accepts_open_gripper_but_never_switches_actor():
    c=config();c['mode']='shadow';p=TakeoverProtocol(c);sig=EpisodeSignals()
    for t in (1,1.1,1.21):
        s=sample(t);s['gripper_width']=.085
        result=p.poll(sig,['b'],int(t*100),s,t)
    assert result['trigger'] and sig.critical_start_tick is None


@pytest.mark.parametrize('key',['q','x','e','c','a'])
def test_stop_or_terminal_wins_on_trigger_tick(key):
    p=TakeoverProtocol(config());sig=EpisodeSignals()
    p.poll(sig,['g'],1,sample(1),1)
    p.poll(sig,[],2,sample(1.1),1.1)
    p.poll(sig,[key],3,sample(1.21),1.21)
    assert sig.outcome is not None and sig.critical_start_tick is None


def test_auto_sets_original_critical_tick_only_after_arming():
    p=TakeoverProtocol(config());sig=EpisodeSignals()
    p.poll(sig,['b'],1,sample(1),1)
    assert sig.critical_start_tick is None
    p.poll(sig,['g'],2,sample(1.1),1.1)
    p.poll(sig,[],3,sample(1.21),1.21)
    p.poll(sig,[],4,sample(1.32),1.32)
    assert sig.critical_start_tick == 4


@pytest.mark.parametrize('field,value',[('dwell_s',0),('max_joint_speed',float('nan')),
                                        ('frame','plug_tip'),('center',np.zeros((4,4)))])
def test_invalid_configuration_rejected(field,value):
    c=config();c[field]=value
    with pytest.raises(ValueError):PoseGate(c)


def test_tool_change_rejected():
    with pytest.raises(ValueError,match='Desk tool differs'):
        observation_sample({'calibration':{'expected_desk_tool':{'transformation':
            np.diag([1,1,1,2]).flatten(order='F').tolist()}}},config())
