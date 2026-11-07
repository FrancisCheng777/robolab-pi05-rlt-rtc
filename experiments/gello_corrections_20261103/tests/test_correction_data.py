import numpy as np
import pytest
from pi05_rlt_rtc.deployment.correction_data import RelativeGello


def test_full_trigger_handover_preserves_closed_gripper_without_joint_jump():
    robot=np.array([.2,-.2,0,-2,0,2,.5]);leader=np.ones(7)
    mapper=RelativeGello(robot,leader)
    a=mapper.action(leader,0.,robot)
    np.testing.assert_array_equal(a[:7],0.)
    assert a[7]==-1.


def test_relative_motion_rate_limit_and_anti_windup():
    mapper=RelativeGello(np.zeros(7),np.ones(7))
    for _ in range(30):
        a=mapper.action(np.ones(7)+.5,0.,np.zeros(7))
        assert np.max(np.abs(a[:7]))<=.2+1e-8
        assert np.max(np.abs(mapper.reference))<=.05+1e-8
    a=mapper.action(np.ones(7)-.5,0.,np.zeros(7))
    assert np.all(a[:7]<0)


def test_releasing_trigger_holds_instead_of_opening_plug():
    mapper=RelativeGello(np.zeros(7),np.zeros(7))
    with pytest.raises(ValueError,match='扳机'):
        mapper.action(np.ones(7),.04,np.zeros(7))

from pi05_rlt_rtc.deployment.correction_data import demo_windows
from pi05_rlt_rtc.contracts.spaces import HorizonContract

def rows(n=20):
    return [dict(source='human_gello',observation=dict(tick=200+i,captured_monotonic=10+i/15),
                 audit=dict(normalized_action=[i/100]*7+[-1.],bridge_acknowledged=True)) for i in range(n)]

def test_bc_target_is_after_delay_not_same_tick_and_never_overlaps_prefix():
    windows=list(demo_windows(rows(),HorizonContract(10,5,5),hz=15))
    assert len(windows)==3
    i,prefix,target=windows[0]
    np.testing.assert_allclose(prefix[:,0],np.arange(5)/100)
    np.testing.assert_allclose(target[:,0],np.arange(5,10)/100)

def test_temporal_gaps_are_not_treated_as_contiguous_demonstrations():
    data=rows(10);data[5]['observation']['captured_monotonic']+=.2
    assert list(demo_windows(data,HorizonContract(10,5,5),hz=15))==[]

def test_unacknowledged_actions_are_not_supervision():
    data=rows(10);data[5]['audit']['bridge_acknowledged']=False
    with pytest.raises(ValueError,match='unacknowledged'):
        list(demo_windows(data,HorizonContract(10,5,5),hz=15))
