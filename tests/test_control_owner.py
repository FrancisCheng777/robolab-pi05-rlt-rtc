from types import SimpleNamespace
import numpy as np
import pytest
from pi05_rlt_rtc.deployment.control_owner import ControlOwner

class Bridge:
    def __init__(self):
        self.q=np.zeros(7); self.commands=[]
    def state(self):
        return {'q':self.q.tolist(),'state_sequence':1,
                'calibration':{'joint_lower':[-2]*7,'joint_upper':[2]*7,'gripper_max_width':.085}}
    def command(self, state, target, width):
        self.commands.append(target.copy()); return {'ok':True}


def test_refresh_does_not_reapply_delta_to_moving_feedback():
    bridge=Bridge()
    life=SimpleNamespace(phase='standby',held=lambda s:True)
    owner=ControlOwner(bridge,life)
    owner.action([.5]*7+[0])
    bridge.q[:]=.01
    owner.tick(); owner.tick()
    np.testing.assert_allclose(bridge.commands,np.full((3,7),.025))
    owner.action([.5]*7+[0])
    np.testing.assert_allclose(bridge.commands[-1],.035)


def test_stale_action_faults_instead_of_refreshing_forever():
    owner=ControlOwner(Bridge(),SimpleNamespace(phase='standby',held=lambda s:True))
    owner.action([0]*8);owner.last_action=0
    with pytest.raises(TimeoutError):owner.tick()


def test_action_and_home_require_confirmed_hold():
    owner=ControlOwner(Bridge(),SimpleNamespace(phase='homing',held=lambda s:False))
    with pytest.raises(RuntimeError):owner.action([0]*8)
    with pytest.raises(RuntimeError):owner.home()
