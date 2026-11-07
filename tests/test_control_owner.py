from types import SimpleNamespace
import numpy as np
import pytest
from pi05_rlt_rtc.deployment.control_owner import ControlOwner

class Bridge:
    def __init__(self):
        self.q=np.zeros(7); self.commands=[]
    def state(self):
        return {'q':self.q.tolist(),'state_sequence':1,
                'calibration':{'joint_lower':[-2]*7,'joint_upper':[2]*7,
                               'max_tracking_error':[.05]*7,'gripper_max_width':.085}}
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


def test_bounded_reference_preserves_displacement_but_never_winds_up():
    bridge = Bridge()
    life = SimpleNamespace(phase='standby', held=lambda s: True,
                           _settle=lambda *a: None)
    owner = ControlOwner(bridge, life, joint_command_mode='bounded_reference')
    # A stalled joint must not cause an unbounded remote target.
    for _ in range(100):
        audit = owner.action([.1]*7+[1])
        assert np.max(np.abs(bridge.commands[-1]-bridge.q)) <= .05+1e-12
    np.testing.assert_allclose(owner.target, .05)
    assert audit['reference_limited']
    bridge.q[:] = .01
    owner.tick(); owner.tick()
    np.testing.assert_allclose(owner.target, .05)  # refresh adds no delta
    owner.action([.1]*7+[1])
    np.testing.assert_allclose(owner.target, .055)
    # Saturation discarded excess; a reversal responds immediately.
    owner.action([-.1]*7+[1])
    np.testing.assert_allclose(owner.target, .05)
    owner.hold()
    owner.action([.1]*7+[1])
    np.testing.assert_allclose(owner.target, .015)  # no cross-episode residual
    bridge.q[:] = -.1
    owner.tick()
    np.testing.assert_allclose(owner.target, -.05)  # also bound between policy ticks


def test_bounded_reference_recovers_small_steps_in_deadzone_fixture():
    # Deterministic controller fixture, not a calibrated robot simulation.
    def run(mode):
        b = Bridge()
        owner = ControlOwner(b, SimpleNamespace(phase='standby', held=lambda s: True),
                             joint_command_mode=mode)
        for _ in range(60):
            owner.action([.1]*7+[0])
            error = owner.target-b.q
            b.q += np.sign(error)*np.minimum(np.maximum(np.abs(error)-.01, 0), .3/15)
        return b.q
    np.testing.assert_allclose(run('measured_delta'), 0)
    assert np.min(run('bounded_reference')) > .25


def test_bounded_reference_matches_measured_decode_under_perfect_tracking():
    b = Bridge()
    owner = ControlOwner(b, SimpleNamespace(phase='standby', held=lambda s: True),
                         joint_command_mode='bounded_reference')
    for value in (.1, -.2, .3, 0):
        expected = b.q+.05*value
        owner.action([value]*7+[0])
        np.testing.assert_allclose(owner.target, expected)
        b.q = owner.target.copy()


def test_bounded_reference_honors_calibration_and_joint_limits():
    b = Bridge()
    original = b.state
    def state():
        s = original(); s['calibration']['max_tracking_error'] = [.02]*7
        return s
    b.state = state
    owner = ControlOwner(b, SimpleNamespace(phase='standby', held=lambda s: True),
                         joint_command_mode='bounded_reference')
    for _ in range(10): owner.action([1]*7+[0])
    np.testing.assert_allclose(owner.target, .02)
    owner.tracking = False; b.q[:] = 1.99
    with pytest.raises(ValueError, match='joint limits'): owner.action([1]*7+[0])


@pytest.mark.parametrize('limit', [0, -.01, .051, float('nan')])
def test_reference_lead_cannot_exceed_accepted_bound(limit):
    with pytest.raises(ValueError, match='reference lead'):
        ControlOwner(Bridge(), None, joint_command_mode='bounded_reference', reference_lead_rad=limit)


def test_home_opens_stationary_gripper_then_waits_for_hold_before_arm_motion():
    b=Bridge();b.q[:]=.2
    b.hold=lambda state:None
    calls=[]
    life=SimpleNamespace(phase='standby',held=lambda s:True,
                         config={'open_gripper_after_home':True})
    life._start_home=lambda state,dest:(calls.append('home'),setattr(life,'phase','homing'))
    def settle(state,dest):life.phase='settling'
    life._settle=settle
    life.tick=lambda keys:None
    owner=ControlOwner(b,life)
    owner.home()
    assert life.phase=='opening_gripper'
    np.testing.assert_allclose(life.gripper_open_q,b.q)
    assert calls==[] and not owner.ready(b.state())
    owner.tick()
    assert calls==[]
    life.phase='settling'
    owner.tick()
    assert calls==[]  # arm must not start while the gripper/arm hold is pending
    life.phase='standby'
    owner.tick()
    assert calls==['home']
    assert life.config['open_gripper_after_home'] is False


def test_operator_hold_cancels_pending_home_during_opening():
    b=Bridge();b.hold=lambda state:None
    calls=[]
    life=SimpleNamespace(phase='standby',held=lambda s:True,config={})
    life._start_home=lambda *a:calls.append('home')
    life._settle=lambda *a:setattr(life,'phase','settling')
    life.tick=lambda keys:None
    owner=ControlOwner(b,life);owner.home();owner.hold()
    life.phase='standby';owner.tick()
    assert calls==[] and owner.ready(b.state())
