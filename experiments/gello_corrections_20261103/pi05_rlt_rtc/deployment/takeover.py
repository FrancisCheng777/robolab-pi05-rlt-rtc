"""One-shot EEF gate. All transforms are base->Desk EE, never plug-tip poses."""
import math
import numpy as np


def transform(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError('pose must be finite 4x4')
    if (not np.allclose(a[3], [0, 0, 0, 1], atol=1e-8) or
        not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-6) or
        abs(np.linalg.det(a[:3, :3])-1) > 1e-6):
        raise ValueError('pose must be a rigid transform')
    return a


def panda_ee(q, flange_to_ee):
    q = np.asarray(q, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError('expected seven finite Panda joints')
    result = np.eye(4)
    for theta, a, alpha, d in zip(q, [0,0,0,.0825,-.0825,0,.088],
            [0,-math.pi/2,math.pi/2,math.pi/2,-math.pi/2,math.pi/2,math.pi/2],
            [.333,0,.316,0,.384,0,0]):
        c,s,ca,sa = math.cos(theta),math.sin(theta),math.cos(alpha),math.sin(alpha)
        result = result @ np.array([[c,-s,0,a],[s*ca,c*ca,-sa,-d*sa],
                                   [s*sa,c*sa,ca,d*ca],[0,0,0,1]])
    flange = np.eye(4); flange[2,3] = .107
    return result @ flange @ transform(flange_to_ee)


def observation_sample(state, config):
    tool = np.asarray(state['calibration']['expected_desk_tool']['transformation']).reshape(4,4,order='F')
    if not np.allclose(tool, config['flange_to_ee'], atol=1e-7, rtol=0):
        raise ValueError('Desk tool differs from takeover calibration; recapture pose')
    return dict(pose=panda_ee(state['q'], config['flange_to_ee']).tolist(),
                dq=state['dq'], stamp=state['state_bridge_receive_s'],
                gripper_stamp=state['gripper_bridge_read_s'],gripper_width=state['gripper_width'])


class PoseGate:
    def __init__(self, config):
        self.config = dict(config)
        if config['mode'] not in ('shadow','auto') or config['frame'] != 'panda_base_to_desk_ee':
            raise ValueError('invalid takeover mode/frame')
        self.center = transform(config['center'])
        transform(config['flange_to_ee'])
        self.tolerance = np.asarray(config['position_tolerance_m'],dtype=float)
        if self.tolerance.shape != (3,) or not np.isfinite(self.tolerance).all() or np.any(self.tolerance <= 0):
            raise ValueError('invalid position tolerance')
        for k in ('rotation_tolerance_deg','max_joint_speed','dwell_s','max_sample_gap_s',
                  'max_state_age_s','max_gripper_age_s','max_gripper_width_m'):
            if not math.isfinite(config[k]) or config[k] <= 0:
                raise ValueError('invalid '+k)
        self.since = self.last = None
        self.latched = False

    def update(self, sample, *, now, armed):
        if self.latched:
            return dict(trigger=False,latched=True)
        c = self.config
        try:
            pose = transform(sample['pose'])
            dq = np.asarray(sample['dq'],dtype=float)
            stamp,gs,width = (float(sample[k]) for k in ('stamp','gripper_stamp','gripper_width'))
            if dq.shape != (7,) or not np.isfinite(dq).all() or not all(map(math.isfinite,[stamp,gs,width,now])):
                raise ValueError('invalid state')
            delta = pose[:3,3]-self.center[:3,3]
            angle = math.degrees(math.acos(float(np.clip((np.trace(self.center[:3,:3].T@pose[:3,:3])-1)/2,-1,1))))
            fresh = 0 <= now-stamp <= c['max_state_age_s']
            ordered = self.last is None or stamp > self.last
            gap = self.last is None or stamp-self.last <= c['max_sample_gap_s']
            geometry = bool(np.all(np.abs(delta)<=self.tolerance) and angle<=c['rotation_tolerance_deg'])
            slow = bool(np.max(np.abs(dq))<=c['max_joint_speed'])
            grip = 0 <= now-gs <= c['max_gripper_age_s'] and 0 <= width <= c['max_gripper_width_m']
            allowed = c['mode']=='shadow' or (armed and grip)
            eligible = bool(fresh and ordered and geometry and slow and allowed)
            if not eligible or not gap:
                self.since = None
            if eligible and self.since is None:
                self.since = stamp
            dwell = 0. if self.since is None else stamp-self.since
            self.last = stamp
            trigger = eligible and dwell >= c['dwell_s']
            self.latched = bool(trigger)
            return dict(trigger=bool(trigger),latched=self.latched,eligible=eligible,
                        position_error_m=delta.tolist(),rotation_error_deg=angle,
                        fresh=fresh,ordered=ordered,slow=slow,gripper_ok=bool(grip),armed=bool(armed),
                        dwell_s=dwell,pose=pose.tolist(),stamp=stamp,observed_at=now,
                        max_joint_speed=float(np.max(np.abs(dq))),
                        gripper_width_m=width,gripper_age_s=now-gs)
        except (ValueError,KeyError,TypeError):
            self.since = self.last = None
            return dict(trigger=False,latched=False,eligible=False,invalid_sample=True)


class TakeoverProtocol:
    def __init__(self, config):
        self.gate = PoseGate(config)
        self.armed = False
        self.confirmation_tick = None
        self.events = []

    def confirm_grasp(self, tick):
        if not self.armed:
            self.armed = True
            self.confirmation_tick = tick

    def poll(self, signals, keys, tick, sample, now):
        # Stop/terminal/timeout precedence is identical to the original protocol.
        signals.poll([k for k in keys if k not in ('g','b')], tick)
        if signals.outcome is not None or signals.critical_start_tick is not None:
            return None
        if 'g' in keys:
            self.confirm_grasp(tick)
        event = self.gate.update(sample,now=now,armed=self.armed)
        self.events.append(dict(tick=tick,**event))
        if event['trigger'] and self.gate.config['mode']=='auto':
            signals.poll(['b'],tick)
        return event
