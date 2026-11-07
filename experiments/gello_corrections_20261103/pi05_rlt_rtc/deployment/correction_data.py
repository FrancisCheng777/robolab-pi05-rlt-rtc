"""Human correction actions are demonstrations, never invented RL transitions."""
import numpy as np

class TriggerReleased(ValueError):
    pass


class RelativeGello:
    def __init__(self, robot_q, leader_q, *, hz=15., speed=.15, lead=.05):
        self.robot_origin=np.asarray(robot_q,dtype=float).copy()
        self.leader_origin=np.asarray(leader_q,dtype=float).copy()
        self.reference=self.robot_origin.copy()
        self.hz,self.speed,self.lead=hz,speed,lead
    def action(self, leader_q, width, measured_q):
        leader=np.asarray(leader_q,dtype=float)
        measured=np.asarray(measured_q,dtype=float)
        if any(a.shape!=(7,) or not np.isfinite(a).all() for a in
               (leader,measured,self.robot_origin,self.leader_origin)):
            raise ValueError('invalid robot/GELLO joint state')
        if not np.isfinite(width) or not 0<=width<=.005:
            raise TriggerReleased('请持续按住 GELLO 扳机；已停止示范，未发送张爪指令')
        desired=self.robot_origin+(leader-self.leader_origin)
        limited=np.clip(desired,self.reference-self.speed/self.hz,self.reference+self.speed/self.hz)
        limited=np.clip(limited,measured-self.lead,measured+self.lead)
        action=np.r_[(limited-self.reference)/.05,-1.].astype(np.float32)
        if not np.isfinite(action).all() or np.any(np.abs(action)>1):
            raise ValueError('GELLO reference discontinuity exceeds action contract')
        self.reference=limited
        return action


def demo_windows(frames, contract, *, hz):
    """Hindsight queue reconstruction for BC ONLY; never construct a Q transition."""
    D,C,H=contract.delay,contract.chunk,contract.horizon
    if H!=D+C or D!=C:
        raise ValueError('correction importer requires the validated H/C/D=10/5/5 layout')
    for i in range(0,len(frames)-D-C+1,C):
        rows=frames[i:i+D+C]
        ticks=[r['observation']['tick'] for r in rows]
        stamps=np.array([r['observation']['captured_monotonic'] for r in rows])
        if ticks!=list(range(ticks[0],ticks[0]+H)) or np.any(np.diff(stamps)<=0) or np.any(np.abs(np.diff(stamps)-1/hz)>.5/hz):
            continue
        actions=np.asarray([r['audit']['normalized_action'] for r in rows],dtype=np.float32)
        if actions.shape!=(H,8) or not np.isfinite(actions).all() or np.any(np.abs(actions)>1):
            raise ValueError('invalid recorded human actions')
        if any(r['source']!='human_gello' or r['audit'].get('bridge_acknowledged') is not True for r in rows):
            raise ValueError('human demonstration contains an unacknowledged/nonhuman action')
        yield i,actions[:D].copy(),actions[D:D+C].copy()
