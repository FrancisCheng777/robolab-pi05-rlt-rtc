"""CPU-only owner of the accepted bridge; never imports torch or camera SDKs.

The policy supplies one delta at 15 Hz. Its absolute target is refreshed at the
accepted teleop rate. Home/open is delegated unchanged to CollectionSession.
"""
import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
from pi05_rlt_rtc.adapters.robolab import activate
from pi05_rlt_rtc.contracts.physical import JOINT_DELTA


class ControlOwner:
    def __init__(self, bridge, lifecycle, *, action_timeout=.25):
        self.bridge, self.lifecycle = bridge, lifecycle
        self.tracking = False
        self.target = self.width = None
        self.last_action = 0.
        self.action_timeout = action_timeout

    def ready(self, state):
        return bool(not self.tracking and self.lifecycle.phase == 'standby' and self.lifecycle.held(state))

    def action(self, action):
        state = self.bridge.state()
        if not self.tracking and not self.ready(state):
            raise RuntimeError('policy action requires completed hold/home')
        a = np.asarray(action, dtype=float)
        if a.shape != (8,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1):
            raise ValueError('action must be finite [-1,1]^8')
        c = state['calibration']
        target = np.asarray(state['q']) + JOINT_DELTA*a[:7]
        if np.any(target < c['joint_lower']) or np.any(target > c['joint_upper']):
            raise ValueError('policy target violates accepted joint limits')
        self.target, self.width = target, float((a[7]+1)*.5*c['gripper_max_width'])
        self.last_action = time.monotonic()
        self.tracking = True
        ack = self.bridge.command(state, self.target, self.width)
        return {'measured_q': state['q'], 'q_target': target.tolist(),
                'gripper_width_target': self.width, 'normalized_action': a.tolist(),
                'state_sequence': state['state_sequence'], 'bridge_reply': ack,
                'bridge_acknowledged': True}

    def hold(self):
        self.tracking = False
        self.lifecycle._settle(self.bridge.state(), 'standby')

    def home(self):
        state = self.bridge.state()
        if not self.ready(state):
            raise RuntimeError('home requires completed hold')
        self.lifecycle._start_home(state, 'standby')

    def tick(self):
        state = self.bridge.state()
        if self.tracking:
            if time.monotonic()-self.last_action > self.action_timeout:
                raise TimeoutError('policy action heartbeat expired')
            # Re-send exactly the latched target, never integrate delta again.
            self.bridge.command(state, self.target, self.width)
        elif self.lifecycle.phase in ('init', 'standby'):
            self.bridge.hold(state)
            if self.lifecycle.held(state):
                self.lifecycle.phase = 'standby'
        else:
            self.lifecycle.tick([])

    def status(self):
        state = self.bridge.state()
        return {'state': state, 'ready': self.ready(state),
                'phase': 'tracking' if self.tracking else self.lifecycle.phase}


def serve(args):
    activate()
    from pi05_demo.transport import Bridge
    from pi05_demo.session import CollectionSession
    config = json.loads(Path(args.collection_config).read_text())
    config['recording_enabled'] = False
    bridge = Bridge(args.socket, mock=args.mock)
    requests = queue.Queue()
    def reader():
        try:
            for line in sys.stdin:
                requests.put(json.loads(line))
        finally:
            requests.put({'op': 'close'})
    try:
        lifecycle = CollectionSession(bridge, None, None, config, None, 'rlt-control',
                                     mock=args.mock, announce=lambda *a, **k: None)
        owner = ControlOwner(bridge, lifecycle)
        threading.Thread(target=reader, daemon=True).start()
        period = 1/config['teleop_hz']
        while True:
            started = time.monotonic()
            try:
                request = requests.get_nowait()
            except queue.Empty:
                request = None
            if request:
                op = request['op']
                if op == 'close':
                    break
                if op not in ('status', 'action', 'hold', 'home'):
                    raise ValueError('unknown control operation')
                result = owner.action(request['action']) if op == 'action' else getattr(owner, op)()
                print(json.dumps({'ok': True, 'result': result}), flush=True)
            owner.tick()
            time.sleep(max(0., period-(time.monotonic()-started)))
    except BaseException as exc:
        print(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}), flush=True)
        raise
    finally:
        bridge.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', required=True)
    parser.add_argument('--collection-config', required=True)
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.mock and not args.execute:
        parser.error('real bridge ownership requires --execute')
    serve(args)
