"""CPU-only owner of the accepted bridge; never imports torch or camera SDKs.

The policy supplies one delta at 15 Hz. Its absolute target is refreshed at the
accepted teleop rate. Home/open is delegated unchanged to CollectionSession.
"""
import argparse
import faulthandler
import json
import queue
import sys
import signal
import threading
import time
from pathlib import Path

import numpy as np
from pi05_rlt_rtc.adapters.robolab import activate
from pi05_rlt_rtc.contracts.physical import JOINT_DELTA


class ControlOwner:
    def __init__(self, bridge, lifecycle, *, action_timeout=.25,
                 joint_command_mode='measured_delta', reference_lead_rad=.05):
        if joint_command_mode not in ('measured_delta', 'bounded_reference'):
            raise ValueError('unknown joint command mode')
        if not np.isfinite(reference_lead_rad) or not 0 < reference_lead_rad <= .05:
            raise ValueError('reference lead must be in (0, .05] rad')
        self.bridge, self.lifecycle = bridge, lifecycle
        self.joint_command_mode = joint_command_mode
        self.reference_lead_rad = reference_lead_rad
        self.tracking = False
        self.target = self.width = None
        self.last_action = 0.
        self.action_timeout = action_timeout
        self._home_pending = False
        self._home_retry_started = None

    def ready(self, state):
        return bool(not self.tracking and not self._home_pending and self.lifecycle.phase == 'standby' and self.lifecycle.held(state))

    def _bound_reference(self, target, state):
        from pi05_rlt_rtc.adapters.policy_reference import bound_joint_reference
        return bound_joint_reference(target, state['q'],
            tracking_limits=state['calibration']['max_tracking_error'],
            lead_limit=self.reference_lead_rad)

    def action(self, action):
        state = self.bridge.state()
        if not self.tracking and not self.ready(state):
            raise RuntimeError('policy action requires completed hold/home')
        a = np.asarray(action, dtype=float)
        if a.shape != (8,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1):
            raise ValueError('action must be finite [-1,1]^8')
        c = state['calibration']
        measured = np.asarray(state['q'], dtype=float)
        if measured.shape != (7,) or not np.isfinite(measured).all():
            raise ValueError('invalid measured joint state')
        anchor = measured
        if self.joint_command_mode == 'bounded_reference' and self.tracking:
            anchor = self.target
        requested = anchor + JOINT_DELTA*a[:7]
        target = requested.copy()
        if self.joint_command_mode == 'bounded_reference':
            # Realized-motion labels prescribe per-tick displacement. Retain
            # untracked reference motion once per NEW action, never per refresh.
            # Anti-windup: discard everything beyond the accepted lead bound.
            target = self._bound_reference(target, state)
        if np.any(target < c['joint_lower']) or np.any(target > c['joint_upper']):
            raise ValueError('policy target violates accepted joint limits')
        self.target, self.width = target, float((a[7]+1)*.5*c['gripper_max_width'])
        self.last_action = time.monotonic()
        self.tracking = True
        ack = self.bridge.command(state, self.target, self.width)
        return {'measured_q': state['q'], 'q_target': target.tolist(),
                'gripper_width_target': self.width, 'normalized_action': a.tolist(),
                'joint_command_mode': self.joint_command_mode,
                'reference_lead_rad': (target-measured).tolist(),
                'reference_limited': bool(np.any(np.abs(target-requested) > 1e-10)),
                'state_sequence': state['state_sequence'], 'bridge_reply': ack,
                'bridge_acknowledged': True}

    def hold(self):
        self._home_pending = False
        self._home_retry_started = None
        self.tracking = False
        self.target = self.width = None
        self.lifecycle._settle(self.bridge.state(), 'standby')

    def home(self):
        state = self.bridge.state()
        if not self.ready(state):
            raise RuntimeError('home requires completed hold')
        self._home_retry_started = None
        self.lifecycle._start_home(state, 'standby')

    def tick(self):
        state = self.bridge.state()
        if self.tracking:
            if time.monotonic()-self.last_action > self.action_timeout:
                raise TimeoutError('policy action heartbeat expired')
            # Re-send exactly the latched target, never integrate delta again.
            if self.joint_command_mode == 'bounded_reference':
                # External displacement must not defeat the lead bound between
                # policy ticks. This only removes excess; it adds no motion.
                self.target = self._bound_reference(self.target, state)
            self.bridge.command(state, self.target, self.width)
        elif self._home_pending and self.lifecycle.phase == 'standby' and self.lifecycle.held(state):
            if time.monotonic()-self._home_retry_started > 10:
                raise TimeoutError('home resume could not obtain stable hold within 10 seconds')
            self._home_pending = False
            self.lifecycle._start_home(state, 'standby')
        elif self.lifecycle.phase in ('init', 'standby'):
            self.bridge.hold(state)
            if self.lifecycle.held(state):
                self.lifecycle.phase = 'standby'
        else:
            try:
                self.lifecycle.tick([])
            except RuntimeError as exc:
                # Bridge validates this condition BEFORE enqueuing the command.
                # Do not retry timeouts or commands whose acceptance is uncertain.
                if (self.lifecycle.phase not in ('homing','opening_gripper') or
                    str(exc) != 'wait until arm and gripper are held before resuming'):
                    raise
                rejected_sequence = self.bridge.seq
                fresh = self.bridge.state()
                if fresh.get('last_sequence') != rejected_sequence-1:
                    raise
                self.bridge.seq = rejected_sequence-1
                if self._home_retry_started is None:
                    self._home_retry_started = time.monotonic()
                if time.monotonic()-self._home_retry_started > 10:
                    raise TimeoutError('home resume could not obtain stable hold within 10 seconds') from exc
                self.lifecycle._settle(fresh, 'standby')
                self._home_pending = True
                print(json.dumps({'event':'home_waiting_for_hold','reason':str(exc)}),
                      file=sys.stderr,flush=True)

    def status(self):
        state = self.bridge.state()
        return {'state': state, 'ready': self.ready(state),
                'phase': 'tracking' if self.tracking else self.lifecycle.phase}


def serve(args):
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
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
    operation = 'initialization'
    request = None
    try:
        lifecycle = CollectionSession(bridge, None, None, config, None, 'rlt-control',
                                     mock=args.mock, announce=lambda *a, **k: None)
        owner = ControlOwner(bridge, lifecycle, joint_command_mode=args.joint_command_mode,
                             reference_lead_rad=args.reference_lead_rad)
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
                operation = op
                if op == 'close':
                    break
                if op not in ('status', 'action', 'hold', 'home'):
                    raise ValueError('unknown control operation')
                result = owner.action(request['action']) if op == 'action' else getattr(owner, op)()
                print(json.dumps({'ok': True, 'request_id': request.get('request_id'), 'result': result}), flush=True)
            operation = 'heartbeat'
            owner.tick()
            elapsed = time.monotonic()-started
            if elapsed > .05:
                print(json.dumps({'event': 'slow_control_cycle', 'elapsed_ms': elapsed*1000,
                                  'operation': operation}), file=sys.stderr, flush=True)
            time.sleep(max(0., period-(time.monotonic()-started)))
    except BaseException as exc:
        detail = f'{operation}: {type(exc).__name__}: {exc}'
        print(json.dumps({'event': 'control_fault', 'operation': operation,
                          'error': detail}), file=sys.stderr, flush=True)
        print(json.dumps({'ok': False, 'error': detail}), flush=True)
        raise
    finally:
        primary_error = sys.exc_info()[1]
        try:
            bridge.close()
        except Exception as exc:
            print(f'bridge cleanup failed: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
            if primary_error is None:
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', required=True)
    parser.add_argument('--collection-config', required=True)
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--joint-command-mode', choices=('measured_delta', 'bounded_reference'), default='measured_delta')
    parser.add_argument('--reference-lead-rad', type=float, default=.05)
    args = parser.parse_args()
    if not args.mock and not args.execute:
        parser.error('real bridge ownership requires --execute')
    serve(args)
