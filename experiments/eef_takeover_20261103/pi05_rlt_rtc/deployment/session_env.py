"""Persistent control-process client and per-episode camera environment."""
import contextlib
import json
import os
import select
import signal
import subprocess
import time
import threading
from pathlib import Path
import numpy as np
from pi05_rlt_rtc.adapters.cameras import validate_frames
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
from pi05_rlt_rtc.runtime.scheduler import StepOutcome


class OperatorInterrupt(Exception):
    """A keyboard stop has latched; no more policy commands are allowed."""


class OwnerClient:
    def __init__(self, cfg, *, mock=False):
        self._rpc_lock = threading.RLock()
        self.interrupted = threading.Event()
        self._failure = None
        self._reply_buffer = b''
        self._request_id = 0
        self._diagnostic_ready = False
        settings = cfg.raw['session']
        command = [settings.get('control_python', 'python3'), '-m',
                   'pi05_rlt_rtc.deployment.control_owner', '--socket', cfg.raw['franka']['socket'],
                   '--collection-config', str(cfg.asset(settings['collection_config'], 'collection config')),
                   '--joint-command-mode', cfg.raw['franka'].get('joint_command_mode', 'measured_delta'),
                   '--reference-lead-rad', str(cfg.raw['franka'].get('reference_lead_rad', .05)),
                   '--mock' if mock else '--execute']
        directory = Path.home()/'.cache/pi05-stage1-8000/control-owner'
        directory.mkdir(parents=True, exist_ok=True)
        self.log_path = directory/f'{time.time_ns()}.log'
        with self.log_path.open('a') as log:
            self.process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=log, bufsize=0)
        print(f'CPU control owner log: {self.log_path}', flush=True)
        try:
            state = self.wait_ready()["state"]
            expected = json.loads(cfg.asset(cfg.raw["franka"]["bridge_config"], "bridge config").read_text())
            if state["calibration"] != expected:
                raise ContractError("bridge calibration differs from accepted runtime profile")
        except BaseException:
            self.close()
            raise

    def interrupt(self):
        # Latch BEFORE waiting for an in-flight RPC; action() cannot overtake hold.
        self.interrupted.set()
        self.rpc('hold')

    def arm_episode(self):
        with self._rpc_lock:
            self.wait_ready()
            self.interrupted.clear()

    def rpc(self, op, **fields):
        with self._rpc_lock:
            if op == 'action' and self.interrupted.is_set():
                raise OperatorInterrupt('operator requested hold')
            return self._rpc(op, **fields)

    def _rpc(self, op, **fields):
        if self._failure is not None:
            raise ContractError(f'CPU control channel fault latched: {self._failure}')
        started = time.monotonic()
        self._request_id += 1
        try:
            if self.process.poll() is not None:
                raise ContractError('CPU control owner exited; restart session explicitly')
            payload = json.dumps({'op': op, 'request_id': self._request_id, **fields}, allow_nan=False)
            self.process.stdin.write((payload+'\n').encode())
            self.process.stdin.flush()
            reply = self._read_reply(started+2.)
            self._diagnostic_ready = True
            if not reply['ok']:
                raise ContractError(reply['error'])
            if reply.get('request_id') != self._request_id:
                raise ContractError('CPU control reply request ID mismatch')
            return reply['result']
        except Exception as exc:
            self._failure = (f'{op} request {self._request_id}, '
                             f'{1000*(time.monotonic()-started):.1f} ms: '
                             f'{type(exc).__name__}: {exc}; log={self.log_path}')
            self.interrupted.set()
            # Never send another command into a channel with uncertain ACKs.
            # EOF asks the owner to close; the bridge watchdog remains enabled.
            with contextlib.suppress(OSError):
                if self._diagnostic_ready and self.process.poll() is None:
                    self.process.send_signal(signal.SIGUSR1)
            with contextlib.suppress(OSError):
                self.process.stdin.close()
            raise ContractError(self._failure) from exc

    def _read_reply(self, deadline):
        # TextIO.readline after select can block forever on a partial JSON line,
        # or hide a second line in its private read-ahead buffer.
        while b'\n' not in self._reply_buffer:
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError('CPU control owner did not acknowledge a complete reply')
            chunk = os.read(self.process.stdout.fileno(), 16384)
            if not chunk:
                raise ContractError('CPU control owner disconnected')
            self._reply_buffer += chunk
            if len(self._reply_buffer) > 131072:
                raise ContractError('oversized CPU control reply')
        line, self._reply_buffer = self._reply_buffer.split(b'\n', 1)
        return json.loads(line)

    def wait_ready(self, timeout=12.):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            status = self.rpc('status')
            if status['ready']:
                return status
            time.sleep(.02)
        raise TimeoutError('control owner failed to enter confirmed hold')

    def hold(self):
        self.rpc('hold')
        return self.wait_ready()

    def close(self):
        if self.process.poll() is None:
            with contextlib.suppress(BrokenPipeError):
                self.process.stdin.close()  # EOF stops the bridge in child finally
            try:
                self.process.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=3.)


class SessionEnv:
    def __init__(self, cfg, owner, cameras, episode_id):
        self.cfg, self.owner, self.cameras, self.episode_id = cfg, owner, cameras, episode_id
        self._tick = 0
        self.rlt_active = False
        self.command_log = []
        self.stop_request_acknowledged = False

    def reset(self, *, seed=0):
        self.owner.wait_ready()
        return self.observe()

    def observe(self):
        status = self.owner.rpc('status')
        state = status['state']
        frames = self.cameras.capture()
        now = time.monotonic()
        settings = self.cfg.raw['franka']
        validate_frames(frames, now=now, max_age=settings['max_camera_age_s'], max_skew=settings['max_camera_skew_s'])
        if now-state['state_bridge_receive_s'] > settings['max_state_age_s']:
            raise ContractError('robot feedback expired during image capture')
        return ObservationSnapshot(env_id='franka', episode_id=self.episode_id, tick=self._tick,
            states=np.asarray([*state['q'], state['gripper_width']/2, state['gripper_width']/2], np.float32),
            images={k:f.rgb for k,f in frames.items()}, image_masks={'wrist': True},
            prompt=self.cfg.raw['prompt'], captured_monotonic=now,
            source_timestamps={'robot':state['state_bridge_receive_s'], **{k:f.captured_monotonic for k,f in frames.items()}},
            sensor_metadata={'rlt_active':self.rlt_active, **self.takeover_metadata(state), **{k:f.metadata for k,f in frames.items()}})

    def takeover_metadata(self, state):
        config = self.cfg.raw['session'].get('takeover_gate')
        if config is None:
            return {}
        from .takeover import observation_sample
        return {'takeover_sample': observation_sample(state, config)}

    def step(self, action):
        audit = self.owner.rpc('action', action=np.asarray(action).tolist())
        self.command_log.append({'tick':self._tick, **audit})
        self._tick += 1
        return StepOutcome(self.observe(), reward=0., terminated=False, truncated=False,
                           info={'reward_valid':False})

    def stop(self):
        self.stop_request_acknowledged = False
        self.owner.hold()
        self.stop_request_acknowledged = True

    def close(self):
        self.stop()
