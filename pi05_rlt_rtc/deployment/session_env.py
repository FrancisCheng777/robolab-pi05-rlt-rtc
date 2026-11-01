"""Persistent control-process client and per-episode camera environment."""
import contextlib
import json
import select
import subprocess
import time
from pathlib import Path
import numpy as np
from pi05_rlt_rtc.adapters.cameras import validate_frames
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
from pi05_rlt_rtc.runtime.scheduler import StepOutcome


class OwnerClient:
    def __init__(self, cfg, *, mock=False):
        settings = cfg.raw['session']
        command = [settings.get('control_python', 'python3'), '-m',
                   'pi05_rlt_rtc.deployment.control_owner', '--socket', cfg.raw['franka']['socket'],
                   '--collection-config', str(cfg.asset(settings['collection_config'], 'collection config')),
                   '--mock' if mock else '--execute']
        self.process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            state = self.wait_ready()["state"]
            expected = json.loads(cfg.asset(cfg.raw["franka"]["bridge_config"], "bridge config").read_text())
            if state["calibration"] != expected:
                raise ContractError("bridge calibration differs from accepted runtime profile")
        except BaseException:
            self.close()
            raise

    def rpc(self, op, **fields):
        if self.process.poll() is not None:
            raise ContractError('CPU control owner exited; restart session explicitly')
        self.process.stdin.write(json.dumps({'op': op, **fields}, allow_nan=False)+'\n')
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 2.)[0]:
            raise TimeoutError('CPU control owner did not acknowledge')
        line = self.process.stdout.readline()
        if not line:
            raise ContractError('CPU control owner disconnected')
        reply = json.loads(line)
        if not reply['ok']:
            raise ContractError(reply['error'])
        return reply['result']

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
            sensor_metadata={'rlt_active':self.rlt_active, **{k:f.metadata for k,f in frames.items()}})

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
