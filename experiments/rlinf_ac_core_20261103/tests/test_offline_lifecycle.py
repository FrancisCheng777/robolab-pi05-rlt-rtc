"""No hardware: replay exact recorded planner inputs and fail closed on I/O loss."""
import dataclasses
import threading

import numpy as np
import pytest

from pi05_rlt_rtc import PLANNING_STATE_VERSION, SCHEMA_VERSION
from pi05_rlt_rtc.config.runtime_config import load_runtime_config
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
from pi05_rlt_rtc.contracts.plan import PlanRequest, PrevPlanContext
from pi05_rlt_rtc.runtime.scheduler import PlanResult


def sample():
    cfg = load_runtime_config('configs/runtime_franka.yaml')
    obs = ObservationSnapshot('franka', 'recorded', 5, np.zeros(9, np.float32),
        {k: np.zeros((224,224,3), np.uint8) for k in ('main','wrist')},
        {'main': True, 'wrist': True}, cfg.raw['prompt'], 100., {'main': 99.99},
        sensor_metadata={'main': {'source_clock': 'kinect_device', 'frame_id': 42}})
    prev = PrevPlanContext.empty()
    req = PlanRequest(0, 'franka', 'recorded', 1, 5, 7, 1, 'policy-test', 'feature-test',
                      SCHEMA_VERSION, PLANNING_STATE_VERSION, prev, 123, 'full_vjp', 5)
    metadata = {'mock': True, 'policy_version_id': 'policy-test', 'feature_version_id': 'feature-test',
                'runtime_contract': cfg.wire_contract()}
    return cfg, obs, req, np.zeros((2,8), np.float32), metadata


def test_recording_roundtrip_and_offline_no_hardware(tmp_path, monkeypatch):
    from pi05_rlt_rtc.runtime.recording import AsyncRequestRecorder, load_request
    from pi05_rlt_rtc.deployment.offline import replay_recordings
    from pi05_rlt_rtc.deployment import runtime
    cfg, obs, req, pending, meta = sample()
    monkeypatch.setattr(runtime, 'make_env', lambda *_: pytest.fail('hardware construction'))
    with AsyncRequestRecorder(tmp_path / 'recordings', meta) as recorder:
        recorder.submit(req, obs, pending)
    payload = load_request(next((tmp_path / 'recordings').glob('request-*.json')))
    assert payload['observation'].sensor_metadata['main']['frame_id'] == 42
    np.testing.assert_array_equal(payload['observation'].images['main'], obs.images['main'])
    assert payload['request'].rng_seed == 123
    calls = []
    class Backend:
        def plan(self, request, observation, prefix, previous):
            calls.append(request)
            np.testing.assert_array_equal(prefix, pending)
            return PlanResult(np.zeros((5,8)), np.zeros((10,8)), 'test')
    result = replay_recordings(cfg, Backend(), meta, tmp_path / 'recordings', tmp_path / 'output', allow_mock=True)
    assert len(calls) == 1 and result['robot_commands_sent'] == 0
    assert result['mock'] and not result['hardware_validated']
    assert (tmp_path / 'output/plans.npz').exists()
    with pytest.raises(ContractError, match='mock'):
        replay_recordings(cfg, Backend(), meta, tmp_path / 'recordings', tmp_path / 'rejected')


def test_recorder_disk_error_is_not_silent(tmp_path, monkeypatch):
    from pi05_rlt_rtc.runtime import recording
    cfg, obs, req, pending, meta = sample()
    monkeypatch.setattr(recording, 'write_request', lambda *a: (_ for _ in ()).throw(OSError('disk full')))
    recorder = recording.AsyncRequestRecorder(tmp_path / 'records', meta)
    recorder.submit(req, obs, pending)
    with pytest.raises(ContractError, match='disk full'):
        recorder.close()


def test_recorder_overflow_fails_without_waiting(tmp_path, monkeypatch):
    from pi05_rlt_rtc.runtime import recording
    cfg, obs, req, pending, meta = sample()
    entered, release = threading.Event(), threading.Event()
    real_write = recording.write_request
    def slow(*args):
        entered.set()
        assert release.wait(3)
        real_write(*args)
    monkeypatch.setattr(recording, 'write_request', slow)
    recorder = recording.AsyncRequestRecorder(tmp_path / 'records', meta, capacity=1)
    try:
        recorder.submit(req, obs, pending)
        assert entered.wait(2)
        recorder.submit(dataclasses.replace(req, request_id=1), obs, pending)
        with pytest.raises(ContractError, match='full'):
            recorder.submit(dataclasses.replace(req, request_id=2), obs, pending)
    finally:
        release.set()
        with pytest.raises(ContractError, match='full'):
            recorder.close()


def test_offline_smoke_requires_recording_before_model_or_environment(monkeypatch):
    from types import SimpleNamespace
    from pi05_rlt_rtc.deployment import runtime
    monkeypatch.setattr(runtime, 'make_backend', lambda *_: pytest.fail('loaded model before validation'))
    monkeypatch.setattr(runtime, 'make_env', lambda *_: pytest.fail('hardware construction'))
    with pytest.raises(ContractError, match='recording'):
        runtime.smoke_or_profile(SimpleNamespace(runtime_config='configs/runtime_franka.yaml', seed=0))


def test_camera_timing_survives_rpc_and_caller_mutation():
    from pi05_rlt_rtc.deployment.rpc import encode_observation
    from pi05_rlt_rtc.tools.planner_server import _observation_from_json
    cfg, obs, req, pending, meta = sample()
    original = {'main': {'frame_id': 7, 'source_clock': 'device'}}
    frozen = dataclasses.replace(obs, sensor_metadata=original)
    original['main']['frame_id'] = 999
    decoded = _observation_from_json(encode_observation(frozen))
    assert decoded.sensor_metadata['main']['frame_id'] == 7


def test_recorder_failure_stops_controller_before_next_command(contract, monkeypatch):
    from tests.helpers import RecordingEnv
    from pi05_rlt_rtc.runtime.controller import WallClockController
    from pi05_rlt_rtc.runtime.queue import ActionQueue
    from pi05_rlt_rtc.runtime.scheduler import NumberedReferencePlanner
    class Recorder:
        def submit(self, *args): pass
        def check(self): raise ContractError('disk full')
    class Env(RecordingEnv):
        stopped = False
        steps = 0
        def stop(self): self.stopped = True
        def step(self, action): self.steps += 1; return super().step(action)
    env = Env()
    controller = WallClockController(contract=contract, queue=ActionQueue(contract),
        backend=NumberedReferencePlanner(contract), env=env, tick_hz=15, observation_recorder=Recorder())
    with pytest.raises(ContractError, match='disk full'):
        controller.run(3)
    assert env.stopped and env.steps == 0
