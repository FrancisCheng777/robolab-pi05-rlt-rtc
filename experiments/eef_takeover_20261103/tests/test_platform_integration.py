"""Real deployment paths with explicitly identified CPU dependency doubles."""
import copy
import dataclasses
import json
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest

from pi05_rlt_rtc.config.runtime_config import load_runtime_config
from pi05_rlt_rtc.contracts.errors import ContractError, MockPolicyError
from pi05_rlt_rtc.deployment.planner import FeaturePlanner
from tests.test_review_controller import manual_runtime


def backend_double():
    from pi05_rlt_rtc.evaluation.toy_pipeline import build_toy
    toy = build_toy(z_dim=16)
    toy.feature_service.is_mock = False  # test-only protocol double
    toy.feature_service.enforce_single_prefix = False  # IDs restart across independent trials
    backend = FeaturePlanner(service=toy.feature_service, spec=toy.spec, contract=toy.contract,
                             control_hz=10, policy_version_id="toy-untrained-v1")
    cfg = load_runtime_config("configs/runtime_franka.yaml")
    metadata = {"protocol": 1, "mock": False, "policy_kind": "teacher", "feature_version_id": "MOCK-feature-v1",
                "policy_version_id": backend.policy_version_id,
                "norm_stats_sha256": toy.feature_service.norm_stats_sha256,
                "runtime_contract": cfg.wire_contract(), "fingerprint": {"test_double": True}}
    return cfg, toy, backend, metadata


def test_real_collection_warmup_online_update_and_policy_reload(tmp_path, monkeypatch, manual_runtime):
    from pi05_rlt_rtc.deployment import runtime
    from pi05_rlt_rtc.deployment.training import QueueLearner
    from pi05_rlt_rtc.learning.checkpoint import PolicyBundle
    from pi05_rlt_rtc.evaluation.toy_env import ToyDelayedControlEnv
    cfg, toy, backend, metadata = backend_double()
    class EnvDouble(ToyDelayedControlEnv):
        stop_request_acknowledged = True
        def close(self): pass
    monkeypatch.setattr(runtime, "make_env", lambda _: EnvDouble())
    replay, report = runtime.collect(cfg, backend, metadata, ticks=35, seed=1, out=tmp_path / "teacher")
    from pi05_rlt_rtc.deployment.rollout import load_rollout, label_rollout
    replay, annotation = label_rollout(load_rollout(tmp_path / "teacher/rollout.pt"), outcome="success", annotator="CPU boundary test")
    assert len(replay.learnable_transitions()) >= 2
    learner = QueueLearner(cfg, toy.spec, device="cpu", seed=1)
    learner.warmup(replay, 2)
    learner.update(replay, 2)
    saved = learner.save(tmp_path / "learner", metadata, replay)
    loaded = PolicyBundle.load(tmp_path / "learner/policy.pt")
    assert loaded.actor_state_sha256 == saved.actor_state_sha256
    backend.actor, backend.policy_version_id = loaded.build_actor(), loaded.policy_version_id
    _, actor_report = runtime.collect(cfg, backend, {**metadata, "policy_version_id": loaded.policy_version_id},
                                      ticks=25, seed=2, out=tmp_path / "actor")
    actor_replay, _ = label_rollout(load_rollout(tmp_path / "actor/rollout.pt"), outcome="failure", annotator="CPU boundary test")
    assert len(actor_replay.learnable_transitions()) >= 2


def test_remote_planner_images_identity_and_contract(tmp_path):
    from pi05_rlt_rtc.deployment.rpc import RemotePlanner
    from pi05_rlt_rtc.tools.planner_server import PlannerService, make_handler
    from pi05_rlt_rtc.tools.cli import _toy_request
    cfg, toy, backend, metadata = backend_double()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(PlannerService(backend, toy.contract, metadata=metadata)))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        remote = RemotePlanner(f"http://127.0.0.1:{server.server_address[1]}", cfg)
        obs = dataclasses.replace(toy.env.reset(), env_id="toy0", episode_id="toy0-ep1")
        request = _toy_request(toy)
        result = remote.plan(request, obs, np.zeros((2,8)), request.rtc_context)
        assert result.reference_env_full.shape == (10,8)
        assert result.bc_target_plan.shape == (5,8)
        assert toy.feature_service.prefix_calls == 1
        bad = dataclasses.replace(cfg, num_steps=999)
        with pytest.raises(ContractError, match="contracts differ"):
            RemotePlanner(remote.url, bad)
        original = remote._call
        def wrong_identity(path, payload=None):
            data = original(path, payload)
            data["identity"]["queue_generation"] += 1
            return data
        remote._call = wrong_identity
        with pytest.raises(ContractError, match="identity"):
            remote.plan(request, obs, np.zeros((2,8)), request.rtc_context)
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_camera_encoding_is_lossless_and_rejects_wrong_byte_count():
    from pi05_rlt_rtc.deployment.rpc import encode_observation, decode_image
    _, toy, _, _ = backend_double()
    obs = toy.env.reset()
    wire = encode_observation(obs)
    np.testing.assert_array_equal(decode_image(wire["images"]["main"]), obs.camera("main"))
    wire["images"]["main"]["data"] = "AA=="
    with pytest.raises(ContractError, match="byte count"):
        decode_image(wire["images"]["main"])


def test_actor_training_exploration_is_seeded_and_deployment_is_deterministic():
    import torch
    from pi05_rlt_rtc.learning.actor import QueueAwareRLTActor
    from pi05_rlt_rtc.tools.cli import _toy_request
    _, toy, backend, _ = backend_double()
    actor = QueueAwareRLTActor(toy.spec)
    with torch.no_grad():
        for parameter in actor.parameters(): parameter.zero_()
    backend.actor = actor
    obs = dataclasses.replace(toy.env.reset(), env_id="toy0", episode_id="toy0-ep1")
    req = _toy_request(toy)
    args = (req, obs, np.zeros((2,8)), req.rtc_context)
    assert np.count_nonzero(backend.plan(*args).submitted) == 0
    backend.deterministic = False
    noisy = backend.plan(*args).submitted
    assert np.count_nonzero(noisy) > 0
    np.testing.assert_array_equal(noisy, backend.plan(*args).submitted)


def test_upstream_codec_preserves_upstream_epsilon_and_zero_padding():
    from pi05_rlt_rtc.deployment.model_loader import UpstreamActionCodec
    def norm(data):
        return {"actions": (data["actions"] - .3) / (2 + 1e-6)}
    codec = UpstreamActionCodec(norm, "stats-file-hash")
    actions = np.ones((10,8), np.float32)
    result = codec.encode_env_for_guidance(actions)
    np.testing.assert_allclose(result[:,:8], norm({"actions": actions})["actions"])
    assert np.count_nonzero(result[:,8:]) == 0


def test_deployed_planner_rejects_a_different_rtc_mode():
    from pi05_rlt_rtc.tools.cli import _toy_request
    _, toy, backend, _ = backend_double()
    backend.rtc_mode = "full_vjp"
    request = _toy_request(toy)
    with pytest.raises(ContractError, match="RTC mode"):
        backend.plan(request, toy.env.reset(), np.zeros((2,8)), request.rtc_context)


def robot_fixture(tmp_path):
    from pi05_rlt_rtc.adapters.cameras import CameraFrame
    cfg = load_runtime_config("configs/runtime_franka.yaml")
    calibration = json.loads((cfg.path.parent / "franka_bridge.example.json").read_text())
    calibration.update(robot_model="panda", joint_lower=[-2]*7, joint_upper=[2]*7,
        max_velocity=[.5]*7, max_acceleration=[1]*7, max_jerk=[10]*7, joint_impedance=[100]*7,
        max_tracking_error=[.1]*7, collision_torque=[1]*7, collision_force=[1]*6,
        gripper_max_width=.085, gripper_speed=64, gripper_force=64,
        hardware_verified=True, gripper_device="/dev/test-only", robotiq_open_position=0, robotiq_closed_position=255,
        gripper_grasp_epsilon_inner=.005, gripper_grasp_epsilon_outer=.005,
        load_mass=0, load_center=[0]*3, load_inertia=[0]*9)
    path = tmp_path / "calibration.json"; path.write_text(json.dumps(calibration))
    raw = copy.deepcopy(cfg.raw); raw["franka"]["bridge_config"] = str(path)
    cfg = dataclasses.replace(cfg, raw=raw)
    class Transport:
        def __init__(self):
            self.calls = []
            self.state = {"ok": True, "protocol": 1, "session": "test-session", "ready": True,
                "mock": False, "armed": False, "claimed": False, "stopped": False, "fault": "",
                "q": [0]*7, "state_sequence": 10, "gripper_width": .06,
                "state_age_s": .001, "gripper_age_s": .001, "calibration": calibration}
        def request(self, payload):
            self.calls.append(payload)
            if payload["op"] == "state": return copy.deepcopy(self.state)
            return {"ok": True, "sequence": payload.get("sequence")}
    class Cameras:
        age, closed = 0., False
        def capture(self):
            return {k: CameraFrame(np.zeros((4,4,3), np.uint8), time.monotonic() - self.age) for k in ("main", "wrist")}
        def close(self): self.closed = True
    return cfg, Transport(), Cameras()


def test_robot_measured_state_mapping_and_single_trial(tmp_path):
    from pi05_rlt_rtc.adapters.franka import FrankaEnvAdapter
    cfg, transport, cameras = robot_fixture(tmp_path)
    env = FrankaEnvAdapter(cfg, transport=transport, cameras=cameras)
    try:
        obs = env.reset()
        np.testing.assert_allclose(obs.states[-2:], [.03, .03])
        transport.state["q"] = [.2]*7
        env.step(np.zeros(8))
        command = next(c for c in transport.calls if c["op"] == "command")
        np.testing.assert_allclose(command["q_target"], [.2]*7)
        assert command["gripper_width"] == pytest.approx(.0425)
        assert command["controller_id"]
        with pytest.raises(ContractError, match="automatically reset"):
            env.reset()
    finally:
        env.close()
    assert cameras.closed and transport.calls[-1]["op"] == "stop"


@pytest.mark.parametrize("problem", ["camera", "robot", "restart", "nan", "mock", "ack"])
def test_robot_bad_feedback_or_rpc_stops_owned_trial(tmp_path, problem):
    from pi05_rlt_rtc.adapters.franka import FrankaEnvAdapter
    cfg, transport, cameras = robot_fixture(tmp_path)
    if problem == "mock":
        transport.state["mock"] = True
        with pytest.raises(ContractError, match="mock"):
            FrankaEnvAdapter(cfg, transport=transport, cameras=cameras)
        assert not any(c["op"] == "command" for c in transport.calls)
        return
    env = FrankaEnvAdapter(cfg, transport=transport, cameras=cameras); env.reset()
    if problem == "camera": cameras.age = 3
    if problem == "robot": transport.state["state_age_s"] = 3
    if problem == "restart": transport.state["session"] = "different"
    if problem == "nan": transport.state["q"][0] = float("nan")
    if problem == "ack":
        original = transport.request
        def altered(payload):
            reply = original(payload)
            if payload["op"] == "command": reply["sequence"] = 999
            return reply
        transport.request = altered
    with pytest.raises(ContractError):
        if problem == "ack": env.step(np.zeros(8))
        else: env.observe()
    assert transport.calls[-1]["op"] == "stop"
    if problem != "ack": assert not any(c["op"] == "command" for c in transport.calls)
    env.close()


def test_constructing_second_robot_controller_does_not_stop_existing_owner(tmp_path):
    from pi05_rlt_rtc.adapters.franka import FrankaEnvAdapter
    cfg, transport, cameras = robot_fixture(tmp_path)
    transport.state["claimed"] = True
    with pytest.raises(ContractError, match="active trial"):
        FrankaEnvAdapter(cfg, transport=transport, cameras=cameras)
    assert [c["op"] for c in transport.calls] == ["state"]


def test_fresh_hardware_observation_and_stop_hooks(manual_runtime):
    from pi05_rlt_rtc.runtime.controller import WallClockController
    from pi05_rlt_rtc.runtime.queue import ActionQueue
    from pi05_rlt_rtc.evaluation.toy_env import ToyDelayedControlEnv
    _, toy, backend, metadata = backend_double()
    class LiveEnv(ToyDelayedControlEnv):
        observations = 0
        stopped = 0
        def observe(self):
            self.observations += 1
            return dataclasses.replace(self._snapshot(), tick=self.observations)
        def stop(self): self.stopped += 1
    # Separate hook test uses a minimal snapshot supplier independent of toy internals.
    env = LiveEnv(); initial = env.reset()
    def observe():
        env.observations += 1
        return initial
    env.observe = observe
    controller = WallClockController(contract=toy.contract, queue=ActionQueue(toy.contract), backend=toy.backend,
                                     env=env, tick_hz=10)
    controller.run(3)
    assert env.observations == 3 and env.stopped == 1


def test_slow_hardware_capture_stops_before_sending_overdue_command(manual_runtime, contract):
    from pi05_rlt_rtc.runtime.controller import WallClockController
    from pi05_rlt_rtc.runtime.queue import ActionQueue
    from pi05_rlt_rtc.runtime.scheduler import NumberedReferencePlanner
    from tests.helpers import RecordingEnv
    env = RecordingEnv()
    snapshot = env.reset()
    stopped = []
    def slow_observation():
        manual_runtime.sleep(.11)
        return snapshot
    env.observe = slow_observation
    env.stop = lambda: stopped.append(True)
    report = WallClockController(contract=contract, queue=ActionQueue(contract), env=env,
        backend=NumberedReferencePlanner(contract), tick_hz=10).run(3)
    assert report.safety_stops == 1 and report.ticks_run == 0
    assert not env.executed and stopped == [True]
