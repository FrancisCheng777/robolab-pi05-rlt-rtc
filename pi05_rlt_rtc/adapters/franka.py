"""Franka environment using the local C++ libfranka bridge (no SDK in Python)."""
from __future__ import annotations

import contextlib
import json
import socket
import time
import uuid

import numpy as np

from pi05_rlt_rtc.adapters.cameras import create_cameras, validate_frames
from pi05_rlt_rtc.contracts.errors import ContractError, EnvironmentNotReadyError, MissingAssetError
from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
from pi05_rlt_rtc.contracts.physical import JOINT_DELTA, BRIDGE_CONTRACT_ID
from pi05_rlt_rtc.runtime.scheduler import StepOutcome


def vector7(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (7,) or not np.isfinite(a).all():
        raise ContractError(f"{name} must contain seven finite values")
    return a


def map_joint_delta_action(action, q, *, delta_scale, joint_lower, joint_upper, finger_range, max_width):
    a = np.asarray(action, dtype=float)
    if a.shape != (8,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1):
        raise ContractError("Franka requires a finite normalized command in [-1,1]^8")
    q = vector7(q, "q")
    delta = vector7(delta_scale, "joint_delta_scale")
    lower, upper = vector7(joint_lower, "joint_lower"), vector7(joint_upper, "joint_upper")
    if np.any(delta <= 0) or np.any(upper <= lower):
        raise ContractError("invalid joint scale/limits")
    target = q + a[:7] * delta  # relative to this execution step's measured q
    if np.any(target < lower) or np.any(target > upper):
        raise ContractError("joint target violates the configured soft limit")
    fingers = np.asarray(finger_range, float)
    if fingers.shape != (2,) or not np.isfinite(fingers).all() or fingers[0] >= fingers[1] or not 0 < max_width:
        raise ContractError("invalid finger range or maximum gripper width")
    finger = fingers[0] + (a[7] + 1) / 2 * (fingers[1] - fingers[0])
    # One measured Robotiq aperture; the state uses two derived half-widths.
    width = float(np.clip(2 * finger, 0, max_width))
    return target, width


def load_bridge_config(path):
    config = json.loads(path.read_text())
    from pi05_rlt_rtc.contracts.physical import validate_calibration_contract
    validate_calibration_contract(config)
    required = ("robot_model", "action_contract_id", "joint_lower", "joint_upper", "joint_delta_scale",
                "max_velocity", "max_acceleration", "max_jerk", "joint_impedance", "max_tracking_error",
                "collision_torque", "collision_force", "gripper_max_width", "gripper_speed", "gripper_force",
                "load_mass", "load_center", "load_inertia", "gripper_grasp_epsilon_inner", "gripper_grasp_epsilon_outer")
    missing = [k for k in required if config.get(k) is None]
    if missing:
        raise MissingAssetError(f"fill hardware/calibration fields in {path}: {missing}")
    if config["robot_model"] not in ("panda", "fr3"):
        raise ContractError("bridge robot_model must explicitly be panda or fr3")
    for name in ("joint_lower", "joint_upper", "joint_delta_scale", "max_velocity", "max_acceleration",
                 "max_jerk", "joint_impedance", "max_tracking_error", "collision_torque"):
        a = vector7(config[name], name)
        if name not in ("joint_lower", "joint_upper") and np.any(a <= 0):
            raise ContractError(f"{name} must be positive")
    if np.any(np.asarray(config["joint_upper"]) <= np.asarray(config["joint_lower"])):
        raise ContractError("joint_upper must exceed joint_lower")
    for name, size in (("collision_force", 6), ("load_center", 3), ("load_inertia", 9), ("finger_range", 2)):
        a = np.asarray(config.get(name), float)
        if a.shape != (size,) or not np.isfinite(a).all():
            raise ContractError(f"{name} must contain {size} finite values")
    for name in ("gripper_max_width", "gripper_speed", "gripper_force", "command_timeout_s", "state_timeout_s",
                 "gripper_grasp_epsilon_inner", "gripper_grasp_epsilon_outer"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ContractError(f"{name} must be positive and finite")
    if not np.isfinite(config["load_mass"]) or config["load_mass"] < 0:
        raise ContractError("load_mass must be finite and nonnegative")
    if config.get("hardware_verified") is not True:
        raise MissingAssetError("complete tool/Robotiq calibration and set hardware_verified=true")
    if not isinstance(config.get("gripper_device"), str) or not config["gripper_device"].strip():
        raise MissingAssetError("configure the calibrated Robotiq serial device")
    lo, hi = config.get("robotiq_open_position"), config.get("robotiq_closed_position")
    if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= 255:
        raise ContractError("invalid calibrated Robotiq register endpoints")
    for key in ("gripper_speed", "gripper_force"):
        if type(config[key]) is not int or not 1 <= config[key] <= 255:
            raise ContractError(f"{key} must be an integer register value in [1,255]")
    return config


class UnixFrankaTransport:
    """One bounded JSON-line transaction per connection. No network work in RT callback."""
    def __init__(self, path, *, timeout=.05):
        if not 0 < timeout <= 1:
            raise ContractError("bridge timeout must be in (0,1] seconds")
        self.path, self.timeout = path, timeout

    def request(self, payload):
        data = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect(self.path)
                sock.sendall(data)
                response = bytearray()
                while b"\n" not in response:
                    block = sock.recv(4096)
                    if not block:
                        raise OSError("bridge closed the connection")
                    response.extend(block)
                    if len(response) > 65536:
                        raise ContractError("oversized bridge response")
        except (OSError, TimeoutError) as exc:
            raise EnvironmentNotReadyError(f"libfranka bridge unavailable: {exc}") from exc
        reply = json.loads(response.split(b"\n", 1)[0])
        if not reply.get("ok"):
            raise ContractError(f"libfranka bridge rejected request: {reply.get('error')}")
        return reply


class FrankaEnvAdapter:
    """A single manually positioned trial. reset() never moves the arm or homes the gripper."""
    def __init__(self, cfg, *, transport=None, cameras=None):
        self.cfg = cfg
        self.settings = cfg.raw["franka"]
        for name in ("rpc_timeout_s", "max_state_age_s", "max_camera_age_s", "max_camera_skew_s"):
            value = self.settings.get(name)
            if not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
                raise ContractError(f"franka.{name} must be positive and finite")
        self.calibration = load_bridge_config(cfg.asset(self.settings.get("bridge_config"), "franka.bridge_config"))
        if self.calibration["action_contract_id"] != BRIDGE_CONTRACT_ID:
            raise ContractError("hardware calibration action contract differs from the policy")
        if self.calibration["command_timeout_s"] <= 1 / cfg.control_hz:
            raise ContractError("bridge command timeout must exceed one controller period")
        self.transport = transport or UnixFrankaTransport(self.settings["socket"], timeout=self.settings["rpc_timeout_s"])
        self._tick, self._sequence, self._used, self._stopped = 0, 0, False, False
        self.cameras = None
        self._session = None
        self._owns_bridge = False
        self._controller_id = uuid.uuid4().hex
        self.stop_request_acknowledged = False
        self.command_log = []
        try:
            state = self._state()
            self._session = state["session"]
            if state.get("armed") or state.get("claimed") or state.get("last_sequence", 0) != 0:
                raise ContractError("bridge already belongs to an active trial; restart explicitly")
            if state.get("mock") is not False:
                raise ContractError("real Franka adapter refuses a mock bridge")
            if state.get("calibration") != self.calibration:
                raise ContractError("Python and C++ bridge calibration files differ")
            self.cameras = cameras or create_cameras(self.settings["cameras"])
        except BaseException:
            self.close()
            raise

    def _state(self):
        state = self.transport.request({"op": "state", "protocol": 1})
        if state.get("protocol") != 1 or not state.get("ready") or state.get("fault") or state.get("stopped"):
            raise ContractError(f"bridge is not ready: {state.get('fault') or 'stopped/uninitialized'}")
        if self._session is not None and state.get("session") != self._session:
            raise ContractError("bridge restarted; the current trial is invalid")
        for key, limit in (("state_age_s", self.settings["max_state_age_s"]),
                           ("gripper_age_s", self.settings.get("max_gripper_state_age_s", .5))):
            age = state.get(key, float("inf"))
            if not np.isfinite(age) or not 0 <= age <= limit:
                raise ContractError(f"stale {key}: {age}")
        vector7(state["q"], "robot q")
        width = state["gripper_width"]
        if not np.isfinite(width) or not 0 <= width <= self.calibration["gripper_max_width"] + .001:
            raise ContractError("invalid measured gripper width")
        return state

    def reset(self, *, seed=0):
        if self._used or self._stopped:
            raise ContractError("Franka trial cannot automatically reset; reposition and restart explicitly")
        self._used = True
        obs = self.observe()
        # A separate controller cannot stop or command a bridge owned by this
        # trial. Claim is idempotent for our ID, and never itself starts motion.
        self._owns_bridge = True
        try:
            self.transport.request({"op": "claim", "protocol": 1, "session": self._session,
                                    "controller_id": self._controller_id})
        except BaseException:
            self.stop()
            raise
        return obs

    def observe(self):
        try:
            state = self._state()
            frames = self.cameras.capture()
            now = time.monotonic()
            validate_frames(frames, now=now, max_age=self.settings["max_camera_age_s"],
                            max_skew=self.settings["max_camera_skew_s"])
            return ObservationSnapshot(env_id="franka", episode_id=self._session, tick=self._tick,
                states=np.asarray([*state["q"], state["gripper_width"] / 2, state["gripper_width"] / 2], np.float32),
                images={k: f.rgb for k, f in frames.items()}, image_masks={"wrist": True},
                prompt=self.cfg.raw["prompt"], captured_monotonic=now,
                source_timestamps={**{k: f.captured_monotonic for k, f in frames.items()},
                                   "robot": now - state["state_age_s"]},
                sensor_metadata={**{k: f.metadata for k, f in frames.items()},
                    "robot": {"state_sequence": state["state_sequence"], "state_age_s": state["state_age_s"],
                              "clock": "host_monotonic_estimate"}})
        except BaseException:
            self.stop()
            raise

    def step(self, action):
        if not self._used or self._stopped:
            raise ContractError("Franka is not in an active trial")
        try:
            state = self._state()
            validate_frames(self.cameras.capture(), now=time.monotonic(), max_age=self.settings["max_camera_age_s"],
                            max_skew=self.settings["max_camera_skew_s"])
            c = self.calibration
            target, width = map_joint_delta_action(action, state["q"], delta_scale=[JOINT_DELTA]*7,
                joint_lower=c["joint_lower"], joint_upper=c["joint_upper"],
                finger_range=c["finger_range"], max_width=c["gripper_max_width"])
            self._sequence += 1
            audit = {"tick": self._tick, "sequence": self._sequence,
                     "state_sequence": state["state_sequence"], "measured_q": list(state["q"]),
                     "normalized_action": np.asarray(action, float).tolist(), "q_target": target.tolist(),
                     "gripper_width_target": width, "bridge_acknowledged": False}
            self.command_log.append(audit)
            audit["send_monotonic_s"] = time.monotonic()
            reply = self.transport.request({"op": "command", "protocol": 1, "session": self._session,
                "controller_id": self._controller_id,
                "sequence": self._sequence, "state_sequence": state["state_sequence"],
                "q_target": target.tolist(), "gripper_width": width})
            audit["ack_receive_monotonic_s"] = time.monotonic()
            audit["bridge_reply"] = dict(reply)
            if reply.get("sequence") != self._sequence:
                raise ContractError("bridge command ACK sequence mismatch")
            audit["bridge_acknowledged"] = True
            self._tick += 1
            return StepOutcome(self.observe(), reward=0., terminated=False, truncated=False,
                               info={"reward_valid": False, "bridge_sequence": self._sequence})
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if not self._stopped:
            self._stopped = True
            if self._owns_bridge:
                with contextlib.suppress(Exception):
                    reply = self.transport.request({"op": "stop", "protocol": 1, "session": self._session,
                                                    "controller_id": self._controller_id})
                    self.stop_request_acknowledged = bool(reply.get("ok"))

    def close(self):
        self.stop()
        if self.cameras is not None:
            self.cameras.close()
            self.cameras = None
