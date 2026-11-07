"""Panda/Robotiq demonstration and deployment semantics, versioned together."""
import numpy as np

from .errors import ContractError

ACTION_CONTRACT_ID = "panda_delta_q0.05_robotiq_width0.085_v2"
BRIDGE_CONTRACT_ID = "panda_delta_q0.03_robotiq_width0.085_v1"
MODEL_CONFIG = "pi05_rlt_panda_robotiq_joint"
CONTROL_HZ = 15
JOINT_DELTA = .05
MAX_WIDTH = .085
PROMPT = "insert the plug into the power strip"


def encode_action(q, target, width):
    q, target = np.asarray(q, float), np.asarray(target, float)
    if q.shape != (7,) or target.shape != (7,) or not np.isfinite([q, target]).all():
        raise ContractError("q and target require seven finite joint positions")
    if not np.isfinite(width) or not 0 <= width <= MAX_WIDTH:
        raise ContractError("Robotiq width must be in [0,0.085] metres")
    action = np.r_[(target-q)/JOINT_DELTA, 2*width/MAX_WIDTH-1]
    if np.any(np.abs(action) > 1 + 1e-7):
        raise ContractError("demonstration target exceeds action contract; do not clip labels")
    return np.clip(action, -1, 1).astype(np.float32)


def validate_calibration_contract(config):
    if (config.get("robot_model") != "panda" or config.get("gripper_type") != "robotiq_2f85"
            or config.get("action_contract_id") != BRIDGE_CONTRACT_ID
            or np.asarray(config.get("joint_delta_scale")).shape != (7,)
            or not np.allclose(config["joint_delta_scale"], .03, rtol=0, atol=1e-10)
            or config.get("gripper_max_width") != MAX_WIDTH
            or config.get("finger_range") != [0, MAX_WIDTH / 2]):
        raise ContractError("calibration contradicts Panda/Robotiq action contract")


def validate_frame_pairs(frames, now):
    from pi05_rlt_rtc.adapters.cameras import CameraFrame, validate_frames
    validate_frames({k: CameraFrame(*v) for k, v in frames.items()}, now=now,
                    max_age=.2, max_skew=.08)
