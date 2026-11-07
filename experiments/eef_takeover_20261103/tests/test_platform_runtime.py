"""Integration boundaries without downloading a model or moving a robot."""
import dataclasses

import numpy as np
import pytest

from pi05_rlt_rtc.contracts.errors import ContractError, MissingAssetError


def test_runtime_templates_validate_without_assets():
    from pi05_rlt_rtc.config.runtime_config import load_runtime_config
    robot = load_runtime_config("configs/runtime_franka.yaml")
    assert robot.platform == "franka"
    with pytest.raises(MissingAssetError):
        robot.require_model_assets()


def test_real_planner_uses_one_feature_pass_and_attaches_the_learning_state():
    from pi05_rlt_rtc.deployment.planner import FeaturePlanner
    from pi05_rlt_rtc.evaluation.toy_pipeline import build_toy
    from pi05_rlt_rtc.tools.cli import _toy_request
    toy = build_toy(z_dim=16)
    # This is explicitly a test double, not a real-model acceptance result.
    service = toy.feature_service
    service.is_mock = False
    backend = FeaturePlanner(service=service, spec=toy.spec, contract=toy.contract,
                             control_hz=10, policy_version_id="toy-untrained-v1")
    obs = dataclasses.replace(toy.env.reset(), env_id="toy0", episode_id="toy0-ep1")
    req = _toy_request(toy)
    pending = np.full((2, 8), 0.2, np.float32)
    result = backend.plan(req, obs, pending, req.rtc_context)
    assert service.prefix_calls == 1
    np.testing.assert_array_equal(result.planning_state.pending_prefix[0, :2], pending)
    np.testing.assert_array_equal(result.submitted, result.reference_env_full[2:7])
    assert not hasattr(backend, "_features")  # no growing inference cache


def test_robot_joint_delta_and_gripper_mapping():
    from pi05_rlt_rtc.adapters.franka import map_joint_delta_action
    a = np.array([1, -1, 0, 0, 0, 0, 0, 0], np.float32)
    q = np.zeros(7)
    target, width = map_joint_delta_action(a, q, delta_scale=np.full(7, .1),
        joint_lower=np.full(7, -2), joint_upper=np.full(7, 2),
        finger_range=(-.01, .04), max_width=.08)
    np.testing.assert_allclose(target[:2], [.1, -.1])
    assert width == pytest.approx(.03)  # two fingers; normalized 0 != half-open
    a[0] = 1.1
    with pytest.raises(ContractError):
        map_joint_delta_action(a, q, delta_scale=np.full(7, .1),
            joint_lower=np.full(7, -2), joint_upper=np.full(7, 2),
            finger_range=(-.01, .04), max_width=.08)


def test_robot_limits_are_rejected_instead_of_clipping_the_executed_action():
    from pi05_rlt_rtc.adapters.franka import map_joint_delta_action
    with pytest.raises(ContractError, match="limit"):
        map_joint_delta_action(np.ones(8), np.full(7, .95), delta_scale=np.full(7, .1),
            joint_lower=np.full(7, -1), joint_upper=np.full(7, 1),
            finger_range=(-.01, .04), max_width=.08)
