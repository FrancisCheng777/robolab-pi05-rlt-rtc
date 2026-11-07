import json
import numpy as np
import pytest
from pi05_rlt_rtc.contracts.physical import ACTION_CONTRACT_ID, encode_action
from pi05_rlt_rtc.adapters.panda_policy import PandaInputs
from pi05_rlt_rtc.adapters.cameras import CameraFrame, validate_frames
from pi05_rlt_rtc.contracts.errors import ContractError


def test_v2_preserves_measured_delta_above_legacy_scale():
    q = np.zeros(7); target = q.copy(); target[5] = .036
    a = encode_action(q, target, .04)
    assert ACTION_CONTRACT_ID == 'panda_delta_q0.05_robotiq_width0.085_v2'
    np.testing.assert_allclose(a[:7] * .05, target, atol=1e-8)


def test_wrist_occupies_exactly_one_model_slot():
    image = np.full((224,224,3), 83, np.uint8)
    out = PandaInputs()({'observation/state': np.zeros(9), 'observation/wrist_image': image})
    assert list(out['image_mask'].values()) == [True, False, False]
    np.testing.assert_array_equal(out['image']['base_0_rgb'], image)
    assert not out['image']['left_wrist_0_rgb'].any()
    assert not out['image']['right_wrist_0_rgb'].any()


def test_wrist_only_freshness_rejects_stale_or_missing_frame():
    frame = CameraFrame(np.zeros((224,224,3),np.uint8), 10.)
    validate_frames({'wrist':frame}, now=10.01, max_age=.2, max_skew=.08)
    with pytest.raises(ContractError):
        validate_frames({'wrist':frame}, now=10.3, max_age=.2, max_skew=.08)
    with pytest.raises(ContractError):
        validate_frames({}, now=10.01, max_age=.2, max_skew=.08)
