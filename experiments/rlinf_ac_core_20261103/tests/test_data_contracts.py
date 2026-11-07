"""Data-space contract tests T01-T10.

T01/T02/T07/T10 need the real LeRobot dataset and the RLinf transforms; they
are explicit SKIPs with reasons.  T03/T04/T05/T06/T08/T09 run on CPU project
code (fixtures where a dataset would otherwise be needed).
"""

from __future__ import annotations

import numpy as np
import pytest

from pi05_rlt_rtc.action_codec.codec import ActionCodec, NormMethod, NormStats
from pi05_rlt_rtc.contracts.errors import ContractError, SchemaMismatchError
from pi05_rlt_rtc.contracts.spaces import HorizonContract, reference_env_suffix
from pi05_rlt_rtc.contracts.transition import build_action_windows
from pi05_rlt_rtc.data.prompts import resolve_prompt


@pytest.mark.skip(
    reason="T01 requires the real LeRobot dataset; no dataset_audit can substitute for the "
    "local schema check (guide 5.2)"
)
def test_t01_state_action_dims_real_dataset():
    raise AssertionError("NOT_RUN: real dataset required")


@pytest.mark.skip(
    reason="T02 requires the real two-camera dataset and the RLinf image transforms"
)
def test_t02_camera_mapping_and_rgb_range():
    raise AssertionError("NOT_RUN: real dataset required")


def test_t03_episode_windows_with_tail_mask():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    actions = np.arange(12 * 8, dtype=np.float32).reshape(12, 8) / 100.0
    windows = build_action_windows(actions, "ep-a", contract)
    assert len(windows) == 12
    for w in windows:
        assert w.actions.shape == (10, 8)
        assert w.valid.shape == (10,)
    assert windows[0].valid.all()
    assert windows[0].anchor_index == 0
    # tail window: only rows 10 and 11 are real, the rest must be masked
    assert windows[-1].anchor_index == 11
    assert windows[-1].valid[:1].all()
    assert not windows[-1].valid[1:].any()
    assert np.all(windows[-1].actions[1:] == 0.0)
    with pytest.raises(ContractError):
        build_action_windows(actions, "ep-a", contract, tail_padding="zero")


def test_t04_default_prompt_and_unknown_task_rejected():
    assert resolve_prompt(None) == "insert the plug into the power strip"
    assert resolve_prompt("insert the plug into the power strip") == "insert the plug into the power strip"
    with pytest.raises(ContractError):
        resolve_prompt("pick up the cube")


def test_t05_env_encode_decode_roundtrip(codec: ActionCodec):
    rng = np.random.default_rng(0)
    env_actions = rng.uniform(-1.0, 1.0, size=(3, 10, 8)).astype(np.float32)
    encoded = codec.encode_env_for_guidance(env_actions)
    assert encoded.shape == (3, 10, 32)
    assert np.all(encoded[..., 8:] == 0.0), "model padding dims must stay exactly zero"
    decoded = codec.decode_full_horizon(encoded)
    assert decoded.actions.shape == (3, 10, 8)
    assert not decoded.clipped.any()
    assert np.max(np.abs(decoded.actions - env_actions)) < 1e-5
    # boundaries and special values
    special = np.zeros((5, 10, 8), dtype=np.float32)
    special[0, :, :] = -1.0
    special[1, :, :] = 1.0
    special[2, :, :] = 0.0
    special[3, :, :] = 0.5
    special[4, :, -1] = 1.0  # gripper extreme
    roundtrip = codec.decode_full_horizon(codec.encode_env_for_guidance(special)).actions
    assert np.max(np.abs(roundtrip - special)) < 1e-5
    # clipping is reported, never hidden as float noise
    overshoot = np.full((1, 10, 8), 2.0, dtype=np.float32)
    result = codec.decode_full_horizon(codec.encode_env_for_guidance(overshoot))
    assert result.clipped.all()
    assert np.all(result.actions == 1.0)
    # NaN input is rejected
    with pytest.raises(ContractError):
        codec.encode_env_for_guidance(np.full((1, 10, 8), np.nan, dtype=np.float32))
    # wrong dim is rejected
    with pytest.raises(ContractError):
        codec.encode_env_for_guidance(np.zeros((1, 10, 7), dtype=np.float32))
    # state anchors are refused for the delta codec
    with pytest.raises(ContractError):
        codec.encode_env_for_guidance(env_actions, state_anchor=np.zeros(9))


def test_t06_padding_dimensions_do_not_influence_guidance():
    import torch

    from pi05_rlt_rtc.rtc.full_vjp import FullVJPGuidance

    codec = ActionCodec(NormStats.symmetric())
    mask_np = codec.valid_model_mask.astype(np.float32)
    mask = torch.as_tensor(np.broadcast_to(mask_np, (1, 4, 32)).copy())
    x0 = torch.zeros(1, 4, 32)
    target = torch.zeros(1, 4, 32)
    target[..., 8:] = 1e6  # padding dims contain garbage

    def velocity(x, tau):
        del tau
        return x * 0.0

    sampler = FullVJPGuidance(num_steps=2)
    result = sampler.sample(velocity, x0, target_model=target, guidance_mask=mask)
    # constant velocity => J = I => guided update is w * mask * target, so the
    # padding dims must remain exactly zero regardless of their target values.
    assert torch.all(result.actions[..., 8:] == 0.0)


@pytest.mark.skip(
    reason="T07 requires the real dataset plus the RLinf PandaOutputs transform to prove no "
    "double delta application"
)
def test_t07_no_extra_delta_transform_on_real_data():
    raise AssertionError("NOT_RUN: real dataset required")


def test_t08_norm_stats_hash_mismatch_is_rejected():
    from pi05_rlt_rtc.learning.replay import ReplayBuffer

    buffer = ReplayBuffer(
        capacity=10,
        feature_version_id="feature-v1",
        norm_stats_sha256="hash-a",
    )
    with pytest.raises(SchemaMismatchError):
        buffer.require_compatible(
            schema_version=buffer.schema_version,
            feature_version_id="feature-v1",
            norm_stats_sha256="hash-b",
        )
    with pytest.raises(SchemaMismatchError):
        buffer.require_compatible(
            schema_version="pi05-rlt-rtc.schema.v0",
            feature_version_id="feature-v1",
            norm_stats_sha256="hash-a",
        )


def test_t09_full_h_suffix_selection_is_d_dot_dot_c():
    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    full = np.arange(10 * 8, dtype=np.float32).reshape(1, 10, 8)
    suffix = reference_env_suffix(full, contract)
    assert suffix.shape == (1, 5, 8)
    assert np.array_equal(suffix[0], full[0, 2:7]), "must be [D:D+C] = 2..6, not 0..4"
    with pytest.raises(ContractError):
        reference_env_suffix(full[0, :5], contract)


@pytest.mark.skip(
    reason="T10 requires the real RLinf training/inference state preprocessing paths to be "
    "compared on the same raw state"
)
def test_t10_proprio_preprocessing_equivalence_real_paths():
    raise AssertionError("NOT_RUN: RLinf runtime required")


def test_codec_zscore_method_roundtrip():
    stats = NormStats(
        NormMethod.ZSCORE,
        8,
        mean=np.arange(8, dtype=np.float64),
        std=np.arange(1, 9, dtype=np.float64),
    )
    codec = ActionCodec(stats, model_action_dim=8, env_action_dim=8)
    actions = np.random.default_rng(1).uniform(-1, 1, (2, 4, 8)).astype(np.float32)
    decoded = codec.decode_full_horizon(codec.encode_env_for_guidance(actions)).actions
    assert np.max(np.abs(decoded - actions)) < 1e-5
