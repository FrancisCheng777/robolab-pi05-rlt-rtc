"""Representation/gradient contract tests T11-T20.

T11-T13/T15-T18 need the real RLinf RLT token transformer and Stage 1/2
runtimes; they are explicit SKIPs.  T14/T19/T20 exercise project code paths.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from pi05_rlt_rtc.adapters.feature_reference import check_prefix_length
from pi05_rlt_rtc.contracts.errors import ContractError, SchemaMismatchError
from pi05_rlt_rtc.learning.checkpoint import PolicyBundle


@pytest.mark.skip(reason="T11 requires the real RLT encoder/decoder and VLA parameters")
def test_t11_rlt_loss_does_not_train_vla_real_modules():
    raise AssertionError("NOT_RUN: RLinf runtime required")


@pytest.mark.skip(reason="T12 requires the real pi0.5 flow-matching loss")
def test_t12_vla_loss_does_not_train_rlt_modules():
    raise AssertionError("NOT_RUN: RLinf runtime required")


@pytest.mark.skip(reason="T13 requires the real causal RLT reconstruction decoder")
def test_t13_decoder_teacher_shifting_real_module():
    raise AssertionError("NOT_RUN: RLinf runtime required")


def test_t14_prefix_over_capacity_is_an_error():
    check_prefix_length(1024, 1024)
    with pytest.raises(ContractError):
        check_prefix_length(1025, 1024)
    with pytest.raises(ContractError):
        check_prefix_length(10, 0)


@pytest.mark.skip(reason="T15 requires the real prefix padding masks from the tokenizer")
def test_t15_prefix_padding_mask_semantics():
    raise AssertionError("NOT_RUN: RLinf runtime required")


@pytest.mark.skip(reason="T16 requires a real Stage 1 run on real data")
def test_t16_stage1_small_sample_overfit():
    raise AssertionError("NOT_RUN: real data + weights required")


@pytest.mark.skip(
    reason="T17 requires the real frozen pi0.5/RLT Stage 2 parameter checksums; the toy "
    "gradient-boundary audit is a different (labelled) result and cannot substitute"
)
def test_t17_stage2_frozen_parameter_checksums_real_model():
    raise AssertionError("NOT_RUN: RLinf runtime required")


@pytest.mark.skip(reason="T18 requires a real RLinf Stage 2 checkpoint loader")
def test_t18_missing_rlt_weights_fail_to_load():
    raise AssertionError("NOT_RUN: RLinf runtime required")


def test_t19_save_reload_same_observation_same_noise(toy_build, tmp_path):
    spec = toy_build.spec
    torch.manual_seed(0)
    from pi05_rlt_rtc.learning.actor import QueueAwareRLTActor

    actor = QueueAwareRLTActor(spec, hidden_dims=(32, 32))
    bundle = PolicyBundle(
        spec=spec,
        contract=toy_build.contract,
        actor_state={k: v.detach().clone() for k, v in actor.state_dict().items()},
        critic_state=None,
        feature_version_id="feature-v1",
        norm_stats_sha256="norm-v1",
        policy_version_id="policy-v1",
        actor_hidden_dims=(32, 32),
    )
    path = bundle.save(tmp_path / "actor.pt", include_critic=False)

    state = torch.randn(1, sum(d for _, d in spec.actor_state_layout()))
    ref = torch.randn(1, spec.chunk * spec.env_action_dim)
    generator = torch.Generator().manual_seed(7)
    first = actor.act(state=state, reference_suffix=ref, deterministic=True)

    loaded = PolicyBundle.load(path, expected_norm_stats_sha256="norm-v1")
    reloaded_actor = loaded.build_actor()
    second = reloaded_actor.act(state=state, reference_suffix=ref, deterministic=True)
    assert torch.equal(first.actions, second.actions)
    # frozen on load
    assert all(not p.requires_grad for p in reloaded_actor.parameters())

    # version mismatches are rejected
    with pytest.raises(SchemaMismatchError):
        PolicyBundle.load(path, expected_norm_stats_sha256="other")
    with pytest.raises(SchemaMismatchError):
        PolicyBundle.load(path, expected_feature_version_id="other-feature")


def test_t20_one_prefix_pass_per_request(toy_build):
    build = toy_build
    obs = build.env.reset(seed=0)
    from tests.helpers import make_request

    request = make_request(build.contract, request_id=5, capture_tick=5)
    result = build.feature_service.extract(request, obs)
    result.validate(build.contract, expected_batch=1)
    assert build.feature_service.prefix_calls == 1
    assert result.prefix_calls == 1

    # a second extraction for the same request id is refused
    with pytest.raises(RuntimeError):
        build.feature_service.extract(request, obs)
    assert build.feature_service.prefix_calls == 1

    # the full reference horizon survives; the only slicing is [D:D+C]
    assert result.reference_env_full.shape == (1, build.contract.horizon, 8)
    assert result.vla_model_actions.shape == (1, build.contract.horizon, 32)
    assert result.reference_suffix(build.contract).shape == (1, build.contract.chunk, 8)

    # z and reference come from the same call: disabling further post-processing
    # must not change z or the reference for this request.
    assert result.planning_metadata()["z_sha256"]
    with pytest.raises(ContractError):
        bad = copy.copy(result)
        bad.prefix_calls = 2
        bad.validate(build.contract)
