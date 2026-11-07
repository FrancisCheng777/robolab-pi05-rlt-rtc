"""RTC math tests T21-T32 (CPU, no upstream runtime required)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pi05_rlt_rtc.contracts.errors import ContractError, InferenceModeError
from pi05_rlt_rtc.rtc.approx import (
    build_rtc_target_and_mask,
    sample_actions_with_rtc_guidance_approx,
)
from pi05_rlt_rtc.rtc.canonical import (
    canonical_time_grid,
    canonical_velocity_from_native,
    guidance_coefficient,
)
from pi05_rlt_rtc.rtc.dispatcher import RTCDispatcher
from pi05_rlt_rtc.rtc.full_vjp import (
    FullVJPGuidance,
    check_outer_context,
    plain_canonical_euler,
    plain_native_euler,
    vjp_correction,
)
from pi05_rlt_rtc.rtc.prefix_weights import (
    build_guidance_weight_matrix,
    get_prefix_weights,
    guidance_mask_from_previous_plan,
)


def _linear_velocity(A: torch.Tensor, b: torch.Tensor):
    def velocity(x: torch.Tensor, tau: float) -> torch.Tensor:
        del tau
        return x @ A.T + b

    return velocity


def test_t21_zero_mask_degenerates_to_plain_sampler():
    torch.manual_seed(0)
    B, H, A = 2, 4, 3
    x0 = torch.randn(B, H, A)
    A_mat = 0.3 * torch.eye(A)
    b = torch.zeros(A)
    velocity = _linear_velocity(A_mat, b)
    plain = plain_canonical_euler(velocity, x0, num_steps=5)

    sampler = FullVJPGuidance(num_steps=5, guidance_weight_cap=5.0)
    target = torch.randn(B, H, A)
    zero_mask = torch.zeros(B, H, A)
    guided = sampler.sample(velocity, x0, target_model=target, guidance_mask=zero_mask)
    assert torch.equal(guided.actions, plain)
    assert guided.diagnostics.guided is False

    no_context = sampler.sample(velocity, x0)
    assert torch.equal(no_context.actions, plain)


def test_t22_no_previous_plan_samples_normally():
    torch.manual_seed(1)
    x0 = torch.randn(1, 5, 3)
    velocity = _linear_velocity(0.1 * torch.eye(3), torch.zeros(3))
    sampler = FullVJPGuidance(num_steps=4)
    result = sampler.sample(velocity, x0)
    assert result.actions.shape == x0.shape
    assert torch.isfinite(result.actions).all()


def test_t23_native_time_direction_regression():
    torch.manual_seed(2)
    x0 = torch.randn(2, 4, 3)
    A_mat = 0.2 * torch.randn(3, 3)
    b = 0.1 * torch.ones(3)

    def velocity_native(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return x @ A_mat.T + b

    canonical = canonical_velocity_from_native(velocity_native)
    ours = plain_canonical_euler(canonical, x0, num_steps=6)
    reference = plain_native_euler(velocity_native, x0, num_steps=6)
    assert torch.allclose(ours, reference, atol=1e-6)

    # sign guard: if the sign were not flipped, the trajectories would differ.
    wrong = plain_canonical_euler(
        lambda x, tau: velocity_native(x, torch.as_tensor(1.0 - tau)), x0, num_steps=6
    )
    assert not torch.allclose(ours, wrong, atol=1e-4)


def test_t24_constant_velocity_jacobian_is_identity():
    x = torch.randn(1, 3, 2)
    e = torch.randn(1, 3, 2)
    c = torch.tensor([0.4, -0.2])
    tau = 0.37
    g = vjp_correction(lambda xx: xx + (1 - tau) * c, x, e)
    assert torch.allclose(g, e, atol=1e-6)


def test_t25_linear_velocity_analytic_vjp():
    torch.manual_seed(3)
    n = 4
    A_mat = torch.randn(n, n)
    b = torch.randn(n)
    x = torch.randn(1, 2, n)
    e = torch.randn(1, 2, n)
    tau = 0.4
    g = vjp_correction(lambda xx: xx + (1 - tau) * (xx @ A_mat.T + b), x, e)
    expected = e + (1 - tau) * (e @ A_mat)
    assert torch.allclose(g, expected, atol=1e-5)


def test_t26_nonlinear_velocity_matches_finite_difference():
    torch.manual_seed(4)
    x = torch.randn(1, 3, 3, dtype=torch.float64)

    def field(xx: torch.Tensor) -> torch.Tensor:
        return torch.tanh(xx) + 0.3 * xx * xx

    e = torch.randn(1, 3, 3, dtype=torch.float64)

    def xhat(xx: torch.Tensor) -> torch.Tensor:
        tau = 0.35
        v = field(xx)
        return xx + (1 - tau) * v

    g = vjp_correction(xhat, x, e)

    direction = torch.randn_like(x)
    eps = 1e-6

    def phi(perturb: torch.Tensor) -> float:
        return float((e * xhat(perturb)).sum())

    fd = (phi(x + eps * direction) - phi(x - eps * direction)) / (2 * eps)
    analytic = float((g * direction).sum())
    assert abs(fd - analytic) < 1e-8


def test_t27_soft_mask_applied_once_not_squared():
    torch.manual_seed(5)
    n = 3
    A_mat = torch.randn(n, n) * 0.5
    b = torch.randn(n) * 0.1
    x = torch.randn(1, 2, n)
    e_raw = torch.randn(1, 2, n)
    W = torch.tensor([[[0.3, 0.7, 0.9], [0.1, 0.5, 0.2]]])
    tau = 0.5
    e = W * e_raw
    g = vjp_correction(lambda xx: xx + (1 - tau) * (xx @ A_mat.T + b), x, e)
    expected = e + (1 - tau) * (e @ A_mat)
    assert torch.allclose(g, expected, atol=1e-5)
    wrong = (W * W * e_raw) + (1 - tau) * ((W * W * e_raw) @ A_mat)
    assert not torch.allclose(g, wrong, atol=1e-4)

    # the guidance mask built from a previous plan also uses W exactly once
    mask = guidance_mask_from_previous_plan(
        horizon=10,
        chunk=5,
        delay=2,
        previous_valid_len=5,
        action_valid_mask=np.ones((1, 32), dtype=np.float32),
    )
    assert mask.shape == (10, 32)
    assert np.allclose(mask[6:, :], 0.0)
    assert np.allclose(mask[0, :], 1.0)


def test_t28_tau_boundary_and_single_step_no_nan():
    cap = 5.0
    assert guidance_coefficient(0.0, cap) == cap
    for tau in (0.25, 0.5, 0.75):
        w = guidance_coefficient(tau, cap)
        assert math.isfinite(w) and 0 < w <= cap
    with pytest.raises(ContractError):
        guidance_coefficient(1.0, cap)

    torch.manual_seed(6)
    x0 = torch.randn(1, 3, 2)
    velocity = _linear_velocity(0.2 * torch.eye(2), torch.zeros(2))
    sampler = FullVJPGuidance(num_steps=1, guidance_weight_cap=cap)
    target = torch.randn(1, 3, 2)
    mask = torch.ones(1, 3, 2)
    result = sampler.sample(velocity, x0, target_model=target, guidance_mask=mask)
    assert torch.isfinite(result.actions).all()


def test_t29_frozen_params_with_input_gradients():
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
    before = [p.detach().clone() for p in model.parameters()]

    def velocity_native(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        flat = x.reshape(-1, 3)
        return model(flat).reshape(x.shape)

    sampler = FullVJPGuidance(num_steps=4, guidance_weight_cap=5.0)
    x0 = torch.randn(2, 5, 3)
    target = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5, 3)
    result = sampler.sample(
        canonical_velocity_from_native(velocity_native),
        x0,
        target_model=target,
        guidance_mask=mask,
    )
    assert any(n > 0 for n in result.diagnostics.correction_norms)
    for old, new in zip(before, model.parameters()):
        assert torch.equal(old, new.detach())
        assert new.grad is None

    # A velocity/xhat path with no dependence on x_tau must fail loudly.
    with pytest.raises(ContractError):
        vjp_correction(lambda xx: torch.ones_like(xx), x0, torch.ones_like(x0))


def test_t30_outer_inference_mode_is_detected():
    if not hasattr(torch, "is_inference_mode_enabled"):  # pragma: no cover
        pytest.skip("torch version lacks inference-mode introspection")
    x0 = torch.randn(1, 3, 2)
    velocity = _linear_velocity(0.1 * torch.eye(2), torch.zeros(2))
    sampler = FullVJPGuidance(num_steps=2)
    with torch.inference_mode():
        with pytest.raises(InferenceModeError):
            sampler.sample(velocity, x0)
    # sanity: outside inference mode the same call works
    sampler.sample(velocity, x0)
    check_outer_context()


def test_t31_no_cross_step_graph_or_retained_state():
    torch.manual_seed(8)
    model = torch.nn.Linear(2, 2)
    x0 = torch.randn(1, 4, 2)
    target = torch.randn(1, 4, 2)
    mask = torch.ones(1, 4, 2)
    sampler = FullVJPGuidance(num_steps=5)
    result = sampler.sample(
        canonical_velocity_from_native(lambda x, t: model(x)), x0, target_model=target, guidance_mask=mask
    )
    assert result.actions.grad_fn is None
    assert result.actions.requires_grad is False
    with pytest.raises(RuntimeError):
        result.actions.backward()
    # repeated requests must not accumulate graphs
    for _ in range(10):
        again = sampler.sample(
            canonical_velocity_from_native(lambda x, t: model(x)),
            x0 + 0.01,
            target_model=target,
            guidance_mask=mask,
        )
        assert again.actions.grad_fn is None


def test_t32_mode_dispatch_reaches_distinct_implementations():
    torch.manual_seed(9)
    x0 = torch.randn(1, 4, 3)

    def velocity_native(x, t):
        del t
        return -0.2 * x

    def velocity_canonical(x, tau):
        del tau
        return -0.2 * x

    dispatcher = RTCDispatcher(num_steps=4)
    none_out = dispatcher.sample("none", x_init=x0, velocity_native=velocity_native)
    assert none_out.implementation == "native_flow_ode"

    approx_out = dispatcher.sample(
        "approx",
        x_init=x0,
        velocity_native=velocity_native,
        prev_model_actions=torch.zeros(1, 4, 3) + 0.5,
        executed_horizon=1,
        delay_steps=2,
    )
    assert approx_out.implementation == "rlinf_approximate_rtc"
    assert not torch.allclose(none_out.actions, approx_out.actions)

    full_out = dispatcher.sample(
        "full_vjp",
        x_init=x0,
        velocity_canonical=velocity_canonical,
        target_model=torch.zeros(1, 4, 3),
        guidance_mask=torch.ones(1, 4, 3),
    )
    assert full_out.implementation == "kinetix_full_vjp_port"
    assert full_out.diagnostics["guided"] is True

    with pytest.raises(ContractError):
        dispatcher.sample("full_vjp", x_init=x0)
    with pytest.raises(ContractError):
        dispatcher.sample("bogus", x_init=x0, velocity_native=velocity_native)


def test_prefix_weights_reference_example():
    weights = get_prefix_weights(start=2, end=6, total=10, schedule="linear")
    expected = np.asarray([1, 1, 4 / 5, 3 / 5, 2 / 5, 1 / 5, 0, 0, 0, 0], dtype=np.float32)
    assert np.allclose(weights, expected, atol=1e-6)
    zeros = get_prefix_weights(start=2, end=6, total=10, schedule="zeros")
    assert np.allclose(zeros, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])


def test_approx_sampler_matches_plain_euler_without_guidance():
    torch.manual_seed(10)
    x0 = torch.randn(1, 4, 3)

    def velocity_native(x, t):
        del t
        return 0.3 * x

    approx = sample_actions_with_rtc_guidance_approx(
        velocity_native, x0, num_steps=5, prev_model_actions=None
    )
    plain = plain_native_euler(velocity_native, x0, num_steps=5)
    assert torch.allclose(approx, plain, atol=1e-6)


def test_approx_target_mask_matches_upstream_construction():
    prev = torch.arange(1 * 6 * 2, dtype=torch.float32).reshape(1, 6, 2)
    target, mask = build_rtc_target_and_mask(prev, horizon=8, action_dim=2, delay_steps=2)
    assert torch.allclose(target[:, :6], prev)
    assert torch.allclose(target[:, 6:], torch.zeros(1, 2, 2))
    assert float(mask[0, 0, 0]) == 1.0 and float(mask[0, 1, 0]) == 1.0
    assert float(mask[0, 6, 0]) == 0.0 and float(mask[0, 7, 0]) == 0.0
    # soft weights decay (expm1 shape) and never exceed one
    assert 0 < float(mask[0, 2, 0]) < 1.0
    assert float(mask[0, 2, 0]) > float(mask[0, 5, 0])


def test_guidance_matrix_multiplies_action_mask_once():
    time_w = get_prefix_weights(2, 5, 10, "exp")
    action_mask = np.zeros((1, 32), dtype=np.float32)
    action_mask[0, :8] = 1.0
    W = build_guidance_weight_matrix(time_w, action_mask)
    assert W.shape == (10, 32)
    assert np.allclose(W[:, 8:], 0.0)
    assert np.allclose(W[:, 0], time_w)
    assert np.allclose(canonical_time_grid(5).numpy(), [0.0, 0.2, 0.4, 0.6, 0.8], atol=1e-6)
