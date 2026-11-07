"""True inference-time RTC guidance via vector-Jacobian products (full_vjp).

Ported math (official Kinetix ``FlowPolicy.realtime_action``,
``src/model.py`` @ ``9296f31d...``):

    x_hat_1 = x_tau + (1 - tau) * v(x_tau, tau)
    e       = W * (Y - x_hat_1)                # W applied exactly once
    g       = J_{x_hat_1}(x_tau)^T e           # vector-Jacobian product
    w_tau   = min(beta, (1 - tau) / (tau * r_tau^2))
    v_guided = v + w_tau * g
    x_next  = x_tau + dtau * v_guided

Gradient boundary (guide section 9.4):

1. prefix/KV are computed once, outside this function, under ``no_grad``;
2. each step detaches the noisy action and makes it a fresh differentiable leaf;
3. ``autograd.grad`` with ``create_graph=False`` yields the VJP; the cotangent
   ``e`` is detached;
4. the update is detached and the per-step graph is released;
5. model parameters are frozen in Stage 2 and never receive gradients, while
   the noisy-action input gradient must exist.

The outer context must not be ``torch.inference_mode()``: ``enable_grad()``
cannot undo it, and the resulting guidance would be silently wrong (T30).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import numpy as np
import torch

from pi05_rlt_rtc.contracts.errors import ContractError, InferenceModeError
from pi05_rlt_rtc.rtc.canonical import canonical_time_grid, guidance_coefficient

VelocityCanonical = Callable[[torch.Tensor, float], torch.Tensor]


@dataclasses.dataclass
class VJPDiagnostics:
    mode: str
    num_steps: int
    guidance_weight_cap: float
    taus: list[float]
    guidance_coefficients: list[float]
    correction_norms: list[float]
    input_grad_norms: list[float]
    value_norms: list[float]
    guided: bool
    finite: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class VJPResult:
    actions: torch.Tensor  # [B, H, 32], detached
    diagnostics: VJPDiagnostics

    def max_abs(self) -> float:
        return float(self.actions.abs().max().item()) if self.actions.numel() else 0.0


def check_outer_context(*, allow_inference_mode: bool = False) -> None:
    """Refuse to run full_vjp when autograd input gradients are impossible."""
    if not allow_inference_mode and torch.is_inference_mode_enabled():
        raise InferenceModeError(
            "full_vjp cannot run under torch.inference_mode(): enable_grad() does not "
            "restore autograd for inference tensors. Move the sampler out of the "
            "inference-mode context instead of silently falling back to approx."
        )


def plain_canonical_euler(
    velocity_canonical: VelocityCanonical,
    x_init: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """No-guidance canonical Euler used as the exact degeneration oracle (T21)."""
    if x_init.ndim != 3:
        raise ContractError(f"x_init must be [B, H, A], got {tuple(x_init.shape)}")
    dt = 1.0 / float(num_steps)
    taus = canonical_time_grid(num_steps)
    x = x_init.detach().clone().to(torch.float32)
    with torch.no_grad():
        for tau in taus:
            v = velocity_canonical(x, float(tau))
            if not torch.isfinite(v).all():
                raise ContractError(f"velocity is non-finite at tau={float(tau)}")
            x = x + dt * v
    return x.detach()


def plain_native_euler(
    velocity_native: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_init: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """Pinned RLinf RTC sampler Euler loop (``get_timesteps``, N updates)."""
    from pi05_rlt_rtc.rtc.approx import rlinf_timesteps

    timesteps = rlinf_timesteps(num_steps, device=x_init.device)
    dt = 1.0 / float(num_steps)
    x = x_init.detach().clone().to(torch.float32)
    with torch.no_grad():
        for idx in range(num_steps):
            t = timesteps[idx].to(x.dtype)
            v = velocity_native(x, t.expand(x.shape[0]))
            x = x - dt * v
    return x.detach()


def vjp_correction(
    xhat_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    error: torch.Tensor,
) -> torch.Tensor:
    """``J^T e`` for a differentiable ``xhat_fn`` evaluated at ``x``.

    Exposed for the analytic/VJP unit tests (T24-T27).  ``error`` must already
    contain the mask (applied once) and is detached.
    """
    x_leaf = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        xhat = xhat_fn(x_leaf)
    try:
        (g,) = torch.autograd.grad(
            xhat,
            x_leaf,
            grad_outputs=error.detach(),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
    except RuntimeError as exc:  # pragma: no cover - message wrapper
        raise ContractError(
            "full_vjp requires a differentiable noisy-action path; autograd could not "
            f"differentiate x_hat wrt x_tau: {exc}"
        ) from exc
    return g.detach()


class FullVJPGuidance:
    """Per-step VJP guidance sampler with no cross-step autograd graph."""

    def __init__(
        self,
        *,
        num_steps: int = 5,
        guidance_weight_cap: float = 5.0,
        tau_eps: float = 0.0,
        allow_inference_mode: bool = False,
    ) -> None:
        if num_steps <= 0:
            raise ContractError("num_steps must be > 0")
        if guidance_weight_cap <= 0:
            raise ContractError("guidance_weight_cap must be > 0")
        self.num_steps = int(num_steps)
        self.guidance_weight_cap = float(guidance_weight_cap)
        self.tau_eps = float(tau_eps)
        self.allow_inference_mode = bool(allow_inference_mode)

    def sample(
        self,
        velocity_canonical: VelocityCanonical,
        x_init: torch.Tensor,
        *,
        target_model: torch.Tensor | None = None,
        guidance_mask: torch.Tensor | None = None,
        return_diagnostics: bool = True,
    ) -> VJPResult:
        check_outer_context(allow_inference_mode=self.allow_inference_mode)
        if x_init.ndim != 3:
            raise ContractError(f"x_init must be [B, H, A], got {tuple(x_init.shape)}")
        if not torch.isfinite(x_init).all():
            raise ContractError("x_init contains NaN/Inf")
        B, H, A = x_init.shape

        guided = target_model is not None and guidance_mask is not None
        if (target_model is None) != (guidance_mask is None):
            raise ContractError(
                "target_model and guidance_mask must be provided together; a partial "
                "RTC context is a bug, not a reason to fall back"
            )
        if guided:
            if target_model.shape != x_init.shape or guidance_mask.shape != x_init.shape:
                raise ContractError(
                    "RTC target/mask must match x_init shape: "
                    f"target={tuple(target_model.shape)} mask={tuple(guidance_mask.shape)} "
                    f"x={tuple(x_init.shape)}"
                )
            if not torch.isfinite(target_model).all() or not torch.isfinite(guidance_mask).all():
                raise ContractError("RTC target/mask contains NaN/Inf")
            # `any()` on a detached bool tensor is fine; used only to pick the path.
            guided = bool((guidance_mask.detach() != 0).any().item())

        diagnostics = VJPDiagnostics(
            mode="full_vjp",
            num_steps=self.num_steps,
            guidance_weight_cap=self.guidance_weight_cap,
            taus=[],
            guidance_coefficients=[],
            correction_norms=[],
            input_grad_norms=[],
            value_norms=[],
            guided=guided,
            finite=True,
        )

        if not guided:
            # Exact degeneration: same plain Euler function used by the no-RTC path.
            actions = plain_canonical_euler(velocity_canonical, x_init, self.num_steps)
            diagnostics.finite = bool(torch.isfinite(actions).all().item())
            return VJPResult(actions=actions, diagnostics=diagnostics)

        dt = 1.0 / float(self.num_steps)
        x = x_init.detach().clone().to(torch.float32)
        target = target_model.detach().to(torch.float32)
        mask = guidance_mask.detach().to(torch.float32)

        for idx in range(self.num_steps):
            tau = float(idx) / float(self.num_steps)
            x_leaf = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                v = velocity_canonical(x_leaf, tau)
                if not torch.isfinite(v).all():
                    raise ContractError(f"velocity is non-finite at tau={tau}")
                xhat = x_leaf + (1.0 - tau) * v
            e = (mask * (target - xhat)).detach()
            try:
                (g,) = torch.autograd.grad(
                    xhat,
                    x_leaf,
                    grad_outputs=e,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
            except RuntimeError as exc:
                raise ContractError(
                    "full_vjp requires a differentiable noisy-action path; autograd "
                    f"could not differentiate x_hat wrt x_tau at tau={tau}: {exc}"
                ) from exc
            w = guidance_coefficient(tau, self.guidance_weight_cap, eps=self.tau_eps)
            v_guided = v.detach() + w * g.detach()
            if not torch.isfinite(v_guided).all():
                raise ContractError(
                    f"guided velocity is non-finite at tau={tau}; refusing to clamp away "
                    "an upstream error"
                )
            with torch.no_grad():
                x = x_leaf.detach() + dt * v_guided
            # No cross-step graph: `x` is a plain detached tensor.
            diagnostics.taus.append(tau)
            diagnostics.guidance_coefficients.append(w)
            diagnostics.correction_norms.append(float(g.detach().norm().item()))
            diagnostics.input_grad_norms.append(float(g.detach().norm().item()))
            diagnostics.value_norms.append(float(v.detach().norm().item()))

        diagnostics.finite = bool(torch.isfinite(x).all().item())
        if not diagnostics.finite:
            raise ContractError("full_vjp produced non-finite actions")
        if x.requires_grad or x.grad_fn is not None:
            raise ContractError(
                "full_vjp leaked an autograd graph across denoising steps (T31)"
            )
        return VJPResult(actions=x.detach(), diagnostics=diagnostics)


def zeros_guidance_mask(shape: tuple[int, int, int], *, device: Any = "cpu") -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float32, device=device)


def numpy_to_tensor(array: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=dtype)
