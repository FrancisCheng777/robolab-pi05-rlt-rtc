"""Canonical (noise -> data) time coordinate and guidance coefficients.

The official RTC reference uses ``tau: 0 -> 1`` (noise to data):

    x_hat_1 = x_tau + (1 - tau) * v(x_tau, tau)
    r_tau^2 = (1 - tau)^2 / ((1 - tau)^2 + tau^2)
    w_tau   = min(beta, (1 - tau) / (tau * r_tau^2))

The pinned RLinf pi0.5 sampler uses the opposite direction
(``t: 1 -> 0``, ``dt = -1/N``).  The adapter here is
``t_native = 1 - tau`` and ``v_canonical = -v_native`` (guide section 9.3).
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from pi05_rlt_rtc.contracts.errors import ContractError


def canonical_time_grid(num_steps: int) -> torch.Tensor:
    """``tau_i = i / N`` for ``i = 0..N-1`` (tau = 1 is never evaluated)."""
    if num_steps <= 0:
        raise ContractError(f"num_steps must be > 0, got {num_steps}")
    return torch.arange(num_steps, dtype=torch.float32) / float(num_steps)


def native_time_grid(num_steps: int) -> torch.Tensor:
    """Pinned RLinf native Euler grid ``[1, (N-1)/N, ..., 1/N, 0]``."""
    from pi05_rlt_rtc.rtc.approx import rlinf_timesteps

    return rlinf_timesteps(num_steps)


def guidance_coefficient(tau: float, cap: float, *, eps: float = 0.0) -> float:
    """``w_tau`` with finite boundary handling (T28).

    Exact reference-code arithmetic::

        inv_r2 = (t^2 + (1-t)^2) / (1-t)^2
        c      = (1-t)/t        (posinf -> cap)
        w      = min(c * inv_r2, cap)

    At ``tau = eps = 0`` the reference produces ``c = posinf``; ``nan_to_num``
    turns it into ``cap`` so ``w = cap``.  We implement that branch directly.
    ``tau >= 1`` returns 0 (the loop never evaluates it; calling it is a bug we
    surface instead of returning Inf).
    """
    if cap <= 0:
        raise ContractError(f"guidance weight cap must be > 0, got {cap}")
    t = float(tau)
    if t <= float(eps):
        return float(cap)
    if t >= 1.0:
        raise ContractError(
            f"guidance coefficient evaluated at tau={t}; the canonical loop must stop "
            "before tau = 1 (T28)"
        )
    inv_r2 = (t * t + (1.0 - t) * (1.0 - t)) / ((1.0 - t) * (1.0 - t))
    c = (1.0 - t) / t
    w = min(c * inv_r2, cap)
    if not math.isfinite(w):
        raise ContractError("guidance coefficient produced a non-finite value")
    return float(w)


def canonical_velocity_from_native(
    velocity_native: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor, float], torch.Tensor]:
    """Adapt ``v_native(x, t_native)`` to ``v_canonical(x, tau) = -v_native(x, 1-tau)``.

    The returned callable keeps the autograd graph of ``velocity_native`` with
    respect to ``x``; this is what makes the full-VJP path work while model
    parameters stay frozen.
    """

    def velocity_canonical(x: torch.Tensor, tau: float) -> torch.Tensor:
        t_native = 1.0 - float(tau)
        t_tensor = torch.as_tensor(t_native, dtype=x.dtype, device=x.device)
        if t_tensor.ndim == 0:
            t_tensor = t_tensor.expand(x.shape[0])
        v = velocity_native(x, t_tensor)
        if v.shape != x.shape:
            raise ContractError(
                f"native velocity returned {tuple(v.shape)} for input {tuple(x.shape)}"
            )
        return -v

    return velocity_canonical
