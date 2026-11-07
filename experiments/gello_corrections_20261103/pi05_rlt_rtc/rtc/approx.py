"""Faithful port of the pinned RLinf *approximate* RTC sampler.

Source: ``rlinf/models/embodiment/openpi/sampling/rtc_guidance.py`` @
``c70606f08cdca259b8dec03d4430926b5b8fac9d``.  The behaviour is retained for
regression/performance comparison and must always be reported as
"RLinf approximate RTC", never as the paper-form inference-time RTC.

Upstream properties preserved here:

- Euler loop over ``rl_sampler.get_timesteps`` (native ``t: 1 -> 0``);
- soft overlap mask built by ``build_rtc_target_and_mask`` (time-only mask);
- guidance applied to the flow-ODE mean with
  ``scale = clamp((1 - tau) / (tau + 1e-4), max = guidance_clip)``;
- ``paper_tau`` advances ``0 -> 1`` while native ``t`` goes ``1 -> 0``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable

import torch

from pi05_rlt_rtc.contracts.errors import ContractError

NativeVelocity = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def rlinf_timesteps(num_steps: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """``[1, (N-1)/N, ..., 1/N, 0]`` (length ``N + 1``), see upstream ``rl_sampler``."""
    if num_steps <= 0:
        raise ContractError(f"num_steps must be > 0, got {num_steps}")
    steps = torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device)
    return torch.cat([steps, torch.zeros(1, device=device)])


@dataclasses.dataclass
class ApproxRTCContext:
    """Upstream-compatible RTC context (see ``RTCGuidanceContext``)."""

    prev_model_actions: torch.Tensor | None = None  # [B, H, 32]
    executed_horizon: int = 0
    delay_steps: int = 0

    def get_prev_remaining(self) -> torch.Tensor | None:
        if self.prev_model_actions is None:
            return None
        executed = int(max(self.executed_horizon, 0))
        if executed >= self.prev_model_actions.shape[1]:
            return None
        return self.prev_model_actions[:, executed:, :]


def build_rtc_target_and_mask(
    prev_remaining: torch.Tensor | None,
    horizon: int,
    action_dim: int,
    delay_steps: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact port of upstream ``build_rtc_target_and_mask`` (time-only mask)."""
    batch_size = 1 if prev_remaining is None else prev_remaining.shape[0]
    target = torch.zeros((batch_size, horizon, action_dim), device=device, dtype=dtype)
    mask = torch.zeros((batch_size, horizon, 1), device=device, dtype=dtype)
    if prev_remaining is None or prev_remaining.numel() == 0:
        return target, mask
    overlap = min(prev_remaining.shape[1], horizon)
    if overlap <= 0:
        return target, mask
    target[:, :overlap] = prev_remaining[:, :overlap].to(device=device, dtype=dtype)
    hard_end = min(max(int(delay_steps), 0), overlap)
    if hard_end > 0:
        mask[:, :hard_end, 0] = 1.0
    if hard_end < overlap:
        i = torch.arange(hard_end, overlap, device=device, dtype=dtype)
        denom = max(float(overlap - hard_end + 1), 1.0)
        c_i = (overlap - i) / denom
        soft = c_i * (torch.expm1(c_i) / (math.e - 1.0))
        mask[:, hard_end:overlap, 0] = soft
    return target, mask


def sample_actions_with_rtc_guidance_approx(
    velocity_native: NativeVelocity,
    x_init: torch.Tensor,
    *,
    num_steps: int,
    prev_model_actions: torch.Tensor | None = None,
    executed_horizon: int = 0,
    delay_steps: int = 0,
    guidance_clip: float = 5.0,
    guidance_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Euler sampling with the upstream approximate overlap guidance.

    Implemented from the upstream source with ``noise_method='flow_ode'`` and
    ``noise_level=0``, i.e. the deterministic mean update
    ``x_mean = x0_pred * (1 - (t - delta)) + x1_pred * (t - delta)`` which is
    algebraically identical to ``x - delta * v``.
    """
    if x_init.ndim != 3:
        raise ContractError(f"x_init must be [B, H, A], got {tuple(x_init.shape)}")
    if guidance_clip <= 0:
        raise ContractError("guidance_clip must be > 0")
    timesteps = rlinf_timesteps(num_steps, device=x_init.device)
    B = x_init.shape[0]
    x = x_init.detach().clone().to(torch.float32)

    context = ApproxRTCContext(
        prev_model_actions=None if prev_model_actions is None else prev_model_actions.detach(),
        executed_horizon=executed_horizon,
        delay_steps=delay_steps,
    )
    prev_remaining = context.get_prev_remaining()
    target, mask = build_rtc_target_and_mask(
        prev_remaining=prev_remaining,
        horizon=x_init.shape[1],
        action_dim=x_init.shape[2],
        delay_steps=delay_steps,
        device=x_init.device,
        dtype=x.dtype,
    )
    if guidance_mask is not None:
        if guidance_mask.shape != x.shape or not torch.isfinite(guidance_mask).all() or (guidance_mask < 0).any():
            raise ContractError("approx guidance mask must be finite, nonnegative and match x_init")
        # Explicit project mask includes both time validity and action padding.
        # Without it, retain the pinned upstream time-only behavior.
        mask = guidance_mask.detach().to(device=x.device, dtype=x.dtype)

    paper_tau = 0.0
    dt = 1.0 / float(num_steps)
    with torch.no_grad():
        for idx in range(num_steps):
            t = timesteps[idx].to(x.dtype)
            t_expanded = t.expand(x.shape[0])
            v_t = velocity_native(x, t_expanded)
            delta = (timesteps[idx] - timesteps[idx + 1]).to(x.dtype).expand_as(x)
            x0_pred = x - v_t * t
            x1_pred = x + v_t * (1 - t)
            x_mean = x0_pred * (1 - (t - delta)) + x1_pred * (t - delta)
            if prev_remaining is not None and prev_remaining.numel() > 0:
                guidance_term = (target - x_mean) * mask
                # Upstream: torch.clamp((1 - paper_tau) / max(paper_tau, 1e-4), max=clip).
                # Implemented with Python floats because torch.clamp rejects a Python
                # scalar input; numeric behaviour is identical.
                guidance_scale = min(
                    (1.0 - paper_tau) / max(paper_tau, 1e-4), float(guidance_clip)
                )
                x_mean = x_mean + guidance_scale * guidance_term
            # flow_ode => x_t_std == 0, so the update is deterministic.
            x = x_mean
            paper_tau += dt
    if not torch.isfinite(x).all():
        raise ContractError("approximate RTC sampler produced non-finite actions")
    return x
