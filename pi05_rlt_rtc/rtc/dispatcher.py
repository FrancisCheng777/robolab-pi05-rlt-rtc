"""Explicit RTC mode dispatch.

Three modes are separately named and must reach different implementations
(guide section 9.1 / T32):

- ``none``: native flow ODE, no guidance;
- ``approx``: pinned RLinf approximate guidance (regression/performance only);
- ``full_vjp``: the paper-form inference-time guidance ported in
  :mod:`pi05_rlt_rtc.rtc.full_vjp`.

There is deliberately **no automatic fallback**: requesting ``full_vjp``
without a differentiable velocity function is an error, not a reason to run
``approx``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import torch

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.rtc.approx import sample_actions_with_rtc_guidance_approx
from pi05_rlt_rtc.rtc.full_vjp import FullVJPGuidance, VJPResult

NativeVelocity = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
CanonicalVelocity = Callable[[torch.Tensor, float], torch.Tensor]


@dataclasses.dataclass
class RTCDispatchResult:
    actions: torch.Tensor
    mode: str
    implementation: str
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if self.mode not in ("none", "approx", "full_vjp"):
            raise ContractError(f"unknown RTC mode {self.mode!r}")


class RTCDispatcher:
    """Named mode dispatch with an auditable used-mode record."""

    def __init__(
        self,
        *,
        num_steps: int = 5,
        guidance_weight_cap: float = 5.0,
        approx_guidance_clip: float = 5.0,
        tau_eps: float = 0.0,
        allow_inference_mode_full_vjp: bool = False,
    ) -> None:
        self.num_steps = int(num_steps)
        self.guidance_weight_cap = float(guidance_weight_cap)
        self.approx_guidance_clip = float(approx_guidance_clip)
        self._full_vjp = FullVJPGuidance(
            num_steps=num_steps,
            guidance_weight_cap=guidance_weight_cap,
            tau_eps=tau_eps,
            allow_inference_mode=allow_inference_mode_full_vjp,
        )

    def sample(
        self,
        mode: str,
        *,
        x_init: torch.Tensor,
        velocity_native: NativeVelocity | None = None,
        velocity_canonical: CanonicalVelocity | None = None,
        target_model: torch.Tensor | None = None,
        guidance_mask: torch.Tensor | None = None,
        prev_model_actions: torch.Tensor | None = None,
        executed_horizon: int = 0,
        delay_steps: int = 0,
    ) -> RTCDispatchResult:
        if mode == "none":
            if velocity_native is None:
                raise ContractError("mode='none' requires velocity_native")
            from pi05_rlt_rtc.rtc.full_vjp import plain_native_euler

            actions = plain_native_euler(velocity_native, x_init, self.num_steps)
            return RTCDispatchResult(
                actions=actions,
                mode=mode,
                implementation="native_flow_ode",
                diagnostics={"guidance": "disabled"},
            )

        if mode == "approx":
            if velocity_native is None:
                raise ContractError("mode='approx' requires velocity_native")
            actions = sample_actions_with_rtc_guidance_approx(
                velocity_native,
                x_init,
                num_steps=self.num_steps,
                prev_model_actions=prev_model_actions,
                executed_horizon=executed_horizon,
                delay_steps=delay_steps,
                guidance_clip=self.approx_guidance_clip,
                guidance_mask=guidance_mask,
            )
            return RTCDispatchResult(
                actions=actions,
                mode=mode,
                implementation="rlinf_approximate_rtc",
                diagnostics={
                    "guidance": "approx_mean_correction",
                    "executed_horizon": int(executed_horizon),
                    "delay_steps": int(delay_steps),
                },
            )

        if mode == "full_vjp":
            if velocity_canonical is None:
                raise ContractError(
                    "mode='full_vjp' requires a differentiable canonical velocity function. "
                    "Refusing to substitute the approximate sampler (T32)."
                )
            result: VJPResult = self._full_vjp.sample(
                velocity_canonical,
                x_init,
                target_model=target_model,
                guidance_mask=guidance_mask,
            )
            return RTCDispatchResult(
                actions=result.actions,
                mode=mode,
                implementation="kinetix_full_vjp_port",
                diagnostics=result.diagnostics.to_dict(),
            )

        raise ContractError(
            f"unknown rtc mode {mode!r}; expected one of ('none', 'approx', 'full_vjp')"
        )
