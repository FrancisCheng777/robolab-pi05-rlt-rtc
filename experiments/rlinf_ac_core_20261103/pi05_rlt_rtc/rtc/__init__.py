"""Inference-time RTC guidance: canonical math, masks and mode dispatch."""

from pi05_rlt_rtc.rtc.approx import (
    ApproxRTCContext,
    build_rtc_target_and_mask,
    rlinf_timesteps,
    sample_actions_with_rtc_guidance_approx,
)
from pi05_rlt_rtc.rtc.canonical import (
    canonical_time_grid,
    canonical_velocity_from_native,
    guidance_coefficient,
    native_time_grid,
)
from pi05_rlt_rtc.rtc.dispatcher import RTCDispatcher, RTCDispatchResult
from pi05_rlt_rtc.rtc.full_vjp import (
    FullVJPGuidance,
    VJPDiagnostics,
    VJPResult,
    check_outer_context,
    plain_canonical_euler,
    plain_native_euler,
    vjp_correction,
)
from pi05_rlt_rtc.rtc.prefix_weights import (
    VALID_SCHEDULES,
    PrefixSchedule,
    build_guidance_weight_matrix,
    get_prefix_weights,
    guidance_mask_from_previous_plan,
)

__all__ = [
    "ApproxRTCContext",
    "FullVJPGuidance",
    "PrefixSchedule",
    "RTCDispatcher",
    "RTCDispatchResult",
    "VALID_SCHEDULES",
    "VJPDiagnostics",
    "VJPResult",
    "build_guidance_weight_matrix",
    "build_rtc_target_and_mask",
    "canonical_time_grid",
    "canonical_velocity_from_native",
    "check_outer_context",
    "get_prefix_weights",
    "guidance_coefficient",
    "guidance_mask_from_previous_plan",
    "native_time_grid",
    "plain_canonical_euler",
    "plain_native_euler",
    "rlinf_timesteps",
    "sample_actions_with_rtc_guidance_approx",
    "vjp_correction",
]
