# Deferred runtime changes — 2026-11-07

This branch preserves the uncommitted acceleration and home-order changes from
`robolab-pi05-stage2-prep` as a draft, based on the CPU-validated stable integration.
It is not approved for deployment or merge.

- Frozen vision CUDA Graph capture/replay and planner timing instrumentation.
- Independent CPU Actor inference device selection while the teacher uses CUDA.
- Home command opens the gripper at the held pose before returning home.

Required before promotion: compare CUDA graph outputs with the same unaccelerated
model and recorded inputs; test repeated replay, input replacement and output
isolation; run production full-VJP latency/underrun acceptance; compare actor device
outputs and checkpoint identity; verify controlled open-before-home cancellation,
opening failure and hold transitions on the prepared station with an operator.
The 4090 host was unreachable during this audit. CPU tests do not satisfy these gates.

Stable main keeps home-then-open and the original Actor device. Reward comparisons,
EEF/GELLO takeover and other independent experimental overlays remain under
`codex/archive/*`; their historical test reports are not new validation.
