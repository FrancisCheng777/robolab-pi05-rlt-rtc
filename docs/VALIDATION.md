# Validation — 2026-11-01

Tests ran on the existing local RTX4090 workstation, in a new
`robolab-pi05-rlt-rtc` directory. No robot, gripper or camera device was opened.
The real C++ backend was compiled against installed libfranka 0.9.0, but only
its pure core/fault-history unit tests were run. Process tests explicitly
launched the **mock-only** C++ binary on temporary Unix sockets.

| Check | Observed result |
|---|---|
| Main Python regression suite | 189 passed, 19 skipped (32.83 s in the new CUDA runtime; also passed in the CPU validation environment) |
| Stage-1 v2 synthetic data/config tests | 5 passed, Python 3.12 / isolated LeRobot environment |
| Real upstream tiny random numerical layers | 4 passed on CPU: model/RLT gradients, inference, RTC modes, tokenizer agreement |
| Real v2 dataset, read-only load | 336 contiguous windows, action `[10,32]`, state `[32]`, image masks `[true,false,false]` |
| Accepted bridge real-backend build | Compiled with libfranka 0.9.0; CTest 2/2 passed; no connection |
| Persistent bridge owner | Two mock episodes with hold heartbeats and accepted home/open; no target re-integration |
| Critical-phase replay | Pre-handoff requests excluded, terminal partial chunk handled, no cross-episode bootstrap |
| Sequential learning | BC then actor–critic; replay history, optimizer/RNG resume, atomic round publication and no duplicate committed round |
| Exploration | Seeded pre-tanh noise, bounded action, unchanged pending prefix, eval deterministic |

The 19 skips are asset/environment-gated legacy tests, including full trained
model, true RLT/VLA parameters and concurrent learner/planner cases. They are not
reported as successful GPU or hardware checks. The single-GPU session here
intentionally uses sequential learning.

CUDA validation completed on the actual RTX4090 with PyTorch 2.7.1+cu128 / CUDA
12.8. Real upstream tiny random numerical layers passed `none`, `approx` and
`full_vjp` inference with finite features and reproducible seeded outputs. The
frozen model parameters remained gradient-free. A queue actor/critic with the
production z dimension 2048 completed two BC updates and two RL updates on
explicitly mock replay. This is a numerical integration test, not full-model
memory/latency acceptance. Machine-readable output is in
[CUDA validation](../provenance/cuda_validation.json).

The documented `scripts/setup_runtime.sh` completed in a new environment
without modifying the parent OpenPI environment. Production RLinf `get_model`
and the registered Panda dataconfig import successfully without loading model
weights or opening devices; the CLI session dry run also passed.

The actual OpenPI vs Stage-1 transform audit also passed 32 synthetic samples:
state/action max absolute error 0, identical tokens, masks and images. It used
v2 exported train statistics and only the wrist image.

The current Stage-1 loader accepted the existing real v2 export without editing
it. Its 336 windows come from a small checkout recording, not a completed
100-demo training dataset. Synthetic fixtures and toy replay were explicitly
labelled mock and were not promoted to real data.

## Remaining acceptance work

- Run full π0.5 + RLT SFT on the two A800 80GB GPUs with a reviewed real dataset.
- Strict-load the resulting task checkpoint and check full-size forward/memory
  on the 4090. Tiny random weights validate code paths only.
- Profile full-size, real-input full-VJP latency against D=2 at 15Hz (~133ms),
  including camera freshness and the execution loop. A missed deadline stops
  the trial and invalidates ordinary training replay.
- Measure actual physical insertion success and time under fixed-policy
  evaluation. No success-rate or autonomous safety claim is made here.
- The human `b` handoff remains required. Automatic critical-phase recognition,
  reward classification and teleoperated intervention learning are not enabled.

## Reproduction

See the root README for commands. Main tests use the pinned project source;
Stage-1 data tests use a separate LeRobot 0.4.4 environment; model numerical tests
use the provisioned OpenPI environment. `scripts/validate_cuda.py` requires CUDA
and reports the actual GPU, torch/CUDA versions, finite numerical results and
peak allocated memory. It never substitutes CPU for GPU or opens a device.

The accepted control and collection configurations are retained verbatim;
[dependency hashes](../provenance/robolab_dependency.json) identify the reused
files. Existing control/collection worktrees and their tuned parameters were
not modified by this integration.
