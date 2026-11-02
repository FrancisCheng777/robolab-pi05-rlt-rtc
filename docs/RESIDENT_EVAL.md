# Stage-1 resident inference and operator lifecycle

On the configured 4090 host:

```sh
bash ~/run_pi05_stage1_8000.sh eval
```

This reuses (or first loads and warms) a local model-only service, starts the
accepted bridge, then enters a held operator session. It never starts an episode
without `r`. It remains Stage-1 teacher evaluation with inference-time full-VJP
RTC, H10/C5/D5, five denoising steps at 15Hz. No Stage-2 update is performed.

- `r`: start a trial when confirmed held. Keys pressed during home are not queued.
- `e`: stop policy commands, confirm hold, return using the existing collection
  `arm_home` trajectory and open the gripper, then wait. It does not label success.
- `x`: hold/abort, including during home; cancel pending home.
- `q`: hold and close the camera/control-owner/managed bridge. Clean the abandoned
  socket only after the bridge exits. The next `eval` creates a new hardware session.
- `b`, `a`, `c`: retain critical-phase / failure / success labels.

The keyboard reader runs independently of inference, camera capture and rollout
serialization. A stop latches before waiting for an in-flight owner RPC. Further
policy actions are rejected until explicit `r`. The CPU owner's existing action
watchdog remains active; software keys are not a replacement for the physical stop.
Terminal flow control is disabled during the session and restored on normal exit.
SIGHUP/SIGTERM/EOF close the owned hardware session. Live/unowned sockets are never
blindly removed, and a global launcher lock prevents duplicate managed sessions.

`q` deliberately retains the **model-only** process on loopback so another `eval`
does not reload weights or repeat CUDA warmup. It holds no robot/camera connection.
The full first startup was measured at 86 seconds on this host; later model RPC
connections measured 3–6ms (this excludes bridge/camera startup). Model/source/config
changes invalidate reuse. Reboot or `stop-model` also requires a new initial load.

```sh
bash ~/run_pi05_stage1_8000.sh status
bash ~/run_pi05_stage1_8000.sh stop-model  # release GPU memory when done
bash ~/run_pi05_stage1_8000.sh prewarm    # model only, never opens hardware
```

Logs: `~/.cache/pi05-stage1-8000/model.log`, `bridge.log`; trial records remain in
`outputs/stage1_8000_eval`. The model state file contains a local RPC credential;
do not share it. The service binds only 127.0.0.1 and runs CUDA load/warmup/planning
on one executor thread. Planning states and identity are preserved over JSON RPC.
Remote sessions are limited to teacher eval; Stage-2 updates require a separate
validated training path.

## Tracking correction

A policy delta is realized motion `(q[t+1]-q[t])/0.05`, whereas the bridge accepts
an absolute target. Replacing that target with `current_measured_q + delta` every
15Hz tick discarded residual motion when the torque servo lagged. The arm could
stall before reaching the visual state where the policy closes the gripper. The
prompt and gripper mapping were present; offline recorded states predicted closure.

`bounded_reference` retains the reference once per **new** policy action. Its lead
is bounded by min(0.05rad, calibrated tracking limit), and saturation is discarded.
The 100Hz heartbeat only refreshes/rebounds; it does not reapply the action. Hold
clears the reference. No gain/speed/collision-limit changes are made. The pure
helper in `adapters/policy_reference.py` mirrors the reviewed helper introduced by
robolab-control-system commit `87996db`; GELLO's absolute targets do not use it.

## Verification and limits

- Control repository: mock C++ build/core test, 115 Python tests passed, nine skipped.
- Runtime suite: 248 passed, 19 skipped. Real C++ mock confirms `e/x/q`
  hold even when the main thread does not poll for 500ms (beyond the action watchdog).
- Full 8000-step model, recorded validation observations: 900 ticks plus a second
  120-tick RPC client, zero deadline misses, underruns, rejected plans or planner
  errors. P95 228ms and max248ms against333ms; predicted closure at tick105.
- These are offline results. The operator previously reported grasp, placement and
  insertion attempt with bounded reference; complete insertion and the new physical
  keyboard lifecycle still need operator evaluation. One trial is not a success-rate
  estimate or sufficient evidence to launch Stage-2 training automatically.
