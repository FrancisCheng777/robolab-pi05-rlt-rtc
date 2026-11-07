# RLinf-core AC comparison, 2026-11-03

This isolated experiment answers whether the RLinf real-world RLT AC core and
learning recipe improve this station's insertion behavior. It does not replace
A_fixed or claim an exact reproduction of RLinf's robot setup.

## Scope

- Upstream reference: RLinf c70606f08cdca259b8dec03d4430926b5b8fac9d,
  `realworld_rlt_stage2_ac_mlp.yaml`.
- Actor consumes frozen RL token, 9D proprio and reference suffix. Critic
  consumes frozen RL token, 9D proprio and candidate action. Queue/time fields
  remain in the transport for RTC identity validation but are not network inputs.
- Original MLP hidden layers, Adam, learning rates 3e-4, batch 256, Q/BC weights
  0.1/5, reference dropout 0.5, fixed std 0.002, target tau 0.005, gradient clip
  10, FP32 and Q1/min-twin-Q objectives are retained.
- Success +1, failure -1; gamma 0.96 is recomputed from primitive rewards and
  actual durations, including terminal partial chunks. Original files are immutable.
- Fresh actor, critic and optimizers, seed 1234, no BC warmup. Feed the fixed
  round-10 source list chronologically, run 8 critic updates per episode after
  the second episode, actor every fourth update. 100 episodes produce 792/198
  updates. Last-200-episode uniform-transition replay; all 100 fit in this window.
- Architecture tag `rlinf_core_rtc_v1` rejects the original queue-network weights.

## Differences that remain

This is offline replay of data collected by earlier policies, not new on-policy
rollouts. Human takeover data are unavailable. The current frozen Stage-1 model,
normalization and camera mapping are retained. Physical control remains 8D joint
actions, 9D proprio, H/C/D=10/5/5, 15Hz, full-VJP RTC. RLinf's reference setup uses
19D TCP/velocity/wrench/gripper state and 7D Cartesian actions, so these dimensions
cannot be substituted into existing data or the control interface.

The inherited queue-specific toy delayed-control performance test T60 fails
after removing actor queue inputs (0/5 against its >=3/5 requirement). This is
recorded, not hidden or treated as evidence of real-robot performance. Focused
tests verify input isolation, bounded actions, reward/discount migration,
checkpoint roundtrip and runtime/session compatibility. Real success remains
unverified until operator evaluation.

## Execution

`scripts/train_rlinf_core_comparison.py` takes a fixed `training_sources.json`,
the existing runtime config, and a new output directory. It acquires the same
session lock as the real robot, performs no robot calls, and saves source hashes,
episode audit, metrics, checkpoint and reload validation. It never auto-switches
the running policy.

`scripts/eval_pi05_rlinf_core.sh` checks completed status, uses this isolated
code directory and reuses the resident frozen teacher. The operator starts motion
with r and switches to the new actor with b. Evaluation outputs use a separate
directory.

## Completed validation

The real replay run completed in 64.38 seconds: 100 episodes (38 success,
62 failure), 3573 transitions, 792 critic and 198 actor updates. All training
metrics were finite; reloaded checkpoint predictions exactly matched the trained
network; original source hashes were unchanged. Maximum absolute normalized
action over the saved states was 0.97473.

28 distinct targeted tests passed across the focused commands. The inherited
T60 toy performance failure described above remains an explicit limitation.
Actual session policy binding and before/after-b routing were checked on 103
recorded feature states without hardware. CPU actor latency was 1.32 ms on the
first call and 0.45 ms at P95; this is not end-to-end VLA latency. The read-only
resident-teacher compatibility check passed.

On the 4090, run `bash ~/eval_pi05_rlinf_core.sh` for the new candidate or
`bash ~/eval_pi05_afixed_round10.sh` for the fixed old baseline. Exit one session
with q before starting the other. Both commands are eval-only. Reports and the
exact config/code hashes are copied under `results/` in this local experiment.
