# 4090 Stage-2 preparation — 2026-11-02

Integration note (2026-11-07): the station observations below are historical,
not validation of the current commit. Current CPU/mock checks do not validate
GPU latency or physical home stability. The station used D=5; the repository
template remains D=2 and requires measured acceptance before deployment.
CUDA Graphs, an independent Actor device override, reward-comparison tools and
open-before-home remain archived/deferred. The main home path holds, returns
home, then opens the gripper. No robot or real learner was started in this audit.

The operator reports three physical trials: one complete insertion, two near-hole
failures. This supports preparing a small Stage-2 pilot, not a reliable success-rate
estimate. No real-policy RL update or automatic robot trial was run during setup.

## Preserved inference

`/home/heinrich/robolab-pi05-stage2-prep` is a separate checkout. The existing
`~/run_pi05_stage1_8000.sh eval` and its resident teacher are unchanged.
The new client reuses that model's authenticated loopback RPC. π0.5 and the trained
RLT module remain frozen. Only the local MLP actor/critic are trainable; the remote
teacher never receives actor weights. Policy IDs in captured features are explicitly
rebound to the local actor; the teacher's own identity remains fixed on the wire.

Teacher actions remain in charge before `b`; after `b`, a trained actor supplies
chunks. During the initial labelled-data phase, the teacher remains in charge.
Reference targets and features continue to use full-VJP RTC, H/C/D=10/5/5, five
steps, 15Hz, and bounded reference lead <=0.05rad. Gains and limits are unchanged.

## Commands (prepared, not started)

```sh
bash ~/run_pi05_stage2.sh check                    # no hardware, no training
bash ~/run_pi05_stage2.sh collect --execute        # initial teacher labels, no updates
bash ~/run_pi05_stage2.sh train --execute          # held updates + next trial
```

Both launchers share the same session lock. Exit Stage-1 before using either
Stage-2 hardware command. `collect` and `train` share a persistent session on the
500GB SSD: `/media/heinrich/SANDISK SSD/pi05-stage2/plug_rlt_20261102`.
A train launch reuses committed labelled episodes and resumes prior rounds.
It never replaces a committed round; interrupted pending rounds are not promoted.

`r` starts; `b` marks the insertion-critical phase; `c` means actual successful
insertion; `a` means confirmed task failure; `e` aborts/holds then home+open;
`x` holds and cancels home; `q` closes hardware. Label with `a/c` BEFORE `e` if a
trial is to enter RL replay. Aborted/faulted/unlabelled trials are excluded.
At inspection, all six saved Stage-1 labels were ineligible. The user's verbal
success report was not converted into an invented per-trial training label.
Stage-1 trials outside the Stage-2 session are not silently imported.

## Initial recipe

Matches the pinned RLinf real-world AC scalar recipe already audited in
`RLINF_PARAMETER_AUDIT.md`: batch256, actor/critic LR3e-4, critic:actor updates4:1,
Q weight0.1, BC weight5, reference dropout0.5, target tau0.005, gradient clip10,
pre-tanh exploration std0.002. The existing local sparse reward is success1,
otherwise0; per-control-tick gamma0.99 with duration-aware discounting.

The current template batches ten NEW eligible episodes with at least 32 total
learnable transitions, then 1000 critic / 250 actor updates; BC warmup100 runs
only for the first batch. Existing configs without update_every_episodes retain
per-episode scheduling. Config derivation preserves the accepted source settings.
These scheduling numbers are local engineering choices, not verbatim RLinf settings.
Updates run only between rollouts while held; they do not run concurrently with
physical model inference. Obtain some genuine successes as well as failures.

## Communication fault and pending station work

The failure log records Franka `communication_constraints_violation`, 0.779 command
success rate and five consecutive lost packets. Python then saw stale robot state
50.5ms against50ms. The physical trials before home had zero planner/deadline errors.
The current evidence does not identify the gripper's own fault register/LED cause.

The station has PREEMPT_RT and a1Gbps full-duplex wired robot link, with no interface
error/drop counters. However the earlier temporary collection settings are absent:
Robotiq FTDI latency16ms (accepted preparation1ms); network ksoftirqd/9 SCHED_OTHER0
(previous preparation SCHED_FIFO50). The old inference launcher also omitted the
collection launcher's CPU12 control-thread binding. Stage-2 restores the CPU binding
and excludes CPU12 from its client/learner workers. The existing model is untouched.

After exiting bridge, restore the prior temporary settings using:

```sh
bash ~/prepare_pi05_stage2_station.sh
```

This invokes the repository's existing reversible scripts, requires interactive
sudo, refuses an active bridge, and sends no motion commands. Automated setup had
no passwordless sudo and did not modify these settings. Stage-2 hardware launch
checks the known station prerequisites and refuses if they remain absent. Restoring
settings is not proof the fault is fixed: operator-supervised repeated home/open and
communication checks remain necessary. Do not raise the stale-state threshold.

## Verification and remaining work

CPU regression tests cover remote-teacher identity, numerical parity with the local actor path,
critical-only switching, seeded exploration, immutable teacher targets, learner
cadence and checkpoint reload. The production2048-D / batch256 synthetic CPU smoke
uses4 BC warmup and8 critic/2 actor updates; all saved metrics finite and actor
reload identical. It is marked mock and cannot serve as a physical actor checkpoint.
Actor725544 parameters; online twin critic1516034 parameters. The smoke's complete
saved state/replay occupied33.3MB; real replay and multiple rounds will add storage.

Pending before online promotion: restore and verify communication/home stability;
collect eligible labelled episodes; measure the combined real teacher+GPU actor path
while the current physical session is stopped; inspect first BC/AC losses and actor
outputs. No second VLA or real GPU learner was launched while the user evaluated.
