# 4090 Stage-2 preparation — 2026-11-02

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

The small Actor runs on CPU (`model.actor_inference_device=cpu`) while teacher
inference and AC training remain on the4090. A same-checkpoint comparison over
recorded states measured first-use GPU Actor latency286.5ms versus CPU2.2ms;
deterministic action disagreement was at most8.94e-8. This removes an observed
critical-handoff initialization spike, not all possible teacher-service stalls.
Published checkpoint directories are immutable: the client reuses the same Actor
between trials, and announces new weights only when loading a different round.
After this change, recorded-observation replay passed600 ticks (max241ms), then
stopped on tick460 of a second600-tick trial. Service timing localized that miss
to teacher prefix preparation/cache construction254.5ms, with331.6ms total
service time and345.4ms end-to-end planning. No GC pause occurred in that request.
Teacher latency spikes remain unresolved; do not claim uninterrupted real-time
operation from the first passing replay. Optional `PI05_PROFILE_PLANNER=1`
adds data-free service/component timings without changing actions or limits.

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

User-selected rollout scheduling: each 10 NEW eligible episodes and at least32
accumulated learnable transitions triggers one held update batch. The first batch
adds BC warmup100; each batch runs1000 critic/250 actor updates on cumulative replay.
`session.warmup_episodes=10`, `update_every_episodes=10`, and
`updates_per_episode=100`. Aborted trials do not advance the count; a published
batch is not trained again after restart. The next batch requires 10 additional
eligible episodes. Terminal notices show progress, readiness, elapsed training
time, and when the next actor is loaded. No mid-episode weight swap occurs.
These scheduling numbers are local engineering choices, not verbatim RLinf settings.
Updates run only between rollouts while held; they do not run concurrently with
physical model inference. Obtain some genuine successes as well as failures.

A hardware-free 4090 benchmark using production2048-D cached synthetic features,
batch256 and the full100 BC +1000 critic/250 actor workload took99.8 seconds:
5.1s BC,91.3s AC,0.2s saving, plus startup. This benchmark has only8 synthetic
transitions and is not evidence of learning on the physical task. Allow roughly
1–3 minutes for an initial real batch; its actual load/save and training time is
reported separately in `batch_status.json`. Synthetic weights are kept outside
the real session and are never promoted.

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

## Frozen vision acceleration (2026-11-02)

Observed pre-critical queue underruns were planner deadline misses, not actor
publication: trial 41 needed 436 ms for a request with a 333 ms (5/15 s) budget.
Fine offline timings localized the variable cost to eager vision/prefix execution,
not image preprocessing or GC. Keep H/C/D=10/5/5, 15 Hz, full-VJP and five denoising
steps; do not accept late actions by merely increasing a timeout.

`model.vision_cuda_graph: true` records and replays the **same** frozen SigLIP
operations. It does not compile different operators, change precision, skip masked
cameras, change weights or alter the feature identity. Startup warms and verifies
bitwise output equality before the service is ready. Every request copies current
inputs, checks the tensor layout, and returns independent output tensors. The
whole prefix cannot be captured unchanged because its constant mask construction
performs a host-to-device copy; acceleration is deliberately limited to vision.
Execution acceleration is reported separately in resident metadata. Existing
Stage-2 Actor/replay contracts are unchanged. Shape changes fail explicitly.

Recorded-observation probe: 7 complete prefix/KV comparisons were bitwise equal;
360 full teacher requests had P50 100.2 ms, P95 143.6 ms, maximum 162.9 ms and zero
requests over 333 ms. This probe excludes robot hardware and does not establish
physical task success. Full resident/queue acceptance is recorded separately.

The first resident queue acceptance also exposed a separate client-side failure:
a generation-2 cyclic GC scan paused Python for 78.7 ms and overran a control tick,
with zero planner deadline misses. `WallClockController.run` now defers automatic
cyclic GC for the duration of the episode and restores its previous enabled/disabled
state after stop, including exceptions. Ordinary reference-counted memory release
continues. GC remains available between episodes; no timing or robot safety limit
was relaxed. Regression tests cover normal exit, environment failure and callers
that already disabled GC. This requires a new collection process, not a hot patch.

Final resident acceptance after both fixes: 3 × 1200 control ticks with the real
resident teacher, second-round CPU Actor, full-VJP RTC, simulated wall-clock I/O
and recorded real observations; zero deadline misses, underruns, rejected plans,
planner errors or safety stops. Worst request was 219.7 ms; no measured cyclic-GC
pauses. The GPU service remains resident and the session lock is released.
Raw evidence: `outputs/underrun-diagnosis/1793628671459695248/report.json` on 4090.
CPU suite: 266 passed/21 skipped; the three CUDA graph tests also passed separately.
This is offline timing acceptance; the operator must still verify actual trials.

### Fourth-round reward comparison (2026-11-02)

The requested final comparison uses terminal success +1 / failure -1, replacing
an abandoned +2 / -0.5 proposal. Both comparison branches resume round 3 and use
the same 40 fourth-round source episodes, 1000 critic / 250 actor updates, and
preserve optimizer/RNG state. Historical weights and source labels stay intact.

An audit found 21/40 original replays omitted the terminal transition when the
next planner request was pending at human termination. The assembler now accepts
the last request's actually committed terminal tail up to D+C ticks, validates
command provenance, and disables terminal bootstrapping. It does not invent next
states or executed commands. Eligible labels require one learnable terminal.
`comparisons/round4_repaired_labels/manifest.json` records derived replay and
provenance; its 40 episodes include 10 successes and 30 failures.

Fair reward comparison: `round4_reward_old_repaired` (+1/0) versus
`round4_reward_neg1_pos1` (+1/-1). The untouched original round-000004 remains a
historical reference and differs in both terminal replay assembly and rewards
from the signed branch. Comparison jobs hold the hardware session lock and never
activate hardware or replace the main training policy. Use the workstation
wrappers `~/eval_pi05_reward_a_fixed.sh` and `~/eval_pi05_reward_b.sh` respectively;
`~/eval_pi05_reward_a.sh` retains the original fourth-round policy. Press b for
Actor takeover; before b both use the same frozen teacher. Actual task improvement
requires physical evaluation.
