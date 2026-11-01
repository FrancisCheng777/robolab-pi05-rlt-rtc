# AutoDL 4×A800: real-world Stage-1 preparation

This profile trains π0.5 and RLT on reviewed physical demonstrations. It does
not run ManiSkill or a simulator. The data contract remains Panda + Robotiq
2F-85 + one wrist D405, state9/action8, 0.05 v2 encoding, and horizon 10.

## Parameter provenance

Reference: RLinf commit `c70606f08cdca259b8dec03d4430926b5b8fac9d`,
`examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml`.

| Setting | Four-card profile | Reason |
| --- | --- | --- |
| Global batch | 64 | Official real-world RLT |
| Learning rate | 1e-5 | Official real-world RLT |
| Warmup | 1000 optimizer steps | Official real-world RLT |
| Cosine schedule | 20000 steps, minimum 2.5e-6 | Official real-world RLT |
| Adam betas / epsilon | 0.9 / 0.95 / 1e-8 | Official real-world RLT |
| Weight decay / clip norm | 1e-10 / 1.0 | Official real-world RLT |
| RLT loss coefficient | 1.0 | Official joint objective |
| Seed | 42 | Official real-world RLT |
| Initial run budget | At most 8000 optimizer steps | Local initial budget for about 150×400 frames; validate before extending |
| Save interval | 2000 | Official real-world RLT; storage plan required |
| Micro-batch | 16 | Selected after full-model synthetic benchmarks on these four A800s; global batch remains 64 |
| FSDP | Four ranks, FULL_SHARD, FP32 storage / BF16 autocast | Existing full-SFT precision contract, expanded from two ranks |

At four ranks and global batch 64, micro-batch 1/2/4/8/16 means 16/8/4/2/1
accumulation passes per rank. Do not increase the global batch or LR just
because there are more GPUs. Keep `enable_gradient_accumulation=false`: in
this backend it controls `no_sync`, not whether loss accumulation occurs.

For approximately 135 training episodes of 400 frames and 10-frame complete
windows, 52785 windows give approximately 2.4/4.9/7.3/9.7 window passes at
2000/4000/6000/8000 optimizer steps. Actual preflight window counts take
precedence. These heavily overlapping windows are not independent episodes.
Eight thousand updates with batch 64 are not optimization-equivalent to two
thousand with batch 256, despite the same total sample presentations.

The official real-world example uses a different action space and horizon.
Those fields cannot be copied over the established Panda data/deployment
contract. This is an adaptation, not an exact reproduction of its robot task.

## Launch and validation

```bash
export PI05_BASE_MODEL=/root/autodl-tmp/assets/pi05_base
export PI05_DATASET=/path/to/complete/export
export PI05_RUN_NAME=plug_realworld_001
bash scripts/start_stage1_4xa800.sh inspect
bash scripts/start_stage1_4xa800.sh preflight
# Only after /root/autodl-fs is mounted, with enough quota/free space,
# and dataset/memory/throughput/save-restore checks pass:
bash scripts/start_stage1_4xa800.sh train runner.max_steps=2000 \
  runner.logger.log_path=/root/autodl-fs/pi05-stage1
```

The default micro-batch 16 completed the full-model synthetic throughput test.
Confirm throughput again with real LeRobot decoding and the actual dataset. The initial AutoDL topology has PCIe links and a
cross-NUMA fourth GPU, so do not assume ideal four-card speedup.

Stopping at 2000 is a validation checkpoint, not the end of the LR schedule.
After independent evaluation, resume the same run/checkpoint to 4000, then
6000 and 8000, keeping the scheduler horizon 20000 and all data/statistics
unchanged. Use `runner.resume_dir=.../checkpoints/global_step_2000`.
Restore requires the same GPU world size. Production data is only accepted in
`train`; the explicitly synthetic `smoke` mode must never produce a deployment
policy. The full-model smoke includes checkpoint writing and needs the same
disk planning as training.

The standalone evaluator defaults to the complete held-out set.
`--max-batches` is an explicit diagnostic limit; the report records
`complete_validation` so partial evaluations cannot be mistaken for full coverage.
Use session-disjoint data, train-only statistics, a fixed evaluation seed, and
both physical8 flow loss and RLT reconstruction loss. Offline loss is not robot
success rate. Do not set the training runner's validation interval expecting
automatic evaluation: this worker intentionally requires separate held-out
evaluation.

## Verified tensor-size calculation (2026-11-01)

The production architecture was instantiated on PyTorch's meta device, without
allocating weights: 4,100,198,160 parameters, including 746,764,288 in RLT.
The FP32 state dictionary payload is 16,400,792,640 bytes.

| Saved object | Decimal GB |
| --- | ---: |
| Model in resumable DCP checkpoint | 16.4008 |
| Adam first and second moments | 32.8016 |
| Separate full model for deployment | 16.4008 |
| Total per current checkpoint | 65.6032 (61.10 GiB) |

The original table is the tensor-payload calculation. Full-model synthetic
save/restore now measured 65,625,135,224 bytes per complete checkpoint
(about 65.625 GB / 61.118 GiB), including serialization and auxiliary files. FSDP
splits the state across ranks; it does not multiply this total by four. Base
`lerobot/pi05_base` weights at revision
`22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0` are another 14,467,165,872 bytes.

The initial 50 GiB data disk cannot hold even one current checkpoint plus
environment, base weights, and data. Four retained checkpoints require about
262.4 GB before those other files. Keeping only latest/best still needs room
to finish a new checkpoint before deleting an old one. There is currently no
automatic checkpoint deletion; retention and disk capacity must be agreed
before training. Never split the training data into successive 80/70-episode
runs merely to address checkpoint size: the complete export is small and
should be sampled together.

Preparation is separate from training. Check the preparation logs before
claiming dependencies, weights, full-model GPU smoke, or throughput are ready.

## One configuration and storage boundaries

All active launchers load `stage1_integration/configs/stage1_realworld.yaml`.
The two-/four-card names are topology aliases only. Generic launchers accept
`PI05_WORLD_SIZE` or `stage1.expected_world_size=N`; placement follows automatically.
Exact resume rejects changes to batch, world size, model, optimizer and schedule.
Stage-2 batch 256 is intentional and unrelated to the Stage-1 global batch 64.

The AutoDL container has a 480 GiB RAM quota and 240 GiB `/dev/shm` tmpfs.
Tmpfs consumes that same RAM quota, is not extra memory, and is not durable.
Synthetic full-model save/resume may use tmpfs while monitoring aggregate RAM.
Production requires verified persistent checkpoint storage before launch.
AutoDL `autodl-fs` is separate shared storage. The A800-region 200 GB store
is initialized and empty; this running instance has not mounted it.
Do not restart or release the scarce GPUs to mount storage without an agreed plan.
Cloud-to-4090 transfers should be direct; the Mac is not a checkpoint destination.

## Full-model synthetic measurements (2026-11-01)

| Micro-batch per GPU | Accumulation | Approx. seconds/update |
| --- | --- | --- |
| 1 | 16 | 58 |
| 4 | 4 | 16 |
| 8 | 2 | 9.2 |
| 16 (selected) | 1 | 5.5 after first-step startup |

Each measurement used 3 optimizer steps, global batch 64, the real 4.1B model,
FP32 storage, BF16 autocast, FULL_SHARD and gradient checkpointing. These short
synthetic-input measurements exclude real video decoding and cannot establish
convergence or robot success. The data path and task metrics require the user's
reviewed LeRobot export. Original `lerobot/pi05_base` weights remain unchanged;
synthetic checkpoints are rejected by the real-data resume/deployment gates.

At 5.5 seconds/update, compute alone would be roughly 3.1 hours per 2000 steps
or 12.2 hours per 8000. This is an extrapolation, not a real-data timing promise;
include data loading, checkpoint I/O, evaluation and any runtime variability.

AutoDL-specific preparation includes `/root/autodl-tmp/preparation/start-real-stage1.sh`.
It refuses to run without an explicit dataset/run name, a mounted fs, and at least
70 GiB reported free. It stops the initial segment at 2000 updates; evaluate and
manage completed checkpoints before resuming to 4000/6000/8000. No automatic
deletion is configured. The fs quota must also be checked in the console; four
full checkpoints (about 262.5 GB) do not fit its default 200 GB quota.

## Data compatibility with RLinf

Both use LeRobot datasets and the same pi0.5/RLT SFT model interface. This repo
uses a Panda-specific `PlugDataset` rather than the stock OpenPI dataloader.
The official real-world `pi05_franka_state` example repacks `image`, `state`,
`actions`, `prompt`, with 7D end-effector actions and horizon 20. Our adapter reads
`observation.images.wrist`, `observation.state`, `action`, `task`, with state9,
action8 joint deltas at 0.05 scale, and horizon 10. LeRobot container compatibility
does not make the physical action conventions interchangeable.

Point `PI05_DATASET` at the complete export root containing `train/`, `validation/`,
`manifest.json`, `norm_stats.json` and `action_codec.json`. Each split is a LeRobot
dataset. This project's additional checks require genuine `source_time`, session
split provenance, single-wrist mapping and train-only statistics; they must not be
invented merely to make arbitrary data pass. These interfaces were not changed
by the four-card adaptation. Inspect the user's actual export before deciding
whether any conversion or metadata repair is necessary.
