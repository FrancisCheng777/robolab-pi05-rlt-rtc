# 两张 A800 80GB 全量微调

本入口在独立项目内使用 RLinf 原生 `SFTRunner → PlugSftWorker(FSDPVlaSftWorker) → FSDP`，同时训练 π0.5 与 RLT。原 roboLab、OpenPI、bridge 和 Mac 主项目均不修改。

固定的 RLinf 源码在 `vendor/rlinf_stage1`，来源 commit `c70606f08cdca259b8dec03d4430926b5b8fac9d`。FSDP 仅通过本项目的 allow_openpi_full_sft 显式启用，其他上游 OpenPI 配置的限制保留。修改和来源指纹记录在 `reports/rlinf_stage1_*.json`。这里不使用普通 pi05_droid checkpoint 冒充已完成 Stage 1。

## 租到机器后先测试

选择同一台主机的 2 张 A800 80GB，建议主机内存至少 128GB、可用磁盘至少 300GB。完整模型约 16.4GB（FP32），每个训练 checkpoint 还包含 Adam 状态、分片与完整导出，多次保存会继续占用磁盘。确认 CUDA 12.6 wheel 所需的驱动已安装；本脚本不安装或修改驱动。

把本项目复制到 AutoDL，保留 `vendor/rlinf_stage1`、`stage1/assets`、`reports/rlinf_stage1_upstream.json`。不要复制本机的 `.venv*`、`build`、合成测试 checkpoint 或 `outputs`。训练数据另行传输。传输后执行：

```bash
cd /你的路径/pi05_plug_demo
./scripts/setup_stage1.sh
# RLinf 文档指定的基础权重；版本固定，避免不同下载日期产生变化。
.venv-stage1/bin/hf download lerobot/pi05_base \
  --revision 22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0 \
  --include model.safetensors config.json --local-dir assets/pi05_base
export CUDA_VISIBLE_DEVICES=0,1
export PI05_BASE_MODEL="$PWD/assets/pi05_base"
export PI05_RUN_NAME=plug_memory_smoke_001
./scripts/stage1.sh preflight stage1.synthetic=true
./scripts/stage1.sh smoke
```

`PI05_BASE_MODEL` 指向含 `model.safetensors` 的 π0.5 基础 PyTorch checkpoint，或已有的完整 RLT Stage-1 checkpoint。JAX/Orbax 权重目录不能直接使用。基础权重尚未随此项目下载或转换；使用与固定 RLinf 版本兼容的完整 π0.5 权重，模型加载器会检查缺失键。权重来源为 [RLinf 文档指定的 lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base/tree/22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0)。

`smoke` 使用合成双相机输入和真实基础模型，执行 5 个 optimizer steps，包含梯度累积、完整分片、反向重计算、Adam 更新和 checkpoint 保存。它将 effective batch 临时改为 8，warmup 改为 1，输出标记为 synthetic，不能作为真机 policy 或成功率结果。

输出在 `outputs/stage1/$PI05_RUN_NAME/`：
- `diagnostics_rank_0.jsonl`、`diagnostics_rank_1.jsonl`：各卡显存峰值、loss、梯度范数；前两步检查视觉、语言、动作专家、动作投影、RLT encoder/decoder 都有有限且非零的梯度。
- `checkpoints/global_step_5/actor/dcp_checkpoint/`：用于恢复 optimizer、scheduler、模型和 RNG。
- `checkpoints/global_step_5/actor/model_state_dict/full_weights.pt`：完整模型，保留 RLT 权重。
- actor 目录同时保存配置、tokenizer、动作契约和 synthetic 标记。真实数据训练还会保存 train-only norm_stats 和 manifest。

可以把 `runner.max_steps=50 actor.optim.total_training_steps=50` 传给 smoke 延长检查。先查看两个 rank 的峰值，而不是仅看某一时刻的 nvidia-smi。

完整权重结构检查（不使用 GPU）：

```bash
export PYTHONPATH="$PWD:$PWD/vendor/rlinf_stage1"
.venv-stage1/bin/python -m stage1.checkpoint \
  outputs/stage1/plug_memory_smoke_001/checkpoints/global_step_5/actor --allow-synthetic
```

继续一步，验证真实双 GPU 恢复：

```bash
export PI05_RUN_NAME=plug_memory_resume_001
./scripts/stage1.sh smoke runner.max_steps=6 \
  runner.resume_dir="$PWD/outputs/stage1/plug_memory_smoke_001/checkpoints/global_step_5"
```

恢复时 scheduler/optimizer 从 checkpoint 读取；修改 total_training_steps 不是重新规划已保存 scheduler 的方式。生产长任务恢复应保留原学习率计划。

## 有真机示范后训练

本阶段是固定示范上的离线 SFT（π0.5 动作学习 + RL-token 辅助目标），不会在线采集机器人经验。采集与导出仍使用项目原有入口。将成功示范按采集 session 分为 train/validation，训练统计仅从 train 生成。同一人同一天的两批数据即可作为两个 session；阶段/干预标签不是必填项。

训练只取 episode 内完整 H 步动作窗口，不重复末尾非零增量、不跨 episode。短于 H 帧的 episode 不贡献窗口；末尾真实动作仍作为最后完整窗口的标签。`preflight` 分别报告原始帧数 `dataset_frames` 和可训练窗口数 `training_windows`。此规则与推理时 RTC 的重叠动作约束无关。

```bash
export PI05_DATASET=/你的路径/plug_v1
export PI05_BASE_MODEL=/你的路径/pi05_base
export PI05_RUN_NAME=plug_full_sft_001
./scripts/stage1.sh preflight
./scripts/stage1.sh train
```

数据目录必须包含 `train/`、`validation/`、`norm_stats.json`、`manifest.json`。生产入口拒绝 mock 数据以及被修改的统计文件。训练不读取 validation 来更新梯度；本次不将监督 loss 当成真机 SR，也未添加真机自动评测。

默认参数：
- 全部约 4.10B 参数参与训练；无 LoRA、量化、模块冻结或缩小网络。
- FP32 参数、梯度与 Adam 状态；BF16 autocast 计算。
- FULL_SHARD 分片参数、梯度和优化器；逐层包装，限制并发 all-gather，不开启预取。
- micro-batch 每卡 1，effective global batch 256，累积 128 次。保留上游示例的 effective batch，日后可以基于数据量另行调整。
- `enable_gradient_accumulation: false` 是 RLinf 的 no_sync 开关：禁用未分片梯度滞留，**不是取消数学上的梯度累积**。每个 micro-batch reduce-scatter，最后统一 optimizer.step。
- 视觉、语言/动作专家、RLT encoder/decoder 均启用 checkpointing；使用 non-reentrant 重计算。
- 双有效相机 RGB224，第三个 OpenPI 图像槽为 mask=false；动作 8 维、state 9 维，归一化和 tokenization 后补到模型 32 维；horizon 10。
- RLT 的 detach、损失定义与 alpha 保持上游语义。RLT 重构损失训练 RLT 模块，VLA flow loss 更新 VLA。
- 默认关闭 torch.compile，先建立可复现的显存/梯度基线；不影响可训练参数。

若显存有余量，可将 `actor.micro_batch_size=2`，global batch 不变时自动变成 64 次累积。改变 micro-batch 或卡数会改变浮点归约顺序，不能承诺与 8 卡逐 bit 相同，但不会削减模型容量或全量微调能力。代价主要是重计算和更多通信，以及比 8 卡低的总吞吐。

## 验证范围

本机使用独立 `.venv-stage1-check` 复用 CPU Torch，只在新项目内安装补充依赖。它不是 AutoDL 的 CUDA 训练环境。

```bash
.venv-stage1-check/bin/python -m unittest discover -s tests/stage1 -v
PI05_DISTRIBUTED_TEST_OUTPUT="$PWD/data/stage1_cpu_checkpoint" \
  .venv-stage1-check/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/test_stage1_distributed.py
```

CPU 小模型测试覆盖真实模型实现的重计算、梯度等价、数据适配、两进程 FULL_SHARD、梯度累积、BF16 autocast、RLinf DCP 保存恢复及完整权重严格重载。CPU 测试关闭 CUDA 缓存管理，并对已在 CPU 上的完整权重直接保存，不模拟 CUDA→CPU offload。完整 4.10B 模型的 CUDA/NCCL、GPU checkpoint offload、实际显存峰值和吞吐必须在租到 A800 后通过上面的 smoke 和 resume 检查。

原始采集程序和既有推理服务不会被这些入口启动或停止。后续 Stage 2 必须使用真实数据 Stage-1 产物和相同的动作、相机、归一化及 tokenizer 契约。
