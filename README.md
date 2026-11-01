# RoboLab π0.5 + RLT + RTC

面向 **Franka Emika Panda + Robotiq 2F-85 + 单 wrist RealSense D405** 的真实示范 SFT、关键阶段人工评分 RL 和 RTC 推理集成。

本仓库是训练/推理层；控制与数采沿用已接受的 RoboLab 实现。没有用 FR3 配置替换 Panda，也不要求先在 ManiSkill 训练。

- 控制：[robolab-control-system](https://github.com/FrancisCheng777/robolab-control-system)
- 数采：[robolab-collecting-pipeline](https://github.com/FrancisCheng777/robolab-collecting-pipeline)，以只读 submodule 固定到 `eb9a6c5855a26025841204c715880f4ea3182709`。
- 模型：[RLinf RLT 文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html)，固定 RLinf `c70606f08cdca259b8dec03d4430926b5b8fac9d` 的本地补丁快照。
- 流程与奖励：[Physical Intelligence RLT 论文](https://arxiv.org/html/2604.23073v2)，采用 VLA 接近 → 人工切入关键阶段 → 人工终止评分。默认成功 `1`，失败/中间步骤 `0`；不是 RLinf 通用键盘 wrapper 的失败 `-1`。兼容选项 `rlinf_signed` 可显式选择，但不同奖励配置不能混合 replay。

**当前交付是接口、运行闭环及无真机验证。** 仓库不包含示范、基础权重或已训练任务 checkpoint；完整 π0.5 的 4090 显存与 RTC 延迟验收、2×A800 全量训练及插入成功率仍需实际数据/权重与后续实验。tiny 随机模型测试不能代替这些验收。详见 [验证记录](docs/VALIDATION.md)。

## 硬件与软件基线

| 部分 | 基线 |
|---|---|
| 机械臂 | Franka Emika Panda；沿用 Desk 4.2.2 的既有部署记录（本次不读取/更新固件） |
| 夹爪 | Robotiq 2F-85，标定最大开口 0.085 m；固定仓库内官方 Robotiq SDK 1.2.0 |
| 相机 | 单 wrist D405，RGB → letterbox 224×224；沿用数采预处理与时间戳 |
| 本地 GPU | NVIDIA GeForce RTX 4090，`nvidia-smi` 实测 24564 MiB（24 GB 级） |
| 本地系统 | Ubuntu 20.04.6，Linux 5.15.0-rt17，NVIDIA driver 570.133.07 |
| 机械臂 SDK | 已安装 libfranka 0.9.0；本次 real backend 仅编译，不连接机械臂 |
| Stage 1 | 2×A800 **80 GB**；启动器要求恰好两卡、每卡 ≥75 GiB；全量参数、FSDP full shard、BF16 运算 |
| Runtime | Python 3.11；PyTorch 2.7.1+cu128，CUDA runtime 12.8；验证见记录 |
| Stage 1 依赖 | [固定 requirements](stage1_integration/stage1/requirements.txt)，独立环境；LeRobot 0.4.4 |
| 控制与采样 | 1 kHz C++ servo；100 Hz CPU 目标刷新；15 Hz 策略/数据 |

保持 `responsive_recording` 中已调好的速度、增益、前馈、碰撞阈值、回位轨迹和夹爪顺序。运行时校验 bridge 的完整 calibration 与配置一致；不通过改动作编码来提高控制器限速。

## 统一的数据与动作接口

`panda_delta_q0.05_robotiq_width0.085_v2`：

```text
state[9]  = q_measured[7] + [measured_width/2, measured_width/2]
action[:7] = (q_measured_next - q_measured_now) / 0.05
action[7]  = 2 * command_width / 0.085 - 1
runtime_q_target = execution_time_measured_q + 0.05 * action[:7]
```

训练标签来自下一采样时刻的实测位移，不是 GELLO 目标与当前关节之差。执行时每个 15 Hz action 只解码一次，100 Hz 刷新保持该绝对目标。禁止重复积分、DROID 速度转换或二次 DeltaActions。

D405 `observation.images.wrist` → `base_0_rgb`，图像掩码 `[true,false,false]`。另两个 slot 填零且无效；不复制 wrist 来伪造多相机。`main_images` 在 RLinf API 中只是第一个槽位的字段名，不代表 Kinect。

训练/推理共享 FP32 分位数归一化，统计仅来自 train；先把 9 个真实状态维度离散化/tokenize，再补到 32。动作也仅在归一化后补零到 32。checkpoint 必须带 `manifest.json`、`norm_stats.json`、`action_codec.json`、训练配置和 tokenizer。

**Bridge 仍保留 v1/0.03 calibration 标识**：其 wire API 发送的是绝对 `q_target` 和米制夹爪宽度，0.05 在上层解码；不能据此修改已接受的 bridge 参数。旧 v1 示范导出与双相机 checkpoint 会被新训练/部署入口拒绝，需要从原始 HDF5 重新导出或重新训练。

## 从 100 条示范到最终部署

1. **采集与筛选。** 在数采仓库用已接受的采集入口收集约 100 条成功插入示范，覆盖合理初始位姿、插排位置与抓持差异。保留 session、原始 HDF5 和人工审核结果。把完整 session 留作验证，例如约 90 条训练、10 条验证；不要将同一 episode 的帧拆到两边。这个规模是启动方案，不保证任务覆盖充分。
2. **导出 v2。** 用 pinned 数采 exporter 导出 `train/`、`validation/` 与统计文件。只在 train 拟合 norm stats。先运行 `scripts/check_v2_dataset.py`，确认动作、窗口、单相机映射和 mock 标识；生产训练禁止 `--allow-mock`。
3. **2×A800 Stage 1。** 用 π0.5 基础权重初始化，全量训练 π0.5 + RLT 表征/重建模块。先预检和独立 synthetic smoke，再运行真实 SFT。RTC 在此阶段关闭。选择 checkpoint 时同时看独立验证集 flow loss、RLT 重建 loss 和后续冻结策略 rollout；2000 steps 是配置起点，不是最佳值承诺。
4. **冻结与无运动验收。** 把完整 actor 导出目录拷到 4090。配置 checkpoint、norm stats 和本机解释器。先做资产校验、录制输入离线推理和全尺寸 RTC profile，不启动设备。当前 H/C/D=`10/5/2`，15 Hz，对应规划 deadline 约 **133 ms**；必须测到实际 full-VJP 延迟满足要求才用于连续控制，不能用 tiny 测试时间代替。
5. **固定 Stage-1 基线评估。** 在人工准备的初始条件下，用 `session --mode eval` 跑固定 teacher。每轮 `r` 接近，`b` 标记关键阶段，`a/c` 结束判定；未进入关键阶段的失败也保留在整轮成功率分母。评分后先 hold，不自动回位。需要回位时明确按 `h`，确认完成后再 `r`。
6. **4090 Stage 2。** 新训练 session 先用 teacher 采样。默认至少 5 个有效关键阶段 episode 且至少 32 条可学习 chunk transition 后，actor 做 100 次 BC warmup，再做 100 次 critic 更新、25 次 actor 更新。之后每个新增有效 episode 都从累积 replay 更新 100 次 critic、25 次 actor，配置可调；BC warmup 单独计数。π0.5 和 RLT 全程冻结。
7. **独立评估与定版。** 固定一个 Stage-2 `policy.pt`，用新目录 `--mode eval --policy ...`；探索关闭，整个评估过程不更新权重。报告整轮成功率、关键阶段耗时、终止原因与安全/规划失败，保留所有开始的 episode。选择模型后按相同输入/归一化/RTC 配置部署。

### 推荐的训练调度

单台 4090 使用 **按 episode 顺序交替采样与更新**：rollout → 确认 hold → 人工终止标签落盘 → GPU 更新 → 下一轮。独立 CPU 子进程继续刷新 hold 心跳，GPU learner 不与机器人 rollout 同时争抢 GPU。每轮内 actor 权重固定，只在轮间更换。

RTC 解决执行队列与推理延迟，因此 **Stage 1 不需要 RTC，但 Stage 2 rollout 就要开启最终部署使用的 RTC**，不能等 RL 结束后才换入不同执行语义。本项目的 queue-aware actor/critic、延迟 transition assembler 和 RTC 组合是本地集成扩展，不宣称是上游完整复现。

Stage-2 学习标量已对齐固定 RLinf 示例：actor/critic LR=3e-4、batch=256、Q/BC 权重=0.1/5、reference dropout=0.5、梯度裁剪=10。AC 主体已恢复官方三层 256/Tanh、critic LayerNorm 和初始化，actor:critic 更新比为 1:4，计数跨 episode 与 resume 连续。队列输入、gamma=0.99 和 episode 采样调度保留。详细来源、差异理由及测试见 [参数与结构审计](docs/RLINF_PARAMETER_AUDIT.md)。请为新 AC 使用新的 Stage-2 训练目录；旧两层 AC checkpoint 会明确拒绝加载，Stage-1 π0.5/RLT 权重仍可沿用。改变训练参数的 resume 也会被拒绝，防止旧优化器恢复旧学习率。

### 人工评分与探索

| 键/选项 | 行为 |
|---|---|
| `r` | confirmed hold 后开始新 episode，VLA 负责接近 |
| `b` | 进入关键插入阶段；下一次规划请求切 actor，已承诺前缀不改写；teacher warmup 仍用 VLA |
| `c` / `a` | 成功 / 正常失败，停止后分别写 terminal reward 1 / 0 |
| `x` | 放弃/干预，不进入正常 RL replay |
| `q` | 退出；进行中的 episode 记 aborted，最终关闭 bridge |
| `h` | 仅 held 时调用原数采 `CollectionSession` 的 home → hold → open gripper → hold |
| `--exploration` / `--no-exploration` | 训练 actor 探索开/关；默认开；eval 无条件关闭 |

探索是让 actor 在原有动作附近尝试小变化，便于从成功/失败中学习；沿用 RLinf 的 **tanh 前 logits 高斯噪声，std=0.002**。它不是直接加 0.002 rad，也不改 committed prefix、限速、增益或碰撞阈值。所有输出保持在 `[-1,1]`；仍由既有 bridge 检查与限制执行。中间 reward 全为 0；超时为正常失败，故障/干预/没有进入关键阶段的 episode 不作为普通 RL 训练样本。

## 安装与命令

```bash
git clone --recurse-submodules https://github.com/FrancisCheng777/robolab-pi05-rlt-rtc.git
cd robolab-pi05-rlt-rtc
```

采集/export 使用数采环境；Stage 1 使用独立训练环境；runtime 使用独立 GPU 环境。不要让旧 `pi05_demo` 覆盖 pinned 数采模块。两个采集/训练模块有同名包，所以 CPU owner 与 GPU 模型/训练分进程。

```bash
# 只查看 exporter 帮助；此命令不会连接设备
PI05_DATA_PYTHON=/path/to/lerobot/python bash scripts/data_pipeline.sh export --help
PI05_DATA_PYTHON=/path/to/lerobot/python bash scripts/data_pipeline.sh export \
  --input /absolute/path/to/reviewed_raw --output /absolute/path/to/export \
  --validation-session heldout_session_name
# 多个验证 session 可重复 --validation-session；正式数据不要使用 --checkout。
/path/to/stage1/python scripts/check_v2_dataset.py /absolute/path/to/export

# A800 主机
bash stage1_integration/scripts/setup_stage1.sh
export PI05_DATASET=/absolute/path/to/export
export PI05_BASE_MODEL=/absolute/path/to/pi05_base
bash scripts/start_stage1_integrated.sh inspect
bash scripts/start_stage1_integrated.sh preflight
bash scripts/start_stage1_integrated.sh smoke runner.logger.experiment_name=synthetic_smoke
bash scripts/start_stage1_integrated.sh train runner.logger.experiment_name=plug_real_sft
```

生产训练默认 `stage1.synthetic=false`，不从 synthetic smoke checkpoint 恢复。推荐主入口是 `start_stage1_integrated.sh`；旧通用上游 SFT 入口仅保留供开发审查。

Runtime 安装使用已按官方 OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479` 配好的 Python 3.11 环境作为只读父环境：

```bash
OPENPI_PYTHON=/path/to/openpi/python bash scripts/setup_runtime.sh
```

该脚本新建 `.venv-runtime`，不升级父环境；JAX 固定使用 CPU，仅 PyTorch 使用 GPU。Stage-1 的 LeRobot 0.4.4 / NumPy 2 环境与 OpenPI runtime 的 NumPy 1.26.4 环境分开。

4090 上配置 `configs/runtime_franka.yaml` 的 `model.checkpoint`、`model.norm_stats` 和 `session.control_python`。相对路径以 YAML 所在目录解析。`control_python` 只需 pinned 数采控制依赖，不加载 CUDA；应使用明确的绝对路径。当前机器人 IP、相机序列号来自既有本机配置，换机时显式修改。

```bash
# 不打开硬件的检查
bash scripts/build_bridge.sh --mock
bash scripts/build_bridge.sh --real  # 仅编译；使用本机 libfranka
PI05_RUNTIME_PYTHON=/path/to/runtime/python bash scripts/runtime.sh runtime-validate --runtime-config configs/runtime_franka.yaml
PI05_RUNTIME_PYTHON=/path/to/runtime/python bash scripts/runtime.sh session --runtime-config configs/runtime_franka.yaml --out outputs/train

# 后续人工准备好真机后：终端 A（本次交付测试没有执行）
bash scripts/start_franka_bridge.sh --execute
# 终端 B
PI05_RUNTIME_PYTHON=/path/to/runtime/python bash scripts/runtime.sh session \
  --runtime-config configs/runtime_franka.yaml --out outputs/train --annotator operator --mode train --execute
# 同一目录重启会恢复累计 replay、优化器与最近完成的更新轮；不自动开始动作。

# 固定模型评估：另开 session 目录、重启已停止的 bridge
PI05_RUNTIME_PYTHON=/path/to/runtime/python bash scripts/runtime.sh session \
  --runtime-config configs/runtime_franka.yaml --out outputs/eval --annotator operator \
  --mode eval --policy /absolute/path/to/policy.pt --execute
```

`session` 没有 `--execute` 时仅打印操作说明，不加载模型或打开设备。按下 `h` 会运动；评分不会自动调用它。CLI 不做自动解锁、reflex 恢复或碰撞后继续。

每个 episode 保存 observations、执行 ledger、命令 ACK、terminal label 和 replay。只有已确认 hold 且无故障的 success/failure 样本能学习；更新轮通过临时目录完成后原子发布，session 目录加独占锁，同一组 episode 不重复应用已提交的更新。异常中断的未完成目录保留供审查，不自动补造奖励或训练标签。

## 开发验证

```bash
python -m pytest tests -q
python scripts/validate_cuda.py  # CUDA，随机 tiny 上游模型与显式 mock replay，无设备
PI05_TEST_DATASET=/path/to/synthetic/export \
  PYTHONPATH=.:stage1_integration:stage1_integration/vendor/rlinf_stage1 \
  python -m unittest discover -s stage1_integration/tests/stage1 -q
```

源码与许可证见 [NOTICE](NOTICE.md)、[LICENSE](LICENSE)、[provenance](provenance)。`stage1_integration/pi05_demo` 中旧采集实现仅作历史/测试兼容；采集/export 主入口已指向 submodule。旧设计材料保留在 `docs/`，若与本 README 的 v2/单 D405 定义冲突，以本 README 和当前配置为准。
