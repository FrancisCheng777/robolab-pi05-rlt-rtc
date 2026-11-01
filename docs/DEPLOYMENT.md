> Historical integration notes. Current hardware, v2 data contract and session commands: [README](../README.md). Current validation: [VALIDATION](VALIDATION.md).

> 更新：当前整合版和本次检查见 [INTEGRATION.md](INTEGRATION.md)。下文保留原阶段记录；本次新增 smoke/profile 必须提供保存的观测，不会创建机器人环境。历史直接给动作记录不构成模型推理验收。

# Panda + Robotiq 2F-85 + D405 真机流程

## 1. 数据与 Stage 1

沿用 `heinrich:/home/heinrich/pi05_plug_demo/` 的 GELLO/HDF5/LeRobot v3 数采链路。训练根目录包含 `train/`、`validation/`、`manifest.json`、`norm_stats.json`。按 session 划分；统计量仅来自 train。使用本项目契约，不能混入旧 DROID demo 或其他动作缩放的数据。

当前唯一生产配置为 `stage1_integration/configs/stage1_realworld.yaml`，默认四卡 FULL_SHARD、FP32 参数与 BF16 autocast，启用非重入 gradient checkpointing。全局 batch 64、LR 1e-5、warmup 1000、cosine 20000 步；首轮最多 8000 步。旧 `stage1_panda.yaml` 已归档，不再从外部 RLinf runner 启动。

```bash
bash stage1_integration/scripts/setup_stage1.sh
export PI05_DATASET=/absolute/path/to/export
export PI05_BASE_MODEL=/absolute/path/to/pi05_base
export PI05_RUN_NAME=plug_realworld_001
bash scripts/start_stage1_4xa800.sh inspect
bash scripts/start_stage1_4xa800.sh preflight
# 完整模型 smoke/save/resume 和持久存储验收后才运行 train。
```

只有一个 wrist 视角映射到 `base_0_rgb`，mask 为 `[true,false,false]`；9D 状态/8D 动作、0.05 v2 编码、10 帧窗口。详细启动、恢复、评估与存储说明见 [四卡训练说明](STAGE1_4XA800.md)。loader 丢弃跨 episode、时间不连续、干预和不足 H 的窗口，统计量仅从 train 计算。

Stage-1 输出须含训练后的 `rlt_module.*`，部署使用 `full_weights.pt` 与对应的同一份 norm stats。普通 π0.5 基础权重或原有 `pi05_droid` 服务不能代替它。

## 2. 真机配置与 bridge

在 4090 的项目副本安装 `.[runtime,cameras]`。单腕部 D405 需 pyrealsense2。C++ 需要与 Panda FCI 匹配的 libfranka 和 libmodbus。

填写 `configs/franka_bridge_panda.example.json` 的工具质量、重心、惯量、Robotiq 串口路径、实测全开/全闭 gPO 端点，核验限速/阻抗/碰撞阈值后设置 `hardware_verified=true`。Robotiq speed/force 使用 **1–255 整数寄存器值**。新 bridge 直接连接 Robotiq Modbus，绝不调用 Franka Hand。

填写 `configs/runtime_franka.yaml` 的 D405 序列号和 Stage-1 路径，保持数采的相机方向/裁剪一致。D405 设备型号会被检查，不能自动回退到 D435i 或零图。Python 控制器与 bridge 使用同一个标定文件；文件内容不一致则拒绝。

```bash
cmake -S cpp/franka_bridge -B build/franka \
  -DPI05_WITH_FRANKA=ON -DPI05_FETCH_DEPS=ON
cmake --build build/franka -j4
ctest --test-dir build/franka --output-on-failure
bash scripts/start_franka_bridge.sh --robot 172.16.0.40
```

每次 bridge 生命周期仅对应一个 trial；人工摆回机器人与物体，并显式重新启动 bridge。没有自动 reset、自动解除 reflex 或自动重试运动。Robotiq 启动含激活过程，应在台架验收时核实。

`python scripts/test_bridge_gentle.py` 默认只读状态；显式 `--execute` 才执行单个 0.003 rad 的关节测试，保持原夹爪开口。本次软件修改没有执行该运动测试。

## 3. 教师 rollout 与人工奖励

控制器与 C++ bridge 必须在同一主机（Unix socket）。用于训练的 rollout 使用本地 feature planner，在采样时记录冻结 z/reference；不事后重新生成特征。

```bash
pi05-rlt-rtc collect-teacher --runtime-config configs/runtime_franka.yaml \
  --ticks 150 --out runs/physical/teacher_001
pi05-rlt-rtc annotate-rollout --rollout runs/physical/teacher_001/rollout.pt \
  --outcome success --annotator operator_name --out runs/physical/labelled_001
```

人工在 trial 完成后标注 `success / failure / aborted / intervened`。成功只在终局 tick 给 1，失败全程 0。`--terminal-tick N` 可以标明实际任务终止 tick，之后的采样不参加学习。没有终止 tick 时使用最后一个实际执行 tick。不要把普通批次截断标成任务失败；只有确认任务结局时才选 success/failure，否则选 aborted。

故障、deadline miss、停止未确认的 trial 只能归为 aborted/intervened，不进入正常策略 replay。未标注的物理 `reward=0` 无效。保存的文件包括 `rollout.pt`、`hardware_commands.json`、`ledger.jsonl`、运行 manifest 和报告；annotate 输出 replay 和标注来源。当前保存特征与动作/反馈，不保存完整视频；结果由现场人工或独立录制确认。

## 4. Actor–critic

```bash
pi05-rlt-rtc warmup-actor --runtime-config configs/runtime_franka.yaml \
  --replay runs/physical/labelled_001 --warmup-updates 100 \
  --learner-device cpu --out runs/physical/warmup_001
pi05-rlt-rtc train-queue-rlt --runtime-config configs/runtime_franka.yaml \
  --replay runs/physical/labelled_001 runs/physical/labelled_002 \
  --resume runs/physical/warmup_001/latest --warmup-updates 0 --updates 150 \
  --learner-device cpu --out runs/physical/update_001
```

learner 仅消费明确标注的缓存特征，不连接机器人、不加载 VLA。CPU 默认避免与 4090 推理争显存，也可显式选择 `cuda:0`。它可以在另一个终端与下一条 rollout 并行运行；实际同卡竞争需要测量。

使用 `--resume` 保留 actor、critic、优化器与随机生成器状态。每个输出目录都是新目录，策略文件原子写入。下一条人工准备的 trial 加载完整的新 snapshot，不在已经承诺的动作块中途换权重。

```bash
pi05-rlt-rtc run-controller --runtime-config configs/runtime_franka.yaml \
  --policy runs/physical/update_001/latest/policy.pt \
  --ticks 150 --out runs/physical/actor_001
```

然后同样人工标注 actor trial，再进行下一轮 learner 更新。更新次数不等于新增真机 episode 数。初始 actor 必须先 BC warmup。推理默认确定性；没有自动随机真机探索开关。

## 5. 评估与性能

```bash
pi05-rlt-rtc evaluate --runtime-config configs/runtime_franka.yaml \
  --trials runs/physical/eval_labelled_001 runs/physical/eval_labelled_002 \
  --out runs/physical/evaluation.json
```

该入口只聚合标注后的真实 trial，不驱动机器人。要求同一策略版本、相同运行契约，拒绝重复 trial；中止/干预保留在总数分母中并单独报告。训练数据不能冒充独立评估数据。

RTC 的 H/C/D=10/5/2 是起始值，15 Hz 下 D=2 只有约 133 ms 提前量。用 `profile-planner` 和单次 `run-controller` 测真实模型时延与控制抖动；失败即停，不能把晚结果贴进旧时间槽。硬件、真实数据、GPU 训练及现场成功率尚待实际验收。

## 已同步的 4090 检查副本

`/home/heinrich/pi05-rlt-rtc-physical-20261029-review` 是本次新建的独立副本，内有重新编译的 `build/franka/pi05_franka_bridge`；没有启动它的真实控制模式。它与数采 bridge 共享协议，不能同时占用同一 socket/机器人。

`.venv-check` 仅供 CPU 软件检查，读取现有 Python 3.12 依赖目录且单独安装 pytest；它不是已配置的 GPU 推理环境。实际模型仍需独立、已审核的固定 RLinf/OpenPI checkout 与真实 Stage-1 checkpoint。不要把 `pi05_plug_demo/vendor/rlinf_stage1` 填进当前严格校验的 rlinf_root：该树是另一项训练工作维护的非 Git 优化副本。

已核对的训练输出位置为 `actor/model_state_dict/full_weights.pt`，并附 `stage1_metadata.json`、`stage1_resolved.yaml`、`norm_stats.json` 和 tokenizer；只有真实训练产物才可进入机器人部署。当前没有加载该类真实权重验收，profile 名称不同不自动代表 checkpoint 不兼容，架构与数据变换须同时一致。

完整核对及未统一项见 [SYNC_4090.md](SYNC_4090.md)。
