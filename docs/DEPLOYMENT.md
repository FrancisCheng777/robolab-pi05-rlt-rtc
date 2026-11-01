> Historical integration notes. Current hardware, v2 data contract and session commands: [README](../README.md). Current validation: [VALIDATION](VALIDATION.md).

> 更新：当前整合版和本次检查见 [INTEGRATION.md](INTEGRATION.md)。下文保留原阶段记录；本次新增 smoke/profile 必须提供保存的观测，不会创建机器人环境。历史直接给动作记录不构成模型推理验收。

# Panda + Robotiq 2F-85 + D405 真机流程

## 1. 数据与 Stage 1

沿用 `heinrich:/home/heinrich/pi05_plug_demo/` 的 GELLO/HDF5/LeRobot v3 数采链路。训练根目录包含 `train/`、`validation/`、`manifest.json`、`norm_stats.json`。按 session 划分；统计量仅来自 train。使用本项目契约，不能混入旧 DROID demo 或其他动作缩放的数据。

在已安装匹配 RLinf/OpenPI/PyTorch/LeRobot v3 依赖的训练环境中：

```bash
python -m pip install -e '.[runtime]'
export RLINF_ROOT=/absolute/path/to/isolated/RLinf
export PI05_DATASET_ROOT=/absolute/path/to/export
export PI05_BASE_CHECKPOINT=/absolute/path/to/pi05_base
bash scripts/start_stage1.sh
```

RLinf 固定提交为 `c70606f08cdca259b8dec03d4430926b5b8fac9d`。脚本只在所指定的隔离 checkout 加入两处显式注册：独立模型数据配置和真实 LeRobot loader；不修改全局 site-packages，不重写模型骨干。预先存在的其他源码修改会被拒绝，不能与另一份正在改动的训练副本直接混用。

`configs/stage1_panda.yaml` 保留 π0.5 + RLT 全量训练，使用 FULL_SHARD、micro-batch 1。它不等于两张 A800 上的显存验收；固定上游的 SFT 会内部关闭 gradient checkpointing，因此本模板没有把该开关冒充为已经生效。另行准备的分片/checkpointing 优化需要合并审核后使用。

模型 profile：`pi05_rlt_panda_robotiq_joint`。loader 校验真实数据标志、动作契约、15 Hz、train-only norm stats 哈希，使用真实 `source_time`，丢弃跨 episode、时间不连续、干预和不足 H 的窗口。不会把末尾关节增量重复成十步动作。两个真实视角必须存在，第三视角仅作为带 false mask 的模型占位。

Stage-1 输出须含训练后的 `rlt_module.*`，部署使用 `full_weights.pt` 与对应的同一份 norm stats。普通 π0.5 基础权重或原有 `pi05_droid` 服务不能代替它。

## 2. 真机配置与 bridge

在 4090 的项目副本安装 `.[runtime,cameras]`。Kinect 另需本机 libfreenect2/pylibfreenect2，D405 需 pyrealsense2。C++ 需要与 Panda FCI 匹配的 libfranka 和 libmodbus。

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
