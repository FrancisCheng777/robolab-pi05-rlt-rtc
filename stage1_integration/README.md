> 历史快照说明，以下旧双相机/0.03内容已被根目录 [README](../README.md) 的单 D405/v2 训练接口替代。采集请使用 vendor/robolab-collecting-pipeline。

# π0.5 插排 Demo：独立采集与 Stage-1 全量微调

目录：heinrich:/home/heinrich/pi05_plug_demo。
代码、配置、原始数据、导出数据均在这里；未修改原 roboLab、OpenPI、桥接目录或 Mac 主项目。

## 已实现

GELLO → Panda/Robotiq bridge → Kinect + D405 → HDF5 → 浏览器检查 → LeRobot v3。
每次运行一条 episode，人工标记成功、失败或放弃；关键阶段和干预标记可选。原始数据保留实测关节、目标关节、夹爪宽度、双路 RGB、分开的时间信息、命令 ACK 和标记。
复制的 pi05-rlt-rtc 在 runtime/，动作和相机接口已对齐。

当前目标是同一操作者采集约 100 条 Franka 示范，之后在 GPU 上进行离线 SFT。Stage 1 使用固定示范，同时训练 π0.5 和为后续 RLT 准备的 RL-token 表征；它不在线收集 rollout 或使用 actor–critic。阶段细分、力信息、深度、高分辨率归档、reward classifier、多操作者元数据均不是本次基础采集的必需项。

尚未验证真实硬件、实际 GPU SFT、真机异步 RLT 或 RTC 性能。runtime/ 原 Stage-2 入口仍只支持 ManiSkill；本次交付是采集和训练数据准备链路。

## 固定动作约定

action_contract_id: panda_delta_q0.03_robotiq_width0.085_v1

- action[:7] = (目标 q - 该步实测 q) / 0.03，范围 [-1,1]，单位由 rad 转成控制量。
- action[7] = 2 × 夹爪命令宽度 / 0.085 - 1；-1 闭合，+1 张开。
- state[9] = q[7] + 两个 width/2；后两维派生自同一夹爪开口，不是两路独立传感器。
- GELLO 的闭合比例先转换成米；越界拒绝，不静默裁剪标签。
- 15 Hz 采集/控制，bridge 独立执行 1 kHz 回调。
- 图像 RGB 224×224；main 按配置旋转/裁方；wrist 保持方向；两路 letterbox。
- 主机事件使用 monotonic 时间；相机设备帧时间另存并标注时钟域。两者没有自动完成时钟/曝光标定。
- bridge 的 gripper_speed/force 是 Robotiq 1–255 整数寄存器值，不是 m/s 或 N。

不能再套 DROID 速度转换或重复 DeltaActions。chunk 每行相对该执行步的实测 q。
ACK 证明目标被接收，不代表机器人已到达目标。原始数据不是含 pending queue 的完整 RLT replay。

## 时间记录与原始图像

新采集文件记录 `timing_version=2`：
- `camera_time/{main,wrist}`：SDK 返回帧后的主机接收时间，早于 Python 的颜色转换、裁剪和缩放；不是曝光时间。
- `camera_ready_time/*`：224×224 预处理完成时间。
- `camera_source_time/*`：设备帧时间，单位统一为秒；`camera_source_clock/*` 属性注明设备时钟域。Kinect 原生计数单位为 0.125 ms，RealSense API 原生单位为 ms。缺失值通过 `camera_source_time_valid/*=false` 明确表示，不冒充曝光时间。
- `camera_frame_id/*`：设备帧编号，无法读取时为 -1。
- `timing/state_bridge_receive_s`、`state_client_receive_s`：bridge 收到机器人反馈、采集端收到该反馈的主机时间；`gripper_bridge_read_s` 是夹爪读数完成时间。
- `timing/gello_read_s`、`command_send_s`、`command_enqueue_s`、`command_ack_s`：读取遥操作输入、发送命令、bridge 入队、客户端收到 ACK 的时间。ACK 不表示物理执行完成。

导出 manifest 每条源记录附带 `timing_quality`，汇总周期分布、相机接收年龄/时间差、重复帧及 ACK 往返时间。相邻采样间隔偏离标称周期超过 50%，或相对固定 FPS 累积漂移超过一帧时拒绝导出。这个阈值是初期数据质量检查，不是硬件同步精度保证；设备时钟映射及真实曝光/执行延迟仍待实机测量。旧记录继续可读，其缺失的时间信息不自动补造。

本次仍保存 RGB224；这可用于基础 SFT，前提是缩放后能看清插头与插孔。源相机的高分辨率画面在缩放/裁剪后未归档，所以以后无法恢复被丢失的细节。高分辨率/深度归档可按实际需求另行增加。

## 无硬件检查

在项目根目录运行：
```bash
.venv/bin/python -m unittest discover -s tests -q
./build/bridge_core_test
.venv/bin/python scripts/smoke.py
```

smoke 自动使用独立 socket 上的 mock bridge，结果见 reports/latest_smoke.json。
合成数据在 data/smoke_*/，导出默认拒绝 mock。显式 --allow-mock 也始终保留 mock 标记。

## 硬件到位后

configs/collection.json：
填写 D405 序列号；核验 Kinect 序列号、方向、GELLO 路径和 vendor/gello 中的关节标定。

configs/bridge.real.json：
填写 Robotiq 串口路径、实测 gPO 全开/全闭端点、工具总质量/重心/惯量；核验限速、阻抗、碰撞阈值。
完成硬件验收后才把两个配置的 hardware_verified 改为 true。空值没有被自动猜测。
相同相机参数同步到 configs/runtime_franka.yaml；不同裁剪/标定版本不要混用。
首轮使用断电排插，先明确插入深度/保持条件，再人工判定成功。

只读预检：
```bash
LD_LIBRARY_PATH=$HOME/freenect2_prefix/lib .venv/bin/python -m pi05_demo.preflight --hardware-config
```
未填硬件参数时非零退出是预期行为。

## 采集一条示范

先关闭其他占用 Franka FCI 的控制程序，人工使 Panda 与 GELLO 初始姿态匹配。不会自动 home 或解除 reflex。

终端 A：
```bash
cd ~/pi05_plug_demo
./scripts/start_bridge.sh --execute
```

终端 B：
```bash
cd ~/pi05_plug_demo
./scripts/collect.sh --execute --session plug_day1_train
```

Enter 开始；s 成功、f 失败、a 放弃；c 切换关键阶段，i 切换干预。
本次仅需 s/f/a；c/i 不作为离线 SFT 的必填标签。
结束后先停止 bridge，保存 episode。退出终端 A 的旧 bridge，人工复位，再启动下一条。没有自动复位。
初始 GELLO 偏差超过 0.03 rad 或运行中输入越界、图像过期、严重超时会拒绝/停止。
崩溃的 .inprogress 文件不导出，异常 episode 保留诊断；SFT 只选成功示范。
建议先采 10 条验证链路，再采 100 条，使用独立 session 采验证集。
同一人同一天可以分成 `plug_train`、`plug_val` 两批，例如总共 100 条中的 90 条训练、10 条验证；不要求不同操作者或不同日期。按整条 episode 分开，避免相邻帧混入训练与验证。

## 浏览器检查

```bash
.venv/bin/python -m pi05_demo.review --input data/raw --port 8766
```

Mac 转发：
```bash
ssh -L 8766:127.0.0.1:8766 heinrich
```

打开 http://127.0.0.1:8766。支持双相机播放/逐帧、目标关节、夹爪和阶段标记；入口只读，不裁剪/修改原始数据，也不启动机器人。

## 导出 SFT 数据

```bash
./scripts/export.sh --input data/raw --output data/datasets/plug_v1 \
  --validation-session plug_day2_val
```

输出 train/、validation/、norm_stats.json、manifest.json。
按采集 session 划分，统计只用 train；manifest 保留源文件 SHA256 和官方 loader/视频验证结果。
已有输出不覆盖，请使用新版本名。不同采集/标定配置不能静默合并。
采样缺口和明显时间抖动会拒绝导出；失败输出留在 .inprogress-*，不会发布成训练集。

configs/training_contract.json 定义数据接口。
pi05_demo.training_data.ChunkDataset 返回 state[9]、actions[H,8]、OpenPI 图像和掩码、prompt；第三路是 mask=false 的 padding。
两个 SFT loader（`pi05_demo.training_data.ChunkDataset` 与 `stage1.data.PlugDataset`）只取 episode 内真实存在的完整动作块，不填充末尾。例如 12 帧、H=10 时得到 3 个样本，末尾动作仍包含在最后一个完整窗口中。少于 H 帧的 episode 不产生训练窗口；整个 split 没有完整窗口时明确报错。原始数据和导出的帧数保持完整。
已接入独立 RLinf Stage-1 SFT 入口，支持 2×A800 80GB FULL_SHARD 和梯度检查点；CPU 单元及双进程测试见 reports/stage1_*。完整模型的 GPU 显存和吞吐仍待 AutoDL 验证，不能直接套 DROID 速度配置。
Stage 1 必须同时训练 RL-token 表征，普通 pi05_droid 权重不等价。

末尾窗口处理属于离线训练取样；inference-time RTC 则是在机器人执行当前动作块时，利用已承诺动作约束下一动作块的生成。遥操作采集不运行 RTC，也不需要补造 RTC 队列；训练和执行保持同一动作契约、频率、相机处理和归一化即可。

## RLT/RTC 模板

```bash
./scripts/runtime.sh runtime-validate --runtime-config configs/runtime_franka.yaml
```

checkpoint、norm_stats、rlinf_root 需填真实产物。H=10/C=5/D=2 是待 profile 初值。
训练与推理使用同一个动作契约和 camera factory。
GPU 训练、真机 reward/复位、异步 learner、队列 replay 是下一阶段，不会被采集程序启动。

未来真机 Stage 2 可以由操作者人工判定成功/失败并提供稀疏奖励，无须先补齐阶段标注或训练 reward classifier。接入时需正确写入 transition 的 reward、终止/超时标记和最后观察。RLinf 的 `single_stage` 键盘奖励与 `rlt_policy_switch` 的策略切换是不同功能；后者的切换按键不能直接当成成功标签。当前 runtime 真机奖励接口尚未完成。

参考：[RLinf RLT 两阶段流程](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html)、[Franka 人工键盘奖励](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/franka.html#label-rewards-from-a-keyboard-optional)、[推理时 RTC](https://www.pi.website/research/real_time_chunking)。

## 环境和构建

.venv 新建并只读继承 openpi 的 site-packages；h5py/dynamixel-sdk 安装在新环境内。
scripts/export.sh 只读复用 roboLab/.venv-lerobot 的 CPU LeRobot 安装；没有修改它。本版本不是脱离原机的容器。
vendor/ 保留 GELLO、camera helper、C++ 依赖副本及 GELLO LICENSE。
原始文件指纹见 reports/source_manifest.json；交付复核见 reports/source_integrity.json。

```bash
cmake -S cpp/bridge -B build -DPI05_WITH_FRANKA=ON -DPI05_FETCH_DEPS=ON \
  -DFETCHCONTENT_SOURCE_DIR_RUCKIG="$PWD/vendor/cppdeps/ruckig-0.15.3" \
  -DFETCHCONTENT_SOURCE_DIR_JSON="$PWD/vendor/cppdeps/json-3.11.3" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
(cd build && ctest --output-on-failure)
```

真实 bridge 使用 kEnforce。只有一个线程访问 Robotiq Modbus，避免并发串口事务；通讯失败不会伪装成有效夹爪反馈。

## 2×A800 80GB 全量微调

准备、smoke、训练、恢复和验证边界见 [Stage-1 操作说明](docs/STAGE1_2XA800.md)。

入口：scripts/setup_stage1.sh、scripts/stage1.sh inspect|preflight|smoke|train。新 RLinf 副本只在 vendor/rlinf_stage1/，原项目未改动。
