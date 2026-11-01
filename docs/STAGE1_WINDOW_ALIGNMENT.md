> 当前 Stage-1 以 [四卡真机训练说明](STAGE1_4XA800.md) 和 `stage1_integration/configs/stage1_realworld.yaml` 为准：4 卡、global batch 64、单 wrist、0.05 v2。下文的双相机、0.03、双卡或旧 batch 数字仅记录历史，不作为启动配置。

> 更新：当前整合版和本次检查见 [INTEGRATION.md](INTEGRATION.md)。下文保留原阶段记录；本次新增 smoke/profile 必须提供保存的观测，不会创建机器人环境。历史直接给动作记录不构成模型推理验收。

# Stage-1 完整连续窗口对齐（4090 独立副本，2026-10-29）

本次修改位于 `/home/heinrich/pi05-rlt-rtc-physical-20261029-review`。
已读取最新 `/home/heinrich/pi05_plug_demo`，将其 Stage-1、数采数据接口、配置、tokenizer 和优化后的 RLinf 源码复制到独立的 `stage1_integration/`。原项目 2,245 个被复制的文件经前后 SHA256 核对，无变化；未覆盖正在开发的训练代码。

Mac 主源码此次未回写；Mac 上保留核对报告、补丁和 `.codex-work/window-alignment-20261029/working` 审阅副本。不要用旧 Mac 源码直接覆盖这份已经更新的 4090 独立目录。

## 已统一的窗口规则

新版数采已经取消重复末尾动作补齐。本次进一步让以下三个入口共用 `pi05_rlt_rtc/data/windows.py`：

- `PhysicalChunkDataset`（RLT/OpenPI 数据入口）
- `pi05_demo.training_data.ChunkDataset`（独立数采副本）
- `stage1.data.PlugDataset`（独立优化 Stage-1 副本）

统一策略 ID：`complete_contiguous_no_intervention_v1`。

每个样本必须包含完整 H 步，且不跨 episode、不补零或重复最后一个动作。时序来自真实接收时间列 `source_time`；15 Hz 下允许相邻间隔偏差不超过半个周期（约 33.33 ms），相对 episode 固定频率网格的累计漂移不超过一帧（约 66.67 ms）。这两个上限对齐新版导出的 `audit_timing`。

窗口包含 `intervention=true` 时排除。该标记仍是可选字段，缺省表示未标记干预；`critical` 不改变窗口筛选。缺失 `source_time`、非法元数据或无可用窗口会明确拒绝。短 episode 被跳过，不把其他 episode 拼接进来。

窗口规则写入独立副本的训练契约、后续导出的 manifest 与 Stage-1 checkpoint metadata。续训会检查 checkpoint 的 window_policy；旧 checkpoint 没有该字段或规则不同，必须新开训练，而不能沿用不兼容的 dataloader 进度。

## 与最新数采的其他对齐

已同步新版相机 `read_with_metadata`：使用取帧时记录的主机接收时刻，保留设备时钟域和 frame ID，避免把图像处理结束时刻当成采集时刻。

独立 C++ bridge 增加新版协议的 `state_bridge_receive_s`、`gripper_bridge_read_s`、`timestamp_clock` 和 `command_enqueue_s`；保留此前本项目的 Robotiq 标定 helper。现有数采与独立 runtime 客户端对两套 bridge 的 4 组 --mock 互通通过。

这些时间戳不等于已经验证硬件曝光同步。本次没有打开真实相机、串口或 FCI，也没有启动真机运动／恢复命令。

## 接入和验证

最新导出 fixture 保持 `mock=true`，从原项目复制到独立副本用于 CPU 软件验证，没有改成真实数据。

| 检查 | 结果 |
|---|---|
| train / validation，各120帧 | 三个 loader 均得到111个窗口，起点0–110，索引逐项相同 |
| 首／中／末窗口 | 原始动作、归一化动作、状态、token和图像一致 |
| 注入干预 | 三个 loader 均保留101个窗口 |
| 注入缺帧间隔 | 三个 loader 均保留102个窗口 |
| 注入累计时钟漂移 | 三个 loader 均保留58个窗口 |
| 缺少source_time | 三个 loader 均拒绝 |
| mock生产门禁 | 三个 loader 默认均拒绝；仅诊断调用显式allow_mock=True |
| 实际PlugSftWorker.build_dataloader | CPU模拟两个rank，各55样本，互不重叠；drop_last丢弃1个以均匀分片；batch为[1,10,32] |
| 集成启动入口inspect | 两worker、FULL_SHARD、gradient checkpointing配置检查通过；未启动Ray训练 |
| 实际OpenPI CPU变换 | 32组输入，状态／动作最大误差0，token／mask／图像一致 |
| Python运行库 | 167 passed，15 skipped |
| 数采单元测试 | 21 passed |
| Stage-1单元测试 | 7 passed |
| 独立C++真实分支编译／CTest | 编译通过，CTest 1/1通过 |

Stage-1 worker 检查使用真实 StatefulDataLoader 和 DistributedSampler，只有诊断数据构造边界显式选择了 mock fixture；没有初始化模型、训练进程组或 GPU。15 个跳过项仍需要真实数据、权重或硬件／GPU验收。

## 独立入口

只检查配置：

```bash
cd /home/heinrich/pi05-rlt-rtc-physical-20261029-review
PYTHONDONTWRITEBYTECODE=1 PI05_TRAIN_PYTHON="$PWD/.venv-check/bin/python" \
  bash scripts/start_stage1_integrated.sh inspect
```

复现数据接入检查（CPU、软件fixture）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-check/bin/python scripts/verify_stage1_windows.py
```

`start_stage1_integrated.sh` 指向 `stage1_integration/scripts/stage1.sh`，同时设置集成副本及公共窗口包的导入路径。Stage-1 的配置、优化 worker、vendor 源码都取自独立快照。未来训练仍需在有匹配环境和两张A800的机器上提供实际数据与基础权重；本次没有执行train/smoke训练模式。旧的 `scripts/start_stage1.sh` 是原独立profile入口，不是本次验证的优化Stage-1启动路径。

## 证据与保留

远端证据目录：`reports/stage1_alignment/`。

- `windows_red.log` → `windows_green.log`：窗口行为先失败后修复。
- `timestamp_red.log` → `timestamp_green.log`：新版时间戳接口先失败后修复。
- `loader_integration.json`：实际三路数据入口、异常元数据和worker batch验证。
- `runtime_pytest.log`、`collection_tests.log`、`stage1_tests.log`：完整测试。
- `model_transforms.json`、`bridge_interop.json`：实际变换与mock协议互通。
- `changes.json`、`review.patch`：改动清单与补丁；`source_snapshot.json`记录原始文件哈希。
- `independent_before.tar.gz`：独立副本修改前备份。

这次已消除独立副本中的训练窗口规则差异。真实数据GPU SFT、真实权重加载、模型时延和现场硬件参数仍未验收；旧Franka Hand控制目录也没有被替换。
