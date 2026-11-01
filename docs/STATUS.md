> 当前 Stage-1 以 [四卡真机训练说明](STAGE1_4XA800.md) 和 `stage1_integration/configs/stage1_realworld.yaml` 为准：4 卡、global batch 64、单 wrist、0.05 v2。下文的双相机、0.03、双卡或旧 batch 数字仅记录历史，不作为启动配置。

> Historical integration notes. Current hardware, v2 data contract and session commands: [README](../README.md). Current validation: [VALIDATION](VALIDATION.md).

> 更新：当前整合版和本次检查见 [INTEGRATION.md](INTEGRATION.md)。下文保留原阶段记录；本次新增 smoke/profile 必须提供保存的观测，不会创建机器人环境。历史直接给动作记录不构成模型推理验收。

本文件保留前次核对记录。训练窗口及新版数采时间戳的最新对齐结果见 [STAGE1_WINDOW_ALIGNMENT.md](STAGE1_WINDOW_ALIGNMENT.md)。

# 纯真机迁移状态（2026-10-29）

当前工程链路：真实示教 → π0.5 + RLT SFT → 冻结表示 → 真机 RTC rollout → 人工终局标注 → actor–critic 更新。硬件：Franka Emika Panda、Robotiq 2F-85、D405 腕部相机、Kinect v2 外部相机。

## 已实现

- 移除旧仿真运行入口、配置和训练适配器，替换为单台真机、人工复位的 trial。
- 与 `heinrich:/home/heinrich/pi05_plug_demo/` 统一 15 Hz、0.03 rad 关节增量、0.085 m 夹爪开口、9D 状态和双相机预处理。
- C++ bridge 改用 libfranka + Robotiq Modbus；Python 校验动作契约、标定一致性、反馈时间戳及控制权。
- 真实 LeRobot v3 SFT loader、独立 Panda profile、固定 RLinf 注册脚本；只使用完整连续 H 步窗口及 train-only 统计量。
- rollout 保存真实时序和采样时的冻结特征。人工 success 终局奖励为 1，failure 为 0；aborted/intervened 或故障轨迹不进入正常训练。
- learner 从标注 replay 读取数据，验证原始记录和 replay 指纹，支持 BC、actor–critic、优化器/RNG 续训；不连接机器人。
- 评估按同一策略的人工标注 trial 聚合，中止/干预计入分母。没有成功率实验结果。

## 实际验证

- Python 全套：**153 通过、15 跳过**，见 [pytest.log](../runs/physical_migration/pytest.log)。具名物理接口测试也包含在全套中；跳过项需要真实数据、权重或 GPU。
- 历史编号 T01–T68：53 通过、15 跳过。此编号台账不等于全套用例数，详见 [RESULTS.md](../tests/RESULTS.md)。
- Mac C++ mock 构建和 CTest：1/1 通过；Python 测试实际启动 mock bridge 验证 socket 协议。
- 4090 的隔离目录 `/tmp/pi05-physical-build.MmdmIF`：真实 libfranka + libmodbus 分支编译通过、CTest 1/1 通过。日志前半段含一次旧版 ctest 参数不支持而未发现测试，末尾是在 build 目录正确执行的结果。见 [构建日志](../runs/physical_migration/remote_bridge_build.log)。
- 固定 RLinf 静态接口检查 13/13 通过，见 [P0_SOURCE_AUDIT.md](P0_SOURCE_AUDIT.md)。注册补丁已在本地隔离 checkout 校验。

以上都是软件验证。本次没有运行真实相机、执行机器人动作、GPU SFT 或加载真实 Stage-1 权重进行端到端 rollout。

## 与 4090 旧入口的关系

检查用户给出的 `/home/heinrich/pi05_franka_bridge_src/` 后发现，原 bridge 仍使用 Franka Hand 夹爪路径，不能作为 Robotiq 2F-85 的适配完成证据。本地代码已改为新数采项目采用的 Robotiq 链路，但没有覆盖该目录、重启服务或替换原二进制。

Mac 是代码开发位置。部署到 4090 后，模型、观测和动作接口仍须使用同一份物理契约；不能因为代码放在 Mac 就跳过适配。训练可在 GPU 训练机，真实 controller 与 Unix socket bridge 必须同机。

## 尚待现场／真实资产验证

填写并核验工具负载/惯量、Robotiq 串口和实测 gPO 开闭端点、D405 序列号，才能将 hardware_verified 设为 true。模板保留 null，不猜测现场值。

真实导出数据与 train-only norm stats、Stage-1 RLT 权重、GPU 全量 SFT、full-VJP 梯度和时延均未实际验收。H/C/D=10/5/2 为起始值，15 Hz 下 D=2 约 133 ms；需测得时延后确认。Stage-1 模板使用 FULL_SHARD，但 gradient checkpointing 未宣称生效；其他正在准备的训练优化需要独立合并验证。

修改前完整源码及旧规范保存在 `.codex-work/physical-migration-20261029-123629/`。当前目录没有 Git 仓库；本次没有创建提交或远端部署。

完整操作步骤：[DEPLOYMENT.md](DEPLOYMENT.md)。

## 4090 独立副本与一致性检查

已同步至 `/home/heinrich/pi05-rlt-rtc-physical-20261029-review`。未覆盖 `pi05_plug_demo`、其 Stage-1/vendor 训练代码或旧控制目录；未启动真机运动。

这次实际 OpenPI CPU 对照发现 FP64/FP32 分位数精度差异，已将本项目归一化统计转换为与现有训练一致的 FP32；32 组状态、动作、token、mask、RGB 完全一致。新增精度回归测试和 Linux 测试包导入修复。

动作/状态、相机、模型架构检查和 4 组 mock bridge 互通已通过。已有 Stage-1 仍重复末尾动作补齐窗口，本项目采用完整连续窗口，因此不能宣称整条链路完全一致。详见 [4090 同步核对报告](SYNC_4090.md)。
