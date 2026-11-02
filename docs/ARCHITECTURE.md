# 真机架构

`physical_sft.PhysicalChunkDataset → PandaInputs → OpenPI normalization/tokenization → RLinf π0.5 + RLT SFT`

`FrankaEnvAdapter → ObservationSnapshot → FeaturePlanner → RLinfFeatureReferenceService + RTC → WallClockController → Unix socket → C++ libfranka/Robotiq`

`recorded rollout → human annotation → DelayedTransitionAssembler → ReplayBuffer → QueueLearner → atomic policy snapshot`

## 保留的算法语义

VLA 模型空间 `[H,32]`，真实命令空间 `[C,8]`。一个 prefix pass 同时产生 z 与完整 H 参考。Stage 2 冻结 VLA/RLT；full_vjp 只对 noisy action 输入求梯度，不能用外层 inference_mode 包住它。

Actor、teacher 和 BC 共用 `[D:D+C]`。最终计划为 `P || U || reference_tail`；下一次 RTC 使用经过 Actor 修改的最终计划。控制器是 tick 与动作队列唯一所有者，不能覆盖已经承诺的前 D 步。晚结果、状态过期、相机失败、桥接错误触发停止。

Critic 的动作是提交计划 U，实际混合执行序列 E 另存。奖励窗从 capture_tick 开始，使用实际 K 步与 gamma^K；不是把奖励移动到 U 开始执行时。下一规划状态是在真实下一次请求时捕获的同版本特征。无有效下一状态的非终止样本被 censor。

## 物理观测与动作

模型动作是相邻 15Hz 实测关节位移除以 `0.05`，夹爪宽度是 `(action[7]+1)*0.085/2`。9D 状态中的两个半宽只是同一个 Robotiq aperture 的派生表示。不要再次乘时间步长或施加 DeltaActions。

执行方式由 `franka.joint_command_mode` 明确区分。默认 `measured_delta` 使用 `q_measured + action[:7]*0.05`。对低层力矩跟踪器，微小目标可能来不及执行就被下一次实测位置重置；`session` 支持 `bounded_reference`，仅在新的 15Hz 动作到达时向参考目标增加一次位移，100Hz 保活重复发送同一目标。参考领先实测位置的幅度不超过 `min(reference_lead_rad, max_tracking_error)`，配置最大为 0.05rad；截断的残余直接丢弃，hold 后重新以实测位置初始化。关节位置约束、速度和碰撞阈值仍有效。这是执行控制器适配，不是更改训练标签；配置写入 runtime contract，旧的策略/验收不能静默复用。直接 `FrankaEnvAdapter` 不支持该模式，会要求使用持久 owner 的 `session`。

Kinect 与 D405 后台取帧，控制线程只读最新缓存。时间戳为主机接收时钟，不代表硬件曝光同步。相机方向、裁剪、letterbox 与采集流程一致。

## 实验生命周期

每次显式启动一个 bridge 和 trial；人工复位。controller 保存真实时序，训练不自动采集。人工标签让奖励变为有效；成功/失败为任务终止，中止/干预/控制故障不进入正常 replay。

learner 与 rollout 通过持久化文件解耦，可以并行运行；策略仅在下一条 trial 载入，不热替换 in-flight 动作。HTTP planner 保留用于推理，但训练 rollout 需要本地 planner 记录 z/reference。

虚拟时钟、toy 与 mock 只属于软件验证，不是物理仿真或训练环境。
