# Stage-2 参数与结构对照（2026-11-01）

本次仅对齐可直接映射的 RLinf 学习标量。gamma、网络结构、规划状态、更新调度和奖励定义均未改动。因此这是带本地 RTC 扩展的配置对齐，不能称为完整复现 RLinf，也不代表这些参数已在本任务验证最优。

## 固定参考来源

- RLinf commit：`c70606f08cdca259b8dec03d4430926b5b8fac9d`；[真机 RLT Stage-2 配置](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml)。采用仓库已固定的版本，不追踪浮动 main。
- [RLT actor 输入定义](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py)、[MLP 网络](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/models/embodiment/mlp_policy/mlp_policy.py)、[Q 网络](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/models/embodiment/modules/q_head.py)。
- [更新实现](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/workers/actor/fsdp_sac_policy_worker.py)。
- [PI RLT 论文](https://arxiv.org/html/2604.23073v1)：附录 B 支持训练时 50% reference masking；不能把下述所有 RLinf 数值都称为 PI 对本任务的推荐。

## 本次改动及理由

| 项目 | 原运行配置 | 本次 | 理由 |
|---|---:|---:|---|
| actor learning rate | 1e-4 | 3e-4 | 采用 RLinf 示例的 actor 优化器学习率，移除无实验依据的本地偏差 |
| replay batch size | 32 | 256 | 采用示例每次更新的样本数；这是小型 AC 的 batch，不是 Stage-1 VLA 的 batch |
| BC 权重 | 1 | 5 | 与示例 Q 权重 0.1 配合，恢复其相对约束强度 |
| reference dropout | 0.1 | 0.5 | 与 RLinf 配置及 PI 论文的 50% reference masking 一致 |
| actor / critic 梯度范数裁剪 | 硬编码 1 | 配置化 10 / 10 | 与示例优化器设置一致，确保 YAML 数值确实进入优化步骤 |

critic LR=3e-4、Q 权重=0.1、tau=0.005、tanh 前探索 std=0.002 原已一致，保持不变。设计 YAML 的旧 critic LR=1e-4 也同步更正为运行值，避免文档误导。主运行 YAML 与仓库内 Stage-1 集成目录所附运行 YAML 同步。

集中默认值位于 `pi05_rlt_rtc/learning/defaults.py`，避免配置缺项时落回旧参数。新增训练参数校验，以及恢复时的参数一致性检查：旧 checkpoint 的 optimizer 含旧 LR，不能悄悄覆盖新配置。改变配置后应使用新训练目录；本次没有实现旧优化器的自动迁移。网络 checkpoint 格式未改。

gamma 保持 **0.99 / control tick**，本地 TD target 使用真实持续时间的 `gamma^K`。官方配置为 0.96；在折扣时间单位和调度不同的情况下，不能只比较两个小数。本次没有调整折扣或奖励：本地仍是成功 terminal=1、失败/中间=0。

## “改了网络结构”具体指什么

主要指 **Stage-2 AC 小网络**：actor 生成动作 chunk，critic 评估动作；不是重新设计 π0.5 视觉/语言/动作主干。

| 项目 | 固定 RLinf 实现 | 本地已有实现（本次未改） |
|---|---|---|
| actor 输入 | z_rl、本体状态、参考动作 chunk 拼接 | 规划状态分支与参考动作分支分别投影，再拼接 |
| actor 隐藏层 | 256→256→256，每层 Tanh | 状态分支 256→256，中间 SiLU；参考分支单层 128 线性投影；拼接后线性输出，再 tanh |
| actor 初始化 | layer_init、输出层小增益 | PyTorch Linear 默认初始化，输出 bias 置零 |
| critic | twin Q；每个 Q 三层 256，Tanh、LayerNorm、专门初始化 | twin Q；每个 Q 两层 256，SiLU，无 LayerNorm，默认 Linear 初始化 |
| critic 输入 | z_rl、本体状态、候选动作 chunk | 完整规划状态、当前完整参考计划及 mask、候选动作 chunk |

actor 实现见 `learning/actor.py`，critic 见 `learning/critic.py`（均相对于 `pi05_rlt_rtc/`）。保留了双 Q、target EMA、min-Q bootstrap、tanh 前固定探索噪声等算法要素，但保留算法要素不等于结构一致。

**动机与证据界限：** 独立 reference 分支便于只遮蔽参考动作而保留队列信息；规划状态输入用于描述延迟执行。但这两点都不要求把官方三层改为两层、Tanh 改为 SiLU，或删除 LayerNorm。现有仓库没有消融结果证明这些网络选择优于官方；它们是本地实现选择，不能称为 RLinf / PI 推荐，也不能称为 RTC 必需。只增加输入特征也可以继续使用官方 MLP 主体。

Stage-1 则沿用 π0.5 + RLT 模块体系。核对 provenance 哈希，`openpi/pi0_config.py`、`openpi/rlt_config.py` 与固定上游一致；RLT 配置保留 2048 维、2 层、8 heads。直接对比上游 `rlt_token_transformer.py` blob `bed210d2272f4368ec67006f31c5e140b373c29f`，本地增加的是 encoder/decoder 的 non-reentrant gradient checkpointing 开关与调用，不改变层数、注意力或可训练参数结构。Stage-1 仍有 full-SFT、精度/梯度与显存工程补丁，见 `stage1_integration/stage1/patches/rlinf-full-sft.patch`；不能笼统说上游代码完全没改。本次未修改任何 Stage-1 模型文件。

## 队列状态为何加入

`learning/planning_state.py` 在 z_rl、本体状态以外记录：

| 状态 | 用途 / 动机 |
|---|---|
| committed pending prefix 与 mask | 表示机器人已承诺、尚未执行完且新动作不能覆盖的部分 |
| 上一次组合计划的尾部与 mask | 表示影响下一次 RTC 重叠引导的历史计划 |
| 最后执行的命令 | 提供执行边界上的动作上下文 |
| 时间特征 | 描述规划与执行的时间位置 |
| 当前完整参考计划与 mask（critic） | 使 Q 估计看到候选动作以外仍会参与执行的参考部分 |

理由是：即使图像、本体状态相同，队列里已有的动作不同，下一段实际执行也可能不同。显式输入这些记忆，是为了减少遗漏执行状态造成的歧义。**这是本地延迟控制的建模假设**，不是官方 RLT 要求，也没有证明它足以恢复 Markov 性或一定提升成功率。官方 AC 输入没有这些显式队列字段。延迟 transition 的奖励窗口、执行确认和实际持续时间也与此设计配合；本次全部保持原样。

## 更新调度为何不同

| 项目 | 本地现状（本次未改） | 理由与不足 |
|---|---|---|
| 采样与学习 | rollout → confirmed hold → 人工标签 → GPU 更新 → 下一 episode | 避免单 4090 上 VLA/RTC 推理与 learner 同时争抢 GPU；是部署取舍，不是 PI/RLinf 算法要求 |
| actor:critic 更新比 | 1:1；官方配置 1:4 | 本地简化；没有证据证明更好。属于用户要求暂不动的更新调度 |
| 启动门槛 | 5 个有效 episode 且 32 条可学习 transition | 工程起点，没有官方实验支持这两个具体值 |
| BC warmup | 100 次 | 先模仿 teacher 再交给 AC 的本地流程；具体次数无消融支持 |
| 每轮 AC 更新 | 初次与后续每个有效 episode 各 100 次 | 便于交互；不等于官方 update_epoch=8，统计单位不同 |
| replay 容量 | 10000 条 transition | 官方 cache/sample window=200 对应其 rollout 存储单位，不能直接照抄为 200 条本地 transition |

官方 `train_actor_steps=2` 是 replay 就绪门槛的一部分，不能解读为每轮只做两次 actor 更新。batch 调成 256 后，本地仍允许从不足 256 条 transition 的池中有放回采样；因此早期 batch 内会重复样本，256 不代表 256 条独立经验。由于启动门槛与调度明确要求不动，本次保留这一行为。

其他尚未对齐的配置：本地 H/C/D=10/5/2、15 Hz、full-VJP RTC；官方参考 horizon=20、actor chunk=10，不能机械迁移其控制时序。Stage-1 本地 LR=2.5e-5、global batch=256、micro batch=1、2000 steps，与官方示例 1e-5、64、16、20000 steps 不同；这些关系到先前训练预算，均未纳入本次 Stage-2 改动。

## 验证与局限

在 4090 主机的 `/home/heinrich/robolab-pi05-rlt-rtc`，使用 `.venv-runtime/bin/python`、PyTorch 2.7.1+cu128：

- 针对性测试：25 passed（参数实际进入 learner、梯度裁剪量级、跨调用 actor/critic 1:1 更新、resume 参数不一致拒绝、session/platform 集成）。
- 完整 `OMP_NUM_THREADS=1 .venv-runtime/bin/python -m pytest tests -q --tb=short`：**193 passed, 19 skipped**，30.88 秒。
- 日志：`reports/official_params_20261101/{focused,full}.log`；修改前备份：同目录 `before.tgz`。
- `git diff --check` 与 Python 语法检查通过。

以上是代码与 mock 回归，不是插入成功率、真实训练收敛、完整模型 RTC 延迟或真机安全验收。本次没有启动机器人或训练真实任务模型。
