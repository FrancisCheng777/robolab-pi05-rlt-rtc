# Stage-2 参数与结构对照（2026-11-01）

当前已对齐 RLinf 学习标量、AC 主体结构与 actor:critic=1:4 更新比例。保留队列输入、gamma=0.99、episode 采样调度和奖励定义。因此仍不能称为完整复现 RLinf，也不代表这些参数已在本任务验证最优。

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

集中默认值位于 `pi05_rlt_rtc/learning/defaults.py`，避免配置缺项时落回旧参数。新增训练参数校验，以及恢复时的参数一致性检查：旧 checkpoint 的 optimizer 含旧 LR，不能悄悄覆盖新配置。改变配置后应使用新训练目录；本次没有实现旧优化器的自动迁移。后续结构恢复新增 `ac_architecture=rlinf_mlp_queue_v2`；缺少此标识的旧两层 AC checkpoint 明确拒绝加载。新训练目录从新 AC 开始，Stage-1 π0.5/RLT checkpoint 可继续使用。

gamma 保持 **0.99 / control tick**，本地 TD target 使用真实持续时间的 `gamma^K`。官方配置为 0.96；在折扣时间单位和调度不同的情况下，不能只比较两个小数。本次没有调整折扣或奖励：本地仍是成功 terminal=1、失败/中间=0。

## “改了网络结构”具体指什么

主要指 **Stage-2 AC 小网络**：actor 生成动作 chunk，critic 评估动作；不是重新设计 π0.5 视觉/语言/动作主干。

| 项目 | 固定 RLinf 实现 | 当前本地实现 |
|---|---|---|
| actor 输入 | z_rl、本体状态、参考动作 chunk | 保留规划状态扩展，与参考动作拼接；取消独立 128 维分支 |
| actor 隐藏层 | 256→256→256，每层 Tanh | 相同；reference dropout 在拼接前只作用于参考输入 |
| actor 初始化 | 正交初始化 gain=√2，输出 gain=0.01√2，bias=0 | 相同 |
| critic | twin Q；每个 Q 三层 256，Linear→LayerNorm→Tanh，最后线性输出 | 相同 |
| critic 初始化 | 隐藏层 Xavier uniform，gain 为 Tanh gain；输出 Normal(0,0.02)，bias=0 | 相同 |
| critic 输入 | z_rl、本体状态、候选动作 chunk | 保留完整规划状态、当前完整参考计划及 mask、候选动作 chunk |

actor 实现见 `learning/actor.py`，critic 见 `learning/critic.py`（均相对于 `pi05_rlt_rtc/`）。保留双 Q、target EMA、min-Q bootstrap、tanh 前固定探索噪声和现有损失。本地控制动作维度及输入布局不同，因此不会直接加载官方 AC 权重。

此前本地使用两层 256/SiLU，actor 独立参考分支 128 维，critic 无 LayerNorm，主要采用 Linear 默认初始化。原有动机是隔离 reference dropout 和表达队列状态，但这些功能并不要求上述结构选择；没有消融结果支持它们。因此按用户确认恢复官方主体，仅保留有明确功能需求的输入扩展。

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

## 已恢复的更新比例与保留的 episode 调度

| 项目 | 当前本地设置 | 理由与不足 |
|---|---|---|
| 采样与学习 | rollout → confirmed hold → 人工标签 → GPU 更新 → 下一 episode | 避免单 4090 上 VLA/RTC 推理与 learner 同时争抢 GPU；是部署取舍，不是 PI/RLinf 算法要求 |
| actor:critic 更新比 | 已由 1:1 恢复官方 1:4 | 全局零起点 update_step % 4 == 0 时更新 actor；跨调用和 resume 连续 |
| 启动门槛 | 5 个有效 episode 且 32 条可学习 transition | 工程起点，没有官方实验支持这两个具体值 |
| BC warmup | 100 次 | 先模仿 teacher 再交给 AC 的本地流程；具体次数无消融支持 |
| 每批 AC 更新 | 每新增 10 个有效 episode，1000 次 critic、250 次 actor；首批额外 100 次 BC | 用户选择的分批调度，保持每条 100 次 critic 的更新预算；不等于官方 update_epoch=8，统计单位不同 |
| replay 容量 | 10000 条 transition | 官方 cache/sample window=200 对应其 rollout 存储单位，不能直接照抄为 200 条本地 transition |

更新计数 `updates` 表示 critic 迭代，`actor_updates` 表示 AC actor 更新，`warmup_updates` 单独记录 BC。target Q 每个 critic 迭代都 EMA 更新，actor 跳过的迭代不会伪造 actor loss。

官方 `train_actor_steps=2` 是 replay 就绪门槛的一部分，不能解读为每轮只做两次 actor 更新。batch 调成 256 后，本地仍允许从不足 256 条 transition 的池中有放回采样；因此早期 batch 内会重复样本，256 不代表 256 条独立经验。由于启动门槛与调度明确要求不动，本次保留这一行为。

其他尚未对齐的配置：本地 H/C/D=10/5/2、15 Hz、full-VJP RTC；官方参考 horizon=20、actor chunk=10，不能机械迁移其控制时序。Stage-1 本地 LR=2.5e-5、global batch=256、micro batch=1、2000 steps，与官方示例 1e-5、64、16、20000 steps 不同；这些关系到先前训练预算，均未纳入本次 Stage-2 改动。

## 验证与局限

在 4090 主机的 `/home/heinrich/robolab-pi05-rlt-rtc`，使用 `.venv-runtime/bin/python`、PyTorch 2.7.1+cu128：

- 先运行新回归测试，确认旧网络的上游数值对照、旧 1:1 cadence 和缺失结构版本检查失败；之后实施修改。
- 与固定上游模块直接数值对照：actor 与 critic 在相同随机种子下初始化/前向输出一致，使用扩展输入维度；reference dropout 不改变队列字段。
- 连续五个 episode 的保存/恢复检查：actor 更新数为 1/1/1/1/2，critic 为 1/2/3/4/5；BC warmup 仅首轮执行。验证 target Q 在不更新 actor 时仍更新。
- 保存/加载后 actor、critic 输出一致；旧 AC 架构版本明确拒绝。
- Toy 训练复用正式 QueueLearner，移除独立两层/1:1 更新循环；保留 fixture 显式优化参数。原测试的随机动作 Q 排序错误地包含终止 next_state 的补零，并要求 teacher-only replay 对未见动作排序；替换为全部实际 replay 的 TD 误差和终止状态/动作的真实回报误差检查，仍使用 <1 门槛，保留 BC/critic 下降与闭环成功率门槛。
- 完整回归：`OMP_NUM_THREADS=1 .venv-runtime/bin/python -m pytest tests -q --tb=short -rs`，**198 passed, 19 skipped**，42.51 秒。跳过项依赖真实数据、权重或未配置的 bridge 测试环境。日志见 `reports/official_ac_20261101/full.log`；修改前备份：同目录 `before.tgz`。

以上是代码与 mock 回归，不是插入成功率、真实训练收敛、完整模型 RTC 延迟或真机安全验收。本次没有启动机器人或训练真实任务模型。

## Stage-1 真机 SFT（2026-11-01）

当前配置唯一来源：`stage1_integration/configs/stage1_realworld.yaml`。参考固定 RLinf `c70606f08cdca259b8dec03d4430926b5b8fac9d` 的 `examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml`：global batch 64、LR 1e-5、warmup 1000、cosine 20000 步、min LR 2.5e-6、Adam 0.9/0.95、eps 1e-8、WD 1e-10、clip 1、RLT alpha 1、seed 42。首轮预算 8000 步属于本任务决定，不改变 20000 步衰减周期。

硬件适配为默认四卡 FULL_SHARD；micro-batch 经完整模型合成输入实测由 1 提升为 16，global batch 保持 64。保留 Panda 的 state9/action8、单 wrist、horizon 10，不照搬官方另一机器人动作接口。此处 batch 64 与本文 Stage-2 actor–critic 的 batch 256 分属两个阶段。
