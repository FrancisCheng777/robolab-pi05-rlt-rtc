# 软件测试索引

T01–T68 保留算法回归编号；真机新增测试使用具名测试。实际执行见 ../../tests/RESULTS.md 和 ../../runs/physical_migration/pytest.log。

| ID | 检查 | 预期 |
|---|---|---|
| T01 | 检查实际 state/action shape | 与 9/8 契约一致，不截断容错 |
| T02 | 两相机对应和 RGB 范围 | 不换位、不 RGB/BGR 互换、不重复归一化 |
| T03 | 按 episode 生成 H 步动作 | 通用窗口末尾补零并有 mask，不跨 episode；真机 SFT 完整窗口另由 test_physical_contract.py 覆盖 |
| T04 | 语言字段缺失但有默认 prompt | 正常按配置构造；未知任务必须报错 |
| T05 | 环境动作 encode/decode 回环 | 在声明可逆域和容差内一致 |
| T06 | 32 维 padding guidance | padding 上的误差不影响有效控制维的引导目标定义 |
| T07 | delta 控制数据 | 没有额外 DeltaActions 重复应用 |
| T08 | normalization 哈希改变 | 消费端明确拒绝旧 replay/部署 schema |
| T09 | 完整 H 与 C 后缀 | H=10/C=5/D=2 得到 index 2…6，不是 0…4 |
| T10 | 状态预处理一致性 | 同一原始状态在训练与推理的 proprio 完全同义 |
| T11 | 仅反向 RLT loss | encoder/decoder 有梯度，VLA 无梯度 |
| T12 | 仅反向 VLA loss | 不通过 RLT 模块改造其重构参数 |
| T13 | decoder teacher shifting | 不可访问当前/未来目标 embedding |
| T14 | prefix 长度超过配置 | 明确异常，不能静默裁剪 |
| T15 | prefix padding mask | 改动无效 padding 不改变有效 loss/token 行为到不合理程度 |
| T16 | Stage 1 小样本过拟合 | 两个 loss 有可解释趋势，输出非 NaN；不是性能保证 |
| T17 | Stage 2 更新前后参数 checksum | 冻结 VLA/encoder 不变，Actor/Critic 按训练更新 |
| T18 | Stage 2 RLT 权重缺失 | 加载失败，不能随机初始化继续 |
| T19 | 保存重载，同观察同初始噪声 | 输出在预先声明的精度容差内一致 |
| T20 | 每请求 prefix 调用次数 | 一次；z 与 reference 使用同一观察与模型版本 |
| T21 | 全零 mask | full_vjp 与普通 sampler 退化一致 |
| T22 | 无 previous plan | 正常首次采样，不读取未初始化上下文 |
| T23 | Native t 与 canonical τ | 无引导同噪声回归通过，无符号翻转 |
| T24 | 常量速度场 | Jacobian 为单位阵，VJP 与手工结果一致 |
| T25 | 线性速度 `v=A*x+b` | `J=I+(1-τ)A`，转置积正确 |
| T26 | 小非线性速度有限差分 | VJP 方向导数误差在数值精度阈值内 |
| T27 | 软 mask 权重 | 使用 W 一次，而不是误用 W² |
| T28 | τ=0、末步、单去噪步 | 无 Inf/NaN，clip 边界行为明确 |
| T29 | frozen parameters + input gradient | noisy action 梯度存在、参数梯度不存在 |
| T30 | 外层 inference_mode 错误注入 | 检测/拒绝不合法路径，而不是悄悄退回 approx |
| T31 | 去噪步数扩大/重复请求 | 不保留跨步图，不出现持续显存增长 |
| T32 | approx 与 full_vjp 模式分派 | 到达不同实现，日志与配置一致 |
| T33 | C=5/D=2 编号动作 | 第 12 节时序表精确匹配 |
| T34 | D=0 | 正常同步边界，无负索引/空前缀错误 |
| T35 | D=C | 当前窗口全部 P，U 全部进入下一 pending；不丢信用路径 |
| T36 | D>C 或 D+C>H | 初始化立即拒绝 |
| T37 | 响应早到 | 等待固定 apply_start，不提前覆盖 P |
| T38 | 响应晚到 | 拒绝/安全处理，不挪到随机 offset |
| T39 | 响应重复、乱序、跨 episode | 不重复执行，旧响应拒绝 |
| T40 | reset 时仍有在途推理 | 新 episode 不能收到旧动作 |
| T41 | 消费 cursor 与 plan generation | 正常 tick 消费不错误作废合法请求 |
| T42 | RLT 修改很大的 U | 下一 RTC target 来自组合计划中的 U，不是旧 VLA reference |
| T43 | 新规划生成的前 D 步不相等 | 仍执行旧 P；记录引导误差而非修改承诺 |
| T44 | GPU/网络故意延迟 | 控制循环继续；deadline miss 正确计数 |
| T45 | NaN、错误 shape、错误 stats | 拒绝计划，无命令外溢 |
| T46 | 安全接管/急停 | 作废 generation、记录干预，不伪装正常 prefix 保持 |
| T47 | C=5/D=2 一条 replay | actions 是提交 U，executed_trace 是 P+U 前三步，两者有区分 |
| T48 | 下一 pending | 精确为上个 U 的后 D 步 |
| T49 | 折扣数值样例 | rewards=[1,2,3]、γ=.9 时 R=5.23，K=3 |
| T50 | 非终止 bootstrap，下一 Q=4 | T49 对应 target=8.146；终止时为 5.23 |
| T51 | 提前终止与 reward padding | gamma 指数用真实 K，不是容器 C |
| T52 | terminal obs 与 reset obs | 不错连 episode，不用 reset 后状态 bootstrap |
| T53 | reference dropout | 仅改变 Actor 输入参考，不改监督目标或 Critic 的实际状态 |
| T54 | BC warmup tanh 一致 | 训练 loss 比较的是部署有界命令 |
| T55 | Actor 的 Q 梯度 | Q 参数不更新但 Actor 经 Q 得到动作梯度 |
| T56 | next state 当前参考/队列 | 与 next capture/version 对齐，不用 learner 当前随机重算 |
| T57 | deadline miss/干预样本 | 标记并从普通 learner 过滤，原始日志仍完整 |
| T58 | 保存恢复 replay/schedule | 计数与实际策略匹配；不能隐瞒 replay 被清空 |
| T59 | 旧同步 replay 输入新 learner | schema 不匹配时拒绝 |
| T60 | 固定种子小型延迟控制 toy task | Q target 与策略改进方向符合解析/可枚举预期 |
| T61 | 上游 B0 小规模端到端 | 真实数据、真实权重、训练与独立评估闭环 |
| T62 | B2 虚拟延迟训练 | π0.5+RLT+full_vjp 同时启用，存在有效 Q/BC 更新 |
| T63 | B2 真实墙钟异步 | 模型计算期间控制器仍按 tick 消费命令 |
| T64 | learner 并行负载下 profiling | p50/p95/p99、deadline miss、实际硬件全部报告 |
| T65 | 独立部署导出重载 | 不依赖训练进程内存，不调用 Critic/decoder |
| T66 | 关闭专家的独立评估 | 有清晰 trial 数、成功数、失败原因和种子 |
| T67 | 完整包无 mock 混入 | mock 标签、模型来源和特征 checksum 可追溯 |
| T68 | 超长运行内存与日志 | 资源受控；关键执行账本没有不可解释丢失 |

新增：test_physical_contract.py 覆盖物理动作、真实 SFT 窗口与导出边界；test_physical_rollout.py 覆盖记录、终局奖励、异常剔除、replay 指纹和离线训练续训。这些使用 CPU 测试替身，不是实际相机、GPU 或机器人验证。
