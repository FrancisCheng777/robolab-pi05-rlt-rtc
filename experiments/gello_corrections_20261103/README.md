# GELLO 纠正示范分支（2026-11-03）

远端代码：`/home/heinrich/robolab-pi05-gello-corrections-20261103`。
新数据：`/home/heinrich/pi05-corrections-20261103`（主机内置盘）。
原 Stage1、A_fixed 第 10 批、RLinf 对照和 EEF 验证分支均保留。

## 采集

退出原 shadow 终端 q，再执行：

```bash
bash ~/collect_pi05_gello_corrections.sh
```

1. e 回位/张爪，等待回位完成，r 开始。
2. teacher 负责抓取；抓稳后按 g。没有 g 不进入人工示范。
3. 提前握好 GELLO、把扳机按到底。teacher 阶段 GELLO 不控制机械臂。
4. 区域触发后自动保持、结束 teacher 队列并等待最后的规划返回；确认扳机闭合持续 0.15 秒后自动设定相对起点。
5. 看到“GELLO 已接管”后移动 GELLO 示范纠偏和插入。无需第二次按键接管，机械臂只跟随接管后的 GELLO 相对关节变化。
6. 持续按住扳机；松扳机会停止并保持，不发送张爪指令，此轮不用于 BC。成功 c，失败 a；x 保持，e 停止并回位/张爪，q 退出。c 后仍保持插头，e 才回位/张爪。

保留 V3 区域：原实测 EEF 中心，各轴 ±25 mm，姿态 ≤10°，关节速度 ≤0.10 rad/s，持续 0.2 秒，每轮一次。
示范相对参考速度上限 0.15 rad/s，参考领先上限 0.05 rad；控制频率 15 Hz，既有 CPU owner 继续刷新已提交参考。示范中夹爪目标固定闭合，扳机是持续确认条件。真实 USB 缓存超过 50 ms 会拒绝继续示范。

## 保存与准入

每轮在 `eval/<时间>/episodes/<编号>/` 保存：

- correction.json：完成状态、g/区域/实际交接 tick、相对起点、结果、异常、校验和、训练资格。
- human_frames.jsonl：人控片段的原始观测（含图像）、实际下发动作、ACK、关节目标及 GELLO 关节读数。
- teacher_observations、teacher_ledger.jsonl、teacher_hardware_commands.json：接近阶段审计。
- takeover_events.json：区域判定日志。

仅 g 已确认、实际完成交接、无故障/缺帧、明确 c 成功、至少一个完整 H=10 窗口的示范进入 BC。失败、中止、松扳机和没按 g 的轮次保留诊断，但不进入这批成功示范预热。没有自动删除任何源数据。

## 准备特征并预热新 Actor

先录 3–5 条成功示范并 q 退出采集，保持 teacher 常驻，然后执行：

```bash
bash ~/train_pi05_gello_corrections.sh --steps 500
```

此命令使用与采集相同的会话锁；有真机会话运行时拒绝启动特征提取，避免共享 teacher 抢占。
冻结 teacher 从示范图像/状态提取特征。使用人动作构造 D=5 前缀及之后 C=5 的监督目标；不会把同一时刻动作错误当作延迟后的输出。当前 VLA 参考仍作为输入，BC 输出目标为人动作。

人控时并未真的提交 Actor 未来队列，因此这里显式标记为 hindsight queue reconstruction 的监督学习样本，不写入普通 RL replay，不伪造 Bellman 转移或 Critic 奖励。新 Actor 从头初始化做 BC；Critic 不更新，保存的 bundle 的 critic_state 为 None，后续在线 RL 需另行初始化/训练 Critic。

每次训练在独立 `bc_runs/<时间-随机标识>/` 输出 policy.pt、training_report.json；不会覆盖前一版或原 A_fixed。至少三条示范时按整条 episode 留出一条验证，报告验证 MSE 与夹爪张开预测比例。少于三条时只有训练误差，报告明确标记无留出验证。特征缓存校验原始帧与标注，不是训练有效性保证。

## 新 Actor 评测入口

训练完成后，用输出的完整路径：

```bash
bash ~/eval_pi05_correction_actor.sh /home/heinrich/pi05-corrections-20261103/bc_runs/<本次训练目录>/policy.pt
```

评测保持同样 g 和区域规则；在区域前仍是原 teacher，在区域后是新 Actor，无探索。这是正常 Actor 输出，不是 GELLO 模式。不要把 BC 训练误差当成真机成功率。

## 验证

- 70 项测试通过：相对起点无关节跳变、闭爪映射、限速/防积分累积、松扳机保持、时间窗口与 D/C 对齐、无 g 排除、失败/损坏/模拟数据隔离、完整采集状态机、特征缓存、BC 降损、bundle 加载、既有 owner/bridge 回归。
- 实际 GELLO 7 关节与扳机串口读数已读取成功；探测时扳机处于松开（85 mm）状态。没有驱动 Panda 或 Robotiq。
- 复用 resident teacher 的只读预检、内置盘 fsync、脚本语法和 Python 编译检查通过。
- 原 A_fixed round10 ID 仍为 `queue-rlt:e3315c88621f07c836cdb0c62841a1764088619a674e9cd69077c681870a9481`。
- 实际机械臂接管尚未执行，需要操作员首轮验收。

GELLO 所需纯 Python dynamixel_sdk/serial 来自既有 `/home/heinrich/robolab-control-system/.venv/lib/python3.11/site-packages`，复制至本分支 runtime_support；没有修改共享 Python 环境。
