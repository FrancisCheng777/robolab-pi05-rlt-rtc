# 当前项目说明：纯真机 π0.5 + RLT + RTC

本项目采用 Panda + Robotiq 2F-85 + 腕部 D405 + 外部 Kinect，用真实插排示教做 π0.5/RLT SFT，再用真实 rollout 训练冻结表示之上的 actor–critic。没有仿真训练或 sim2real 阶段。

关键工作是保持示教、SFT、RTC、控制队列与 replay 的动作/时间语义一致。动作是归一化关节增量和夹爪开口；Critic 评价提交动作块，实际执行轨迹单独审计；奖励由人工标注真实任务终局，不由代码虚构。

实现与操作分别见 ARCHITECTURE.md、DEPLOYMENT.md。软件测试、真实 GPU 训练、硬件运行和成功率必须分别报告；不能把 CPU 替身测试或另一个项目的结果当成本项目的实验结论。

本次迁移前的完整面试手册与旧规范已保存在项目 `.codex-work/physical-migration-20261029-123629/before.tar.gz`，便于检索历史内容；不再作为当前执行指南。
