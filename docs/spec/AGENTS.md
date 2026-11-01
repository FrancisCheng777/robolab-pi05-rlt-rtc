# 当前项目约束

用户已将项目明确改为纯真机：Panda + Robotiq 2F-85 + D405 腕部 + Kinect 外部。
读取 ../DEPLOYMENT.md 与 ../ARCHITECTURE.md。不得恢复仿真优先或 sim2real 的旧目标。

保持 H/32 与 C/8 空间区分、每请求一次 prefix、Stage-2 冻结 VLA/RLT、唯一 [D:D+C] 后缀、承诺前缀不变、最终计划作为下次 RTC 上下文、提交 U 作为 Critic 标签、按实际 K 计折扣。

只接受明确标注的真实 rollout 进入 RL replay。人工 reset，故障/干预样本剔除。不得将 mock 测试、空硬件参数、缺失模型或未经执行的 GPU/机器人实验声称为已验收。
