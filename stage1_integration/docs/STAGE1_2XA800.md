# Stage-1 双卡兼容入口

当前权威配置与说明：[`stage1_realworld.yaml`](../configs/stage1_realworld.yaml) 和 [四卡真机训练说明](../../docs/STAGE1_4XA800.md)。默认主机是 4×A800 80GB。

`stage1_2xa800.yaml` 仅覆盖卡数为 2，其余参数继承同一真机配方：global batch 64、LR 1e-5、warmup 1000、cosine 20000 步、首轮最多 8000 步。旧 batch 256 / LR 2.5e-5 / 双有效相机描述已停用。

```bash
CUDA_VISIBLE_DEVICES=0,1 PI05_WORLD_SIZE=2 bash scripts/start_stage1_integrated.sh inspect actor.micro_batch_size=1
```

micro-batch=1 时两卡每卡累积 32 次，四卡每卡累积 16 次。完整续训要求相同卡数，不能直接把双卡 DCP checkpoint 当成四卡 checkpoint 恢复。历史 CPU 双进程回归通过不代表完整 CUDA 模型验收。
