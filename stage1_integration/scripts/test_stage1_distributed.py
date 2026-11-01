"""Two-rank CPU FSDP regression, no CUDA allocation and no Ray cluster."""

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "vendor/rlinf_stage1"),
    str(ROOT / "tests/stage1"),
]
# Import real package initializers here; model fixture only shrinks tensor dimensions.
from model_fixture import batch, tiny_model
from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models.embodiment.openpi import get_model
from rlinf.utils.utils import warmup_optimizer_state

from stage1.config import load_config, validate_contract


def main():
    assert callable(get_model)
    torch.set_num_threads(2)
    from rlinf.scheduler import Worker

    # Only the accelerator cache/RNG boundary is replaced for this CPU regression.
    Worker.torch_platform = SimpleNamespace(
        synchronize=lambda: None,
        ipc_collect=lambda: None,
        empty_cache=lambda: None,
        is_available=lambda: False,
    )
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise ValueError("Run with torchrun --nproc_per_node=2")
    cfg = load_config()
    validate_contract(cfg)
    from rlinf.config import validate_fsdp_cfg, validate_sft_cfg

    validate_sft_cfg(cfg)
    validate_fsdp_cfg(cfg.actor)
    torch.manual_seed(22)
    model = tiny_model().train()
    reference = copy.deepcopy(model)
    model.gradient_checkpointing_enable({"use_reentrant": False})
    policy = get_fsdp_wrap_policy(
        model, cfg.actor.fsdp_config, is_lora=False, model_type="openpi"
    )
    model = FSDP(
        model,
        device_id=torch.device("cpu"),
        auto_wrap_policy=policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=False,
        device_mesh=init_device_mesh("cpu", (2,), mesh_dim_names=("fsdp",)),
    )
    wrappers = [m for m in model.modules() if isinstance(m, FSDP)]
    assert len(wrappers) > 4 and all(
        m.sharding_strategy == ShardingStrategy.FULL_SHARD for m in wrappers
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    warmup_optimizer_state(optimizer)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    # Same effective batch: two ranks x two accumulated micro-batches.
    for micro in range(2):
        torch.manual_seed(100 + 10 * rank + micro)
        loss = model(data=batch())["loss"] / 2
        loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    # Exact FP32 reference for accumulated, averaged gradients.
    reference_optimizer = torch.optim.AdamW(
        reference.parameters(), lr=1e-4, foreach=False
    )
    warmup_optimizer_state(reference_optimizer)
    for r in range(2):
        for micro in range(2):
            torch.manual_seed(100 + 10 * r + micro)
            (reference(data=batch())["loss"] / 4).backward()
    reference_optimizer.step()
    state = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=False)
    )
    state = {k: v.detach().clone() for k, v in state.items()}
    for key, value in state.items():
        torch.testing.assert_close(
            value, reference.state_dict()[key], rtol=2e-4, atol=2e-6, msg=key
        )

    output = Path(os.environ["PI05_DISTRIBUTED_TEST_OUTPUT"])
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    # FSDP CPU->CPU full-state offload is unsupported in this torch build.
    # Exercise native DCP; export the already-CPU full state without that offload.
    FSDPStrategy.save_checkpoint(
        model, optimizer, scheduler, str(output), save_full_model_weights=False
    )
    if rank == 0:
        (output / "model_state_dict").mkdir(parents=True, exist_ok=True)
        torch.save(state, output / "model_state_dict/full_weights.pt")
    # Mutate weights, then exercise RLinf's own checkpoint restore.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    for slot in optimizer.state.values():
        slot["exp_avg"].zero_()
    FSDPStrategy.load_checkpoint(model, optimizer, scheduler, str(output))
    actual_optimizer = optimizer.state_dict()
    # DCP correctly omits state for parameters with no local shard on this rank.
    for snapshot in (saved_optimizer, actual_optimizer):
        snapshot["state"] = {
            k: v for k, v in snapshot["state"].items() if v["exp_avg"].numel()
        }
    torch.testing.assert_close(actual_optimizer, saved_optimizer, rtol=0, atol=0)
    restored = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=False)
    )
    for key, value in restored.items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0, msg=key)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        amp_loss = model(data=batch())["loss"]
    amp_loss.backward()
    assert torch.isfinite(amp_loss)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    if rank == 0:
        full = torch.load(
            output / "model_state_dict/full_weights.pt", weights_only=True
        )
        fresh = tiny_model()
        fresh.load_state_dict(full, strict=True)
        report = {
            "world_size": 2,
            "device": "cpu/gloo",
            "fsdp_wrappers": len(wrappers),
            "full_shard": True,
            "accumulation_matches_reference": True,
            "rlinf_dcp_save_restore": True,
            "optimizer_restore": True,
            "gpu_export_offload_tested": False,
            "full_weights_strict_reload": True,
            "bf16_autocast_backward": True,
            "gpu_memory_measured": False,
        }
        (ROOT / "reports/stage1_distributed_cpu.json").write_text(
            json.dumps(report, indent=2)
        )
        print(json.dumps(report))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
