"""RLinf's native SFT worker with the plug dataset and first-step diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from pi05_rlt_rtc.contracts.physical import ACTION_CONTRACT_ID as CONTRACT_ID
from pi05_demo.chunks import WINDOW_POLICY

from .config import ROOT, validate_contract
from .data import PlugDataset, SyntheticDataset


def gradient_group(name):
    name = name.replace("_fsdp_wrapped_module.", "")
    if name.startswith("img."):
        return "vision"
    if name.startswith("rlt_module.encoder."):
        return "rlt_encoder"
    if name.startswith("rlt_module.decoder."):
        return "rlt_decoder"
    if name.startswith("llm."):
        expert = re.search(
            r"llm\.(?:layers\.\d+\.(?:attn\.[qkvo]_proj|mlps|pre_attention_norms|pre_ffw_norms)|final_norms)\.1\.",
            name,
        )
        return "action_expert" if expert else "language"
    return "action_projection"


class PlugSftWorker(FSDPVlaSftWorker):
    def build_dataloader(self, data_paths, eval_dataset=False):
        if eval_dataset:
            raise ValueError(
                "Use a separate held-out evaluation; no training-data SR metric"
            )
        if self._world_size != self.cfg.stage1.expected_world_size:
            raise ValueError(f"Expected 2 GPU workers, got {self._world_size}")
        s, m = self.cfg.stage1, self.cfg.actor.model.openpi
        if s.synthetic:
            dataset = SyntheticDataset(
                s.tokenizer_path, m.action_horizon, m.max_token_len
            )
        else:
            dataset = PlugDataset(
                s.dataset_root, s.tokenizer_path, m.action_horizon, m.max_token_len
            )
        sampler = DistributedSampler(
            dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=True,
            seed=self.cfg.actor.seed,
            drop_last=True,
        )
        loader = StatefulDataLoader(
            dataset,
            batch_size=self.cfg.actor.micro_batch_size,
            sampler=sampler,
            num_workers=s.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        if not len(loader):
            raise ValueError("Dataset has no complete per-rank micro-batch")
        return loader, None

    def get_max_steps_per_epoch(self):
        return max(1, len(self.data_loader) // self.gradient_accumulation)

    def model_provider_func(self):
        validate_contract(self.cfg)
        model = super().model_provider_func()
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        if frozen:
            raise ValueError(f"Full SFT unexpectedly froze parameters: {frozen[:8]}")
        self._parameter_count = sum(p.numel() for p in model.parameters())
        self._module_counts = {}
        for name, p in model.named_parameters():
            group = gradient_group(name)
            self._module_counts[group] = self._module_counts.get(group, 0) + p.numel()
        return model

    def init_worker(self):
        super().init_worker()
        wrappers = [m for m in self.model.modules() if isinstance(m, FSDP)]
        if len(wrappers) < 4 or any(
            m.sharding_strategy != ShardingStrategy.FULL_SHARD for m in wrappers
        ):
            raise RuntimeError(
                "Expected nested FULL_SHARD wrappers, including RLT layers"
            )
        module = self.model.module
        flags = [
            module.llm.gradient_checkpointing,
            module.img.encoder.gradient_checkpointing,
            module.rlt_module.encoder.gradient_checkpointing,
            module.rlt_module.decoder.gradient_checkpointing,
        ]
        if not all(flags):
            raise RuntimeError(
                "Gradient checkpointing is not active in all four branches"
            )
        self._diagnostic_dir = (
            Path(self.cfg.runner.logger.log_path)
            / self.cfg.runner.logger.experiment_name
        )
        self._diagnostic_dir.mkdir(parents=True, exist_ok=True)
        self._write_record(
            "initialization",
            parameters=self._parameter_count,
            parameter_groups=self._module_counts,
            fsdp_wrappers=len(wrappers),
            precision="FP32 weights/gradients/Adam, BF16 autocast",
        )

    def _write_record(self, event, **fields):
        path = self._diagnostic_dir / f"diagnostics_rank_{self._rank}.jsonl"
        with path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "event": event,
                        "rank": self._rank,
                        "step": self.global_step,
                        **fields,
                    }
                )
                + "\n"
            )

    def get_train_model_output(self, batch):
        loss, metrics = super().get_train_model_output(batch)
        finite = torch.isfinite(loss.detach()).to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("Nonfinite SFT loss on at least one rank")
        return loss, metrics

    def optimizer_step(self):
        # After all accumulation backwards, before clipping/zero_grad.
        if self.global_step < self.cfg.stage1.gradient_audit_steps:
            groups = (
                "vision",
                "language",
                "action_expert",
                "action_projection",
                "rlt_encoder",
                "rlt_decoder",
            )
            norms = torch.zeros(len(groups), device=self.device, dtype=torch.float64)
            counts = torch.zeros_like(norms)
            for name, p in self.model.named_parameters():
                if p.grad is not None and p.grad.numel():
                    i = groups.index(gradient_group(name))
                    norms[i] += torch.linalg.vector_norm(p.grad).double().square()
                    counts[i] += p.grad.numel()
            dist.all_reduce(norms)
            dist.all_reduce(counts)
            if (
                not torch.isfinite(norms).all()
                or (norms <= 0).any()
                or (counts <= 0).any()
            ):
                raise RuntimeError(
                    f"Missing, zero or nonfinite module gradients: {dict(zip(groups, norms.tolist()))}"
                )
            self._write_record(
                "gradient_audit",
                norms=dict(zip(groups, norms.sqrt().tolist())),
                gradient_elements=dict(zip(groups, counts.tolist())),
            )
        return super().optimizer_step()

    def run_training(self):
        torch.cuda.reset_peak_memory_stats()
        metrics = super().run_training()
        torch.cuda.synchronize()
        self._write_record(
            "optimizer_step",
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            **{k: float(v) for k, v in metrics.items()},
        )
        return metrics

    def save_checkpoint(self, save_path, step=0):
        super().save_checkpoint(save_path, step)
        if self._rank == 0:
            path = Path(save_path)
            shutil.copy2(
                ROOT / "configs/training_contract.json", path / "training_contract.json"
            )
            if not self.cfg.stage1.synthetic:
                for name in ("norm_stats.json", "manifest.json", "action_codec.json"):
                    shutil.copy2(Path(self.cfg.stage1.dataset_root) / name, path / name)
            shutil.copy2(
                self.cfg.stage1.tokenizer_path, path / "paligemma_tokenizer.model"
            )
            OmegaConf.save(self.cfg, path / "stage1_resolved.yaml", resolve=True)
            upstream = json.loads(
                (ROOT.parent / "provenance/rlinf_stage1_upstream.json").read_text()
            )
            patched = {}
            for name, original in upstream["files"].items():
                source = ROOT / "vendor/rlinf_stage1" / name
                if source.is_file():
                    digest = hashlib.sha256(source.read_bytes()).hexdigest()
                    if digest != original:
                        patched[name] = digest
            metadata = {
                "action_contract_id": CONTRACT_ID,
                "synthetic": bool(self.cfg.stage1.synthetic),
                "upstream_commit": upstream["commit"],
                "patched_files": patched,
                "normalization": "quantile_q01_q99_then_pad",
                "tokenize_state_before_padding": True,
                "camera_views": ["wrist"],
                "policy_image_mapping": {"base_0_rgb": "wrist"},
                "tokenizer_sha256": hashlib.sha256(Path(self.cfg.stage1.tokenizer_path).read_bytes()).hexdigest(),
                "window_policy": dict(WINDOW_POLICY),
                "parameters": self._parameter_count,
                "world_size": self._world_size,
                "step": step,
            }
            (path / "stage1_metadata.json").write_text(json.dumps(metadata, indent=2))
        dist.barrier()
