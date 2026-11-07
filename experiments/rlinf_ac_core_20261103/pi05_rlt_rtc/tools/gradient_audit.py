"""Toy Stage-1 gradient-boundary audit.

This does **not** claim to smoke-test the real pi0.5/RLT model; it verifies
the *gradient boundary rules* this project relies on (guide section 7.2) on a
tiny coupled model:

- RLT reconstruction loss must not train the VLA backbone (prefix stop-grad);
- VLA loss must train the VLA backbone without touching RLT encoder/decoder;
- the RLT decoder must not see the current target embedding (teacher shifting).

Result is labelled ``toy_gradient_audit``; the real Stage 1 smoke is NOT_RUN
without RLinf + weights.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ToyStage1Model(nn.Module):
    def __init__(self, dim: int = 16, z_dim: int = 8, seq_len: int = 6) -> None:
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.vla = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.rlt_encoder = nn.Sequential(nn.Linear(dim, z_dim), nn.SiLU(), nn.Linear(z_dim, z_dim))
        self.rlt_decoder = nn.Sequential(nn.Linear(z_dim + dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def prefix(self, x: torch.Tensor) -> torch.Tensor:
        return self.vla(x)

    def z(self, prefix: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():  # upstream: stop-gradient on the prefix
            return self.rlt_encoder(prefix.mean(dim=1))

    def reconstruct(self, z: torch.Tensor, teacher_prefix: torch.Tensor) -> torch.Tensor:
        # Teacher-forced causal shifting: predictions for positions 1..T use
        # the detached target embeddings 0..T-1 only.
        shifted = teacher_prefix.detach()[:, :-1, :]
        z_expanded = z[:, None, :].expand(-1, shifted.shape[1], -1)
        logits = self.rlt_decoder(torch.cat([z_expanded, shifted], dim=-1))
        return logits


def run_toy_gradient_audit(seed: int = 0) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = ToyStage1Model()
    x = torch.randn(2, model.seq_len, model.dim)
    vla_target = torch.randn(2, model.seq_len, model.dim)

    # --- case 1: RLT reconstruction loss only
    prefix = model.prefix(x)
    z = model.z(prefix)
    recon = model.reconstruct(z, prefix)
    rlt_loss = torch.nn.functional.mse_loss(recon, prefix.detach()[:, 1:, :])
    model.zero_grad(set_to_none=True)
    rlt_loss.backward()
    vla_grad_none = all(p.grad is None for p in model.vla.parameters())
    encoder_has_grad = all(p.grad is not None for p in model.rlt_encoder.parameters())
    decoder_has_grad = all(p.grad is not None for p in model.rlt_decoder.parameters())

    # --- case 2: VLA loss only
    model.zero_grad(set_to_none=True)
    vla_loss = torch.nn.functional.mse_loss(model.prefix(x), vla_target)
    vla_loss.backward()
    vla_has_grad = all(p.grad is not None for p in model.vla.parameters())
    encoder_grad_none = all(p.grad is None for p in model.rlt_encoder.parameters())
    decoder_grad_none = all(p.grad is None for p in model.rlt_decoder.parameters())

    # --- case 3: decoder cannot see the current/future target embedding
    prefix_a = model.prefix(x)
    z_a = model.z(prefix_a)
    with torch.no_grad():
        prefix_b = prefix_a.clone()
        prefix_b[:, 3:, :] += 5.0  # perturb positions >= 3
    out_a = model.reconstruct(z_a, prefix_a)
    out_b = model.reconstruct(z_a, prefix_b)
    early_unchanged = torch.allclose(out_a[:, :2, :], out_b[:, :2, :], atol=1e-6)

    checks = {
        "rlt_only__vla_grad_none": vla_grad_none,
        "rlt_only__encoder_grad": encoder_has_grad,
        "rlt_only__decoder_grad": decoder_has_grad,
        "vla_only__vla_grad": vla_has_grad,
        "vla_only__encoder_grad_none": encoder_grad_none,
        "vla_only__decoder_grad_none": decoder_grad_none,
        "decoder_teacher_shifting": early_unchanged,
    }
    return {
        "kind": "toy_gradient_audit",
        "mock": True,
        "note": "toy modules only; the real pi0.5/RLT Stage 1 smoke test is NOT_RUN "
        "without the pinned RLinf runtime and real weights",
        "rlt_loss": float(rlt_loss.item()),
        "vla_loss": float(vla_loss.item()),
        "checks": checks,
        "all_passed": all(checks.values()),
    }
