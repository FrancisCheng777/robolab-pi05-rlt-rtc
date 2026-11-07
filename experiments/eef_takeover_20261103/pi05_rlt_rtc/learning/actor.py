"""Queue-aware direct-chunk RLT actor.

The main actor predicts the C-step env command chunk directly (not a residual
or noise policy). It uses the upstream three-layer Tanh MLP trunk,
fixed exploration standard deviation before ``tanh`` and a bounded
final action.  New here is the queue/time conditioning and the rule that the
*reference input* is the only thing reference dropout touches (guide section
11.3/11.4).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.learning.planning_state import PlanningStateSpec


# RLinf c70606f: modules/utils.py::layer_init and mlp_policy.py.
def _actor_linear(in_dim: int, out_dim: int, gain: float = math.sqrt(2)) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


@dataclasses.dataclass
class ActorOutput:
    raw: torch.Tensor  # pre-tanh logits [B, C, A]
    actions: torch.Tensor  # bounded env commands [B, C, A]
    reference_dropped: torch.Tensor  # [B, 1] bool, diagnostics


class QueueAwareRLTActor(nn.Module):
    def __init__(
        self,
        spec: PlanningStateSpec,
        *,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        action_low: Any | None = None,
        action_high: Any | None = None,
        fixed_std: float = 0.002,
        name: str = "queue_aware_direct_chunk_mlp",
    ) -> None:
        super().__init__()
        self.spec = spec
        if not np.isfinite(fixed_std) or fixed_std < 0:
            raise ContractError("fixed_std must be >= 0")
        self.fixed_std = float(fixed_std)
        self.name = name
        self.hidden_dims = tuple(int(d) for d in hidden_dims)
        low = np.full(spec.env_action_dim, -1.0) if action_low is None else np.asarray(action_low, dtype=np.float32)
        high = np.full(spec.env_action_dim, 1.0) if action_high is None else np.asarray(action_high, dtype=np.float32)
        if low.shape != (spec.env_action_dim,) or high.shape != (spec.env_action_dim,):
            raise ContractError("action_low/high must match env_action_dim")
        if np.any(high <= low):
            raise ContractError("action_high must be > action_low")
        self.register_buffer("action_low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high, dtype=torch.float32))

        state_dim = sum(dim for _, dim in spec.actor_state_layout())
        ref_dim = spec.chunk * spec.env_action_dim
        self.state_dim = state_dim
        self.ref_dim = ref_dim
        layers = []
        in_dim = state_dim + ref_dim
        for width in self.hidden_dims:
            layers.extend([_actor_linear(in_dim, width), nn.Tanh()])
            in_dim = width
        self.trunk = nn.Sequential(*layers)
        self.head = _actor_linear(in_dim, ref_dim, gain=.01 * math.sqrt(2))

    # -- layout -----------------------------------------------------------------
    @property
    def layout(self) -> list[tuple[str, int]]:
        return self.spec.actor_state_layout()

    def describe(self) -> str:
        return self.spec.describe_layout("actor")

    # -- forward ----------------------------------------------------------------
    def forward(
        self,
        state: torch.Tensor,
        reference_suffix: torch.Tensor | None,
        *,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> ActorOutput:
        if state.shape[-1] != self.state_dim:
            raise ContractError(
                f"actor state dim {state.shape[-1]} != expected {self.state_dim}; "
                "use the layout printed by describe()"
            )
        B = state.shape[0]
        if reference_suffix is None:
            reference_suffix = torch.zeros(B, self.ref_dim, device=state.device, dtype=state.dtype)
        if reference_suffix.shape != (B, self.ref_dim):
            raise ContractError(
                f"reference_suffix must be [B, {self.ref_dim}], got {tuple(reference_suffix.shape)}"
            )
        dropped = torch.zeros(B, 1, dtype=torch.bool, device=state.device)
        ref = reference_suffix
        if apply_reference_dropout and reference_dropout_prob > 0:
            keep = (
                torch.rand((B, 1), device=state.device, generator=generator)
                >= float(reference_dropout_prob)
            )
            ref = reference_suffix * keep.to(reference_suffix.dtype)
            dropped = ~keep
        raw = self.head(self.trunk(torch.cat([state, ref], dim=-1)))
        raw = raw.reshape(B, self.spec.chunk, self.spec.env_action_dim)
        return ActorOutput(raw=raw, actions=self.bounded_actions(raw), reference_dropped=dropped)

    def bounded_actions(self, raw: torch.Tensor) -> torch.Tensor:
        """Deployment parameterisation shared by BC warmup, RL and evaluation."""
        mid = (self.action_high + self.action_low) / 2.0
        half = (self.action_high - self.action_low) / 2.0
        return mid + half * torch.tanh(raw)

    def act(
        self,
        *,
        state: torch.Tensor,
        reference_suffix: torch.Tensor | None,
        deterministic: bool,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> ActorOutput:
        out = self.forward(
            state,
            reference_suffix,
            apply_reference_dropout=apply_reference_dropout,
            reference_dropout_prob=reference_dropout_prob,
            generator=generator,
        )
        if deterministic or self.fixed_std == 0.0:
            return out
        # Exploration noise only ever perturbs the submitted chunk U, never P.
        noise = torch.randn(
            out.raw.shape, device=out.raw.device, dtype=out.raw.dtype, generator=generator
        )
        raw = out.raw + self.fixed_std * noise
        return ActorOutput(raw=raw, actions=self.bounded_actions(raw), reference_dropped=out.reference_dropped)
