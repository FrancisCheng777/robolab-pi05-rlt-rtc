"""Twin-Q critic over the full queue-aware planning state ``X`` and candidate ``U``.

The critic consumes the complete state including the current full reference
(guide section 11.3); only the *actor's* reference input is subject to
dropout.  Target networks follow the upstream ``rlt_ac`` recipe: min of twin
Q, EMA target update, no entropy term, no CrossQ/CQL variants in the first
implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pi05_rlt_rtc.contracts.errors import ContractError
def _q_head(dims: list[int]) -> nn.Sequential:
    """RLinf c70606f QHead: Linear/LayerNorm/Tanh, then a linear output."""
    layers = []
    for i, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if i < len(dims) - 2:
            layers.extend([nn.LayerNorm(out_dim), nn.Tanh()])
    net = nn.Sequential(*layers)
    for layer in net:
        if isinstance(layer, nn.Linear):
            if layer is net[-1]:
                nn.init.normal_(layer.weight, mean=0., std=.02)
            else:
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(layer.bias)
    return net


class TwinQCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        num_q_heads: int = 2,
    ) -> None:
        super().__init__()
        if num_q_heads < 2:
            raise ContractError(
                "RLT Stage 2 requires at least two Q heads (twin-Q); "
                f"got num_q_heads={num_q_heads}"
            )
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.num_q_heads = int(num_q_heads)
        self.hidden_dims = tuple(hidden_dims)
        self.heads = nn.ModuleList(
            [_q_head([state_dim + action_dim, *hidden_dims, 1]) for _ in range(num_q_heads)]
        )

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.state_dim:
            raise ContractError(
                f"critic state dim {state.shape[-1]} != expected {self.state_dim}"
            )
        actions = actions.reshape(actions.shape[0], -1)
        if actions.shape[-1] != self.action_dim:
            raise ContractError(
                f"critic action dim {actions.shape[-1]} != expected {self.action_dim}"
            )
        x = torch.cat([state, actions], dim=-1)
        return torch.cat([head(x) for head in self.heads], dim=-1)

    def q_min(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q = self.forward(state, actions)
        return torch.minimum(q[..., 0:1], q[..., 1:2])

    def q1(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q = self.forward(state, actions)
        return q[..., 0:1]


class TwinQWithTargets(nn.Module):
    """Online twin-Q plus its EMA target twin-Q."""

    def __init__(self, state_dim: int, action_dim: int, *, hidden_dims: tuple[int, ...] = (256, 256, 256)) -> None:
        super().__init__()
        self.online = TwinQCritic(state_dim, action_dim, hidden_dims=hidden_dims)
        self.target = TwinQCritic(state_dim, action_dim, hidden_dims=hidden_dims)
        self.target.load_state_dict(self.online.state_dict())
        for param in self.target.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def soft_update(self, tau: float) -> None:
        if not 0.0 < tau <= 1.0:
            raise ContractError(f"target EMA tau must be in (0, 1], got {tau}")
        for tgt, src in zip(self.target.parameters(), self.online.parameters()):
            tgt.mul_(1.0 - tau).add_(src.detach(), alpha=tau)
        for tgt_buf, src_buf in zip(self.target.buffers(), self.online.buffers()):
            tgt_buf.copy_(src_buf)
