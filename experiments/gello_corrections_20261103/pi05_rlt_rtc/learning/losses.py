"""RLT actor-critic losses with the queue-aware state (guide section 11.5).

Semantics locked to the upstream ``rlt_ac`` recipe:

- critic target: ``min_j Q_target,j(X', a_theta(X'))`` with a stop-gradient
  target action, ``gamma^K`` using the *real duration* K;
- actor objective: ``-lambda_Q * Q1(X, a_theta(X)) + lambda_BC * BC``;
- no entropy term, no learned variance, no target actor;
- BC compares final bounded (deployed) actions with the reference suffix or a
  teacher plan aligned to the future execution window;
- during the actor update the critic parameters are frozen (no parameter
  gradients) but the critic output stays differentiable wrt the actor action.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from pi05_rlt_rtc.contracts.errors import ContractError, FrozenParameterGradientError
from pi05_rlt_rtc.contracts.transition import DelayedTransition
from pi05_rlt_rtc.learning.actor import QueueAwareRLTActor
from pi05_rlt_rtc.learning.critic import TwinQWithTargets
from pi05_rlt_rtc.learning.planning_state import PlanningStateSpec


@contextmanager
def frozen_params(module: torch.nn.Module) -> Iterator[None]:
    """Temporarily set ``requires_grad_(False)`` on all parameters of a module."""
    flags = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(module.parameters(), flags):
            p.requires_grad_(flag)


@dataclasses.dataclass
class TransitionBatch:
    """Learner tensors: actor branch state is separate from the full critic state."""

    actor_state: torch.Tensor  # [B, D_actor] at the current capture
    critic_state: torch.Tensor  # [B, D_critic] (includes the current reference)
    curr_reference_suffix: torch.Tensor  # [B, C*A]
    next_actor_state: torch.Tensor  # [B, D_actor] at the next capture
    next_critic_state: torch.Tensor  # [B, D_critic]
    next_reference_suffix: torch.Tensor  # [B, C*A]
    actions: torch.Tensor  # [B, C*A] submitted U
    bc_target: torch.Tensor  # [B, C*A]
    bc_valid_mask: torch.Tensor  # [B, C*A]
    reward_sum: torch.Tensor  # [B, 1]
    bootstrap_discount: torch.Tensor  # [B, 1], gamma^K if bootstrap else 0
    learnable: torch.Tensor  # [B, 1]

    @property
    def batch_size(self) -> int:
        return int(self.actor_state.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[-1])


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid entries only (never divides by zero)."""
    if pred.shape != target.shape or pred.shape != valid_mask.shape:
        raise ContractError("masked_mse requires matching pred/target/mask shapes")
    mask = valid_mask.to(pred.dtype)
    error = (pred - target) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return error.sum() / denom


def build_transition_batch(
    transitions: list[DelayedTransition],
    spec: PlanningStateSpec,
    *,
    contract=None,
    device: torch.device | str = "cpu",
) -> TransitionBatch:
    """Convert contract transitions into tensors for the learner."""
    from pi05_rlt_rtc.contracts.spaces import ENV_ACTION_DIM, HorizonContract

    if not transitions:
        raise ContractError("cannot build an empty batch")
    if contract is None:
        raise ContractError("build_transition_batch requires the actual execution contract; max_delay is padding capacity, not D")
    if (contract.horizon, contract.chunk) != (spec.horizon, spec.chunk) or contract.delay > spec.max_delay:
        raise ContractError("execution contract disagrees with the planning-state layout")
    flat_action_dim = spec.chunk * ENV_ACTION_DIM
    B = len(transitions)

    curr_state = np.zeros((B, sum(d for _, d in spec.actor_state_layout())), dtype=np.float32)
    next_actor_state = np.zeros((B, sum(d for _, d in spec.actor_state_layout())), dtype=np.float32)
    curr_critic_state = np.zeros((B, sum(d for _, d in spec.critic_state_layout())), dtype=np.float32)
    curr_ref = np.zeros((B, flat_action_dim), dtype=np.float32)
    next_state = np.zeros((B, sum(d for _, d in spec.critic_state_layout())), dtype=np.float32)
    next_ref = np.zeros((B, flat_action_dim), dtype=np.float32)
    actions = np.zeros((B, flat_action_dim), dtype=np.float32)
    bc_target = np.zeros((B, flat_action_dim), dtype=np.float32)
    bc_mask = np.zeros((B, flat_action_dim), dtype=np.float32)
    reward_sum = np.zeros((B, 1), dtype=np.float32)
    bootstrap = np.zeros((B, 1), dtype=np.float32)
    learnable = np.zeros((B, 1), dtype=np.float32)

    for i, tr in enumerate(transitions):
        ps = tr.planning_state
        if ps is None:
            raise ContractError("transition without a current planning state")
        if ps.spec != spec or ps.batch_size != 1:
            raise ContractError("each replay transition must have one state matching the learner layout")
        expected_pending_mask = np.zeros((1, spec.max_delay, spec.env_action_dim), dtype=bool)
        expected_pending_mask[:, :contract.delay] = True
        for state in (ps, tr.next_planning_state):
            if state is not None and not np.array_equal(state.pending_mask, expected_pending_mask):
                raise ContractError("execution contract delay disagrees with the saved committed queue")
        curr_state[i] = ps.actor_state_vector()[0]
        curr_critic_state[i] = ps.critic_state_vector()[0]
        curr_ref[i] = ps.reference_suffix_vector(contract)[0]
        actions[i] = np.asarray(tr.submitted_plan, dtype=np.float32).reshape(-1)
        if tr.next_planning_state is not None:
            ns = tr.next_planning_state
            if ns.spec != spec or ns.batch_size != 1:
                raise ContractError("next state disagrees with the learner layout")
            next_actor_state[i] = ns.actor_state_vector()[0]
            next_state[i] = ns.critic_state_vector()[0]
            next_ref[i] = ns.reference_suffix_vector(contract)[0]
        if tr.bc_target_plan is not None:
            bc_target[i] = np.asarray(tr.bc_target_plan, dtype=np.float32).reshape(-1)
            bmask = (
                np.ones((spec.chunk, ENV_ACTION_DIM), dtype=np.float32)
                if tr.bc_target_valid_mask is None
                else np.asarray(tr.bc_target_valid_mask, dtype=np.float32)
            )
            bc_mask[i] = bmask.reshape(-1)
        else:
            bc_target[i] = curr_ref[i]
            bc_mask[i] = 1.0
        reward_sum[i, 0] = tr.reward_sum
        if tr.bootstrap_allowed and not tr.terminated and tr.next_planning_state is not None:
            bootstrap[i, 0] = float(tr.gamma_per_tick) ** int(tr.duration_ticks)
        learnable[i, 0] = 1.0 if tr.is_learnable() else 0.0

    if not np.all(learnable > 0):
        # All transitions in a batch must be learnable; callers filter first.
        raise ContractError(
            "batch contains non-learnable transitions; filter with DelayedTransition.is_learnable()"
        )

    def t(arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

    return TransitionBatch(
        actor_state=t(curr_state),
        critic_state=t(curr_critic_state),
        curr_reference_suffix=t(curr_ref),
        next_actor_state=t(next_actor_state),
        next_critic_state=t(next_state),
        next_reference_suffix=t(next_ref),
        actions=t(actions),
        bc_target=t(bc_target),
        bc_valid_mask=t(bc_mask),
        reward_sum=t(reward_sum),
        bootstrap_discount=t(bootstrap),
        learnable=t(learnable),
    )


def critic_loss(
    critic: TwinQWithTargets,
    actor: QueueAwareRLTActor,
    batch: TransitionBatch,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        next_out = actor.act(
            state=batch.next_actor_state,
            reference_suffix=batch.next_reference_suffix,
            deterministic=False,
            apply_reference_dropout=False,
            generator=generator,
        )
        q_next = critic.target.q_min(batch.next_critic_state, next_out.actions)
        target = batch.reward_sum + batch.bootstrap_discount * q_next
    q_all = critic.online(batch.critic_state, batch.actions)
    loss = F.mse_loss(q_all, target.expand_as(q_all))
    td = (q_all.detach() - target)
    metrics = {
        "critic/loss": float(loss.detach().item()),
        "critic/q_mean": float(q_all.detach().mean().item()),
        "critic/q_min": float(q_all.detach().min().item()),
        "critic/q_max": float(q_all.detach().max().item()),
        "critic/target_mean": float(target.detach().mean().item()),
        "critic/td_abs_mean": float(td.abs().mean().item()),
    }
    return loss, metrics


def actor_loss(
    actor: QueueAwareRLTActor,
    critic: TwinQWithTargets,
    batch: TransitionBatch,
    *,
    lambda_q: float,
    lambda_bc: float,
    reference_dropout_prob: float = 0.5,
    generator: torch.Generator | None = None,
    sample_exploration: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """``-lambda_Q * Q1 + lambda_BC * BC`` with critic parameters frozen."""
    if lambda_q < 0 or lambda_bc < 0:
        raise ContractError("loss weights must be >= 0")
    before = [p.detach().clone() for p in critic.online.parameters()]
    with frozen_params(critic.online):
        out = actor.act(
            state=batch.actor_state,
            reference_suffix=batch.curr_reference_suffix,
            deterministic=not sample_exploration,
            apply_reference_dropout=True,
            reference_dropout_prob=reference_dropout_prob,
            generator=generator,
        )
        q1 = critic.online.q1(batch.critic_state, out.actions)
        bc = masked_mse(out.actions.reshape(batch.batch_size, -1), batch.bc_target, batch.bc_valid_mask)
        loss = -lambda_q * q1.mean() + lambda_bc * bc
    after = list(critic.online.parameters())
    for old, new in zip(before, after):
        if not torch.equal(old, new.detach()):
            raise FrozenParameterGradientError(
                "critic parameters changed inside actor_loss; the actor objective must "
                "not update Q"
            )
    metrics = {
        "actor/loss": float(loss.detach().item()),
        "actor/q1_pi": float(q1.detach().mean().item()),
        "actor/bc_loss": float(bc.detach().item()),
        "actor/weighted_q": float((-lambda_q * q1.mean()).detach().item()),
        "actor/weighted_bc": float((lambda_bc * bc).detach().item()),
        "actor/reference_dropout_prob": float(reference_dropout_prob),
        "actor/lambda_q": float(lambda_q),
        "actor/lambda_bc": float(lambda_bc),
    }
    return loss, metrics


def bc_warmup_loss(
    actor: QueueAwareRLTActor,
    batch: TransitionBatch,
) -> tuple[torch.Tensor, dict[str, float]]:
    """BC warmup on the deployed (tanh-bounded) deterministic action.

    The generic upstream ``sft_forward`` compares pre-tanh logits and is
    therefore *not* reused here (T54).
    """
    out = actor.act(
        state=batch.actor_state,
        reference_suffix=batch.curr_reference_suffix,
        deterministic=True,
        apply_reference_dropout=False,
    )
    loss = masked_mse(out.actions.reshape(batch.batch_size, -1), batch.bc_target, batch.bc_valid_mask)
    with torch.no_grad():
        ref_to_target = masked_mse(
            batch.curr_reference_suffix, batch.bc_target, batch.bc_valid_mask
        )
    metrics = {
        "bc/warmup_loss": float(loss.detach().item()),
        "bc/reference_to_target": float(ref_to_target.item()),
    }
    return loss, metrics
