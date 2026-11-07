"""Regularize to a frozen incumbent actor, retaining the VLA reference input."""
import dataclasses
import numpy as np
import torch

from pi05_rlt_rtc.contracts.errors import ContractError


def anchor_targets(transitions, actor, contract):
    rows = list(transitions)
    valid = [i for i, tr in enumerate(rows) if tr.is_learnable()]
    device = next(actor.parameters()).device
    if actor.training or any(p.requires_grad for p in actor.parameters()):
        raise ContractError('anchor actor must be frozen and in eval mode')
    for start in range(0, len(valid), 256):
        indices = valid[start:start+256]
        selected = [rows[i] for i in indices]
        states = np.concatenate([tr.planning_state.actor_state_vector() for tr in selected])
        refs = np.concatenate([tr.planning_state.reference_suffix_vector(contract) for tr in selected])
        with torch.inference_mode():
            targets = actor.act(state=torch.as_tensor(states, device=device),
                reference_suffix=torch.as_tensor(refs, device=device), deterministic=True,
                apply_reference_dropout=False).actions.cpu().numpy()
        if not np.isfinite(targets).all():
            raise ContractError('nonfinite incumbent actor BC target')
        for i, target in zip(indices, targets):
            # These are incumbent-policy regularization targets, not claimed executions.
            rows[i] = dataclasses.replace(rows[i], bc_target_plan=target.copy(),
                bc_target_valid_mask=np.ones_like(target, dtype=bool),
                teacher_source='frozen_incumbent_actor_v1')
    return rows, len(valid)
