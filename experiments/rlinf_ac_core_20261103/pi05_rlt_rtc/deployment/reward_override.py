"""Explicit learner-only terminal rewards; raw human labels stay immutable."""
import dataclasses
import math
import numpy as np
from pi05_rlt_rtc.contracts.errors import ContractError


def reward_definition(training):
    override = training.get('terminal_rewards')
    if override is None:
        scheme = training.get('reward_scheme', 'rlt_binary')
        if scheme not in ('rlt_binary', 'rlinf_signed'):
            raise ContractError('unknown recorded reward scheme')
        return {'success': 1., 'failure': -1. if scheme == 'rlinf_signed' else 0.}
    if not isinstance(override, dict) or set(override) != {'success', 'failure'}:
        raise ContractError('terminal_rewards must specify success and failure')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in override.values()):
        raise ContractError('terminal rewards must be finite numbers')
    return {k: float(v) for k, v in override.items()}


def relabel_terminal_transition(transition, annotation, rewards, *, gamma_per_tick=None):
    gamma = transition.gamma_per_tick if gamma_per_tick is None else gamma_per_tick
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ContractError('gamma_per_tick must be finite and in (0,1]')
    if (annotation.get('eligible') is not True or annotation.get('outcome') not in rewards
            or annotation.get('reward_source') != 'manual_terminal'):
        raise ContractError('reward override requires an eligible manually labelled outcome')
    tick = annotation.get('terminal_tick')
    if type(tick) is not int:
        raise ContractError('terminal tick must identify an executed tick')
    offset = tick - transition.capture_tick
    contained = 0 <= offset < transition.duration_ticks
    valid = contained and bool(transition.reward_valid_mask[offset])
    if transition.terminated and not valid:
        raise ContractError('terminal transition disagrees with labelled terminal tick')
    if valid and transition.is_learnable() and not transition.terminated:
        raise ContractError('labelled terminal reward must not bootstrap')
    values = np.zeros_like(transition.primitive_rewards)
    if valid:
        values[offset] = rewards[annotation['outcome']]
    return dataclasses.replace(transition, primitive_rewards=values, gamma_per_tick=float(gamma))
