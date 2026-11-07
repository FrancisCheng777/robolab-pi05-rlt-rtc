"""Opt-in tracking adapter for 15Hz realized-motion policy labels.

GELLO supplies absolute leader targets and MUST NOT use this adapter. A policy
label is (measured_q[t+1] - measured_q[t]) / 0.05. Its reference advances once
per new policy action, never once per bridge heartbeat. No hardware is opened.
"""
import numpy as np


def _joints(value, name):
    out = np.asarray(value, dtype=float)
    if out.shape != (7,) or not np.isfinite(out).all():
        raise ValueError(f'{name} requires seven finite joints')
    return out


def bound_joint_reference(reference, measured, *, tracking_limits, lead_limit=.05):
    """Anti-windup: discard excess reference beyond measured feedback + bound."""
    if isinstance(lead_limit, bool) or not np.isfinite(lead_limit) or not 0 < lead_limit <= .05:
        raise ValueError('reference lead must be in (0, .05] rad')
    reference = _joints(reference, 'reference')
    measured = _joints(measured, 'measured')
    limits = _joints(tracking_limits, 'tracking limits')
    if np.any(limits <= 0):
        raise ValueError('tracking limits must be positive')
    lead = np.minimum(limits, lead_limit)
    return np.clip(reference, measured-lead, measured+lead)


def advance_joint_reference(measured, normalized_delta, *, previous_reference=None,
                            tracking_limits, lead_limit=.05):
    """Return the next bounded reference; pass None after hold/home/reset.

    The caller retains the returned target (not the unbounded proposal), enforces
    its accepted joint-position limits, and refreshes that same target between
    policy actions. Gripper commands remain separate absolute apertures.
    """
    measured = _joints(measured, 'measured')
    action = _joints(normalized_delta, 'normalized delta')
    if np.any(np.abs(action) > 1):
        raise ValueError('normalized delta must be in [-1,1]')
    anchor = measured if previous_reference is None else _joints(previous_reference, 'previous reference')
    return bound_joint_reference(anchor + .05*action, measured,
                                 tracking_limits=tracking_limits, lead_limit=lead_limit)
