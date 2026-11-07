"""DROID command conversion into the existing, bounded station controller."""
import numpy as np

def gripper_state(width):
    if not np.isfinite(width) or not -.002 <= width <= .087:
        raise ValueError('invalid gripper width')
    # DROID 0=open, 1=closed; this station measures physical opening in metres.
    return np.array([np.clip(1-width/.085,0,1)], dtype=np.float32)

def convert(action):
    a = np.asarray(action, dtype=float)
    if a.shape != (8,) or not np.isfinite(a).all():
        raise ValueError('DROID action must be finite shape (8,)')
    # Official DROID: joint_velocity_to_delta uses 0.2 rad per unit command.
    # Existing owner accepts 0.05 rad per unit. Retain its 0.05 rad bound;
    # report saturation rather than claiming exact DROID hardware execution.
    clipped = np.clip(a, -1, 1)
    requested_delta = .2*clipped[:7]
    command = np.r_[np.clip(requested_delta/.05,-1,1),
                    -1. if a[7] > .5 else 1.]
    return command, {'requested_delta_rad': requested_delta.tolist(),
                     'station_delta_rad': (.05*command[:7]).tolist(),
                     'station_delta_limited': bool(np.any(np.abs(requested_delta)>.05)),
                     'raw_action_clipped': bool(np.any(a!=clipped))}
