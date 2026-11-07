"""One window policy for real demonstration SFT, generic chunks and Stage-1."""
from __future__ import annotations

import numpy as np

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.physical import CONTROL_HZ

WINDOW_POLICY = {
    'id': 'complete_contiguous_no_intervention_v1',
    'control_hz': CONTROL_HZ,
    'tail_padding': False,
    'cross_episode': False,
    'max_interval_error_fraction': .5,
    'max_grid_error_frames': 1.,
    'exclude_intervention': True,
    'timestamp_column': 'source_time',
}


def _column(value, name, dtype=None):
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ContractError(f'{name} must be a vector or [N,1] column')
    return array


def complete_windows(episodes, timestamps, horizon, *, interventions=None, tolerance=None):
    """Keep full H rows with valid time edges, episode clock and intervention flags.

    Timing bounds match the updated exporter's audit_timing: +/-50% of a
    15 Hz period and at most one frame of accumulated episode-grid drift.
    Source receive times are mandatory; synthetic LeRobot frame times are not
    a substitute. Device exposure alignment is not implied by these checks.
    """
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ContractError('positive integer horizon required')
    episodes = _column(episodes, 'episodes', float)
    timestamps = _column(timestamps, 'timestamps', float)
    if len(episodes) != len(timestamps) or not np.isfinite(timestamps).all() or not np.isfinite(episodes).all() or np.any(episodes != np.floor(episodes)):
        raise ContractError('invalid physical episode/timestamp metadata')
    flags = np.zeros(len(episodes), bool) if interventions is None else _column(interventions, 'interventions')
    if len(flags) != len(episodes) or not np.isin(flags, [False, True]).all():
        raise ContractError('interventions must contain one boolean flag per frame')
    flags = flags.astype(bool)
    period = 1 / CONTROL_HZ
    tolerance = period * WINDOW_POLICY['max_interval_error_fraction'] if tolerance is None else tolerance
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ContractError('invalid timestamp tolerance')
    # Break at every episode boundary, even if an earlier ID appears again.
    boundaries = np.r_[0, np.flatnonzero(episodes[1:] != episodes[:-1]) + 1, len(episodes)]
    starts = []
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        size = end - begin
        if size < horizon:
            continue
        times = timestamps[begin:end]
        interval = np.diff(times)
        bad_edge = (interval <= 0) | (np.abs(interval-period) > tolerance + 1e-9)
        bad_frame = flags[begin:end] | (np.abs(times-times[0]-np.arange(size)*period) > period + 1e-9)
        frame_sum = np.r_[0, np.cumsum(bad_frame)]
        edge_sum = np.r_[0, np.cumsum(bad_edge)]
        indices = np.arange(size-horizon+1)
        valid = (frame_sum[indices+horizon] == frame_sum[indices]) & (edge_sum[indices+horizon-1] == edge_sum[indices])
        starts.extend((begin+indices[valid]).tolist())
    return starts


def table_window_starts(table, horizon):
    if 'source_time' not in table.column_names:
        raise ContractError('real source_time timestamps are required for physical SFT')
    starts = complete_windows(table['episode_index'], table['source_time'], horizon,
        interventions=table['intervention'] if 'intervention' in table.column_names else None)
    if not starts:
        raise ContractError('no complete, contiguous physical action windows')
    return np.asarray(starts, dtype=np.int64)


def validate_resume_window_policy(metadata):
    if metadata.get('window_policy') != WINDOW_POLICY:
        raise ContractError('checkpoint window policy differs or is missing; start a new run instead of restoring incompatible dataloader state')
