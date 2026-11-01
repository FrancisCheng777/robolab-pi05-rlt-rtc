"""Common physical SFT windows, exercised through the isolated Stage-1 copy."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'stage1_integration'))
from pi05_demo.chunks import complete_chunk_starts
from pi05_rlt_rtc.data.physical_sft import complete_windows


def both(episodes, timestamps, horizon, interventions=None):
    first = complete_windows(episodes, timestamps, horizon, interventions=interventions)
    second = complete_chunk_starts(episodes, horizon, timestamps=timestamps, interventions=interventions)
    np.testing.assert_array_equal(first, second)
    return list(first)


def test_full_windows_keep_last_real_action_without_crossing_episodes():
    episodes = [0]*12+[1]*4+[2]*11
    assert both(episodes, np.arange(27)/15, 10) == [0,1,2,16,17]


def test_new_exporter_timing_bound_allows_25ms_jitter():
    timestamps = np.arange(12)/15
    timestamps[5] += .025
    assert complete_windows([0]*12, timestamps, 10) == [0,1,2]
    assert both([0]*12, timestamps, 10) == [0,1,2]


def test_time_gaps_and_interventions_are_excluded_in_all_loaders():
    timestamps = np.arange(24)/15
    timestamps[12:] += .04
    mask = np.zeros(24, bool)
    mask[4] = True
    starts = both([0]*24, timestamps, 3, mask)
    assert starts == [0,1,5,6,7,8,9,12,13,14,15,16,17,18,19,20,21]


def test_episode_clock_drift_is_not_hidden_by_locally_regular_intervals():
    assert complete_windows([0]*20, np.arange(20)*(1/15+.01), 3) == [0,1,2,3,4]
    assert both([0]*20, np.arange(20)*(1/15+.01), 3) == [0,1,2,3,4]


def test_missing_timestamps_fail_closed_and_short_episodes_are_not_padded():
    with pytest.raises((TypeError, ValueError), match='timestamp'):
        complete_chunk_starts([0]*12, 10)
    with pytest.raises(ValueError, match='no complete'):
        complete_chunk_starts([0]*9+[1]*9, 10, timestamps=np.arange(18)/15)


@pytest.mark.parametrize('horizon', [0, -1, True, 1.5])
def test_invalid_horizon_rejected(horizon):
    with pytest.raises(ValueError):
        complete_windows([0]*12, np.arange(12)/15, horizon)


def test_resume_requires_the_same_window_rule():
    from pi05_rlt_rtc.data.windows import WINDOW_POLICY, validate_resume_window_policy
    validate_resume_window_policy({'window_policy': dict(WINDOW_POLICY)})
    for metadata in ({}, {'window_policy': {**WINDOW_POLICY, 'tail_padding': True}}):
        with pytest.raises(ValueError, match='window policy'):
            validate_resume_window_policy(metadata)
