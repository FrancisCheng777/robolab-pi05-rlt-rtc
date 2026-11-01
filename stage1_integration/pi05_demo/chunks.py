"""Shared real-time window contract; no tail padding or fabricated timestamps."""
import numpy as np
from pi05_rlt_rtc.data.windows import complete_windows, table_window_starts, WINDOW_POLICY


def complete_chunk_starts(episodes, horizon, *, timestamps=None, interventions=None):
    if timestamps is None:
        raise ValueError("source timestamps are required for complete physical chunks")
    starts = complete_windows(episodes, timestamps, horizon, interventions=interventions)
    if not starts:
        raise ValueError("no complete action chunks with valid timing and no intervention")
    return np.asarray(starts, dtype=np.int64)
