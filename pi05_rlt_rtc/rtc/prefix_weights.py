"""Prefix attention / guidance weights.

Faithful port of ``get_prefix_weights`` from the official inference-time RTC
reference implementation
(``Physical-Intelligence/real-time-chunking-kinetix``,
``src/model.py`` @ ``9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b``).

Docstring example from the reference implementation::

    get_prefix_weights(start=2, end=6, total=10, schedule="linear")
    -> 1  1  4/5 3/5 2/5 1/5 0  0  0  0

``start`` (inclusive) is the first tick the new chunk is allowed to change;
``end`` (exclusive) is where attention to the prefix stops.  ``end`` takes
precedence: ``start = min(start, end)``.

In our fixed-delay protocol the weights are evaluated on the *canonical*
overlap of the previous composed plan: ``start = D`` (committed prefix) and
``end = H - C`` (the previous plan carries at most ``H - C`` remaining ticks
after the shift), with ``total = H``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from pi05_rlt_rtc.contracts.errors import ContractError

PrefixSchedule = Literal["linear", "exp", "ones", "zeros"]
VALID_SCHEDULES: tuple[str, ...] = ("linear", "exp", "ones", "zeros")


def get_prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: PrefixSchedule = "exp",
) -> np.ndarray:
    """Return ``total`` float32 weights, exactly matching the reference code."""
    if total <= 0:
        raise ContractError(f"total must be > 0, got {total}")
    if schedule not in VALID_SCHEDULES:
        raise ContractError(f"invalid schedule {schedule!r}; expected one of {VALID_SCHEDULES}")
    start = min(int(start), int(end))
    end = int(end)
    idx = np.arange(total, dtype=np.float64)
    if schedule == "ones":
        w = np.ones(total, dtype=np.float64)
    elif schedule == "zeros":
        w = (idx < start).astype(np.float64)
    else:  # linear / exp
        w = np.clip((start - 1 - idx) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * np.expm1(w) / (np.e - 1.0)
    w = np.where(idx >= end, 0, w)
    return w.astype(np.float32)


def build_guidance_weight_matrix(
    time_weights: np.ndarray,
    action_valid_mask: np.ndarray,
) -> np.ndarray:
    """Combine a ``[H]`` time mask with a ``[1, 32]`` action-valid mask.

    Returns ``[H, 32]`` float32.  The result is used **once** as
    ``e = W * (Y - xhat1)`` inside the VJP; never squared (T27).
    """
    tw = np.asarray(time_weights, dtype=np.float32).reshape(-1, 1)
    av = np.asarray(action_valid_mask, dtype=np.float32).reshape(1, -1)
    if tw.shape[0] == 0:
        raise ContractError("time weights must be non-empty")
    combined = tw * av
    if not np.all(np.isfinite(combined)):
        raise ContractError("guidance weight matrix contains NaN/Inf")
    return combined


def guidance_mask_from_previous_plan(
    *,
    horizon: int,
    chunk: int,
    delay: int,
    previous_valid_len: int,
    action_valid_mask: np.ndarray,
    schedule: PrefixSchedule = "exp",
) -> np.ndarray:
    """Build the ``[H, model_dim]`` per-request RTC guidance mask.

    ``previous_valid_len`` is the number of valid shifted previous-plan ticks
    (``H - C`` for a full previous plan, less after truncation).  Ticks beyond
    it get weight 0.  The committed prefix ``[0, D)`` gets weight 1 from the
    schedule itself when ``start = D``.

    ``action_valid_mask`` is required so that the returned matrix always has a
    well-defined ``[H, model_dim]`` shape; guidance is never applied to the
    24 model-space padding dimensions.
    """
    av = np.asarray(action_valid_mask, dtype=np.float32).reshape(1, -1)
    if previous_valid_len <= 0:
        return np.zeros((horizon, av.shape[-1]), dtype=np.float32)
    end = min(int(previous_valid_len), int(horizon) - int(chunk))
    if end <= 0:
        return np.zeros((horizon, av.shape[-1]), dtype=np.float32)
    time_weights = get_prefix_weights(start=delay, end=end, total=horizon, schedule=schedule)
    return build_guidance_weight_matrix(time_weights, av)
