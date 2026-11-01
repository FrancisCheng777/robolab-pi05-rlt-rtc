"""Integer tick clocks.

The controller owns the authoritative ``tick_id``.  All action scheduling in
this project is expressed in integer ticks so that the virtual-delay training
mode and the wall-clock deployment mode share the *same* scheduling semantics;
only the clock implementation differs (guide section 15.2).
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Callable

from pi05_rlt_rtc.contracts.errors import ContractError


@dataclasses.dataclass
class TickClock:
    """Monotonic wall-clock ticker with an integer tick id.

    ``tick_to_monotonic`` maps a tick id to the monotonic time at which it
    should start.  The mapping is affine and immutable for the lifetime of the
    clock; a physical clock drift is not "fixed" by rewriting tick ids.
    """

    hz: float
    start_tick: int = 0
    start_monotonic: float | None = None
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.hz <= 0:
            raise ContractError(f"hz must be > 0, got {self.hz}")
        if self.start_monotonic is None:
            self.start_monotonic = time.monotonic()

    @property
    def period_seconds(self) -> float:
        return 1.0 / float(self.hz)

    def tick_to_monotonic(self, tick: int) -> float:
        if tick < self.start_tick:
            raise ContractError(f"tick {tick} precedes clock start {self.start_tick}")
        return float(self.start_monotonic) + (int(tick) - self.start_tick) * self.period_seconds

    def monotonic_to_tick(self, when: float) -> int:
        """First tick whose start time is >= ``when`` (ceil)."""
        elapsed = float(when) - float(self.start_monotonic)
        return self.start_tick + int(np_ceil(elapsed / self.period_seconds - 1e-12))

    def now_tick(self) -> int:
        return self.monotonic_to_tick(time.monotonic())

    def sleep_until(self, tick: int) -> None:
        target = self.tick_to_monotonic(tick)
        while True:
            remaining = target - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(remaining)


def np_ceil(value: float) -> int:
    import math

    return int(math.ceil(value))


@dataclasses.dataclass
class VirtualTickClock:
    """Deterministic discrete clock for training/regression tests.

    Advancing is explicit; there is no sleeping and therefore no wall-clock
    claim attached to any result produced with this clock.
    """

    tick: int = 0

    def advance(self, n: int = 1) -> int:
        if n < 0:
            raise ContractError(f"cannot advance a virtual clock backwards by {n}")
        self.tick += int(n)
        return self.tick

    def now_tick(self) -> int:
        return self.tick


class FrozenClock:
    """Clock pinned at zero; refuses to advance (used by tests as an oracle)."""

    tick = 0

    def advance(self, n: int = 1) -> int:  # pragma: no cover - guard only
        raise ContractError("FrozenClock cannot advance")


@dataclasses.dataclass
class DeadlineTimer:
    """Measures service latency against a tick budget (wall-clock mode only)."""

    clock: Callable[[], float] = time.monotonic

    def timed(self):  # pragma: no cover - trivial context manager
        return _TimerContext(self.clock)


class _TimerContext:
    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self.started: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "_TimerContext":
        self.started = self._clock()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = self._clock() - self.started
