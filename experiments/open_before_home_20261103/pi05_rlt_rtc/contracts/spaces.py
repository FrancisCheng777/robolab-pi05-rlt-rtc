"""Action/observation space constants and the horizon contract.

The single most common integration bug in this project is confusing the VLA
model action space ``[H, 32]`` with the environment command space ``[C, 8]``.
Every new field/function in this package must say which space it belongs to.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from pi05_rlt_rtc.contracts.errors import ContractError

# --- Physical Panda + Robotiq 2F-85 / measured joint delta + aperture ---------
ENV_ACTION_DIM = 8
MODEL_ACTION_DIM = 32
PROPRIO_DIM = 9
DEFAULT_HORIZON = 10
DEFAULT_CHUNK = 5
DEFAULT_DELAY = 2
PROMPT = "insert the plug into the power strip"

SPACE_MODEL = "vla_model"
SPACE_ENV = "env_command"


@dataclasses.dataclass(frozen=True)
class HorizonContract:
    """``0 <= D <= C`` and ``D + C <= H`` (guide section 3.2).

    - ``horizon`` H: full reference plan length produced by one observation.
    - ``chunk`` C: number of new actions the actor submits per request and the
      number of primitive ticks between two requests.
    - ``delay`` D: fixed number of committed primitive ticks a plan takes to
      become effective.  ``D = 0`` is only legal for synchronous regression
      tests; deployments require ``D > 0``.
    """

    horizon: int = DEFAULT_HORIZON
    chunk: int = DEFAULT_CHUNK
    delay: int = DEFAULT_DELAY

    def __post_init__(self) -> None:
        for name, value in (("H", self.horizon), ("C", self.chunk), ("D", self.delay)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise ContractError(f"{name} must be an integer, got {value!r}")
        if self.horizon <= 0:
            raise ContractError(f"H must be > 0, got {self.horizon}")
        if self.chunk <= 0:
            raise ContractError(f"C must be > 0, got {self.chunk}")
        if self.delay < 0:
            raise ContractError(f"D must be >= 0, got {self.delay}")
        if self.delay > self.chunk:
            raise ContractError(f"D must be <= C, got D={self.delay} C={self.chunk}")
        if self.delay + self.chunk > self.horizon:
            raise ContractError(
                f"D + C must be <= H, got D={self.delay} C={self.chunk} H={self.horizon}"
            )

    @property
    def suffix_start(self) -> int:
        """Exclusive-start index of the actor suffix inside the full H plan."""
        return int(self.delay)

    @property
    def suffix_end(self) -> int:
        """Exclusive-end index of the actor suffix inside the full H plan."""
        return int(self.delay + self.chunk)

    @property
    def reference_tail_len(self) -> int:
        return int(self.horizon - self.delay - self.chunk)

    def describe(self) -> str:
        return (
            f"H={self.horizon} C={self.chunk} D={self.delay} "
            f"(suffix=[{self.suffix_start}:{self.suffix_end}], "
            f"tail={self.reference_tail_len})"
        )


def reference_env_suffix(full_env_plan: Any, contract: HorizonContract) -> Any:
    """The one and only ``[D:D+C]`` suffix selector (guide sections 4.3/8.5).

    Actor input, BC target, teacher submission and evaluation visualisation all
    use this function.  Do not re-slice elsewhere.
    """
    _validate_trailing_shape(full_env_plan, contract.horizon, ENV_ACTION_DIM, "reference_env_full")
    return full_env_plan[..., contract.suffix_start : contract.suffix_end, :]


def pending_from_submitted(submitted_env_plan: Any, contract: HorizonContract) -> Any:
    """``P_{i+1} = U_i[C-D:C]``: the last ``D`` commands of a submitted chunk."""
    _validate_trailing_shape(submitted_env_plan, contract.chunk, ENV_ACTION_DIM, "submitted_env_plan")
    if contract.delay == 0:
        return submitted_env_plan[..., 0:0, :]
    return submitted_env_plan[..., contract.chunk - contract.delay : contract.chunk, :]


def _validate_trailing_shape(array: Any, expected_len: int, expected_dim: int, name: str) -> None:
    shape = getattr(array, "shape", None)
    if shape is None:
        raise ContractError(f"{name}: expected an array, got {type(array).__name__}")
    if len(shape) < 2 or shape[-2] != expected_len or shape[-1] != expected_dim:
        raise ContractError(
            f"{name}: expected trailing shape (*, {expected_len}, {expected_dim}), "
            f"got {tuple(shape)}"
        )
