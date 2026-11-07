"""Deterministic toy delayed-control environment (CPU only).

Used to test queue/reward/learning wiring and to run the ``--toy`` CLI paths.
Every artifact produced through this environment is labelled ``toy`` / mock;
it is not a simulator and cannot support any task-performance claim.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
from pi05_rlt_rtc.contracts.spaces import ENV_ACTION_DIM, PROPRIO_DIM
from pi05_rlt_rtc.runtime.scheduler import StepOutcome


@dataclasses.dataclass
class ToyEnvConfig:
    initial_state: float = 1.5
    state_scale: float = 1.0
    target: float = 0.0
    max_state: float = 2.0
    success_radius: float = 0.05
    episode_ticks: int = 200


class ToyDelayedControlEnv:
    """1-D integrator: ``x' = clip(x + action[0], -max, max)``.

    Reward is the negative distance to the target; the first state dimension
    is the controlled one, matching the toy teacher in the mock feature
    service.  The remaining 7 action dims are ignored.
    """

    def __init__(self, config: ToyEnvConfig | None = None, *, env_id: str = "toy0") -> None:
        self.config = config or ToyEnvConfig()
        self.env_id = env_id
        self._tick = 0
        self._episode = 0
        self._state = np.zeros(PROPRIO_DIM, dtype=np.float32)

    def _snapshot(self) -> ObservationSnapshot:
        images = {
            "main": np.zeros((4, 4, 3), dtype=np.uint8),
            "wrist": np.zeros((4, 4, 3), dtype=np.uint8),
        }
        return ObservationSnapshot(
            env_id=self.env_id,
            episode_id=f"{self.env_id}-ep{self._episode}",
            tick=self._tick,
            states=self._state.copy(),
            images=images,
            image_masks={k: True for k in images},
            prompt="insert the plug into the power strip (toy)",
        )

    def reset(self, *, seed: int = 0) -> ObservationSnapshot:
        rng = np.random.default_rng(seed)
        x0 = self.config.initial_state * self.config.state_scale
        x0 = float(np.clip(x0 + 0.05 * rng.standard_normal(), -self.config.max_state, self.config.max_state))
        self._state = np.zeros(PROPRIO_DIM, dtype=np.float32)
        self._state[0] = x0
        self._state[1] = 0.0
        self._tick = 0
        self._episode += 1
        return self._snapshot()

    def step(self, action: np.ndarray) -> StepOutcome:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ENV_ACTION_DIM,):
            raise ValueError(f"toy env expects [{ENV_ACTION_DIM}] action, got {action.shape}")
        x = float(self._state[0]) + float(action[0])
        x = float(np.clip(x, -self.config.max_state, self.config.max_state))
        self._state[0] = x
        self._tick += 1
        distance = abs(x - self.config.target)
        reward = -distance
        terminated = distance < self.config.success_radius
        truncated = self._tick >= self.config.episode_ticks
        return StepOutcome(
            observation=self._snapshot(),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info={"distance": distance, "toy": True, "mock": True},
        )

    def reward_fn(self) -> Any:
        return lambda info: -abs(info["distance"])


class RandomPolicy:
    """Baseline used only for the ``--toy`` evaluation contrast."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(ENV_ACTION_DIM, dtype=np.float32)
        action[0] = float(self.rng.uniform(-1.0, 1.0))
        return action
