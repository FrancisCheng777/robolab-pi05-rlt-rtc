"""Shared test helpers (not collected by pytest).

Builds contract-valid requests/plans by hand so queue tests can construct exact
scenarios without the scheduler.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pi05_rlt_rtc import PLANNING_STATE_VERSION, SCHEMA_VERSION
from pi05_rlt_rtc.contracts.plan import (
    FinalPlan,
    PlanRequest,
    PrevPlanContext,
    SubmittedPlan,
    make_submitted_mask,
)
from pi05_rlt_rtc.contracts.spaces import ENV_ACTION_DIM, HorizonContract
from pi05_rlt_rtc.runtime.ledger import ExecutionLedger
from pi05_rlt_rtc.runtime.queue import ActionQueue

ENV_ID = "env0"
EPISODE_ID = "env0-ep1"
POLICY_ID = "policy-v1"
FEATURE_ID = "feature-v1"


def make_queue(
    contract: HorizonContract,
    *,
    fallback: str = "stop",
    episode_generation: int = 1,
) -> ActionQueue:
    queue = ActionQueue(contract, fallback_policy=fallback, ledger=ExecutionLedger())
    queue.reset(env_id=ENV_ID, episode_id=EPISODE_ID, episode_generation=episode_generation, start_tick=0)
    return queue


def warm_start(queue: ActionQueue, contract: HorizonContract, *, value: float = 0.9) -> int:
    prefix = np.full((contract.delay, ENV_ACTION_DIM), value, dtype=np.float32)
    if contract.delay:
        return queue.commit_initial_prefix(prefix, start_tick=0)
    return -1


def make_request(
    contract: HorizonContract,
    *,
    request_id: int,
    capture_tick: int,
    prev: PrevPlanContext | None = None,
    queue_generation: int = 1,
    episode_generation: int = 1,
    feature_id: str = FEATURE_ID,
    rng_seed: int = 0,
) -> PlanRequest:
    return PlanRequest(
        request_id=request_id,
        env_id=ENV_ID,
        episode_id=EPISODE_ID,
        episode_generation=episode_generation,
        capture_tick=capture_tick,
        apply_start_tick=capture_tick + contract.delay,
        queue_generation=queue_generation,
        policy_version_id=POLICY_ID,
        feature_version_id=feature_id,
        schema_version=SCHEMA_VERSION,
        planning_state_version=PLANNING_STATE_VERSION,
        rtc_context=prev or PrevPlanContext.empty(),
        rng_seed=rng_seed,
        sampler_mode="full_vjp",
        num_denoising_steps=5,
        deadline_tick=capture_tick + contract.delay,
    )


def make_submitted(
    contract: HorizonContract,
    *,
    request: PlanRequest,
    base_value: float,
    raw_actor_plan: np.ndarray | None = None,
) -> SubmittedPlan:
    actions = np.stack(
        [np.full(ENV_ACTION_DIM, base_value + 0.01 * k, dtype=np.float32) for k in range(contract.chunk)]
    )
    return SubmittedPlan(
        request_id=request.request_id,
        env_id=request.env_id,
        episode_id=request.episode_id,
        episode_generation=request.episode_generation,
        capture_tick=request.capture_tick,
        apply_start_tick=request.apply_start_tick,
        actions=actions,
        valid_mask=make_submitted_mask(contract),
        source="reference_teacher",
        policy_version_id=request.policy_version_id,
        feature_version_id=request.feature_version_id,
        schema_version=request.schema_version,
        planning_state_version=request.planning_state_version,
        queue_generation=request.queue_generation,
        raw_actor_plan=raw_actor_plan,
    )


def make_final(
    contract: HorizonContract,
    *,
    request: PlanRequest,
    pending: np.ndarray,
    submitted: SubmittedPlan,
    reference_value: float = 0.7,
    plan_id: int = 0,
) -> FinalPlan:
    reference = np.stack(
        [np.full(ENV_ACTION_DIM, reference_value + 0.001 * l, dtype=np.float32) for l in range(contract.horizon)]
    )
    return FinalPlan.compose(
        request=request,
        pending_prefix=pending,
        submitted_plan=submitted,
        reference_env_full=reference,
        contract=contract,
        plan_id=plan_id,
    )


def prev_context_from(final: FinalPlan, contract: HorizonContract, next_capture: int) -> PrevPlanContext:
    shifted, mask = final.shifted_for_next_request(next_capture)
    return PrevPlanContext(
        prev_env_plan=shifted,
        prev_valid_mask=mask,
        prev_capture_tick=final.capture_tick,
        prev_plan_id=final.plan_id,
        prev_feature_version_id=final.feature_version_id,
    )


class RecordingEnv:
    """Deterministic env that never terminates; records executed actions."""

    def __init__(self, *, env_id: str = "env0") -> None:
        from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
        from pi05_rlt_rtc.runtime.scheduler import StepOutcome

        self.env_id = env_id
        self._step_outcome = StepOutcome
        self._snapshot_cls = ObservationSnapshot
        self.executed: list[np.ndarray] = []
        self._tick = 0

    def _obs(self):
        return self._snapshot_cls(
            env_id=self.env_id,
            episode_id=f"{self.env_id}-ep1",
            tick=self._tick,
            states=np.zeros(9, dtype=np.float32),
            images={"main": np.zeros((4, 4, 3), np.uint8), "wrist": np.zeros((4, 4, 3), np.uint8)},
            image_masks={"main": True, "wrist": True},
            prompt="insert the plug into the power strip",
        )

    def reset(self, *, seed: int = 0):
        self._tick = 0
        return self._obs()

    def step(self, action: np.ndarray):
        self.executed.append(np.asarray(action, dtype=np.float32))
        self._tick += 1
        return self._step_outcome(
            observation=self._obs(), reward=0.0, terminated=False, truncated=False, info={}
        )


def ledger_commands_by_tick(queue: ActionQueue) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for entry in queue.ledger.entries:
        out[entry.tick] = {
            "command": None if entry.command is None else np.asarray(entry.command, dtype=float)[0],
            "kind": entry.kind,
            "plan_id": entry.source_plan_id,
        }
    return out


def make_zero_state_encoder(
    spec,
    contract: HorizonContract,
    *,
    feature_id: str = FEATURE_ID,
    policy_id: str = POLICY_ID,
    norm_hash: str = "norm-v1",
):
    """Assembler state encoder with zero features; preserves queue/reference structure."""
    import numpy as np

    from pi05_rlt_rtc import PLANNING_STATE_VERSION, SCHEMA_VERSION
    from pi05_rlt_rtc.contracts.observation import EnvIDs
    from pi05_rlt_rtc.learning.planning_state import build_planning_state, make_time_features

    def encoder(
        *,
        request,
        observation,
        pending_prefix,
        pending_mask,
        prev_context,
        plan_event=None,
    ):
        pending = np.zeros((1, spec.max_delay, ENV_ACTION_DIM), dtype=np.float32)
        p_mask = np.zeros((1, spec.max_delay, ENV_ACTION_DIM), dtype=bool)
        active = min(pending_prefix.shape[0], spec.max_delay)
        pending[0, :active] = np.asarray(pending_prefix, dtype=np.float32)[:active]
        p_mask[0, :active] = True
        prev = np.zeros((1, spec.horizon, ENV_ACTION_DIM), dtype=np.float32)
        prev_mask = np.zeros((1, spec.horizon), dtype=bool)
        if prev_context.prev_env_plan is not None and prev_context.prev_env_plan.shape[0] > 0:
            n = min(prev_context.prev_env_plan.shape[0], spec.horizon)
            prev[0, :n] = np.asarray(prev_context.prev_env_plan, dtype=np.float32)[:n]
            prev_mask[0, :n] = True
        reference = np.zeros((1, spec.horizon, spec.env_action_dim), dtype=np.float32)
        if plan_event is not None and plan_event.get("plan") is not None:
            plan = plan_event["plan"]
            reference[0] = np.concatenate(
                [plan.pending_prefix, plan.submitted, plan.reference_tail], axis=0
            )
        ids = EnvIDs(
            env_id=request.env_id,
            episode_id=request.episode_id,
            episode_generation=request.episode_generation,
            policy_version_id=policy_id,
            feature_version_id=feature_id,
            schema_version=SCHEMA_VERSION,
            planning_state_version=PLANNING_STATE_VERSION,
        )
        return build_planning_state(
            spec=spec,
            ids=ids,
            capture_tick=request.capture_tick,
            plan_id=None if plan_event is None else plan_event.get("plan_id"),
            z_rl=np.zeros((1, spec.z_dim), dtype=np.float32),
            proprio=np.zeros((1, spec.proprio_dim), dtype=np.float32),
            pending_prefix=pending,
            pending_mask=p_mask,
            prev_plan_tail=prev,
            prev_plan_tail_mask=prev_mask,
            last_command=np.zeros((1, ENV_ACTION_DIM), dtype=np.float32),
            reference_full=reference,
            time_features=make_time_features(request.capture_tick, control_frequency_hz=10, delay_ticks=contract.delay),
            norm_stats_sha256=norm_hash,
        )

    return encoder
