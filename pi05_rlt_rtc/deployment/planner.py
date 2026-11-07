"""One prefix pass -> frozen features/reference -> submitted queue-aware action."""
from __future__ import annotations

import dataclasses
import numpy as np

from pi05_rlt_rtc import PLANNING_STATE_VERSION, SCHEMA_VERSION
from pi05_rlt_rtc.contracts.errors import ContractError, MockPolicyError
from pi05_rlt_rtc.contracts.observation import EnvIDs
from pi05_rlt_rtc.learning.planning_state import build_planning_state, make_time_features
from pi05_rlt_rtc.runtime.scheduler import PlanResult


class FeaturePlanner:
    is_mock = False

    def __init__(self, *, service, spec, contract, control_hz, policy_version_id, actor=None, deterministic=True, rtc_mode=None, critical_only=False):
        if service.is_mock:
            raise MockPolicyError("real planner cannot use a mock feature service")
        self.service, self.spec, self.contract = service, spec, contract
        self.control_hz, self.policy_version_id, self.actor = control_hz, policy_version_id, actor
        self.deterministic = bool(deterministic)
        self.rtc_mode = rtc_mode
        self.critical_only = critical_only

    def plan(self, request, observation, pending_prefix, prev_context):
        request.validate(self.contract, allow_warm_start=True)
        if request.policy_version_id != self.policy_version_id:
            raise ContractError("request policy version differs from the loaded planner")
        if request.request_id >= 0 and self.rtc_mode is not None and request.sampler_mode != self.rtc_mode:
            raise ContractError("request RTC mode differs from the loaded deployment")
        pending_prefix = np.asarray(pending_prefix, np.float32)
        if pending_prefix.shape != (self.contract.delay, 8) or not np.isfinite(pending_prefix).all():
            raise ContractError("pending_prefix must be finite [D,8]")
        fr = self.service.extract(request, observation)
        fr.validate(self.contract, expected_batch=1)
        pending = np.zeros((self.spec.max_delay, 8), np.float32)
        pending_mask = np.zeros_like(pending, bool)
        pending[:self.contract.delay] = pending_prefix
        pending_mask[:self.contract.delay] = True
        prev = np.zeros((self.spec.horizon, 8), np.float32)
        prev_mask = np.zeros(self.spec.horizon, bool)
        if prev_context.prev_env_plan is not None:
            n = min(len(prev_context.prev_env_plan), self.spec.horizon)
            prev[:n] = prev_context.prev_env_plan[:n]
            valid = np.asarray(prev_context.prev_valid_mask, bool)[:n]
            prev_mask[:n] = valid.all(-1) if valid.ndim == 2 else valid
        state = build_planning_state(
            spec=self.spec,
            ids=EnvIDs(env_id=request.env_id, episode_id=request.episode_id,
                       episode_generation=request.episode_generation,
                       policy_version_id=request.policy_version_id, feature_version_id=fr.feature_version_id,
                       schema_version=SCHEMA_VERSION, planning_state_version=PLANNING_STATE_VERSION),
            capture_tick=request.capture_tick, plan_id=None, z_rl=fr.z_rl,
            proprio=fr.proprio_model, pending_prefix=pending, pending_mask=pending_mask,
            prev_plan_tail=prev, prev_plan_tail_mask=prev_mask,
            last_command=np.zeros(8, np.float32) if request.last_command is None else request.last_command,
            reference_full=fr.reference_env_full,
            time_features=make_time_features(request.capture_tick, control_frequency_hz=self.control_hz,
                                             delay_ticks=self.contract.delay),
            norm_stats_sha256=fr.norm_stats_sha256)
        reference = np.asarray(fr.reference_env_full, np.float32).reshape(self.spec.horizon, 8)
        teacher = reference[self.contract.suffix_start:self.contract.suffix_end].copy()
        submitted, raw = teacher.copy(), None
        source = "reference_teacher"
        result = PlanResult(submitted=submitted, reference_env_full=reference, source=source,
                            raw_actor_plan=raw, bc_target_plan=teacher,
                            bc_target_valid_mask=np.ones_like(teacher, bool),
                            teacher_source="frozen_pi05_rtc", planning_state=state)
        apply_actor(result, self.actor, request, observation, self.contract,
                    deterministic=self.deterministic, critical_only=self.critical_only)
        result.validate(self.contract)
        if np.any(np.abs(result.submitted) > 1) or np.any(np.abs(reference) > 1):
            raise ContractError("planner returned commands outside [-1,1]")
        return result


def state_from_plan_event(*, request, plan_event=None, **_):
    if plan_event is None or plan_event.get("result") is None:
        raise ContractError("no captured planning state for this request")
    state = plan_event["result"].planning_state
    if state is None or state.capture_tick != request.capture_tick:
        raise ContractError("planning state does not match capture")
    plan = plan_event.get("plan")
    return dataclasses.replace(state, plan_id=None if plan is None else plan.plan_id)


def apply_actor(result, actor, request, observation, contract, *, deterministic, critical_only):
    """Shared local/resident actor path; frozen teacher targets remain intact."""
    if actor is None or request.request_id < 0 or (critical_only and observation.sensor_metadata.get('rlt_active') is not True):
        return result
    import torch
    state = result.planning_state
    if state is None:
        raise ContractError('actor requires captured frozen planning features')
    device = next(actor.parameters()).device
    generator = torch.Generator(device=device).manual_seed(request.rng_seed)
    with torch.no_grad():
        output = actor.act(state=torch.as_tensor(state.actor_state_vector(), device=device),
            reference_suffix=torch.as_tensor(state.reference_suffix_vector(contract), device=device),
            deterministic=deterministic, generator=generator)
    result.submitted = output.actions[0].cpu().numpy()
    result.raw_actor_plan = output.raw[0].cpu().numpy()
    result.source = 'actor'
    result.validate(contract)
    if not np.isfinite(result.submitted).all() or np.any(np.abs(result.submitted) > 1):
        raise ContractError('actor produced invalid commands')
    return result
