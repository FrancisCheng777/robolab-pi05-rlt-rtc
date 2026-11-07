"""Versioned planner RPC. Cameras use lossless uint8 base64 instead of JSON pixels."""
from __future__ import annotations

import base64
import dataclasses
import json
import urllib.error
import urllib.request

import numpy as np

from pi05_rlt_rtc.contracts.errors import ContractError, EnvironmentNotReadyError, MockPolicyError
from pi05_rlt_rtc.runtime.scheduler import PlanResult


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def encode_observation(obs):
    obs.validate(proprio_dim=9)
    return {"env_id": obs.env_id, "episode_id": obs.episode_id, "tick": obs.tick,
            "states": obs.states.tolist(), "prompt": obs.prompt, "image_masks": dict(obs.image_masks),
            "captured_monotonic": obs.captured_monotonic, "source_timestamps": dict(obs.source_timestamps),
            "sensor_metadata": dict(obs.sensor_metadata),
            "images": {k: {"encoding": "rgb8-base64", "shape": list(v.shape),
                            "data": base64.b64encode(np.ascontiguousarray(v).tobytes()).decode("ascii")}
                       for k, v in obs.images.items()}}


def decode_image(value):
    if not isinstance(value, dict):  # legacy toy client
        a = np.asarray(value)
        if a.ndim != 3 or a.shape[-1] != 3 or not np.isfinite(a).all() or np.any(a != np.floor(a)) or np.any((a < 0) | (a > 255)):
            raise ContractError("invalid RGB image")
        return a.astype(np.uint8)
    shape = value.get("shape")
    if (value.get("encoding") != "rgb8-base64" or not isinstance(shape, list) or len(shape) != 3
            or any(type(s) is not int or s <= 0 for s in shape) or shape[-1] != 3
            or shape[0] > 2048 or shape[1] > 2048):
        raise ContractError("invalid encoded image shape/format")
    raw = base64.b64decode(value["data"], validate=True)
    if len(raw) != int(np.prod(shape)):
        raise ContractError("image byte count disagrees with shape")
    return np.frombuffer(raw, np.uint8).reshape(shape).copy()


def request_identity(request):
    return {k: getattr(request, k) for k in ("request_id", "env_id", "episode_id", "episode_generation",
        "queue_generation", "capture_tick", "apply_start_tick", "feature_version_id", "policy_version_id",
        "schema_version", "planning_state_version", "sampler_mode", "num_denoising_steps", "rng_seed", "deadline_tick")}


class RemotePlanner:
    is_mock = False

    def __init__(self, url, cfg, *, timeout=30., token=None):
        if not url.startswith(("http://", "https://")) or timeout <= 0:
            raise ContractError("planner URL must be HTTP(S) with a positive timeout")
        self.url, self.cfg, self.timeout, self.token = url.rstrip("/"), cfg, timeout, token
        self.metadata = self._call("/metadata")
        if self.metadata.get("mock") is not False:
            raise MockPolicyError("real controller refuses a mock planner service")
        if self.metadata.get("protocol") != 1 or self.metadata.get("runtime_contract") != cfg.wire_contract():
            raise ContractError("planner/controller action, clock or H/C/D contracts differ")
        policy = cfg.raw["policy"]
        if self.metadata.get("policy_kind") != policy["kind"]:
            raise ContractError("remote policy kind differs from the requested teacher/actor")
        for key in ("feature_version_id", "policy_version_id"):
            if policy.get("expected_" + key) and policy["expected_" + key] != self.metadata.get(key):
                raise ContractError(f"remote {key} differs from the configured expected version")

    def _call(self, path, payload=None):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        data = None if payload is None else json.dumps(payload, default=json_value, allow_nan=False).encode()
        request = urllib.request.Request(self.url + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read(2 * 1024 * 1024 + 1)
            if len(body) > 2 * 1024 * 1024:
                raise ContractError("planner response is too large")
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise ContractError(f"planner rejected RPC ({exc.code}): {detail}") from exc
        except (OSError, urllib.error.URLError) as exc:
            raise EnvironmentNotReadyError(f"planner RPC failed: {exc}") from exc

    def plan(self, request, observation, pending_prefix, prev_context):
        result = self._call("/plan", {"request": dataclasses.asdict(request),
            "observation": encode_observation(observation), "pending_prefix": np.asarray(pending_prefix).tolist()})
        if result.get("identity") != request_identity(request):
            raise ContractError("planner response identity/version mismatch")
        out = PlanResult(submitted=np.asarray(result["submitted"], np.float32),
                         reference_env_full=np.asarray(result["reference_env_full"], np.float32),
                         source=result["source"], teacher_source=result.get("teacher_source"))
        for name in ("raw_actor_plan", "bc_target_plan", "bc_target_valid_mask"):
            value = result.get(name)
            if value is not None:
                setattr(out, name, np.asarray(value, np.float32))
        state = result.get('planning_state')
        if state is not None:
            from pi05_rlt_rtc.learning.planning_state import PlanningState, PlanningStateSpec
            from pi05_rlt_rtc.contracts.observation import EnvIDs
            state = dict(state)
            state['spec'] = PlanningStateSpec(**state['spec'])
            state['ids'] = EnvIDs(**state['ids'])
            for key in ('z_rl', 'proprio', 'pending_prefix', 'pending_mask', 'prev_plan_tail',
                        'prev_plan_tail_mask', 'last_command', 'reference_full', 'time_features'):
                state[key] = np.asarray(state[key], bool if key.endswith('mask') else np.float32)
            out.planning_state = PlanningState(**state)
            ps = out.planning_state
            if (any(getattr(ps.ids, k) != getattr(request, k) for k in dataclasses.asdict(ps.ids))
                    or ps.capture_tick != request.capture_tick or ps.plan_id is not None
                    or ps.norm_stats_sha256 != self.metadata['norm_stats_sha256']
                    or ps.spec.horizon != self.cfg.contract.horizon
                    or ps.spec.chunk != self.cfg.contract.chunk or ps.batch_size != 1):
                raise ContractError('remote planning-state identity/spec mismatch')
        out.validate(self.cfg.contract)
        if np.any(np.abs(out.submitted) > 1) or np.any(np.abs(out.reference_env_full) > 1):
            raise ContractError("remote planner returned commands outside [-1,1]")
        return out
