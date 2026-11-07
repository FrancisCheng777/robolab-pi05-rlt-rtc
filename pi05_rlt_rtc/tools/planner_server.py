"""Minimal planner HTTP service (stdlib only).

The service wraps a :class:`PlannerProtocol` backend.  With ``--toy`` it wraps
the explicitly labelled mock backend; the real backend requires the pinned
RLinf runtime.  It is deliberately small: one in-flight request per
environment is the first-version contract, and the service never blocks a
controller tick (the controller does not call this synchronously).
"""

from __future__ import annotations

import json
import dataclasses
import threading
import hmac
import time
import os
import gc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
from pi05_rlt_rtc.contracts.plan import PlanRequest, PrevPlanContext
from pi05_rlt_rtc.runtime.scheduler import PlanResult
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.deployment.rpc import decode_image, request_identity, json_value


def plan_result_to_json(result: PlanResult) -> dict:
    return {
        "planning_state": None if result.planning_state is None else dataclasses.asdict(result.planning_state),
        "submitted": np.asarray(result.submitted).tolist(),
        "reference_env_full": np.asarray(result.reference_env_full).tolist(),
        "source": result.source,
        "teacher_source": result.teacher_source,
        "bc_target_plan": None
        if result.bc_target_plan is None
        else np.asarray(result.bc_target_plan).tolist(),
        "raw_actor_plan": None if result.raw_actor_plan is None else np.asarray(result.raw_actor_plan).tolist(),
        "bc_target_valid_mask": None if result.bc_target_valid_mask is None else np.asarray(result.bc_target_valid_mask).tolist(),
    }


class PlannerService:
    def __init__(self, backend, contract, *, env_action_dim: int = 8, metadata=None, executor=None) -> None:
        self.executor = executor
        self.backend = backend
        self.contract = contract
        self.env_action_dim = env_action_dim
        self.lock = threading.Lock()
        self.metadata = metadata or {"protocol": 1, "mock": bool(getattr(backend, "is_mock", True))}
        self.profile = os.environ.get('PI05_PROFILE_PLANNER') == '1'
        self.gc_seconds = 0.
        self.gc_started = None
        if self.profile:
            gc.callbacks.append(self._gc_callback)

    def _gc_callback(self, phase, info):
        if phase == 'start':
            self.gc_started = time.perf_counter()
        elif self.gc_started is not None:
            self.gc_seconds += time.perf_counter()-self.gc_started
            self.gc_started = None

    def close(self):
        if self.profile and self._gc_callback in gc.callbacks:
            gc.callbacks.remove(self._gc_callback)

    def handle(self, payload: dict) -> dict:
        started = time.perf_counter()
        gc_before = self.gc_seconds
        request = _request_from_json(payload["request"], self.contract)
        obs = _observation_from_json(payload["observation"])
        pending = np.asarray(payload["pending_prefix"], dtype=np.float32)
        request.validate(self.contract, allow_warm_start=True)
        obs.validate(proprio_dim=9)
        if (obs.env_id, obs.episode_id, obs.tick) != (request.env_id, request.episode_id, request.capture_tick):
            raise ContractError("request/observation identity mismatch")
        if pending.shape != (self.contract.delay, 8) or not np.isfinite(pending).all():
            raise ContractError("pending prefix must be finite [D,8]")
        prev = request.rtc_context
        if not self.lock.acquire(blocking=False):
            raise ContractError("planner busy; requests are not queued behind stale inference")
        try:
            dispatched = time.perf_counter()
            if self.executor is None:
                result = self.backend.plan(request, obs, pending, prev)
            else:
                result = self.executor.submit(self.backend.plan, request, obs, pending, prev).result()
        finally:
            self.lock.release()
        result.validate(self.contract)
        response = {**plan_result_to_json(result), "identity": request_identity(request)}
        if self.profile:
            response['timing'] = {'service_ms': (time.perf_counter()-started)*1000,
                'dispatch_to_result_ms': (time.perf_counter()-dispatched)*1000,
                'gc_pause_ms': (self.gc_seconds-gc_before)*1000,
                'feature_components_s': getattr(self.backend, 'last_feature_timings', {})}
            if response['timing']['service_ms'] > 250:
                print(json.dumps({'slow_planner_request': request.request_id, **response['timing']}), flush=True)
        return response


def _request_from_json(data: dict, contract) -> PlanRequest:
    prev = data.get("rtc_context") or {}
    prev_plan = prev.get("prev_env_plan")
    prev_context = PrevPlanContext(
        prev_env_plan=None if prev_plan is None else np.asarray(prev_plan, dtype=np.float32),
        prev_valid_mask=None
        if prev_plan is None
        else np.asarray(prev.get("prev_valid_mask", np.ones(len(prev_plan))), dtype=bool),
        prev_capture_tick=prev.get("prev_capture_tick"),
        prev_plan_id=prev.get("prev_plan_id"),
        prev_feature_version_id=prev.get("prev_feature_version_id"),
    )
    return PlanRequest(
        request_id=int(data["request_id"]),
        env_id=data["env_id"],
        episode_id=data["episode_id"],
        episode_generation=int(data["episode_generation"]),
        capture_tick=int(data["capture_tick"]),
        apply_start_tick=int(data["apply_start_tick"]),
        queue_generation=int(data["queue_generation"]),
        policy_version_id=data["policy_version_id"],
        feature_version_id=data["feature_version_id"],
        schema_version=data["schema_version"],
        planning_state_version=data["planning_state_version"],
        rtc_context=prev_context,
        rng_seed=int(data["rng_seed"]),
        sampler_mode=data["sampler_mode"],
        num_denoising_steps=int(data["num_denoising_steps"]),
        deadline_tick=data.get("deadline_tick"),
        last_command=None if data.get("last_command") is None else np.asarray(data["last_command"], dtype=np.float32),
    )


def _observation_from_json(data: dict) -> ObservationSnapshot:
    return ObservationSnapshot(
        env_id=data["env_id"],
        episode_id=data["episode_id"],
        tick=int(data["tick"]),
        states=np.asarray(data["states"], dtype=np.float32),
        images={k: decode_image(v) for k, v in data["images"].items()},
        image_masks={k: bool(v) for k, v in data.get("image_masks", {}).items()},
        prompt=data["prompt"],
        captured_monotonic=data.get("captured_monotonic"),
        source_timestamps=data.get("source_timestamps", {}),
        sensor_metadata=data.get("sensor_metadata", {}),
    )


def make_handler(service: PlannerService, *, shutdown_after: int | None = None, token=None):
    counter = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def authorized(self):
            return token is None or hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token)

        def do_GET(self):
            if not self.authorized():
                self.send_error(401)
                return
            if self.path != "/metadata":
                self.send_error(404)
                return
            body = json.dumps(service.metadata, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            self.connection.settimeout(10.)
            if not self.authorized():
                self.send_error(401)
                return
            if self.path not in ("/plan", "/"):
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ContractError("planner request exceeds the 16 MiB limit")
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                response = service.handle(payload)
                status = 200
            except Exception as exc:  # noqa: BLE001 - the service must report errors
                response = {"error": f"{type(exc).__name__}: {exc}"}
                status = 500
            body = json.dumps(response, default=json_value, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            counter["n"] += 1
            if shutdown_after is not None and counter["n"] >= shutdown_after:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, *args) -> None:  # silence default logging
            return

    return Handler


def serve(service: PlannerService, *, host: str, port: int, shutdown_after: int | None = None, token=None) -> None:
    """Serve until stopped (``shutdown_after`` requests make this finite for tests)."""
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise ContractError("non-loopback planner binding requires PI05_PLANNER_TOKEN; prefer an SSH tunnel")
    server = ThreadingHTTPServer((host, port), make_handler(service, shutdown_after=shutdown_after, token=token))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()
