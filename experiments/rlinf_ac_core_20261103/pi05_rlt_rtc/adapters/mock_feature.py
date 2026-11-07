"""Explicitly-labelled mock feature service for CPU pipeline tests.

This service exists so that the queue -> codec -> planning-state -> learner
wiring can be exercised on a machine without RLinf/GPU.  It is **not** a
substitute for the real model and every result carries ``is_mock=True`` and a
``MOCK`` feature version.  Formal run directories reject it (see
``pi05_rlt_rtc.tools.run_context``).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np

from pi05_rlt_rtc import SCHEMA_VERSION
from pi05_rlt_rtc.action_codec.codec import ActionCodec
from pi05_rlt_rtc.adapters.feature_reference import FeatureReferenceResult
from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
from pi05_rlt_rtc.contracts.plan import PlanRequest
from pi05_rlt_rtc.contracts.spaces import HorizonContract

MOCK_FEATURE_VERSION_ID = "MOCK-feature-v1"


class MockFeatureReferenceService:
    """Deterministic pseudo-features derived from the observation state.

    ``prefix_calls`` counts extract() invocations so the "one prefix pass per
    request" invariant can be tested (T20).  If ``enforce_single_prefix`` is
    True, calling extract twice for the same request id raises.
    """

    is_mock = True

    def __init__(
        self,
        *,
        contract: HorizonContract,
        codec: ActionCodec,
        spec_z_dim: int = 2048,
        norm_stats_sha256: str = "mock-norm-stats",
        enforce_single_prefix: bool = True,
    ) -> None:
        self.contract = contract
        self.codec = codec
        self.z_dim = int(spec_z_dim)
        self.norm_stats_sha256 = str(norm_stats_sha256)
        self.enforce_single_prefix = bool(enforce_single_prefix)
        self.prefix_calls = 0
        self.extract_counts: dict[int, int] = {}

    def extract(
        self, request: PlanRequest, observation: ObservationSnapshot
    ) -> FeatureReferenceResult:
        # Warm-start pseudo-requests (id < 0) may repeat at every episode reset;
        # real requests must never be extracted twice.
        enforce = self.enforce_single_prefix and request.request_id >= 0
        if enforce and self.extract_counts.get(request.request_id, 0) >= 1:
            raise RuntimeError(
                f"request {request.request_id} asked for a second prefix pass; per-request "
                "prefix count must stay 1"
            )
        self.prefix_calls += 1
        self.extract_counts[request.request_id] = self.extract_counts.get(request.request_id, 0) + 1
        started = time.perf_counter()
        state = np.asarray(observation.states, dtype=np.float32).reshape(1, -1)
        rng = np.random.default_rng(request.rng_seed)
        base = np.zeros((1, self.z_dim), dtype=np.float32)
        base[0, : state.shape[-1]] = state[0, : self.z_dim]
        base[0, : min(self.z_dim, 64)] += rng.standard_normal(min(self.z_dim, 64)).astype(np.float32) * 0.01
        proprio = np.zeros((1, 9), dtype=np.float32)
        proprio[0, : min(9, state.shape[-1])] = state[0, :9]
        # Toy teacher control law: drive its first state dimension to zero.
        # (The test fixture gripper and remaining joints are held at 0 in the toy.)
        phase = float(state[0, 0]) if state.size else 0.0
        u0 = float(np.clip(-phase, -1.0, 1.0))
        env_ref = np.zeros((1, self.contract.horizon, self.codec.env_action_dim), dtype=np.float32)
        env_ref[0, :, 0] = u0
        env_ref = np.clip(env_ref, self.codec.env_min, self.codec.env_max).astype(np.float32)
        model_actions = self.codec.encode_env_for_guidance(env_ref)
        elapsed = time.perf_counter() - started
        return FeatureReferenceResult(
            request_id=request.request_id,
            env_id=request.env_id,
            episode_id=request.episode_id,
            capture_tick=request.capture_tick,
            apply_start_tick=request.apply_start_tick,
            z_rl=base,
            proprio_model=proprio,
            vla_model_actions=model_actions,
            reference_env_full=env_ref,
            feature_version_id=MOCK_FEATURE_VERSION_ID,
            norm_stats_sha256=self.norm_stats_sha256,
            schema_version=SCHEMA_VERSION,
            planning_state_version=request.planning_state_version,
            rng_seed=request.rng_seed,
            sampler_mode=request.sampler_mode,
            num_denoising_steps=request.num_denoising_steps,
            prefix_calls=1,
            timings={"total_seconds": elapsed, "prefix_seconds": elapsed * 0.2},
            is_mock=True,
            codec_provenance=self.codec.provenance(),
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "mock": True,
            "feature_version_id": MOCK_FEATURE_VERSION_ID,
            "prefix_calls": self.prefix_calls,
            "warning": "mock features are for CPU wiring tests only; not a model",
        }
