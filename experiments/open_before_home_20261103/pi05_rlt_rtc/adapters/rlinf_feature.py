"""RLinf-backed feature/reference adapter.

Thin adapter over the *audited* pinned RLinf API.  It does not reimplement the
VLA; it composes the existing methods in the order required by guide section
10.1 so that exactly one prefix pass produces both ``z_rl`` and the full-H
reference:

1. ``env_obs_to_observation`` (single observation transform);
2. ``build_prefix_cache`` once;
3. ``_select_rlt_prefix_embeddings`` + ``_encode_rlt_flat`` for z;
4. the RTC sampler branch (``none`` / ``approx`` / ``full_vjp``) reusing the
   same ``kv_cache`` and ``prefix_mask``;
5. full-H decode through the output transform *without* the
   ``action_chunk`` truncation that ``EnvIO.decode_actions`` applies.

Method names prefixed with ``_`` are audited upstream privates at commit
``c70606f08cdca259b8dec03d4430926b5b8fac9d``; if upstream renames them the
adapter must fail loudly and be re-audited (see ``source_audit.py``).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from pi05_rlt_rtc.adapters.feature_reference import FeatureReferenceResult
from pi05_rlt_rtc.contracts.errors import ContractError, EnvironmentNotReadyError
from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
from pi05_rlt_rtc.contracts.plan import PlanRequest
from pi05_rlt_rtc.contracts.spaces import HorizonContract
from pi05_rlt_rtc.rtc.canonical import canonical_velocity_from_native

PINNED_RLINF_COMMIT = "c70606f08cdca259b8dec03d4430926b5b8fac9d"


class RLinfFeatureReferenceService:
    """Real service; requires the pinned RLinf runtime + weights on device."""

    is_mock = False

    def __init__(
        self,
        *,
        model: Any,
        contract: HorizonContract,
        codec: Any,
        rtc_dispatcher: Any,
        feature_version_id: str,
        norm_stats_sha256: str,
        planning_state_version: str,
        expected_commit: str = PINNED_RLINF_COMMIT,
    ) -> None:
        self.model = model
        self.contract = contract
        self.codec = codec
        self.rtc_dispatcher = rtc_dispatcher
        self.feature_version_id = str(feature_version_id)
        self.norm_stats_sha256 = str(norm_stats_sha256)
        self.planning_state_version = str(planning_state_version)
        self.expected_commit = expected_commit
        self._check_api_surface()
        if codec.norm_stats.sha256 != self.norm_stats_sha256:
            raise ContractError("feature service and ActionCodec norm-stats hashes disagree")
        if model.action_dim != codec.model_action_dim or model.action_horizon != contract.horizon:
            raise ContractError("model action shape disagrees with the codec/horizon contract")
        self.model.eval()
        self.model.requires_grad_(False)

    def _check_api_surface(self) -> None:
        required = (
            "env_obs_to_observation",
            "build_prefix_cache",
            "run_suffix",
            "velocity_from_suffix",
            "_select_rlt_prefix_embeddings",
            "_encode_rlt_flat",
        )
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise EnvironmentNotReadyError(
                f"RLinf pi0.5 eval model is missing audited methods {missing}; either the "
                f"checkout is not at {self.expected_commit} or the API changed. Re-run "
                "PI05_RLT_RTC source audit before proceeding."
            )

    @staticmethod
    def _to_env_obs(observation: ObservationSnapshot) -> dict[str, Any]:
        return {
            "main_images": np.array(observation.camera("wrist"), copy=True)[None, ...],
            "states": np.array(observation.states, copy=True)[None, ...],
            "task_descriptions": [observation.prompt],
        }

    def extract(
        self, request: PlanRequest, observation: ObservationSnapshot
    ) -> FeatureReferenceResult:
        import torch  # local import: torch exists in the RLinf runtime

        request.validate(self.contract, allow_warm_start=True)
        observation.validate(proprio_dim=9)
        if observation.states.shape != (9,) or any(
            observation.camera(k).ndim != 3 or observation.camera(k).shape[-1] != 3
            or not observation.image_masks.get(k, False) for k in ("wrist",)
        ):
            raise ContractError("real feature extraction requires one state [9] and one valid wrist HWC RGB camera")
        if (observation.env_id, observation.episode_id, observation.tick) != (request.env_id, request.episode_id, request.capture_tick):
            raise ContractError("request and observation identity/tick disagree")
        if request.feature_version_id != self.feature_version_id or request.planning_state_version != self.planning_state_version:
            raise ContractError("request targets a different frozen feature/planning-state version")
        if request.num_denoising_steps != self.rtc_dispatcher.num_steps:
            raise ContractError("request denoising steps disagree with the configured sampler")
        if request.rtc_context.prev_env_plan is not None and request.rtc_context.prev_feature_version_id != self.feature_version_id:
            raise ContractError("previous plan uses a different feature version")
        if request.sampler_mode == "full_vjp":
            from pi05_rlt_rtc.rtc.full_vjp import check_outer_context
            check_outer_context()
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        env_obs = self._to_env_obs(observation)
        with torch.no_grad():
            from rlinf.models.embodiment.openpi.modules.model import preprocess_observation

            observation_obj = self.model.env_obs_to_observation(env_obs)
            prepared = preprocess_observation(observation_obj, train=False)
            prefix_output, prefix_mask, kv_cache = self.model.build_prefix_cache(prepared)
            prefix_calls = 1
            timings["prefix_seconds"] = time.perf_counter() - t0

            t1 = time.perf_counter()
            rlt_prefix, rlt_mask = self.model._select_rlt_prefix_embeddings(
                prefix_output, prefix_mask, prepared.tokenized_prompt
            )
            z_rl = self.model._encode_rlt_flat(rlt_prefix, rlt_mask).to(torch.float32)
            timings["rlt_token_seconds"] = time.perf_counter() - t1

            t2 = time.perf_counter()
            vla_model_actions = self._sample_with_mode(
                request, prepared, observation_obj, prefix_mask, kv_cache
            )
            timings["sampling_seconds"] = time.perf_counter() - t2

            t3 = time.perf_counter()
            reference_env_full = self._decode_full_horizon(
                vla_model_actions, observation_obj.state
            )
            timings["codec_seconds"] = time.perf_counter() - t3

        proprio = self._proprio(env_obs, observation_obj.state, env_obs)
        result = FeatureReferenceResult(
            request_id=request.request_id,
            env_id=request.env_id,
            episode_id=request.episode_id,
            capture_tick=request.capture_tick,
            apply_start_tick=request.apply_start_tick,
            z_rl=z_rl.detach().cpu().numpy().astype(np.float32),
            proprio_model=proprio.detach().cpu().numpy().astype(np.float32),
            vla_model_actions=vla_model_actions.detach().cpu().numpy().astype(np.float32),
            reference_env_full=self.codec._to_numpy(reference_env_full).astype(np.float32),
            feature_version_id=self.feature_version_id,
            norm_stats_sha256=self.norm_stats_sha256,
            schema_version=request.schema_version,
            planning_state_version=self.planning_state_version,
            rng_seed=request.rng_seed,
            sampler_mode=request.sampler_mode,
            num_denoising_steps=request.num_denoising_steps,
            prefix_calls=prefix_calls,
            timings=timings,
            is_mock=False,
            codec_provenance=self.codec.provenance(),
        )
        result.validate(self.contract)
        return result

    # -- internals --------------------------------------------------------------
    def _sample_with_mode(self, request, prepared, observation_obj, prefix_mask, kv_cache):
        import torch

        from pi05_rlt_rtc.rtc.approx import build_rtc_target_and_mask

        B = observation_obj.state.shape[0]
        noise = torch.Generator(device=observation_obj.state.device)
        noise.manual_seed(int(request.rng_seed))
        x_init = torch.randn(
            B,
            self.contract.horizon,
            self.model.action_dim,
            generator=noise,
            device=observation_obj.state.device,
            dtype=torch.float32,
        )

        def velocity_native(x, t):
            suffix_act = self.model.run_suffix(prepared, x, t, kv_cache, prefix_mask)
            return self.model.velocity_from_suffix(suffix_act).to(torch.float32)

        prev = request.rtc_context
        prev_model_actions = None
        if prev is not None and prev.prev_env_plan is not None and prev.prev_env_plan.shape[0] > 0:
            prev_model_actions = torch.as_tensor(
                self.codec.encode_env_for_guidance(prev.prev_env_plan),
                device=x_init.device,
                dtype=torch.float32,
            )[None, ...]
        # PrevPlanContext has already been shifted to the current capture tick.
        executed_horizon = 0

        target_model = None
        guidance_mask = None
        if prev_model_actions is not None and prev_model_actions.shape[1] > 0:
            # Full-H reference target from the previous *composed* plan, encoded
            # to model space; the mask also zeroes the padding dimensions.
            target_model = torch.zeros_like(x_init)
            overlap = min(prev_model_actions.shape[1], self.contract.horizon)
            target_model[:, :overlap] = prev_model_actions[:, :overlap]
            from pi05_rlt_rtc.rtc.prefix_weights import guidance_mask_from_previous_plan

            mask_np = guidance_mask_from_previous_plan(
                horizon=self.contract.horizon,
                chunk=self.contract.chunk,
                delay=self.contract.delay,
                previous_valid_len=overlap,
                action_valid_mask=self.codec.valid_model_mask,
            )
            valid = np.asarray(prev.prev_valid_mask, dtype=bool)
            if valid.ndim == 1:
                valid = np.broadcast_to(valid[:, None], prev.prev_env_plan.shape)
            mask_np[:overlap, :self.codec.env_action_dim] *= valid[:overlap]
            guidance_mask = torch.as_tensor(mask_np, device=x_init.device, dtype=torch.float32)
            guidance_mask = guidance_mask[None, ...].expand(B, -1, -1).clone()
        elif request.sampler_mode == "approx":
            # Approx mode uses its own upstream-compatible target/mask builder.
            _, time_mask = build_rtc_target_and_mask(
                prev_model_actions,
                self.contract.horizon,
                self.model.action_dim,
                self.contract.delay,
                device=x_init.device,
                dtype=torch.float32,
            )
            target_model = torch.zeros_like(x_init)
            guidance_mask = time_mask.expand(B, -1, self.model.action_dim).clone()

        canonical = None
        if request.sampler_mode == "full_vjp":
            canonical = canonical_velocity_from_native(velocity_native)
        result = self.rtc_dispatcher.sample(
            request.sampler_mode,
            x_init=x_init,
            velocity_native=velocity_native,
            velocity_canonical=canonical,
            target_model=target_model,
            guidance_mask=guidance_mask,
            prev_model_actions=prev_model_actions,
            executed_horizon=executed_horizon,
            delay_steps=self.contract.delay,
        )
        return result.actions

    def _decode_full_horizon(self, model_actions, state):
        """Decode all H steps, bypassing ``EnvIO.decode_actions``' chunk truncation."""
        import torch

        from rlinf.models.embodiment.openpi.transforms.apply import apply_output_transform

        # Upstream `output_transform` forwards `action_chunk` which truncates to
        # C (audited in env_io.py).  Here we call the transform directly with
        # action_chunk=None so the full H survives to the unique suffix slicer.
        outputs = apply_output_transform(
            self.model._output_transform_fn,
            {"actions": model_actions, "state": state},
            action_chunk=None,
        )
        # The physical action contract bounds each controller input to [-1,1]. Keep the
        # committed plan and RTC memory in the command space actually executed.
        actions = outputs["actions"].to(dtype=torch.float32)
        if not torch.isfinite(actions).all():
            raise ContractError("decoded reference contains NaN/Inf")
        return actions.clamp(-1.0, 1.0)

    @staticmethod
    def _proprio(env_obs, model_state, env_obs_dict):
        state_dim = np.asarray(env_obs_dict["states"]).shape[-1]
        return model_state[..., :state_dim]

    def provenance(self) -> dict[str, Any]:
        return {
            "mock": False,
            "rlinf_commit": self.expected_commit,
            "feature_version_id": self.feature_version_id,
            "norm_stats_sha256": self.norm_stats_sha256,
        }
