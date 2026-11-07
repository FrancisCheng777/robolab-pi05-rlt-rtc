"""Strict parser for the project design config (guide sections 16.1-16.4).

``design_config.example.yaml`` is *our* adapter schema, not an upstream
runnable RLinf config.  This module implements the parsing, factory-facing
defaults and unknown-field validation; it never silently ignores a key and
never maps design fields straight into an upstream launcher.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from pi05_rlt_rtc.contracts.errors import ConfigError, ContractError
from pi05_rlt_rtc.contracts.spaces import HorizonContract
from pi05_rlt_rtc.learning.planning_state import PlanningStateSpec

DESIGN_SCHEMA_VERSION = "pi05-rlt-rtc-design-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _strict(section: str, mapping: Mapping[str, Any], known: set[str]) -> dict[str, Any]:
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise ConfigError(
            f"config section {section!r} has unknown field(s) {unknown}; unknown fields "
            "are rejected, not ignored"
        )
    return dict(mapping)


@dataclasses.dataclass(frozen=True)
class SourceConfig:
    rlinf_repository: str
    rlinf_commit: str
    official_openpi_commit: str
    official_rtc_commit: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SourceConfig":
        d = _strict(
            "source",
            data,
            {"rlinf_repository", "rlinf_commit", "official_openpi_commit", "official_rtc_commit"},
        )
        missing = [k for k in ("rlinf_commit", "official_openpi_commit", "official_rtc_commit") if not d.get(k)]
        _require(not missing, f"source: missing pinned commit(s) {missing}")
        return cls(**d)


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    base_checkpoint_path: str | None
    stage1_checkpoint_path: str | None
    dataset_path: str | None
    dataset_revision: str | None
    norm_stats_path: str | None
    norm_stats_sha256: str | None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AssetsConfig":
        d = _strict(
            "assets",
            data,
            {
                "base_checkpoint_path",
                "stage1_checkpoint_path",
                "dataset_path",
                "dataset_revision",
                "norm_stats_path",
                "norm_stats_sha256",
            },
        )
        return cls(**d)

    def require_for_stage(self, stage: str) -> None:
        if stage in ("stage1", "base"):
            if not self.base_checkpoint_path:
                raise ConfigError("assets.base_checkpoint_path is required for Stage 1")
        if stage in ("stage2", "apply"):
            if not self.stage1_checkpoint_path:
                raise ConfigError(
                    "assets.stage1_checkpoint_path is required for Stage 2; a base model is "
                    "not a Stage 1 feature checkpoint"
                )
        if stage in ("stage2", "data"):
            if not self.dataset_path:
                raise ConfigError("assets.dataset_path is required")
            if not self.norm_stats_path:
                raise ConfigError("assets.norm_stats_path is required")


@dataclasses.dataclass(frozen=True)
class EnvironmentConfig:
    id: str
    control_mode: str
    control_frequency_hz: int
    bridge_frequency_hz: int
    action_dimension: int
    proprio_dimension: int
    cameras: tuple[str, ...]
    prompt: str
    number_of_environments: int
    auto_reset: bool
    expert_takeover: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EnvironmentConfig":
        d = _strict(
            "environment",
            data,
            {
                "id",
                "control_mode",
                "control_frequency_hz",
                "bridge_frequency_hz",
                "action_dimension",
                "proprio_dimension",
                "cameras",
                "prompt",
                "number_of_environments",
                "auto_reset",
                "expert_takeover",
            },
        )
        d["cameras"] = tuple(d.get("cameras", ()))
        env = cls(**d)
        _require(env.id == "PandaRobotiqPlugInsertion-v1", (
            "environment.id must be PandaRobotiqPlugInsertion-v1 in the first version; "
            "substituting a similarly named task while reusing the same data is forbidden"
        ))
        _require(env.control_mode == "measured_joint_delta_robotiq_width", "environment.control_mode must be measured_joint_delta_robotiq_width")
        _require(env.action_dimension == 8, "environment.action_dimension must be 8")
        _require(env.proprio_dimension == 9, "environment.proprio_dimension must be 9")
        _require({"main", "wrist"}.issubset(set(env.cameras)), "cameras must include 'main' and 'wrist'")
        _require(env.control_frequency_hz > 0, "control_frequency_hz must be > 0")
        _require(env.bridge_frequency_hz >= env.control_frequency_hz, "bridge_frequency_hz must be >= control")
        _require(env.prompt == "insert the plug into the power strip", "first version fixes the task prompt")
        _require(env.number_of_environments == 1, "one physical robot per controller is required")
        _require(env.auto_reset is False, (
            "environment.auto_reset must be False for the first correctness loop; terminal and "
            "reset observations must not be mixed in one transition"
        ))
        return env


@dataclasses.dataclass(frozen=True)
class FeatureModelConfig:
    runtime: str
    dataconfig: str
    model_action_dimension: int
    reference_horizon: int
    return_full_reference_horizon: bool
    denoising_steps: int
    precision: str
    token_dimension: int
    prefix_sequence_capacity: int
    frozen_in_stage2: bool
    single_prefix_pass: bool
    use_reconstruction_decoder_in_stage2: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FeatureModelConfig":
        d = _strict(
            "feature_model",
            data,
            {
                "runtime",
                "dataconfig",
                "model_action_dimension",
                "reference_horizon",
                "return_full_reference_horizon",
                "denoising_steps",
                "precision",
                "token_dimension",
                "prefix_sequence_capacity",
                "frozen_in_stage2",
                "single_prefix_pass",
                "use_reconstruction_decoder_in_stage2",
            },
        )
        cfg = cls(**d)
        _require(cfg.model_action_dimension == 32, "model_action_dimension must be 32")
        _require(cfg.reference_horizon > 0, "reference_horizon must be > 0")
        _require(cfg.return_full_reference_horizon, "return_full_reference_horizon must be True")
        _require(cfg.denoising_steps > 0, "denoising_steps must be > 0")
        _require(cfg.token_dimension > 0 and cfg.prefix_sequence_capacity > 0, "token/prefix dims must be > 0")
        _require(cfg.frozen_in_stage2, "the feature model must be frozen in Stage 2")
        _require(cfg.single_prefix_pass, "single_prefix_pass must be True")
        _require(not cfg.use_reconstruction_decoder_in_stage2, "the RLT decoder must not be in the Stage 2 path")
        return cfg


@dataclasses.dataclass(frozen=True)
class IntegrationConfig:
    submitted_chunk_length: int
    request_period_ticks: int
    fixed_delay_ticks: int
    delay_padding_capacity: int
    actor_reference_slice: str
    final_plan_composition: str
    next_rtc_source: str
    delay_policy: str
    late_response_policy: str
    replay_action_semantics: str
    reward_window: str
    actual_duration_discount: bool
    include_pending_queue_in_state: bool
    include_previous_plan_in_state: bool
    include_current_full_reference_in_critic: bool
    reference_dropout_applies_only_to_actor_reference_branch: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "IntegrationConfig":
        d = _strict("integration", data, {f.name for f in dataclasses.fields(cls)})
        cfg = cls(**d)
        _require(cfg.submitted_chunk_length > 0, "submitted_chunk_length must be > 0")
        _require(0 <= cfg.fixed_delay_ticks <= cfg.submitted_chunk_length, "D must satisfy 0 <= D <= C")
        _require(
            cfg.submitted_chunk_length == cfg.request_period_ticks,
            "submitted_chunk_length must equal request_period_ticks (guide section 12.2)",
        )
        _require(cfg.actor_reference_slice == "D:D+C", "actor_reference_slice must be exactly 'D:D+C'")
        _require(
            cfg.final_plan_composition == "pending_prefix || submitted_plan || reference_tail",
            "final plan composition must be P || U || reference_tail",
        )
        _require(
            cfg.next_rtc_source == "previous_final_composed_plan",
            "next RTC target must come from the previous final composed plan",
        )
        _require(cfg.delay_policy == "fixed_per_episode", "first version fixes D per episode")
        _require(
            cfg.late_response_policy in ("reject_and_safe_fallback_or_stop", "reject_and_stop"),
            "late responses must be rejected, never pasted at a random offset",
        )
        _require(
            cfg.replay_action_semantics == "accepted_submitted_plan",
            "the learner action label must be the accepted submitted plan U",
        )
        _require(
            cfg.reward_window == "capture_tick_to_next_capture_tick",
            "reward window is [r_i, r_{i+1}), not the U execution window",
        )
        _require(cfg.actual_duration_discount, "discounting must use the real duration K")
        _require(cfg.include_pending_queue_in_state, "the pending queue must be in the planning state")
        _require(cfg.include_previous_plan_in_state, "the previous plan tail must be in the planning state")
        _require(
            cfg.include_current_full_reference_in_critic,
            "the critic must consume the complete planning state including the current reference",
        )
        _require(
            cfg.reference_dropout_applies_only_to_actor_reference_branch,
            "reference dropout must apply only to the actor reference branch",
        )
        return cfg

    def contract(self) -> HorizonContract:
        raise ConfigError(
            "IntegrationConfig alone cannot validate H; use DesignConfig.horizon_contract()"
        )


@dataclasses.dataclass(frozen=True)
class RTCConfig:
    mode: str
    canonical_time_direction: str
    native_time_adapter: str
    prefix_weight_schedule: str
    guidance_weight_cap: float
    zero_guidance_on_padding_targets: bool
    input_gradients_required: bool
    model_parameter_gradients_enabled: bool
    allow_outer_torch_inference_mode: bool
    retain_graph_across_denoising_steps: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RTCConfig":
        d = _strict("rtc", data, {f.name for f in dataclasses.fields(cls)})
        cfg = cls(**d)
        _require(cfg.mode in ("none", "approx", "full_vjp"), "rtc.mode must be none/approx/full_vjp")
        _require(cfg.canonical_time_direction == "noise_to_data", "canonical time direction must be noise_to_data")
        _require(cfg.prefix_weight_schedule in ("linear", "exp", "ones", "zeros"), "unknown prefix weight schedule")
        _require(cfg.guidance_weight_cap > 0, "guidance_weight_cap must be > 0")
        _require(cfg.zero_guidance_on_padding_targets, "padding dimensions must never be guided")
        if cfg.mode == "full_vjp":
            _require(cfg.input_gradients_required, "full_vjp requires input_gradients_required=True")
            _require(
                not cfg.model_parameter_gradients_enabled,
                "full_vjp must not enable model parameter gradients in Stage 2",
            )
            _require(
                not cfg.allow_outer_torch_inference_mode,
                "full_vjp must not allow an outer torch.inference_mode context",
            )
            _require(
                not cfg.retain_graph_across_denoising_steps,
                "full_vjp must not retain the graph across denoising steps",
            )
        return cfg


@dataclasses.dataclass(frozen=True)
class LearningConfig:
    actor_kind: str
    critic_kind: str
    base_loss_family: str
    actor_precision: str
    critic_precision: str
    actor_learning_rate: float
    critic_learning_rate: float
    gamma_per_control_tick: float
    target_ema_tau: float
    entropy_weight: float
    fixed_exploration_std_before_tanh: float
    reference_dropout_probability: float
    require_bc_warmup_before_actor_takeover: bool
    bc_compares_deployed_bounded_actions: bool
    bc_q_weight_schedule: str
    total_updates: int | None
    replay_capacity: int | None
    save_replay_for_resume: bool
    filter_censored_transitions: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LearningConfig":
        d = _strict("learning", data, {f.name for f in dataclasses.fields(cls)})
        cfg = cls(**d)
        _require(cfg.actor_kind == "queue_aware_direct_chunk_mlp", "actor_kind must be the new registered queue-aware class")
        _require(cfg.critic_kind == "twin_q_augmented_planning_state", "critic_kind must be twin-Q over the augmented state")
        _require(cfg.base_loss_family == "rlt_ac", "base_loss_family must stay rlt_ac")
        _require(cfg.entropy_weight == 0.0, "entropy_weight must stay 0 (no tanh-jacobian-corrected log-prob)")
        _require(0.0 < cfg.gamma_per_control_tick <= 1.0, "gamma_per_control_tick must be in (0, 1]")
        _require(0.0 < cfg.target_ema_tau <= 1.0, "target_ema_tau must be in (0, 1]")
        _require(0.0 < cfg.actor_learning_rate and 0.0 < cfg.critic_learning_rate, "learning rates must be > 0")
        _require(0.0 <= cfg.reference_dropout_probability < 1.0, "reference_dropout_probability must be in [0, 1)")
        _require(cfg.require_bc_warmup_before_actor_takeover, "BC warmup must gate actor takeover")
        _require(cfg.bc_compares_deployed_bounded_actions, "BC must compare deployed bounded actions")
        _require(cfg.filter_censored_transitions, "censored transitions must be filtered from the learner")
        return cfg


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    clock_mode: str
    one_inflight_request_per_environment: bool
    controller_owns_tick_and_queue: bool
    require_acceptance_ack: bool
    atomic_policy_version_swap: bool
    torch_compile: bool
    cuda_graph: bool
    forward_only_acceleration_engine: bool
    measure_with_concurrent_learner: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RuntimeConfig":
        d = _strict("runtime", data, {f.name for f in dataclasses.fields(cls)})
        cfg = cls(**d)
        _require(cfg.clock_mode == "wall_clock", "physical deployment clock_mode must be wall_clock")
        _require(cfg.one_inflight_request_per_environment, "first version keeps one in-flight request per environment")
        _require(
            cfg.controller_owns_tick_and_queue,
            "the controller must own the tick and the queue; planner/learner must not step the env",
        )
        _require(cfg.require_acceptance_ack, "plans must require an acceptance ACK")
        _require(cfg.atomic_policy_version_swap, "policy version swaps must be atomic")
        _require(not cfg.torch_compile, "torch_compile starts disabled")
        _require(not cfg.cuda_graph, "cuda_graph starts disabled")
        _require(not cfg.forward_only_acceleration_engine, "forward-only accelerators are rejected for full_vjp")
        return cfg


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    deterministic_actor: bool
    expert_takeover: bool
    report_failure_trials: bool
    report_deadline_miss_rate: bool
    require_export_reload_test: bool
    results_prepopulated: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvaluationConfig":
        d = _strict("evaluation", data, {f.name for f in dataclasses.fields(cls)})
        cfg = cls(**d)
        _require(cfg.deterministic_actor, "evaluation must use the deterministic actor")
        _require(not cfg.expert_takeover, "main evaluation disables expert takeover")
        _require(cfg.report_failure_trials, "failure trials must be reported")
        _require(cfg.report_deadline_miss_rate, "deadline miss rate must be reported")
        _require(not cfg.results_prepopulated, "results must not be prepopulated")
        return cfg


@dataclasses.dataclass(frozen=True)
class DesignConfig:
    schema_version: str
    implementation_status: str
    source: SourceConfig
    assets: AssetsConfig
    environment: EnvironmentConfig
    feature_model: FeatureModelConfig
    integration: IntegrationConfig
    rtc: RTCConfig
    learning: LearningConfig
    runtime: RuntimeConfig
    evaluation: EvaluationConfig
    raw_path: str | None = None
    raw_sha256: str | None = None

    def __post_init__(self) -> None:
        _require(self.schema_version == DESIGN_SCHEMA_VERSION, f"schema_version must be {DESIGN_SCHEMA_VERSION}")
        # Cross-section constraints (guide section 16.3).
        contract = self.horizon_contract()
        _require(
            self.feature_model.reference_horizon >= contract.delay + contract.chunk,
            "feature reference horizon must be >= D + C",
        )
        _require(
            self.integration.delay_padding_capacity >= contract.delay,
            "delay_padding_capacity must be >= D",
        )
        _require(
            self.environment.number_of_environments == 1 or not self.runtime.one_inflight_request_per_environment,
            "one in-flight request per environment is enforced per environment",
        )

    def horizon_contract(self) -> HorizonContract:
        return HorizonContract(
            horizon=int(self.feature_model.reference_horizon),
            chunk=int(self.integration.submitted_chunk_length),
            delay=int(self.integration.fixed_delay_ticks),
        )

    def planning_state_spec(self) -> PlanningStateSpec:
        contract = self.horizon_contract()
        return PlanningStateSpec(
            z_dim=int(self.feature_model.token_dimension),
            proprio_dim=int(self.environment.proprio_dimension),
            env_action_dim=int(self.environment.action_dimension),
            horizon=contract.horizon,
            chunk=contract.chunk,
            max_delay=max(int(self.integration.delay_padding_capacity), contract.chunk),
        )

    def feature_profile_id(self) -> str:
        payload = {
            "feature_model": dataclasses.asdict(self.feature_model),
            "environment_cameras": list(self.environment.cameras),
            "prompt": self.environment.prompt,
            "source": dataclasses.asdict(self.source),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def to_resolved_dict(self) -> dict[str, Any]:
        resolved = {
            "schema_version": self.schema_version,
            "resolved_by": "pi05_rlt_rtc.config.design_config",
            "resolved_unix_seconds": time.time(),
            "python": sys.version,
            "platform": platform.platform(),
            "source_config_sha256": self.raw_sha256,
            "source_config_path": self.raw_path,
            "feature_profile_id": self.feature_profile_id(),
            "horizon_contract": dataclasses.asdict(self.horizon_contract()),
            "planning_state_layout_fingerprint": _layout_fingerprint(self.planning_state_spec()),
            "sections": {
                "source": dataclasses.asdict(self.source),
                "assets": dataclasses.asdict(self.assets),
                "environment": dataclasses.asdict(self.environment),
                "feature_model": dataclasses.asdict(self.feature_model),
                "integration": dataclasses.asdict(self.integration),
                "rtc": dataclasses.asdict(self.rtc),
                "learning": dataclasses.asdict(self.learning),
                "runtime": dataclasses.asdict(self.runtime),
                "evaluation": dataclasses.asdict(self.evaluation),
            },
        }
        return resolved


def _layout_fingerprint(spec: PlanningStateSpec) -> str:
    import hashlib as _h

    payload = {"actor": spec.actor_state_layout(), "critic": spec.critic_state_layout()}
    return _h.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def load_design_config(path: str | Path, *, require_assets_for: str | None = None) -> DesignConfig:
    """Load and fully validate a design config; unknown keys are errors."""
    import yaml

    path = Path(path)
    if not path.exists():
        raise ConfigError(f"design config not found: {path}")
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes)
    if not isinstance(raw, Mapping):
        raise ConfigError("design config must be a mapping")
    top_keys = {
        "schema_version",
        "implementation_status",
        "source",
        "assets",
        "environment",
        "feature_model",
        "integration",
        "rtc",
        "learning",
        "runtime",
        "evaluation",
    }
    unknown = sorted(set(raw) - top_keys)
    if unknown:
        raise ConfigError(f"unknown top-level config keys: {unknown}")
    missing = sorted(top_keys - set(raw))
    if missing:
        raise ConfigError(f"missing required config sections: {missing}")
    cfg = DesignConfig(
        schema_version=str(raw["schema_version"]),
        implementation_status=str(raw.get("implementation_status", "specification_only")),
        source=SourceConfig.from_mapping(raw["source"]),
        assets=AssetsConfig.from_mapping(raw["assets"]),
        environment=EnvironmentConfig.from_mapping(raw["environment"]),
        feature_model=FeatureModelConfig.from_mapping(raw["feature_model"]),
        integration=IntegrationConfig.from_mapping(raw["integration"]),
        rtc=RTCConfig.from_mapping(raw["rtc"]),
        learning=LearningConfig.from_mapping(raw["learning"]),
        runtime=RuntimeConfig.from_mapping(raw["runtime"]),
        evaluation=EvaluationConfig.from_mapping(raw["evaluation"]),
        raw_path=str(path),
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    if require_assets_for is not None:
        cfg.assets.require_for_stage(require_assets_for)
    return cfg
