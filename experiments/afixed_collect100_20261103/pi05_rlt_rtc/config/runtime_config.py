"""Deployment configuration. Asset paths may remain null until provisioning."""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

from pi05_rlt_rtc.contracts.errors import ConfigError, MissingAssetError
from pi05_rlt_rtc.contracts.spaces import HorizonContract
from pi05_rlt_rtc.learning.defaults import stage2_settings


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    path: Path
    raw: dict[str, Any]
    platform: str
    contract: HorizonContract
    control_hz: int
    rtc_mode: str
    num_steps: int

    def asset(self, value: Any, label: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise MissingAssetError(f"configure {label} in {self.path}")
        p = Path(value).expanduser()
        p = p if p.is_absolute() else self.path.parent / p
        if not p.exists():
            raise MissingAssetError(f"{label} does not exist: {p}")
        return p.resolve()

    def require_model_assets(self) -> tuple[Path, Path, Path]:
        model = self.raw["model"]
        assets = tuple(self.asset(model.get(k), f"model.{k}")
                       for k in ("rlinf_root", "checkpoint", "norm_stats"))
        if not isinstance(model.get("dataset_id"), str) or not model["dataset_id"].strip():
            raise ConfigError("model.dataset_id must identify the Stage-1 training data")
        return assets

    @property
    def action_contract_id(self) -> str:
        return self.raw["action_contract_id"]

    def wire_contract(self) -> dict:
        result = {"horizon": dataclasses.asdict(self.contract), "control_hz": self.control_hz,
                "rtc_mode": self.rtc_mode, "num_steps": self.num_steps,
                "action_contract_id": self.action_contract_id, "prompt": self.raw["prompt"],
                "camera_profile": "d405_wrist_to_base_v1",
                "reward_scheme": self.raw.get("training", {}).get("reward_scheme", "rlt_binary")}
        if self.raw.get('franka', {}).get('joint_command_mode') == 'bounded_reference':
            result['joint_execution'] = {'mode': 'bounded_reference',
                'reference_lead_rad': self.raw['franka'].get('reference_lead_rad', .05)}
        return result


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    import yaml
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise MissingAssetError(f"runtime config not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError("runtime config must be a mapping")
    allowed = {"version", "platform", "horizon", "control_hz", "rtc", "model", "policy",
               "franka", "action_contract_id", "prompt", "training", "session"}
    if set(raw) - allowed:
        raise ConfigError(f"unknown runtime keys: {sorted(set(raw) - allowed)}")
    if raw.get("version") != 1 or raw.get("platform") != "franka":
        raise ConfigError("version=1 and platform=franka are required")
    try:
        h = raw["horizon"]
        contract = HorizonContract(horizon=h["horizon"], chunk=h["chunk"], delay=h["delay"])
        hz = raw["control_hz"]
        steps = raw["rtc"]["num_steps"]
        mode = raw["rtc"]["mode"]
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"missing or malformed runtime field: {exc}") from exc
    if type(hz) is not int or hz <= 0 or type(steps) is not int or steps <= 0:
        raise ConfigError("control_hz and rtc.num_steps must be positive integers")
    if mode not in ("none", "approx", "full_vjp"):
        raise ConfigError("rtc.mode must be none, approx or full_vjp")
    if raw["platform"] == "franka" and contract.delay == 0:
        raise ConfigError("Franka wall-clock control requires delay > 0")
    if not isinstance(raw.get(raw["platform"]), dict):
        raise ConfigError(f"{raw['platform']} settings must be a mapping")
    if not isinstance(raw.get("prompt"), str) or not raw["prompt"].strip():
        raise ConfigError("prompt is required")
    if not isinstance(raw.get("action_contract_id"), str) or not raw["action_contract_id"].strip():
        raise ConfigError("action_contract_id is required")
    if not isinstance(raw.get("model"), dict) or raw["model"].get("config_name") != "pi05_rlt_panda_robotiq_joint":
        raise ConfigError("this runtime requires the 9-state/8-action joint profile pi05_rlt_panda_robotiq_joint")
    from pi05_rlt_rtc.contracts.physical import ACTION_CONTRACT_ID, CONTROL_HZ
    if raw["action_contract_id"] != ACTION_CONTRACT_ID or hz != CONTROL_HZ:
        raise ConfigError("runtime must match the 15 Hz Panda/Robotiq demonstration contract")
    z_dim = raw["model"].get("z_dim", 2048)
    if type(z_dim) is not int or z_dim <= 0:
        raise ConfigError("model.z_dim must be a positive integer")
    for key in ("compile_kernels", "compile_suffix", "vision_cuda_graph"):
        if type(raw["model"].get(key, False)) is not bool:
            raise ConfigError(f"model.{key} must be boolean")
    if raw.get("policy", {}).get("kind") not in ("teacher", "actor"):
        raise ConfigError("policy.kind must be teacher or actor")
    joint_mode = raw.get('franka', {}).get('joint_command_mode', 'measured_delta')
    if joint_mode not in ('measured_delta', 'bounded_reference'):
        raise ConfigError('unknown franka.joint_command_mode')
    lead = raw.get('franka', {}).get('reference_lead_rad', .05)
    if isinstance(lead, bool) or not isinstance(lead, (int, float)) or not math.isfinite(lead) or not 0 < lead <= .05:
        raise ConfigError('franka.reference_lead_rad must be in (0, .05]')
    if type(raw['policy'].get('exploration', False)) is not bool:
        raise ConfigError('policy.exploration must be boolean')
    if raw.get('training', {}).get('reward_scheme', 'rlt_binary') not in ('rlt_binary', 'rlinf_signed'):
        raise ConfigError('unknown reward_scheme')
    noise = raw.get('training', {}).get('exploration_std', .002)
    if not isinstance(noise, (int,float)) or not math.isfinite(noise) or noise < 0:
        raise ConfigError('exploration_std must be finite and nonnegative')
    if 'session' in raw:
        if type(raw['session'].get('exploration', True)) is not bool:
            raise ConfigError('session.exploration must be boolean')
        every = raw['session'].get('update_every_episodes', 1)
        if type(every) is not int or every < 1:
            raise ConfigError('session.update_every_episodes must be a positive integer')
        if 'updates_per_batch' in raw['session']:
            updates = raw['session']['updates_per_batch']
            if type(updates) is not int or updates < 1:
                raise ConfigError('session.updates_per_batch must be a positive integer')
        for key in ('warmup_episodes','min_replay_transitions','warmup_updates','updates_per_episode',
                    'max_approach_ticks','max_critical_ticks'):
            if type(raw['session'].get(key)) is not int or raw['session'][key] <= 0:
                raise ConfigError(f'session.{key} must be a positive integer')
    stage2_settings(raw.get("training", {}))
    gamma = raw.get("training", {}).get("gamma_per_tick", .99)
    if not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ConfigError("training.gamma_per_tick must be in (0,1]")
    return RuntimeConfig(path, raw, raw["platform"], contract, hz, mode, steps)
