"""Shared Stage-2 hyperparameters from the pinned RLinf realworld RLT example.

Source: RLinf c70606f08cdca259b8dec03d4430926b5b8fac9d,
examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml.
These align learning scalars, not the local queue-aware architecture or scheduler.
"""
import math

from pi05_rlt_rtc.contracts.errors import ConfigError


RLINF_STAGE2_DEFAULTS = {
    "batch_size": 256,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "lambda_q": .1,
    "lambda_bc": 5.,
    "reference_dropout": .5,
    "polyak_tau": .005,
    "exploration_std": .002,
    "actor_clip_grad": 10.,
    "critic_clip_grad": 10.,
}


def stage2_settings(overrides):
    settings = {**RLINF_STAGE2_DEFAULTS, **overrides}
    for key in ("batch_size",):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ConfigError(f"training.{key} must be a positive integer")
    for key in ("actor_lr", "critic_lr", "lambda_q", "lambda_bc",
                "reference_dropout", "polyak_tau", "exploration_std",
                "actor_clip_grad", "critic_clip_grad"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConfigError(f"training.{key} must be finite")
        if value < 0:
            raise ConfigError(f"training.{key} must be nonnegative")
    for key in ("actor_lr", "critic_lr", "actor_clip_grad", "critic_clip_grad"):
        if settings[key] == 0:
            raise ConfigError(f"training.{key} must be positive")
    for key in ("polyak_tau",):
        if not 0 < settings[key] <= 1:
            raise ConfigError(f"training.{key} must be in (0,1]")
    if not 0 <= settings["reference_dropout"] <= 1:
        raise ConfigError("training.reference_dropout must be in [0,1]")
    return settings
