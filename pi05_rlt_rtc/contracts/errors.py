"""Typed errors used across the package.

Every failure path in this project must fail loudly with a reason.  Silent
fixups (``nan_to_num``, ``clip`` as "repair", ``strict=False``, automatic
fallback from ``full_vjp`` to ``approx``) are forbidden by the implementation
rules in ``AGENTS.md``.
"""

from __future__ import annotations


class ContractError(ValueError):
    """A schema/shape/time contract was violated."""


class ConfigError(ContractError):
    """A design config is invalid (unknown fields, illegal dimensions, bad mapping)."""


class PlanValidationError(ContractError):
    """A submitted plan is malformed (shape, finiteness, ticks, generation)."""


class SchemaMismatchError(ContractError):
    """A consumer was handed data produced under a different schema version."""


class DeadlineMiss(RuntimeError):
    """A planning response arrived after its fixed ``apply_start_tick``."""


class LateResponse(DeadlineMiss):
    """Convenience alias for a deadline miss detected at commit time."""


class DuplicateResponse(RuntimeError):
    """A request id was already accepted (idempotent ACK is required)."""


class StaleEpisodeError(RuntimeError):
    """A response belongs to an old episode/generation or an out-of-order request."""


class GenerationMismatchError(StaleEpisodeError):
    """Queue generation or policy version mismatch."""


class MissingAssetError(FileNotFoundError):
    """A required checkpoint/dataset/norm-stats asset is missing or malformed."""


class EnvironmentNotReadyError(RuntimeError):
    """The pinned RLinf/openpi/GPU runtime is not available for a real run."""


class InferenceModeError(RuntimeError):
    """``full_vjp`` was invoked under ``torch.inference_mode()``.

    ``torch.enable_grad()`` cannot undo an outer inference-mode context, so the
    noisy-action VJP would silently be wrong.  Refuse instead of degrading.
    """


class FrozenParameterGradientError(RuntimeError):
    """A frozen model received parameter gradients in Stage 2."""


class MockPolicyError(RuntimeError):
    """A mock/toy artifact was about to be used outside an explicitly labelled mode."""


class ControlSafetyError(RuntimeError):
    """A safety check rejected the plan or the controller entered safe fallback."""
