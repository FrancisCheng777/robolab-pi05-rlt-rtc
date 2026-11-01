"""Language/task prompt resolution.

The first version fixes one task prompt.  A dataset row without a prompt uses
the configured default; an *unknown* prompt is an error rather than silently
being replaced (guide section 5.2, test T04).
"""

from __future__ import annotations

from typing import Any, Iterable

from pi05_rlt_rtc.contracts.errors import ContractError

DEFAULT_PROMPT = "insert the plug into the power strip"


def resolve_prompt(task_value: Any | None, *, configured: str = DEFAULT_PROMPT) -> str:
    """Return the prompt for a row, defaulting only when the field is absent."""
    if task_value is None:
        return configured
    text = str(task_value).strip()
    if not text:
        return configured
    if text != configured:
        raise ContractError(
            f"unknown task prompt {text!r}; the first version is scoped to {configured!r} "
            "and refuses to silently relabel the data"
        )
    return text


def resolve_prompts(values: Iterable[Any], *, configured: str = DEFAULT_PROMPT) -> list[str]:
    return [resolve_prompt(v, configured=configured) for v in values]
