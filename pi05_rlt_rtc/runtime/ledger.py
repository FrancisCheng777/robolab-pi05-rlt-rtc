"""Per-tick execution ledger.

The ledger records every command actually handed to the controller, together
with the plan it came from and whether that plan was committed or speculative.
It is the audit source for the timing table in guide section 12.3 and must
never block a control tick: writing is explicit and asynchronous on the caller
side.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LEDGER_KINDS = ("committed", "safety", "hold_last", "zero_delta", "stop", "intervened")


@dataclasses.dataclass
class LedgerEntry:
    tick: int
    env_id: str
    episode_id: str
    episode_generation: int
    queue_generation: int
    command: np.ndarray | None
    kind: str
    source_plan_id: int | None
    request_id: int | None
    policy_version_id: str | None
    note: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "env_id": self.env_id,
            "episode_id": self.episode_id,
            "episode_generation": self.episode_generation,
            "queue_generation": self.queue_generation,
            "command": None if self.command is None else np.asarray(self.command, dtype=float).tolist(),
            "kind": self.kind,
            "source_plan_id": self.source_plan_id,
            "request_id": self.request_id,
            "policy_version_id": self.policy_version_id,
            "note": self.note,
        }


class ExecutionLedger:
    def __init__(self, *, max_entries: int | None = None) -> None:
        self.entries: list[LedgerEntry] = []
        self.events: list[dict[str, Any]] = []
        self._max_entries = max_entries

    def append(self, entry: LedgerEntry) -> None:
        if entry.kind not in LEDGER_KINDS:
            raise ValueError(f"unknown ledger kind {entry.kind!r}")
        self.entries.append(entry)
        if self._max_entries is not None and len(self.entries) > self._max_entries:
            # Keep the execution ledger bounded without silently dropping the
            # important head of the trace; count the overflow explicitly.
            overflow = len(self.entries) - self._max_entries
            self.events.append({"event": "ledger_trimmed", "dropped": overflow})
            del self.entries[:overflow]

    def event(self, name: str, **fields: Any) -> None:
        self.events.append({"event": name, **fields})

    def commands_between(self, start_tick: int, end_tick: int) -> list[np.ndarray | None]:
        return [
            e.command
            for e in self.entries
            if start_tick <= e.tick < end_tick
        ]

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.kind] = counts.get(entry.kind, 0) + 1
        for event in self.events:
            key = f"event:{event['event']}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_records(self) -> list[dict[str, Any]]:
        return [e.to_record() for e in self.entries]

    def write_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.to_records():
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        return path

    def to_jsonl_lines(self) -> Iterable[str]:
        for record in self.to_records():
            yield json.dumps(record, sort_keys=True)
