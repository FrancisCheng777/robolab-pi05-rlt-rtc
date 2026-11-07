"""Shared test fixtures and the machine-readable test-ledger plugin.

Set ``PI05_TEST_LEDGER=/path/to/ledger.json`` to have pytest record the status
of every ``T##`` test from ``TEST_MATRIX.md`` together with its node id and a
reason for skips.  The ledger is generated only from real executions.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

T_ID_RE = re.compile(r"[tT](\d{2})")
_RESULTS: dict[str, dict] = {}


def pytest_runtest_makereport(item, call):  # noqa: ANN001
    if call.when == "setup" and call.excinfo is not None:
        _record(item, "SKIPPED" if call.excinfo.errisinstance(pytest.skip.Exception) else "FAILED", call)
    elif call.when == "call":
        outcome = "PASSED"
        if call.excinfo is not None:
            if call.excinfo.errisinstance(pytest.skip.Exception):
                outcome = "SKIPPED"
            elif call.excinfo.errisinstance(AssertionError):
                outcome = "FAILED"
            else:
                outcome = "ERROR"
        _record(item, outcome, call)


def _record(item, outcome: str, call) -> None:  # noqa: ANN001
    match = T_ID_RE.search(item.name)
    if not match:
        return
    t_id = f"T{match.group(1)}"
    reason = None
    if outcome in ("SKIPPED", "FAILED", "ERROR") and call.excinfo is not None:
        reason = f"{call.excinfo.type.__name__}: {call.excinfo.value}"
    _RESULTS[t_id] = {
        "id": t_id,
        "status": outcome,
        "nodeid": item.nodeid,
        "reason": reason,
    }


def pytest_sessionfinish(session, exitstatus):  # noqa: ANN001
    path = os.environ.get("PI05_TEST_LEDGER")
    if not path:
        return
    payload = {
        "kind": "executed_test_ledger",
        "results": dict(sorted(_RESULTS.items())),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# fixtures


@pytest.fixture
def contract():
    from pi05_rlt_rtc.contracts.spaces import HorizonContract

    return HorizonContract(horizon=10, chunk=5, delay=2)


@pytest.fixture
def codec():
    from pi05_rlt_rtc.action_codec.codec import ActionCodec, NormStats

    return ActionCodec(NormStats.symmetric(), name="test_codec")


@pytest.fixture
def toy_build():
    from pi05_rlt_rtc.evaluation.toy_pipeline import build_toy

    return build_toy(horizon=10, chunk=5, delay=2, z_dim=24, seed=0)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
