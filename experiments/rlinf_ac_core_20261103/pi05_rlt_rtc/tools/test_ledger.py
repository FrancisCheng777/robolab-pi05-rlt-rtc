"""Generate the machine-readable test ledger from a real pytest execution.

Usage::

    python -m pi05_rlt_rtc.tools.test_ledger --out test_status.json \
        --log runs/cpu_tests/pytest.log

The tool runs pytest itself (with the conftest ledger plugin), so statuses
cannot be hand-edited.  CPU results and GPU/upstream results are kept in
separate ``environment`` fields per the implementation rules.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "docs" / "spec" / "test_status.template.json"

# Explicit execution-environment category per test group.  Never merge the
# four categories named in AGENTS.md (CPU mock, real-weight GPU, virtual delay,
# real wall clock).
_ENVIRONMENT = {
    "cpu_unit": "cpu_unit_no_upstream",
    "cpu_virtual_delay": "cpu_virtual_delay_mock_planner",
    "cpu_wallclock": "cpu_wallclock_mock_planner",
    "cpu_toy": "cpu_toy_mock_features",
}
_TEST_ENV = {
    "T05": "cpu_unit",
    "T06": "cpu_unit",
    "T08": "cpu_unit",
    "T09": "cpu_unit",
    "T14": "cpu_unit",
    "T19": "cpu_unit",
    "T20": "cpu_unit",
    "T44": "cpu_wallclock",
    "T65": "cpu_toy",
    "T66": "cpu_toy",
    "T67": "cpu_toy",
    "T68": "cpu_virtual_delay",
}
for _i in list(range(21, 33)):
    _TEST_ENV[f"T{_i:02d}"] = "cpu_unit"
for _i in list(range(33, 44)):
    _TEST_ENV[f"T{_i:02d}"] = "cpu_virtual_delay"
_TEST_ENV["T45"] = "cpu_virtual_delay"
_TEST_ENV["T46"] = "cpu_virtual_delay"
for _i in list(range(47, 62)):
    _TEST_ENV[f"T{_i:02d}"] = "cpu_virtual_delay"
_TEST_ENV["T60"] = "cpu_toy"
for _i in list(range(1, 21)):
    _TEST_ENV.setdefault(f"T{_i:02d}", "cpu_unit")
for _i in list(range(62, 69)):
    _TEST_ENV.setdefault(f"T{_i:02d}", "cpu_toy")

_EXPECTED_SKIP = {
    "T01": "real LeRobot dataset + local schema audit",
    "T02": "real two-camera dataset + RLinf image transforms",
    "T07": "real dataset + RLinf PandaOutputs delta semantics",
    "T10": "real RLinf train/inference proprio preprocessing",
    "T11": "real RLT encoder/decoder + VLA parameter groups",
    "T12": "real pi0.5 flow-matching loss",
    "T13": "real causal RLT reconstruction decoder",
    "T15": "real tokenizer prefix padding masks",
    "T16": "real Stage 1 run on real data",
    "T17": "real frozen pi0.5/RLT Stage 2 checksums",
    "T18": "real RLinf Stage 2 checkpoint loader",
    "T61": "pinned RLinf B0 baseline + real data/weights",
    "T62": "real pi0.5 + RLT + full_vjp with GPU weights",
    "T63": "real wall-clock B2 loop with the real planner",
    "T64": "real learner running concurrently with the planner",
}


@dataclasses.dataclass
class LedgerResult:
    id: str
    test: str
    expected: str
    status: str
    command: str | None
    environment: str | None
    log_path: str | None
    notes: str | None
    nodeid: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def run_pytest(log_path: Path, *, select: str | None = None, quick: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pytest", "tests", "-v", "--tb=short"]
    if select:
        cmd += ["-k", select]
    if quick:
        cmd += ["-x"]
    env = {"PI05_TEST_LEDGER": str(log_path.with_suffix(".ledger.json"))}
    import os

    full_env = dict(os.environ)
    full_env.update(env)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            cmd, cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT, env=full_env
        )
    print(f"pytest exit={completed.returncode}; log={log_path}")
    return completed.returncode


def build_status(
    template_path: Path,
    executed: dict[str, dict[str, Any]],
    *,
    command: str,
    log_path: str,
) -> dict[str, Any]:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    tests = template["tests"]
    results: list[LedgerResult] = []
    counts = {"PASSED": 0, "FAILED": 0, "ERROR": 0, "SKIPPED": 0, "NOT_RUN": 0}
    for entry in tests:
        t_id = entry["id"]
        record = executed.get(t_id)
        if record is None:
            status = "NOT_RUN"
            notes = "no executed test found for this id"
            environment = None
            nodeid = None
        else:
            status = record["status"]
            environment = _ENVIRONMENT.get(_TEST_ENV.get(t_id, ""), "cpu_unit_no_upstream")
            notes = record.get("reason") or ""
            nodeid = record.get("nodeid")
            if status == "SKIPPED":
                needed = _EXPECTED_SKIP.get(t_id)
                notes = (notes + f" | real-environment prerequisite: {needed}").strip(" |")
                environment = "not_executed"
            elif status != "PASSED":
                notes = notes or "see pytest log"
        if status in counts:
            counts[status] += 1
        results.append(
            LedgerResult(
                id=t_id,
                test=entry["test"],
                expected=entry["expected"],
                status=status,
                command=command if status in ("PASSED", "FAILED", "ERROR") else None,
                environment=environment,
                log_path=log_path if status in ("PASSED", "FAILED", "ERROR") else None,
                notes=notes,
                nodeid=nodeid,
            )
        )
    return {
        "kind": "executed_test_status",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": command,
        "log_path": log_path,
        "separation_of_evidence": {
            "cpu_unit_no_upstream": "CPU unit tests; no RLinf/GPU involved",
            "cpu_virtual_delay_mock_planner": "CPU virtual-delay protocol with mock planner/features",
            "cpu_wallclock_mock_planner": "CPU wall-clock timing with an explicitly labelled mock planner",
            "cpu_toy_mock_features": "CPU toy closed loop with mock features; not a robot result",
            "not_executed": "requires real data/weights/GPU/physical robot; no result claimed",
        },
        "counts": counts,
        "tests": [r.to_dict() for r in results],
    }


def write_markdown(status: dict[str, Any], path: Path) -> None:
    lines = [
        "# Executed test results",
        "",
        f"Generated (UTC): {status['generated_utc']}",
        f"Command: `{status['command']}`",
        f"Log: `{status['log_path']}`",
        "",
        "| ID | status | environment | notes |",
        "|---|---|---|---|",
    ]
    for test in status["tests"]:
        note = (test["notes"] or "").replace("|", "/").replace("\n", " ")[:120]
        lines.append(
            f"| {test['id']} | {test['status']} | {test['environment'] or '-'} | {note} |"
        )
    lines += [
        "",
        "PASSED rows were produced by a real CPU execution of this repository's tests.",
        "SKIPPED rows require the real dataset/weights/GPU/physical robot and claim nothing.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate test_status.json from a real run")
    parser.add_argument("--out", default=str(REPO_ROOT / "test_status.json"))
    parser.add_argument(
        "--log", default=str(REPO_ROOT / "runs" / "cpu_tests" / "pytest.log")
    )
    parser.add_argument("--select", default=None, help="pytest -k expression")
    parser.add_argument(
        "--reuse-ledger",
        default=None,
        help="skip running pytest and reuse this ledger JSON",
    )
    args = parser.parse_args(argv)

    log_path = Path(args.log)
    exit_code = 0
    if args.reuse_ledger:
        ledger_path = Path(args.reuse_ledger)
    else:
        exit_code = run_pytest(log_path, select=args.select)
        ledger_path = log_path.with_suffix(".ledger.json")
        if not ledger_path.exists():
            print("error: pytest did not produce a ledger; see log", file=sys.stderr)
            return exit_code or 1
    executed = json.loads(ledger_path.read_text(encoding="utf-8"))["results"]
    status = build_status(
        TEMPLATE,
        executed,
        command=f"PI05_TEST_LEDGER=... python -m pytest tests -v" + (f" -k {args.select}" if args.select else ""),
        log_path=str(log_path.relative_to(REPO_ROOT) if log_path.is_relative_to(REPO_ROOT) else log_path),
    )
    Path(args.out).write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(status, REPO_ROOT / "tests" / "RESULTS.md")
    print(json.dumps(status["counts"], indent=2))
    return exit_code or (0 if status["counts"]["FAILED"] == 0 and status["counts"]["ERROR"] == 0 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
