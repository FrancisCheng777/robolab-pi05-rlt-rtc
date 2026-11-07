"""Static source audit for the pinned RLinf checkout (P0 deliverable).

This tool reads the *actual* checkout and records, for each已知 integration
breakpoint in the guide's section 2.2, whether the pinned source still matches
the audited behaviour.  It is source-reading evidence, not a runtime test.

Usage::

    python -m pi05_rlt_rtc.adapters.source_audit --rlinf /path/to/RLinf \
        --out runs/p0_audit/rlinf_source_audit.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PINNED_RLINF_COMMIT = "c70606f08cdca259b8dec03d4430926b5b8fac9d"
PINNED_OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
PINNED_RTC_COMMIT = "9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b"


@dataclasses.dataclass
class Check:
    id: str
    description: str
    path: str
    pattern: str
    status: str = "NOT_RUN"
    evidence: list[str] = dataclasses.field(default_factory=list)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


CHECKS: list[Check] = [
    Check(
        "C03.extract_rlt_obs_plain_euler",
        "Pi0Eval.extract_rlt_obs samples actions via the plain Euler helper (no RTC context)",
        "rlinf/models/embodiment/openpi/tasks/eval.py",
        r"def extract_rlt_obs\(|_sample_actions_from_prefix_cache\(",
    ),
    Check(
        "C03.rtc_only_approx",
        "RTC branch rejects any guidance mode other than approx",
        "rlinf/models/embodiment/openpi/tasks/eval.py",
        r"Unsupported RTC guidance mode",
    ),
    Check(
        "C03.full_horizon_decode_truncated",
        "EnvIO.decode_actions applies output_transform which can truncate via action_chunk",
        "rlinf/models/embodiment/openpi/env_io.py",
        r"def decode_actions\(|output_transform\(",
    ),
    Check(
        "C06.output_transform_action_chunk_truncation",
        "apply_output_transform slices actions to action_chunk when provided",
        "rlinf/models/embodiment/openpi/transforms/apply.py",
        r"recombined\[\"actions\"\] = recombined\[\"actions\"\]\[:, :action_chunk\]",
    ),
    Check(
        "C04.approx_only",
        "RTC sampler exposes only the approximate guidance implementation",
        "rlinf/models/embodiment/openpi/sampling/rtc_guidance.py",
        r"def sample_actions_with_rtc_guidance\(|build_rtc_target_and_mask\(",
    ),
    Check(
        "C08.ref_chunk_first_C",
        "RLTMLPPolicy._get_ref_chunk slices the first C steps",
        "rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py",
        r"\[:, : self\.chunk_len\]",
    ),
    Check(
        "C08.sft_forward_not_tanh",
        "RLTMLPPolicy.sft_forward compares pre-tanh predictions",
        "rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py",
        r"return F\.mse_loss\(pred_actions, target_actions",
    ),
    Check(
        "C09.learner_ref_chunk_first_C",
        "RLT learner _ref_chunk slices the first C steps",
        "rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py",
        r"return ref_chunk\[:, :chunk_len\]\.reshape",
    ),
    Check(
        "C09.critic_discount_uses_tensor_width",
        "Learner discount uses the padded reward tensor width, not a real duration K",
        "rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py",
        r"chunk_len = rewards\.shape\[-1\]|_discounted_chunk_rewards",
    ),
    Check(
        "C09.rlt_ac_loss_recipe",
        "Actor objective is -q_weight*Q1 + bc_weight*BC with fixed entropy alpha",
        "rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py",
        r"actor_loss = -q_weight \* qf_pi\.mean\(\) \+ bc_weight \* bc_loss",
    ),
    Check(
        "C02.rlt_alpha_multiplies_vla_loss",
        "rlt_alpha multiplies the VLA loss (rlt_loss + alpha * vla_loss)",
        "rlinf/models/embodiment/openpi/pi0.py",
        r"rlt_loss \+ self\.rlt_cfg\.rlt_alpha \* vla_loss",
    ),
    Check(
        "C11.checkpoint_rlt_completeness",
        "checkpoint loader requires complete rlt_module.* weights for Stage 2",
        "rlinf/models/embodiment/openpi/checkpoint.py",
        r"rlt_module\.\*|expect_rlt",
    ),
    Check(
        "C15.rlt_ac_no_training_pipeline",
        "rlt_ac branch rejects runner.use_training_pipeline=True",
        "examples/embodiment/train_embodied_agent.py",
        r"not supported for rlt_ac",
    ),
]


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None
    return None


def run_checks(rlinf: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "kind": "static_source_audit_not_runtime_test",
        "rlinf_path": str(rlinf),
        "rlinf_commit": _git_commit(rlinf),
        "expected_rlinf_commit": PINNED_RLINF_COMMIT,
        "commit_matches": _git_commit(rlinf) == PINNED_RLINF_COMMIT,
        "pinned_references": {
            "openpi": PINNED_OPENPI_COMMIT,
            "real_time_chunking_kinetix": PINNED_RTC_COMMIT,
        },
        "checks": [],
    }
    for check in CHECKS:
        entry = check.to_dict()
        file_path = rlinf / check.path
        if not file_path.exists():
            entry["status"] = "FILE_MISSING"
            report["checks"].append(entry)
            continue
        text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        hits = [
            f"L{i + 1}: {line.strip()}"
            for i, line in enumerate(lines)
            if re.search(check.pattern, line)
        ]
        if hits:
            entry["status"] = "CONFIRMED"
            entry["evidence"] = hits[:5]
        else:
            entry["status"] = "NOT_FOUND"
            entry["evidence"] = []
        report["checks"].append(entry)
    confirmed = sum(1 for c in report["checks"] if c["status"] == "CONFIRMED")
    report["summary"] = {
        "checks": len(report["checks"]),
        "confirmed": confirmed,
        "not_found": len(report["checks"]) - confirmed,
    }
    return report


def runtime_probe() -> dict[str, Any]:
    import importlib.metadata as md

    packages = ["torch", "numpy", "pyyaml", "pytest", "lerobot", "openpi"]
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "cuda_available": _cuda_available(),
        "git_available": shutil.which("git") is not None,
    }


def _cuda_available() -> bool | None:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit-rlinf-source",
        description="Static audit of a pinned RLinf checkout against the recorded breakpoints.",
    )
    parser.add_argument("--rlinf", type=Path, required=True, help="Path to the RLinf checkout")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    parser.add_argument(
        "--environment", type=Path, default=None, help="Optional environment_versions.json path"
    )
    args = parser.parse_args(argv)
    if not (args.rlinf / "rlinf").exists():
        print(f"error: {args.rlinf} does not look like an RLinf checkout", file=sys.stderr)
        return 2
    report = run_checks(args.rlinf)
    report["runtime_probe"] = runtime_probe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    if not report["commit_matches"]:
        print(
            f"WARNING: checkout commit {report['rlinf_commit']} != pinned "
            f"{PINNED_RLINF_COMMIT}; the audit is only valid for the pinned snapshot.",
            file=sys.stderr,
        )
    if args.environment is not None:
        args.environment.write_text(
            json.dumps(report["runtime_probe"], indent=2), encoding="utf-8"
        )
    return 0 if report["summary"]["not_found"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
