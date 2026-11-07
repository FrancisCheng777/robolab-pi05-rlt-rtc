"""Run-directory contract and mock/formal separation (guide section 17.6)."""

from __future__ import annotations

import dataclasses
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pi05_rlt_rtc import SCHEMA_VERSION
from pi05_rlt_rtc.contracts.errors import ContractError, MockPolicyError


@dataclasses.dataclass
class RunContext:
    run_id: str
    root: Path
    kind: str  # "formal" | "toy" | "audit"
    mock: bool = False

    @classmethod
    def create(
        cls,
        base_dir: str | Path,
        *,
        run_id: str | None = None,
        kind: str = "formal",
        mock: bool = False,
    ) -> "RunContext":
        if kind == "formal" and mock:
            raise MockPolicyError(
                "refusing to create a formal run directory for a mock artefact; mock runs "
                "must use kind='toy' so results cannot be mistaken for real evidence"
            )
        run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        root = Path(base_dir) / run_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "execution_ledger").mkdir(exist_ok=True)
        (root / "transition_audit").mkdir(exist_ok=True)
        (root / "checkpoints").mkdir(exist_ok=True)
        (root / "evaluation").mkdir(exist_ok=True)
        (root / "failures").mkdir(exist_ok=True)
        ctx = cls(run_id=run_id, root=root, kind=kind, mock=mock)
        ctx.write_environment_versions()
        ctx.write_source_manifest()
        return ctx

    def write_json(self, name: str, payload: Any) -> Path:
        if self.kind == "formal" and self.mock:
            raise MockPolicyError("mock payload cannot be written into a formal run")
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_environment_versions(self) -> Path:
        import importlib.metadata as md

        versions: dict[str, Any] = {
            "python": sys.version,
            "platform": platform.platform(),
            "schema_version": SCHEMA_VERSION,
        }
        for package in ("torch", "numpy", "pyyaml", "pytest", "lerobot", "openpi"):
            try:
                versions[package] = md.version(package)
            except md.PackageNotFoundError:
                versions[package] = None
        try:
            import torch

            versions["cuda_available"] = bool(torch.cuda.is_available())
            versions["cuda_version"] = getattr(torch.version, "cuda", None)
        except Exception:  # noqa: BLE001
            versions["cuda_available"] = None
        return self.write_json("environment_versions.json", versions)

    def write_source_manifest(self) -> Path:
        manifest = {
            "kind": self.kind,
            "mock": self.mock,
            "created_unix_seconds": time.time(),
            "schema_version": SCHEMA_VERSION,
            "git": _git_describe(Path.cwd()),
            "argv": sys.argv,
        }
        return self.write_json("source_manifest.json", manifest)

    def require_formal(self) -> None:
        if self.kind != "formal":
            raise MockPolicyError(f"this step requires a formal run context, got {self.kind!r}")
        if self.mock:
            raise MockPolicyError("formal run marked as mock")


def _git_describe(cwd: Path) -> dict[str, Any]:
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return {"commit": out.stdout.strip()}
    except Exception:  # noqa: BLE001
        pass
    return {"commit": None}
