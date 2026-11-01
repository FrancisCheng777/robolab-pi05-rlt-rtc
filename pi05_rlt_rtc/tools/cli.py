"""Command-line entry points (guide section 17.5).

Every command has ``--help`` and returns a meaningful exit code:

- ``0`` success; ``1`` a check failed; ``2`` the real runtime is unavailable;
- ``3`` a required asset is missing; ``4`` a contract/config error;
- ``5`` a mock artefact was about to be used in a formal context.

Real-model commands fail with exit code 2 when the pinned RLinf/GPU runtime is
absent; no command fabricates success or silently substitutes ``--toy``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from pi05_rlt_rtc import SCHEMA_VERSION
from pi05_rlt_rtc.contracts.errors import (
    ConfigError,
    ContractError,
    EnvironmentNotReadyError,
    MissingAssetError,
    MockPolicyError,
    SchemaMismatchError,
)
from pi05_rlt_rtc.contracts.plan import PlanRequest, PrevPlanContext
from pi05_rlt_rtc.tools.run_context import RunContext

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# helpers


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _build_toy(args) -> Any:
    from pi05_rlt_rtc.evaluation.toy_pipeline import build_toy

    return build_toy(
        horizon=args.horizon,
        chunk=args.chunk,
        delay=args.delay,
        z_dim=args.z_dim,
        rtc_mode=args.rtc_mode,
        gamma=args.gamma,
        seed=args.seed,
    )


def _toy_request(build, request_id: int = 0, prev: PrevPlanContext | None = None) -> PlanRequest:
    contract = build.contract
    return PlanRequest(
        request_id=request_id,
        env_id="toy0",
        episode_id="toy0-ep1",
        episode_generation=1,
        capture_tick=request_id * contract.chunk,
        apply_start_tick=request_id * contract.chunk + contract.delay,
        queue_generation=1,
        policy_version_id="toy-untrained-v1",
        feature_version_id="MOCK-feature-v1",
        schema_version=SCHEMA_VERSION,
        planning_state_version="queue-planning-state.v1",
        rtc_context=prev or PrevPlanContext.empty(),
        rng_seed=1234 + request_id,
        sampler_mode="none" if request_id < 0 else build.backend.mode,
        num_denoising_steps=5,
    )


# ---------------------------------------------------------------------------
# commands


def cmd_annotate_rollout(args):
    from pi05_rlt_rtc.deployment.rollout import annotate
    return annotate(args)


def cmd_audit_source(args: argparse.Namespace) -> int:
    from pi05_rlt_rtc.adapters import source_audit

    rlinf = Path(args.rlinf)
    if not (rlinf / "rlinf").exists():
        raise MissingAssetError(f"{rlinf} is not an RLinf checkout")
    report = source_audit.run_checks(rlinf)
    report["runtime_probe"] = source_audit.runtime_probe()
    _write_json(Path(args.out), report)
    print(json.dumps(report["summary"], indent=2))
    if not report["commit_matches"]:
        print(
            f"warning: checkout commit {report['rlinf_commit']} != pinned "
            f"{source_audit.PINNED_RLINF_COMMIT}",
            file=sys.stderr,
        )
        return 1
    return 0 if report["summary"]["not_found"] == 0 else 1


def cmd_config_validate(args: argparse.Namespace) -> int:
    from pi05_rlt_rtc.config.design_config import load_design_config

    cfg = load_design_config(args.config, require_assets_for=args.require_assets)
    resolved = cfg.to_resolved_dict()
    if args.out:
        _write_json(Path(args.out), resolved)
    print(f"config OK: {args.config}")
    print(json.dumps({k: resolved[k] for k in ("feature_profile_id", "horizon_contract")}, indent=2))
    return 0


def cmd_inspect_data(args: argparse.Namespace) -> int:
    from pi05_rlt_rtc.data.auditor import DatasetAuditor

    report = DatasetAuditor(args.root, horizon=args.horizon).run()
    out = Path(args.out) if args.out else Path(args.root) / "data_audit.json"
    report.write(out)
    print(report.summary_text())
    return 0 if report.complete else 1


def cmd_inspect_checkpoint(args: argparse.Namespace) -> int:
    import torch

    path = Path(args.path)
    if not path.exists():
        raise MissingAssetError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise MissingAssetError(
            f"cannot load {path} with torch.load(weights_only=True): {exc}. Unknown or "
            "object-containing checkpoints are not treated as safe data."
        ) from exc
    if not isinstance(payload, dict):
        raise ContractError(f"checkpoint payload must be a mapping, got {type(payload).__name__}")
    state = payload.get("actor_state", payload)
    keys = list(state.keys()) if isinstance(state, dict) else []
    n_params = 0
    shapes: dict[str, list[int]] = {}
    for key, value in state.items() if isinstance(state, dict) else []:
        if hasattr(value, "numel"):
            n_params += int(value.numel())
            shapes[key] = list(value.shape)
    report = {
        "path": str(path),
        "top_level_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
        "num_tensors": len(keys),
        "num_parameters": n_params,
        "has_rlt_module": any(k.startswith("rlt_module.") for k in keys),
        "has_vision": any(("vision" in k) or ("img" in k) for k in keys),
        "has_action_expert": any("action" in k for k in keys),
        "key_shapes_sample": dict(list(shapes.items())[:20]),
    }
    failures: list[str] = []
    if args.expect_rlt and not report["has_rlt_module"]:
        failures.append("expect_rlt=True but no rlt_module.* keys found")
    if args.expect_backbone:
        if not report["has_vision"] or not report["has_action_expert"]:
            failures.append("backbone vision/action-expert keys incomplete")
    report["failures"] = failures
    if args.out:
        _write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


def cmd_smoke_stage1(args: argparse.Namespace) -> int:
    if not args.toy:
        import os
        import subprocess
        env = {**os.environ, "PI05_TRAIN_PYTHON": sys.executable}
        result = subprocess.run(["bash", str(REPO_ROOT / "scripts/start_stage1_integrated.sh"),
                                 "smoke", *getattr(args, "overrides", [])], cwd=REPO_ROOT, env=env)
        return result.returncode
    from pi05_rlt_rtc.tools.gradient_audit import run_toy_gradient_audit

    report = run_toy_gradient_audit(seed=args.seed)
    if args.out:
        _write_json(Path(args.out), report)
    print(json.dumps(report["checks"], indent=2))
    return 0 if report["all_passed"] else 1


def cmd_smoke_feature(args: argparse.Namespace) -> int:
    if not args.toy:
        from pi05_rlt_rtc.deployment.runtime import smoke_or_profile
        return smoke_or_profile(args)
    build = _build_toy(args)
    obs = build.env.reset(seed=args.seed)
    request = _toy_request(build, request_id=0)
    prev_result = build.feature_service.extract(_toy_request(build, request_id=-1), obs)
    assert prev_result is not None
    result = build.feature_service.extract(request, obs)
    result.validate(build.contract, expected_batch=1)
    if build.feature_service.prefix_calls != 2:  # warm-start request + request 0
        raise ContractError(
            f"expected exactly one prefix pass per extract (2 total here), got "
            f"{build.feature_service.prefix_calls}"
        )
    report = {
        "kind": "toy_feature_smoke",
        "mock": True,
        "prefix_calls": build.feature_service.prefix_calls,
        "result": result.planning_metadata(),
        "reference_env_suffix_shape": list(result.reference_suffix(build.contract).shape),
    }
    if args.out:
        _write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))
    return 0


def cmd_test_rtc_math(args: argparse.Namespace) -> int:
    import subprocess

    target = REPO_ROOT / "tests" / "test_rtc_math.py"
    if not target.exists():
        raise MissingAssetError(f"test file missing: {target}")
    cmd = [sys.executable, "-m", "pytest", "-q", str(target)]
    if args.select:
        cmd += ["-k", args.select]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


class _SleepyPlanner:
    def __init__(self, backend, seconds: float) -> None:
        self.backend = backend
        self.seconds = float(seconds)

    def plan(self, request, observation, pending_prefix, prev_context):
        if self.seconds > 0:
            time.sleep(self.seconds)
        return self.backend.plan(request, observation, pending_prefix, prev_context)


def cmd_profile_planner(args: argparse.Namespace) -> int:
    if not args.toy:
        from pi05_rlt_rtc.deployment.runtime import smoke_or_profile
        return smoke_or_profile(args, profile=True)
    from pi05_rlt_rtc.runtime.controller import WallClockController
    from pi05_rlt_rtc.runtime.queue import ActionQueue

    build = _build_toy(args)
    queue = ActionQueue(build.contract, fallback_policy="hold_last")
    planner = _SleepyPlanner(build.backend, args.planner_sleep_ms / 1000.0)
    controller = WallClockController(
        contract=build.contract,
        queue=queue,
        backend=planner,
        env=build.env,
        tick_hz=args.hz,
        seed=args.seed,
    )
    report = controller.run(args.ticks)
    payload = {
        "kind": "toy_wallclock_profile",
        "mock": True,
        "tick_hz": args.hz,
        "planner_sleep_ms": args.planner_sleep_ms,
        **report.summary(),
    }
    _write_json(Path(args.out), payload)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_collect_teacher(args: argparse.Namespace) -> int:
    if not args.toy:
        from pi05_rlt_rtc.deployment.runtime import collect_teacher
        return collect_teacher(args)
    from pi05_rlt_rtc.evaluation.toy_pipeline import collect_toy

    ctx = RunContext.create(args.out, kind="toy", mock=True)
    build = _build_toy(args)
    collected = collect_toy(
        build,
        num_ticks=args.ticks,
        planner_latency_ticks=args.planner_latency_ticks,
        seed=args.seed,
        gamma=args.gamma,
    )
    build.queue.ledger.write_jsonl(ctx.root / "execution_ledger" / "ledger.jsonl")
    audit = [t.to_record() for t in collected.transitions]
    _write_json(ctx.root / "transition_audit" / "transitions.json", audit)
    learnable = sum(1 for t in collected.transitions if t.is_learnable())
    _write_json(
        ctx.root / "collection_summary.json",
        {
            "kind": "toy_teacher_collection",
            "mock": True,
            "ticks": collected.log.ticks,
            "requests": len([e for e in collected.log.events if e["kind"] == "request"]),
            "transitions": len(collected.transitions),
            "learnable": learnable,
            "deadline_misses": collected.log.deadline_misses,
            "replay_counts": collected.replay.counts(),
        },
    )
    print(f"toy collection: {len(collected.transitions)} transitions ({learnable} learnable)")
    return 0 if learnable > 0 else 1


def cmd_warmup_actor(args: argparse.Namespace) -> int:
    if not args.toy:
        from pi05_rlt_rtc.deployment.training import train
        return train(args, warmup_only=True)
    from pi05_rlt_rtc.evaluation.toy_pipeline import collect_toy, train_toy_actor_critic

    ctx = RunContext.create(args.out, kind="toy", mock=True)
    build = _build_toy(args)
    collected = collect_toy(build, num_ticks=args.ticks, seed=args.seed, gamma=args.gamma)
    actor, critic, report = train_toy_actor_critic(
        collected,
        updates=1,
        warmup_updates=args.warmup_updates,
        seed=args.seed,
    )
    import torch

    torch.save(
        {
            "actor_state": actor.state_dict(),
            "critic_state": critic.state_dict(),
            "warmup_losses": report.warmup_losses,
            "metrics": report.metrics,
            "mock": True,
        },
        ctx.root / "checkpoints" / "toy_warmup.pt",
    )
    _write_json(ctx.root / "metrics.json", {"kind": "toy_warmup", "mock": True, **report.metrics})
    improved = report.warmup_losses[-1] < report.warmup_losses[0]
    print(json.dumps(report.metrics, indent=2))
    return 0 if improved else 1


def cmd_train_queue_rlt(args: argparse.Namespace) -> int:
    if not args.toy:
        from pi05_rlt_rtc.deployment.training import train
        return train(args)
    from pi05_rlt_rtc.evaluation.toy_pipeline import (
        collect_toy,
        evaluate_toy_actor,
        export_toy_bundle,
        train_toy_actor_critic,
    )

    ctx = RunContext.create(args.out, kind="toy", mock=True)
    build = _build_toy(args)
    collected = collect_toy(build, num_ticks=args.ticks, seed=args.seed, gamma=args.gamma)
    build.queue.ledger.write_jsonl(ctx.root / "execution_ledger" / "ledger.jsonl")
    actor, critic, report = train_toy_actor_critic(
        collected,
        updates=args.updates,
        warmup_updates=args.warmup_updates,
        seed=args.seed,
        gamma=args.gamma,
    )
    export_toy_bundle(build=build, actor=actor, critic=critic, out_dir=str(ctx.root / "checkpoints"))
    evaluation = evaluate_toy_actor(
        build=build, encoder=collected.encoder, actor=actor, episodes=args.eval_episodes, seed=args.seed
    )
    _write_json(ctx.root / "metrics.json", {"kind": "toy_train", "mock": True, **report.metrics})
    _write_json(ctx.root / "evaluation" / "toy_eval.json", {"mock": True, **evaluation.to_dict()})
    print(json.dumps(evaluation.to_dict(), indent=2))
    improved = evaluation.mean_final_abs_state < evaluation.random_mean_final_abs_state
    return 0 if improved else 1


def cmd_serve_planner(args: argparse.Namespace) -> int:
    if not args.toy:
        import os
        from pi05_rlt_rtc.deployment.runtime import config_from_args, make_backend
        from pi05_rlt_rtc.tools.planner_server import PlannerService, serve
        cfg = config_from_args(args)
        backend, metadata = make_backend(cfg, args)
        service = PlannerService(backend, cfg.contract, metadata=metadata)
        print(f"serving real planner on {args.host}:{args.port}; feature={metadata['feature_version_id']}")
        serve(service, host=args.host, port=args.port, shutdown_after=args.once or None,
              token=os.environ.get("PI05_PLANNER_TOKEN"))
        return 0
    from pi05_rlt_rtc.tools.planner_server import PlannerService, serve

    build = _build_toy(args)
    service = PlannerService(build.backend, build.contract)
    print(f"serving toy planner on {args.host}:{args.port} (mock=True)")
    serve(service, host=args.host, port=args.port, shutdown_after=args.once or None)
    return 0


def cmd_run_controller(args: argparse.Namespace) -> int:
    if not args.toy:
        from pi05_rlt_rtc.deployment.runtime import run_controller
        return run_controller(args)
    from pi05_rlt_rtc.runtime.controller import WallClockController
    from pi05_rlt_rtc.runtime.queue import ActionQueue

    ctx = RunContext.create(args.out, kind="toy", mock=True)
    build = _build_toy(args)
    queue = ActionQueue(build.contract, fallback_policy="hold_last")
    planner = _SleepyPlanner(build.backend, args.planner_sleep_ms / 1000.0)
    controller = WallClockController(
        contract=build.contract,
        queue=queue,
        backend=planner,
        env=build.env,
        tick_hz=args.hz,
        seed=args.seed,
    )
    report = controller.run(args.ticks)
    queue.ledger.write_jsonl(ctx.root / "execution_ledger" / "ledger.jsonl")
    payload = {"kind": "toy_controller_run", "mock": True, **report.summary()}
    _write_json(ctx.root / "controller_report.json", payload)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    if not args.toy:
        from pi05_rlt_rtc.deployment.runtime import evaluate
        return evaluate(args)
    if not args.policy:
        raise ContractError("toy evaluation requires --policy")
    from pi05_rlt_rtc.evaluation.toy_pipeline import build_toy, evaluate_toy_actor
    from pi05_rlt_rtc.learning.checkpoint import PolicyBundle

    bundle = PolicyBundle.load(args.policy)
    if not bundle.extra.get("mock"):
        raise MockPolicyError("toy evaluation requires an explicitly marked mock policy bundle")
    if bundle.extra.get("rtc_mode") not in ("none", "approx", "full_vjp"):
        raise ContractError("bundle lacks a verified RTC mode; do not guess evaluation provenance")
    build = build_toy(horizon=bundle.contract.horizon, chunk=bundle.contract.chunk,
                      delay=bundle.contract.delay, z_dim=bundle.spec.z_dim,
                      rtc_mode=bundle.extra["rtc_mode"], seed=args.seed)
    if build.spec != bundle.spec:
        raise ContractError("saved planning-state layout is unsupported by this toy evaluator")
    actor = bundle.build_actor()
    report = evaluate_toy_actor(
        build=build, encoder=None, actor=actor, episodes=args.episodes, seed=args.seed
    )
    _write_json(Path(args.out), {"kind": "toy_evaluation", "mock": True, "policy": str(args.policy), **report.to_dict()})
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def cmd_export_deployment(args: argparse.Namespace) -> int:
    from pi05_rlt_rtc.learning.checkpoint import PolicyBundle

    bundle = PolicyBundle.load(args.policy)
    if not args.toy and bundle.extra.get("mock") is not False:
        raise MockPolicyError("real export requires an explicitly non-mock bundle")
    if args.formal and bundle.extra.get("mock"):
        raise MockPolicyError("refusing to export a mock bundle as a formal deployment")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle.save(out / "actor.pt", include_critic=False)
    manifest = {
        "schema_version": bundle.schema_version,
        "planning_state_version": bundle.planning_state_version,
        "feature_version_id": bundle.feature_version_id,
        "norm_stats_sha256": bundle.norm_stats_sha256,
        "actor_state_sha256": bundle.actor_state_sha256,
        "contract": dataclasses.asdict(bundle.contract),
        "spec": dataclasses.asdict(bundle.spec),
        "codec_name": bundle.codec_name,
        "contains_critic": False,
        "runtime_contract": bundle.extra.get("runtime_contract"),
        "fingerprint": bundle.extra.get("fingerprint"),
        "actor_hidden_dims": list(bundle.actor_hidden_dims),
        "actor_reference_hidden_dims": list(bundle.actor_reference_hidden_dims),
        "rtc_mode": bundle.extra.get("rtc_mode"),
        "num_denoising_steps": bundle.extra.get("num_denoising_steps"),
        "control_frequency_hz": bundle.extra.get("control_frequency_hz"),
        "mock": bool(bundle.extra.get("mock")),
        "toy": bool(bundle.extra.get("mock")),
        "notes": [
            "deployment export excludes the critic and the RLT reconstruction decoder",
            "reload test is required before any real deployment",
        ],
    }
    _write_json(out / "deployment_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


# ---------------------------------------------------------------------------
# parser


def cmd_runtime_validate(args):
    from pi05_rlt_rtc.config.runtime_config import load_runtime_config
    cfg = load_runtime_config(args.runtime_config)
    print(json.dumps({"platform": cfg.platform, **cfg.wire_contract(), "assets_checked": False}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pi05-rlt-rtc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    from pi05_rlt_rtc.deployment.assets import preflight, bundle_command
    from pi05_rlt_rtc.deployment.offline import infer_recording, infer_demo
    p = sub.add_parser("preflight", help="asset-only preflight; never opens devices")
    p.add_argument("--runtime-config", required=True)
    p.add_argument("--out")
    p.set_defaults(func=preflight)
    p = sub.add_parser("infer-recording", help="replay exact saved planner requests without devices")
    p.add_argument("--runtime-config", required=True)
    p.add_argument("--recording", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--planner-url")
    p.add_argument("--rpc-timeout-s", type=float, default=30.)
    p.set_defaults(func=infer_recording)
    p = sub.add_parser("infer-demo", help="cold model inference on one saved HDF5 demonstration frame")
    p.add_argument("--runtime-config", required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=infer_demo)
    p = sub.add_parser("bundle-runtime", help="atomic portable code/model/statistics/tokenizer export")
    p.add_argument("--runtime-config", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=bundle_command)
    p = sub.add_parser("verify-bundle", help="verify deployment bundle integrity without model/device load")
    p.add_argument("path")
    p.set_defaults(func=bundle_command)

    p = sub.add_parser("audit-source", help="static audit of a pinned RLinf checkout")
    p.add_argument("--rlinf", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_audit_source)

    p = sub.add_parser("config-validate", help="strict design-config validation + resolved export")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--require-assets", default=None, choices=["base", "stage1", "stage2", "data", "apply"])
    p.set_defaults(func=cmd_config_validate)

    p = sub.add_parser("inspect-data", help="LeRobot dataset audit")
    p.add_argument("--root", required=True)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_inspect_data)

    p = sub.add_parser("inspect-checkpoint", help="checkpoint key/shape audit")
    p.add_argument("--path", required=True)
    p.add_argument("--expect-rlt", action="store_true")
    p.add_argument("--expect-backbone", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_inspect_checkpoint)

    p = sub.add_parser("smoke-stage1", help="two-GPU full-model synthetic smoke, or CPU toy gradient audit")
    p.add_argument("overrides", nargs="*", help="Hydra key=value overrides for the full-model smoke")
    p.add_argument("--toy", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_smoke_stage1)

    p = sub.add_parser("smoke-feature", help="saved-input z + full-H reference check (no devices)")
    p.add_argument("--recording")
    p.add_argument("--runtime-config", default=None)
    p.add_argument("--planner-url", default=None)
    p.add_argument("--rpc-timeout-s", type=float, default=30.0)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--out", default=None)
    _add_toy_args(p)
    p.set_defaults(func=cmd_smoke_feature)

    p = sub.add_parser("test-rtc-math", help="run the RTC math test file")
    p.add_argument("--select", default=None)
    p.set_defaults(func=cmd_test_rtc_math)

    p = sub.add_parser("profile-planner", help="saved-input planner latency, or toy wall-clock test")
    p.add_argument("--recording")
    p.add_argument("--runtime-config", default=None)
    p.add_argument("--planner-url", default=None)
    p.add_argument("--rpc-timeout-s", type=float, default=30.0)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--ticks", type=int, default=40)
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--planner-sleep-ms", type=float, default=5.0)
    p.add_argument("--out", required=True)
    _add_toy_args(p)
    p.set_defaults(func=cmd_profile_planner)

    p = sub.add_parser("collect-teacher", help="collect fixed-D transitions from a teacher")
    p.add_argument("--runtime-config", default=None)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--ticks", type=int, default=120)
    p.add_argument("--planner-latency-ticks", type=int, default=0)
    p.add_argument("--out", required=True)
    _add_toy_args(p)
    p.set_defaults(func=cmd_collect_teacher)

    p = sub.add_parser("warmup-actor", help="BC warmup on the deployed action parameterisation")
    p.add_argument("--runtime-config", default=None)
    p.add_argument("--replay", nargs="+", help="human-labelled physical trial directories")
    p.add_argument("--resume", help="previous latest/ learner checkpoint")
    p.add_argument("--learner-device", default="cpu")
    p.add_argument("--toy", action="store_true")
    p.add_argument("--ticks", type=int, default=120)
    p.add_argument("--warmup-updates", type=int, default=100)
    p.add_argument("--out", required=True)
    _add_toy_args(p)
    p.set_defaults(func=cmd_warmup_actor)

    p = sub.add_parser("train-queue-rlt", help="queue-aware Actor-Critic training")
    p.add_argument("--runtime-config", default=None)
    p.add_argument("--replay", nargs="+", help="human-labelled physical trial directories")
    p.add_argument("--resume", help="previous latest/ learner checkpoint")
    p.add_argument("--learner-device", default="cpu")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--ticks", type=int, default=160)
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--warmup-updates", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--out", required=True)
    _add_toy_args(p)
    p.set_defaults(func=cmd_train_queue_rlt)

    p = sub.add_parser("serve-planner", help="planner HTTP service")
    p.add_argument("--runtime-config", default=None)
    p.add_argument("--policy", default=None)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--once", action="store_true")
    _add_toy_args(p)
    p.set_defaults(func=cmd_serve_planner)

    p = sub.add_parser("run-controller", help="wall-clock non-blocking controller")
    p.add_argument("--runtime-config", default=None)
    p.add_argument("--planner-url", default=None)
    p.add_argument("--rpc-timeout-s", type=float, default=30.0)
    p.add_argument("--clock", choices=["wall"], default="wall")
    p.add_argument("--policy", default=None)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--ticks", type=int, default=60)
    p.add_argument("--hz", type=float, default=100.0)
    p.add_argument("--planner-sleep-ms", type=float, default=5.0)
    p.add_argument("--out", required=True)
    _add_toy_args(p)
    p.set_defaults(func=cmd_run_controller)

    p = sub.add_parser("evaluate", help="independent evaluation from a saved policy")
    p.add_argument("--runtime-config", default=None)
    p.add_argument("--planner-url", default=None)
    p.add_argument("--rpc-timeout-s", type=float, default=30.0)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--policy", default=None)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--trials", nargs="+", help="labelled real trial directories")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("export-deployment", help="deployment export with manifest")
    p.add_argument("--toy", action="store_true")
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--formal", action="store_true")
    p.set_defaults(func=cmd_export_deployment)

    p = sub.add_parser("annotate-rollout", help="label one recorded physical rollout; no robot connection")
    p.add_argument("--rollout", required=True)
    p.add_argument("--outcome", required=True, choices=["success", "failure", "aborted", "intervened"])
    p.add_argument("--annotator", required=True)
    p.add_argument("--terminal-tick", type=int)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_annotate_rollout)

    p = sub.add_parser("session", help="human critical-phase rollout / hold / update loop")
    p.add_argument("--runtime-config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--annotator", default="")
    p.add_argument("--exploration", action=argparse.BooleanOptionalAction, default=None, help="train default on; eval always off")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--policy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--execute", action="store_true")
    from pi05_rlt_rtc.deployment.session import run_session
    p.set_defaults(func=run_session)

    p = sub.add_parser("runtime-validate", help="validate the physical Panda deployment template")
    p.add_argument("--runtime-config", required=True)
    p.set_defaults(func=cmd_runtime_validate)

    return parser


def _add_toy_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--chunk", type=int, default=5)
    p.add_argument("--delay", type=int, default=2)
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--rtc-mode", default="none", choices=["none", "approx", "full_vjp"])
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except EnvironmentNotReadyError as exc:
        print(f"environment not ready (exit 2): {exc}", file=sys.stderr)
        return 2
    except MissingAssetError as exc:
        print(f"missing asset (exit 3): {exc}", file=sys.stderr)
        return 3
    except (ContractError, ConfigError, SchemaMismatchError) as exc:
        print(f"contract/config error (exit 4): {exc}", file=sys.stderr)
        return 4
    except MockPolicyError as exc:
        print(f"mock/formal separation error (exit 5): {exc}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
