"""Concrete local/remote inference, collection and evaluation entry points."""
from __future__ import annotations

import dataclasses
import json
import os
import uuid
from pathlib import Path


from pi05_rlt_rtc import PLANNING_STATE_VERSION, SCHEMA_VERSION
from pi05_rlt_rtc.config.runtime_config import load_runtime_config
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.runtime.queue import ActionQueue


def config_from_args(args):
    if not getattr(args, "runtime_config", None):
        raise ContractError("non-toy commands require --runtime-config configs/runtime_franka.yaml")
    cfg = load_runtime_config(args.runtime_config)
    policy = getattr(args, "policy", None)
    if policy:
        cfg = dataclasses.replace(cfg, raw={**cfg.raw, "policy": {"kind": "actor", "checkpoint": str(Path(policy).resolve()), "exploration": cfg.raw["policy"].get("exploration", False)}})
    return cfg


def make_backend(cfg, args):
    if getattr(args, "planner_url", None):
        if getattr(args, "policy", None) and not getattr(args, 'resident_teacher', False):
            raise ContractError("select --policy on the planner host; the remote client uses policy.kind and optional expected_policy_version_id")
        from pi05_rlt_rtc.deployment.rpc import RemotePlanner
        teacher_cfg = cfg
        if getattr(args, 'resident_teacher', False):
            teacher_policy = {'kind':'teacher', 'checkpoint':None}
            if cfg.raw['policy'].get('expected_feature_version_id'):
                teacher_policy['expected_feature_version_id'] = cfg.raw['policy']['expected_feature_version_id']
            teacher_cfg = dataclasses.replace(cfg, raw={**cfg.raw, 'policy': teacher_policy})
        backend = RemotePlanner(args.planner_url, teacher_cfg, timeout=getattr(args, "rpc_timeout_s", 30.),
                                token=os.environ.get("PI05_PLANNER_TOKEN"))
        if getattr(args, 'resident_teacher', False):
            from .rpc import ResidentTeacherActor
            backend = ResidentTeacherActor(backend, cfg)
        return backend, backend.metadata
    from pi05_rlt_rtc.deployment.model_loader import load_planner
    return load_planner(cfg)


def make_env(cfg):
    from pi05_rlt_rtc.adapters.franka import FrankaEnvAdapter
    if cfg.platform != "franka":
        raise ContractError("only the physical Panda/Robotiq environment is supported")
    return FrankaEnvAdapter(cfg)


def scheduler_kwargs(cfg, metadata):
    return {"contract": cfg.contract, "rtc_mode": cfg.rtc_mode, "num_denoising_steps": cfg.num_steps,
            "feature_version_id": metadata["feature_version_id"], "policy_version_id": metadata["policy_version_id"],
            "env_id": cfg.platform, "schema_version": SCHEMA_VERSION, "planning_state_version": PLANNING_STATE_VERSION}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run_controller(args):
    cfg = config_from_args(args)
    backend, metadata = make_backend(cfg, args)
    _, report = collect(cfg, backend, metadata, ticks=args.ticks, seed=args.seed, out=args.out)
    print(json.dumps(report, indent=2))
    return 0 if not report["safety_stops"] and report["stop_request_acknowledged"] else 1


def smoke_or_profile(args, *, profile=False):
    if not getattr(args, "recording", None):
        raise ContractError("hardware-free smoke/profile requires --recording saved_requests_directory")
    from .offline import infer_recording
    return infer_recording(args)


def collect(cfg, backend, metadata, *, ticks, seed, out, replay=None):
    """Exactly one manually prepared physical trial; label rewards afterwards."""
    from pi05_rlt_rtc.runtime.controller import WallClockController
    from .rollout import save_rollout
    if replay is not None:
        raise ContractError("label the saved rollout before adding it to replay")
    if type(ticks) is not int or not 1 <= ticks <= 5000:
        raise ContractError("collection ticks must be in [1,5000]")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ContractError("use a new output directory for each physical trial")
    out.mkdir(parents=True, exist_ok=True)
    metadata = {**metadata, "platform": "franka", "clock": "wall",
                "gamma_per_tick": cfg.raw.get("training", {}).get("gamma_per_tick", .99)}
    trial_id = uuid.uuid4().hex
    from pi05_rlt_rtc.runtime.recording import AsyncRequestRecorder
    recorder = AsyncRequestRecorder(out / "observations", metadata)
    try:
        queue, env = ActionQueue(cfg.contract, fallback_policy="stop"), make_env(cfg)
    except BaseException:
        recorder.close()
        raise
    controller = WallClockController(queue=queue, backend=backend, env=env, tick_hz=cfg.control_hz,
        seed=seed, episode_id=trial_id, record_events=True, observation_recorder=recorder,
        **scheduler_kwargs(cfg, metadata))
    report = {"safety_stops": 1, "ticks_run": 0, "learnable": 0,
              "annotation_required": True, "stop_request_acknowledged": False}
    try:
        report.update(controller.run(ticks).summary())
        return None, report
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        env.close()
        recording_error = None
        try:
            recorder.close()
        except ContractError as exc:
            recording_error = exc
            report["error"] = str(exc)
            report["recording_error"] = str(exc)
        report["stop_request_acknowledged"] = bool(getattr(env, "stop_request_acknowledged", False))
        queue.ledger.write_jsonl(out / "ledger.jsonl")
        write_json(out / "runtime_manifest.json", metadata)
        write_json(out / "controller_report.json", report)
        write_json(out / "hardware_commands.json", getattr(env, "command_log", []))
        save_rollout(out / "rollout.pt", events=controller.events, metadata=metadata, report=report,
            contract=cfg.contract, trial_id=trial_id, hardware_commands=getattr(env, "command_log", []))
        if recording_error is not None:
            raise recording_error


def collect_teacher(args):
    cfg = config_from_args(args)
    cfg = dataclasses.replace(cfg, raw={**cfg.raw, "policy": {"kind": "teacher", "checkpoint": None}})
    if getattr(args, "planner_url", None):
        raise ContractError("learning rollout requires captured features; run the local planner on the 4090")
    backend, metadata = make_backend(cfg, args)
    _, report = collect(cfg, backend, metadata, ticks=args.ticks, seed=args.seed, out=args.out)
    print(json.dumps(report, indent=2))
    return 0 if not report["safety_stops"] and report["stop_request_acknowledged"] else 1


def evaluate(args):
    """Aggregate explicitly labelled physical trials; never reset/move a robot."""
    cfg = config_from_args(args)
    paths = getattr(args, "trials", None)
    if not paths:
        raise ContractError("physical evaluation requires --trials labelled_trial_dir [...]")
    results, identities, policies = [], set(), set()
    for value in paths:
        directory = Path(value)
        annotation = json.loads((directory / "annotation.json").read_text())
        metadata = json.loads((directory / "runtime_manifest.json").read_text())
        from .model_loader import file_sha256
        if file_sha256(annotation["rollout"]) != annotation["rollout_sha256"]:
            raise ContractError("annotated rollout changed")
        if metadata.get("runtime_contract") != cfg.wire_contract() or metadata.get("mock") is not False:
            raise ContractError("evaluation trial contract/mock mismatch")
        if annotation["trial_id"] in identities:
            raise ContractError("duplicate evaluation trial")
        identities.add(annotation["trial_id"])
        policies.add(metadata["policy_version_id"])
        results.append(annotation)
    if len(policies) != 1:
        raise ContractError("evaluate one fixed policy version at a time")
    successes = sum(r["outcome"] == "success" for r in results)
    report = {"mock": False, "policy_version_id": next(iter(policies)), "trials": results,
              "success_rate": successes / len(results), "total_trials": len(results),
              "aborted_or_intervened": sum(r["outcome"] in ("aborted", "intervened") for r in results)}
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0
