"""Persist measured wall-clock trials and explicitly label terminal outcomes."""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.observation import EnvIDs
from pi05_rlt_rtc.contracts.plan import PlanRequest, PrevPlanContext, FinalPlan, SubmittedPlan
from pi05_rlt_rtc.contracts.spaces import HorizonContract
from pi05_rlt_rtc.learning.planning_state import PlanningState, PlanningStateSpec
from pi05_rlt_rtc.learning.replay import ReplayBuffer
from pi05_rlt_rtc.learning.serialization import atomic_torch_save
from pi05_rlt_rtc.learning.transition_assembler import DelayedTransitionAssembler
from pi05_rlt_rtc.runtime.scheduler import PlanResult
from .planner import state_from_plan_event

_TYPES = {c.__name__: c for c in (EnvIDs, PlanRequest, PrevPlanContext, FinalPlan,
                                SubmittedPlan, HorizonContract, PlanningState, PlanningStateSpec, PlanResult)}


def pack(value):
    if dataclasses.is_dataclass(value):
        if type(value).__name__ not in _TYPES:
            raise ContractError(f"unsupported rollout type: {type(value).__name__}")
        return {"__type__": type(value).__name__, "fields": {
            f.name: pack(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value, np.ndarray):
        return {"__array__": torch.from_numpy(value.copy())}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [pack(v) for v in value]
    return value


def unpack(value):
    if isinstance(value, dict):
        if "__array__" in value:
            return value["__array__"].numpy().copy()
        if "__type__" in value:
            return _TYPES[value["__type__"]](**{k: unpack(v) for k, v in value["fields"].items()})
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v) for v in value]
    return value


def save_rollout(path, *, events, metadata, report, contract, trial_id, hardware_commands):
    atomic_torch_save(pack({"format": "physical-rollout-v1", "trial_id": trial_id,
        "events": events, "metadata": metadata, "report": report,
        "contract": contract, "hardware_commands": hardware_commands}), path)


def load_rollout(path):
    data = unpack(torch.load(path, map_location="cpu", weights_only=True))
    if data.get("format") != "physical-rollout-v1":
        raise ContractError("unsupported physical rollout format")
    return data


def label_rollout(rollout, *, outcome, annotator, terminal_tick=None, allow_mock=False):
    """Sparse task reward is valid only after a human labels this exact trial.

    Success=1 on the identified final tick; failure=0. Aborted, intervened and
    faulted trials are retained for inspection and never admitted to RL replay.
    """
    if outcome not in ("success", "failure", "aborted", "intervened") or not annotator.strip():
        raise ContractError("choose a valid outcome and identify the human annotator")
    metadata, report = rollout["metadata"], rollout["report"]
    if metadata.get("mock") is not False and not (allow_mock and metadata.get("mock") is True):
        raise ContractError("mock rollout cannot enter physical training replay")
    bad = any(report.get(k, 0) for k in ("safety_stops", "planner_errors", "deadline_misses"))
    if bad or report.get("error") or report.get("stop_request_acknowledged") is not True:
        if outcome in ("success", "failure"):
            raise ContractError("faulted or unacknowledged-stop trial must be aborted/intervened")
    replay = ReplayBuffer(10000, feature_version_id=metadata["feature_version_id"],
                          norm_stats_sha256=metadata["norm_stats_sha256"])
    annotation = {"trial_id": rollout["trial_id"], "outcome": outcome,
                  "annotator": annotator, "terminal_tick": terminal_tick,
                  "reward_source": "manual_terminal", "eligible": outcome in ("success", "failure")}
    start_tick = metadata.get('critical_start_tick', 0)
    scheme = metadata.get('runtime_contract', {}).get('reward_scheme', 'rlt_binary')
    if scheme not in ('rlt_binary', 'rlinf_signed'):
        raise ContractError('unknown reward scheme')
    annotation.update(reward_scheme=scheme, critical_start_tick=start_tick, mock=metadata.get('mock'))
    if start_tick is None:
        annotation['eligible'] = False
        annotation['reason'] = 'critical_phase_not_entered'
    if not annotation["eligible"]:
        return replay, annotation
    events = copy.deepcopy(rollout["events"])
    ticks = [e for e in events if e["kind"] == "tick"]
    if not ticks:
        raise ContractError("cannot label a rollout without executed ticks")
    final_tick = ticks[-1]["tick"] if terminal_tick is None else terminal_tick
    if type(final_tick) is not int or final_tick not in {e["tick"] for e in ticks}:
        raise ContractError("terminal_tick must identify an actually executed tick")
    annotation["terminal_tick"] = final_tick
    for e in ticks:
        e["reward"] = (1. if outcome == "success" else (-1. if scheme == "rlinf_signed" else 0.)) if e["tick"] == final_tick else 0.
        e["terminated"], e["truncated"] = e["tick"] == final_tick, False
        e["info"] = {**e.get("info", {}), "reward_valid": True, "reward_source": "manual_terminal"}
    # Discard requests captured after the declared task end and unresolved jobs.
    requests = {e["request"].request_id for e in events if e["kind"] == "request"
                and start_tick <= e["request"].capture_tick <= final_tick}
    plans = {e["request_id"] for e in events if e["kind"] == "plan" and e.get("accepted")
             and getattr(e.get("result"), "planning_state", None) is not None}
    valid = requests & plans
    filtered = [e for e in events if (
        (e["kind"] == "request" and e["request"].request_id in valid)
        or (e["kind"] == "plan" and e["request_id"] in valid)
        or (e["kind"] == "tick" and e["tick"] <= final_tick))]
    assembler = DelayedTransitionAssembler(contract=rollout["contract"],
        state_encoder=state_from_plan_event,
        gamma_per_tick=metadata.get("gamma_per_tick", .99))
    assembler.ingest_run(filtered)
    for transition in assembler.finalize():
        replay.push(transition)
    if not replay.learnable_transitions():
        annotation.update(eligible=False, reason="no_learnable_critical_transition")
    return replay, annotation


def annotate(args):
    from .runtime import write_json
    source = Path(args.rollout).resolve()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ContractError("annotation output must be a new directory")
    rollout = load_rollout(source)
    replay, annotation = label_rollout(rollout, outcome=args.outcome, annotator=args.annotator,
                                      terminal_tick=args.terminal_tick, allow_mock=getattr(args, "allow_mock", False))
    from .model_loader import file_sha256
    annotation.update(rollout=str(source), rollout_sha256=file_sha256(source), replay_counts=replay.counts())
    replay.save(out / "replay.pt")
    annotation["replay_sha256"] = file_sha256(out / "replay.pt")
    write_json(out / "annotation.json", annotation)
    write_json(out / "runtime_manifest.json", rollout["metadata"])
    print(json.dumps(annotation, indent=2))
    return 0
