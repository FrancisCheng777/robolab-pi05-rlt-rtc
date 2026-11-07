"""Final integration tests T61-T68.

T61-T64 require the real data/weights/simulator/GPU and are explicit SKIPs.
T65/T66/T67/T68 exercise the *project's* export-reload, independent-evaluation,
mock-separation and bounded-ledger mechanisms on CPU (labelled toy/mock).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pi05_rlt_rtc.contracts.errors import MockPolicyError

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skip(reason="T61 requires the pinned RLinf B0 baseline, real data and weights")
def test_t61_upstream_b0_small_end_to_end():
    raise AssertionError("NOT_RUN: RLinf runtime required")


@pytest.mark.skip(
    reason="T62 requires pi0.5 + RLT + full_vjp with real weights; the CPU toy pipeline "
    "exercises the wiring but cannot substitute for the three real components"
)
def test_t62_b2_virtual_delay_training_real_components():
    raise AssertionError("NOT_RUN: RLinf runtime required")


@pytest.mark.skip(
    reason="T63 requires the real wall-clock B2 loop; a CPU wall-clock contract test with an "
    "explicitly labelled mock planner runs under T44"
)
def test_t63_b2_real_wall_clock_async():
    raise AssertionError("NOT_RUN: real backend required")


@pytest.mark.skip(reason="T64 requires the real learner running concurrently with the planner")
def test_t64_profiling_with_concurrent_learner():
    raise AssertionError("NOT_RUN: real backend required")


def test_t65_deployment_export_reload_in_fresh_process(tmp_path):
    """Export a bundle, then reload and run it in a *fresh* Python process."""
    from pi05_rlt_rtc.evaluation.toy_pipeline import (
        build_toy,
        collect_toy,
        export_toy_bundle,
        train_toy_actor_critic,
    )

    build = build_toy(horizon=10, chunk=5, delay=2, z_dim=24, seed=0)
    collected = collect_toy(build, num_ticks=120, seed=0)
    actor, critic, _ = train_toy_actor_critic(
        collected, updates=20, batch_size=16, warmup_updates=20, seed=0
    )
    out_dir = tmp_path / "export"
    bundle = export_toy_bundle(build=build, actor=actor, critic=critic, out_dir=str(out_dir))
    assert (out_dir / "toy_actor.pt").exists()
    assert (out_dir / "deployment_manifest.json").exists()
    manifest = json.loads((out_dir / "deployment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mock"] is True
    assert manifest["actor_state_sha256"] == bundle.actor_state_sha256
    assert manifest["contract"]["horizon"] == 10 and manifest["contract"]["delay"] == 2

    script = (
        "import json, torch;"
        "from pi05_rlt_rtc.learning.checkpoint import PolicyBundle;"
        f"b=PolicyBundle.load({str(out_dir / 'toy_actor.pt')!r});"
        "a=b.build_actor();"
        "out=a.act(state=torch.zeros(1,sum(d for _,d in b.spec.actor_state_layout())),"
        "reference_suffix=torch.zeros(1,b.spec.chunk*b.spec.env_action_dim),deterministic=True);"
        "print(json.dumps({'shape': list(out.actions.shape), 'finite': bool(torch.isfinite(out.actions).all())}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["shape"] == [1, 5, 8]
    assert payload["finite"] is True


def test_t66_independent_evaluation_reports_trials_without_experts(tmp_path):
    from pi05_rlt_rtc.evaluation.toy_pipeline import (
        build_toy,
        collect_toy,
        evaluate_toy_actor,
        train_toy_actor_critic,
    )

    build = build_toy(horizon=10, chunk=5, delay=2, z_dim=24, seed=0)
    collected = collect_toy(build, num_ticks=200, seed=0)
    actor, critic, _ = train_toy_actor_critic(
        collected, updates=150, batch_size=32, warmup_updates=80, seed=0
    )
    evaluation = evaluate_toy_actor(build=build, actor=actor, episodes=4, seed=7)
    report = evaluation.to_dict()
    assert report["episodes"] == 4
    assert 0 <= report["successes"] <= 4
    assert isinstance(report["failures"], list)
    # no expert takeover exists in the evaluation path: the policy backend is
    # the actor, and the toy config has expert_takeover unavailable by design.
    assert "expert" not in json.dumps(report).lower()
    written = tmp_path / "eval.json"
    written.write_text(json.dumps({"mock": True, **report}), encoding="utf-8")
    assert json.loads(written.read_text(encoding="utf-8"))["mock"] is True


def test_t67_mock_artifacts_are_tagged_and_formal_use_is_refused(tmp_path):
    from pi05_rlt_rtc.adapters.mock_feature import MOCK_FEATURE_VERSION_ID, MockFeatureReferenceService
    from pi05_rlt_rtc.action_codec.codec import ActionCodec, NormStats
    from pi05_rlt_rtc.contracts.spaces import HorizonContract
    from pi05_rlt_rtc.evaluation.toy_pipeline import build_toy, collect_toy
    from pi05_rlt_rtc.learning.checkpoint import PolicyBundle
    from pi05_rlt_rtc.tools.run_context import RunContext

    service = MockFeatureReferenceService(
        contract=HorizonContract(10, 5, 2), codec=ActionCodec(NormStats.symmetric())
    )
    assert service.is_mock is True
    assert service.provenance()["mock"] is True
    assert MOCK_FEATURE_VERSION_ID.startswith("MOCK")

    # a formal run directory refuses mock payloads
    with pytest.raises(MockPolicyError):
        RunContext.create(tmp_path / "runs", kind="formal", mock=True)
    ctx = RunContext.create(tmp_path / "runs", kind="toy", mock=True)
    assert ctx.mock is True
    ctx.write_json("metrics.json", {"mock": True, "value": 1})

    # bundle provenance is traceable
    build = build_toy(z_dim=16)
    collected = collect_toy(build, num_ticks=30, seed=0)
    learnable = collected.replay.learnable_transitions()
    assert learnable
    assert learnable[0].feature_version_id == MOCK_FEATURE_VERSION_ID
    assert learnable[0].norm_stats_sha256

    bundle = PolicyBundle(
        spec=build.spec,
        contract=build.contract,
        actor_state={},
        critic_state=None,
        feature_version_id=MOCK_FEATURE_VERSION_ID,
        norm_stats_sha256="mock",
        policy_version_id="toy",
        extra={"mock": True},
    )
    assert bundle.extra["mock"] is True


def test_t68_long_run_bounded_ledger_has_no_unexplained_loss():
    from pi05_rlt_rtc.contracts.spaces import HorizonContract
    from pi05_rlt_rtc.runtime.ledger import ExecutionLedger
    from pi05_rlt_rtc.runtime.queue import ActionQueue
    from pi05_rlt_rtc.runtime.scheduler import NumberedReferencePlanner, VirtualDelayScheduler
    from tests.helpers import RecordingEnv

    contract = HorizonContract(horizon=10, chunk=5, delay=2)
    ledger = ExecutionLedger(max_entries=200)
    queue = ActionQueue(contract, fallback_policy="stop", ledger=ledger)
    scheduler = VirtualDelayScheduler(
        contract=contract,
        queue=queue,
        backend=NumberedReferencePlanner(contract, unit=1e-6),
        env=RecordingEnv(),
        planner_latency_ticks=0,
        seed=0,
    )
    log = scheduler.run(1500)
    assert log.ticks == 1500
    assert len(ledger.entries) <= 200
    assert queue.cursor_tick == 1500
    trims = [e for e in ledger.events if e["event"] == "ledger_trimmed"]
    assert trims, "trimming must be explicit"
    assert sum(t["dropped"] for t in trims) == 1500 - len(ledger.entries)
    assert all(e.kind in ("committed", "stop", "hold_last", "zero_delta") for e in ledger.entries)
