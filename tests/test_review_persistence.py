import json
from pathlib import Path

import pytest
import torch

from pi05_rlt_rtc.evaluation.toy_pipeline import build_toy, export_toy_bundle
from pi05_rlt_rtc.learning.actor import QueueAwareRLTActor
from pi05_rlt_rtc.learning.checkpoint import PolicyBundle
from pi05_rlt_rtc.learning.critic import TwinQWithTargets
from pi05_rlt_rtc.learning.replay import ReplayBuffer


@pytest.fixture
def bundle(tmp_path):
    build = build_toy(z_dim=16)
    actor = QueueAwareRLTActor(build.spec, hidden_dims=(16,))
    critic = TwinQWithTargets(sum(d for _, d in build.spec.critic_state_layout()), 40, hidden_dims=(12,))
    result = export_toy_bundle(build=build, actor=actor, critic=critic, out_dir=str(tmp_path / "source"))
    return result


def test_critic_architecture_survives_save_reload(bundle, tmp_path):
    saved = PolicyBundle.load(bundle.save(tmp_path / "policy.pt"))
    restored = saved.build_critic()
    for key, value in restored.state_dict().items():
        assert torch.equal(value, bundle.critic_state[key])


def test_actor_and_critic_outputs_survive_save_reload(bundle, tmp_path):
    saved = PolicyBundle.load(bundle.save(tmp_path / "policy.pt"))
    state = torch.randn(3, sum(d for _, d in bundle.spec.actor_state_layout()))
    reference = torch.randn(3, bundle.spec.chunk * bundle.spec.env_action_dim)
    actor = bundle.build_actor()
    assert torch.equal(actor(state, reference).actions,
                       saved.build_actor()(state, reference).actions)
    qstate = torch.randn(3, sum(d for _, d in bundle.spec.critic_state_layout()))
    assert torch.equal(bundle.build_critic().online(qstate, reference),
                       saved.build_critic().online(qstate, reference))


def test_cli_deployment_export_is_independently_loadable(bundle, tmp_path):
    from pi05_rlt_rtc.tools.cli import main
    source = bundle.save(tmp_path / "policy.pt")
    out = tmp_path / "deployment"
    assert main(["export-deployment", "--toy", "--policy", str(source), "--out", str(out)]) == 0
    exported = PolicyBundle.load(out / "actor.pt")
    assert exported.critic_state is None
    assert exported.actor_state_sha256 == bundle.actor_state_sha256
    assert exported.extra["rtc_mode"] == "none"
    assert exported.build_actor().hidden_dims == bundle.actor_hidden_dims
    manifest = json.loads((out / "deployment_manifest.json").read_text())
    assert manifest["contains_critic"] is False
    assert manifest["mock"] is True


def test_cli_evaluation_uses_saved_layout_and_rtc_mode(bundle, tmp_path, monkeypatch):
    from pi05_rlt_rtc.tools.cli import main
    from pi05_rlt_rtc.evaluation import toy_pipeline
    seen = []
    original = toy_pipeline.build_toy
    def capture_build(**kwargs):
        result = original(**kwargs)
        seen.append(result)
        return result
    monkeypatch.setattr(toy_pipeline, "build_toy", capture_build)
    bundle.extra["rtc_mode"] = "full_vjp"
    source = bundle.save(tmp_path / "policy.pt")
    assert main(["evaluate", "--toy", "--policy", str(source), "--episodes", "1", "--out", str(tmp_path / "eval.json")]) == 0
    assert seen[0].spec == bundle.spec
    assert seen[0].backend.mode == "full_vjp"


@pytest.mark.parametrize("kind", ["policy", "replay"])
def test_failed_checkpoint_write_preserves_previous_file(bundle, tmp_path, monkeypatch, kind):
    obj = bundle if kind == "policy" else ReplayBuffer(capacity=100, feature_version_id="feature-v1", norm_stats_sha256="norm-v1")
    target = obj.save(tmp_path / "state.pt")
    original = target.read_bytes()

    def interrupted_save(payload, destination):
        if hasattr(destination, "write"):
            destination.write(b"incomplete")
        else:
            Path(destination).write_bytes(b"incomplete")
        raise OSError("injected disk failure")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError):
        obj.save(target)
    assert target.read_bytes() == original


def test_test_ledger_cannot_hide_failure_of_unmapped_regression(tmp_path, monkeypatch):
    from pi05_rlt_rtc.tools import test_ledger
    def failing_run(log_path, **kwargs):
        log_path.with_suffix(".ledger.json").write_text(json.dumps({"results": {"T05": {"status": "PASSED", "nodeid": "fixture", "reason": None}}}))
        return 1  # a supplementary regression test failed
    monkeypatch.setattr(test_ledger, "run_pytest", failing_run)
    monkeypatch.setattr(test_ledger, "write_markdown", lambda *args: None)
    assert test_ledger.main(["--out", str(tmp_path / "status.json"), "--log", str(tmp_path / "pytest.log")]) == 1
