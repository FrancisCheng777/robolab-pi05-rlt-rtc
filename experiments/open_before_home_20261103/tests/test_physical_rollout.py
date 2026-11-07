"""Exercise physical trial recording/annotation/training with explicit CPU doubles."""
import copy
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_review_controller import manual_runtime
from tests.test_platform_integration import backend_double
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.deployment import runtime
from pi05_rlt_rtc.deployment.rollout import load_rollout, label_rollout, annotate


@pytest.fixture
def recorded_trial(tmp_path, monkeypatch, manual_runtime):
    from pi05_rlt_rtc.evaluation.toy_env import ToyDelayedControlEnv
    cfg, toy, backend, metadata = backend_double()
    cfg = dataclasses.replace(cfg, raw={**cfg.raw, "model": {**cfg.raw["model"], "z_dim": 16}})
    class BoundaryEnv(ToyDelayedControlEnv):
        stop_request_acknowledged = True
        def close(self): pass
    monkeypatch.setattr(runtime, "make_env", lambda _: BoundaryEnv())
    runtime.collect(cfg, backend, metadata, ticks=30, seed=4, out=tmp_path / "trial")
    return cfg, tmp_path / "trial/rollout.pt"


def test_only_human_success_receives_terminal_reward(recorded_trial):
    cfg, path = recorded_trial
    data = load_rollout(path)
    assert data["report"]["annotation_required"]
    replay, annotation = label_rollout(data, outcome="success", annotator="test operator")
    transitions = replay.learnable_transitions()
    assert len(transitions) >= 2
    assert sum(float(t.primitive_rewards.sum()) for t in transitions) == 1
    assert transitions[-1].terminated and not transitions[-1].bootstrap_allowed
    assert annotation["terminal_tick"] == data["report"]["ticks_run"] - 1
    assert all(e.get("observation") is None for e in data["events"] if e["kind"] == "request")
    failed, _ = label_rollout(data, outcome="failure", annotator="test operator")
    assert sum(float(t.primitive_rewards.sum()) for t in failed.learnable_transitions()) == 0


@pytest.mark.parametrize("outcome", ["aborted", "intervened"])
def test_intervention_never_becomes_policy_replay(recorded_trial, outcome):
    _, path = recorded_trial
    replay, annotation = label_rollout(load_rollout(path), outcome=outcome, annotator="test")
    assert len(replay) == 0 and not annotation["eligible"]


def test_faults_and_unknown_terminal_ticks_cannot_be_relabelled_success(recorded_trial):
    _, path = recorded_trial
    data = load_rollout(path)
    with pytest.raises(ContractError, match="executed tick"):
        label_rollout(data, outcome="success", annotator="test", terminal_tick=300)
    data["report"]["safety_stops"] = 1
    with pytest.raises(ContractError, match="faulted"):
        label_rollout(data, outcome="success", annotator="test")


def test_labelled_replay_trains_and_resumes_without_model_or_robot(recorded_trial, tmp_path, monkeypatch):
    from pi05_rlt_rtc.deployment import training
    cfg, path = recorded_trial
    labelled = tmp_path / "labelled"
    annotate(SimpleNamespace(rollout=str(path), out=str(labelled), outcome="success", annotator="test", terminal_tick=None))
    monkeypatch.setattr(training, "config_from_args", lambda args: cfg)
    monkeypatch.setattr(training, "make_backend", lambda *a: pytest.fail("learner must not load VLA"))
    monkeypatch.setattr(runtime, "make_env", lambda *a: pytest.fail("learner must not open a robot"))
    args = SimpleNamespace(replay=[str(labelled)], out=str(tmp_path / "train"), resume=None,
                           learner_device="cpu", seed=0, warmup_updates=1, updates=1)
    assert training.train(args) == 0
    args.resume, args.out, args.warmup_updates = str(tmp_path / "train/latest"), str(tmp_path / "resume"), 0
    assert training.train(args) == 0


def test_panda_data_transform_preserves_normalized_physical_actions():
    from pi05_rlt_rtc.adapters.panda_policy import PandaInputs, PandaOutputs
    action = np.linspace(-1, 1, 80).reshape(10, 8).astype(np.float32)
    sample = {"observation/state": np.zeros(9), "observation/image": np.zeros((3, 224, 224)),
              "observation/wrist_image": np.ones((3, 224, 224)), "actions": action}
    output = PandaInputs()(sample)
    np.testing.assert_array_equal(output["actions"], action)
    assert output["image"]["base_0_rgb"].max() == 255
    assert not output["image_mask"]["right_wrist_0_rgb"]
    padded = np.pad(action, ((0,0),(0,24)))
    np.testing.assert_array_equal(PandaOutputs()({"actions": padded})["actions"], action)
    del sample["observation/wrist_image"]
    assert PandaInputs()(sample)["image"]["base_0_rgb"].max() == 0
    del sample["observation/image"]
    with pytest.raises(KeyError):
        PandaInputs()(sample)


def test_training_rejects_replaced_labelled_replay(recorded_trial, tmp_path, monkeypatch):
    from pi05_rlt_rtc.deployment import training
    cfg, path = recorded_trial
    labelled = tmp_path / 'labelled'
    annotate(SimpleNamespace(rollout=str(path), out=str(labelled), outcome='success', annotator='test', terminal_tick=None))
    with (labelled / 'replay.pt').open('ab') as stream:
        stream.write(b'changed')
    monkeypatch.setattr(training, 'config_from_args', lambda args: cfg)
    args = SimpleNamespace(replay=[str(labelled)], out=str(tmp_path / 'train'), resume=None,
                           learner_device='cpu', seed=0, warmup_updates=1, updates=1)
    with pytest.raises(ContractError, match='replay fingerprint'):
        training.train(args)
