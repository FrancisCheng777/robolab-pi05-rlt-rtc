"""Model-factory dependency boundary; uses CPU API doubles, not π0.5 weights."""
import dataclasses
import sys
from types import ModuleType, SimpleNamespace

import pytest

from pi05_rlt_rtc.config.runtime_config import load_runtime_config
from pi05_rlt_rtc.contracts.errors import ContractError
from tests.test_review_adapters import rlinf_boundary  # noqa: F401 - pytest fixture registration


def test_factory_freezes_model_and_binds_weight_stats_and_transform_hashes(tmp_path, monkeypatch, rlinf_boundary):  # noqa: F811
    from pi05_rlt_rtc.deployment import model_loader as loader
    _, model, _ = rlinf_boundary
    model.rlt_module = SimpleNamespace(z_dim=16)
    weights = tmp_path / "full_weights.pt"; weights.write_bytes(b"test-double-weights")
    stats = tmp_path / "norm_stats.json"; stats.write_text('{"test_double": 1}')
    transform_source = tmp_path / "transforms.py"; transform_source.write_text("# CPU test double")
    cfg = load_runtime_config("configs/runtime_franka.yaml")
    raw = dict(cfg.raw)
    raw["model"] = {**raw["model"], "rlinf_root": str(tmp_path), "checkpoint": str(weights),
                    "norm_stats": str(stats), "device": "cpu", "z_dim": 16,
                    "architecture": {"rlt_embed_dim": 16}}
    cfg = dataclasses.replace(cfg, raw=raw)
    from pi05_rlt_rtc.deployment import assets
    monkeypatch.setattr(assets, "validate_assets", lambda _: {
        "tokenizer": "test-double", "tokenizer_sha256": "test-double", "source": {"test_double": True}})
    monkeypatch.setattr(loader, "activate_rlinf", lambda _: None)
    monkeypatch.setattr(loader, "register_panda_profile", lambda **kw: None)
    modules = {}
    for name in ("openpi", "openpi.transforms", "omegaconf", "rlinf.models.embodiment.openpi",
                 "rlinf.models.embodiment.openpi.checkpoint", "rlinf.models.embodiment.openpi.dataconfig"):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    modules["openpi"].transforms = modules["openpi.transforms"]
    modules["openpi.transforms"].__file__ = str(transform_source)
    modules["openpi.transforms"].Normalize = lambda *a, **kw: lambda data: data
    modules["omegaconf"].OmegaConf = SimpleNamespace(create=lambda data: data)
    seen = []
    modules["rlinf.models.embodiment.openpi"].get_model = lambda data: seen.append(data) or model
    modules["rlinf.models.embodiment.openpi.checkpoint"].resolve_full_weights = lambda _: weights
    data_cfg = SimpleNamespace(norm_stats={"state": {}, "actions": {}}, use_quantile_norm=True)
    modules["rlinf.models.embodiment.openpi.dataconfig"].get_openpi_config = lambda *a, **kw: SimpleNamespace(
        data=SimpleNamespace(create=lambda *a: data_cfg), assets_dirs=tmp_path, model=None)
    planner, first = loader.load_planner(cfg)
    assert planner.actor is None and not model.training and not model.weight.requires_grad
    assert seen[-1]["openpi"]["task"] == "eval" and seen[-1]["openpi"]["use_rlt"] is True
    assert seen[-1]["openpi"]["action_chunk"] == cfg.contract.horizon
    assert seen[-1]["openpi"]["torch_compile"] is False
    assert first["policy_kind"] == "teacher" and first["mock"] is False
    stats.write_text('{"test_double": 2}')
    _, second = loader.load_planner(cfg)
    assert second["feature_version_id"] != first["feature_version_id"]
    transform_source.write_text("# changed test-double normalization")
    _, third = loader.load_planner(cfg)
    assert third["feature_version_id"] != second["feature_version_id"]
    model.rlt_module.z_dim = 999
    with pytest.raises(ContractError, match="z_dim"):
        loader.load_planner(cfg)
