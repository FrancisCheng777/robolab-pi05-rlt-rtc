"""CPU checks for the production Stage-1 topology contract."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1_integration"))
from stage1.config import load_config, validate_contract


@pytest.mark.parametrize("world,micro,batch,accum", [(2, 1, 64, 32), (4, 1, 64, 16), (4, 4, 64, 4), (4, 8, 64, 2), (4, 16, 64, 1), (8, 2, 64, 4)])
def test_supported_topologies(world, micro, batch, accum):
    cfg = load_config([
        f"stage1.expected_world_size={world}",
        f"cluster.component_placement.actor=0-{world - 1}",
        f"actor.micro_batch_size={micro}",
        f"actor.global_batch_size={batch}",
    ])
    result = validate_contract(cfg)
    assert result["world_size"] == world
    assert result["accumulation_steps"] == accum


@pytest.mark.parametrize("overrides,match", [
    (["cluster.component_placement.actor=0-1"], "placement"),
    (["stage1.expected_world_size=1"], "workers"),
    (["actor.global_batch_size=65"], "divisible"),
    (["actor.fsdp_config.enable_gradient_accumulation=true"], "sharded accumulated"),
])
def test_invalid_topologies_fail_before_launch(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_contract(load_config(overrides))


def test_realworld_profile_keeps_official_optimizer_and_panda_contract(monkeypatch):
    monkeypatch.delenv("PI05_CONFIG_NAME", raising=False)
    monkeypatch.delenv("PI05_WORLD_SIZE", raising=False)
    cfg = load_config()
    assert validate_contract(cfg)["world_size"] == 4
    assert cfg.actor.global_batch_size == 64
    assert cfg.actor.micro_batch_size == 16
    assert validate_contract(cfg)["accumulation_steps"] == 1
    assert cfg.actor.optim.lr == 1e-5
    assert cfg.actor.optim.lr_warmup_steps == 1000
    assert cfg.actor.optim.total_training_steps == 20000
    assert cfg.runner.max_steps == 8000
    assert cfg.actor.model.action_dim == 8
    assert cfg.actor.model.openpi.action_horizon == 10
    assert cfg.actor.model.openpi.num_images_in_input == 1


@pytest.mark.parametrize("world", [2, 4, 8])
def test_world_size_drives_placement_and_accumulation(monkeypatch, world):
    monkeypatch.setenv("PI05_WORLD_SIZE", str(world))
    cfg = load_config(["actor.micro_batch_size=1"])
    assert str(cfg.cluster.component_placement.actor) == f"0-{world-1}"
    assert validate_contract(cfg)["accumulation_steps"] == 64 // world


@pytest.mark.parametrize("alias,world", [("stage1_2xa800", 2), ("stage1_4xa800_realworld", 4)])
def test_topology_aliases_share_one_realworld_recipe(monkeypatch, alias, world):
    monkeypatch.setenv("PI05_CONFIG_NAME", alias)
    cfg = load_config()
    assert validate_contract(cfg)["world_size"] == world
    assert cfg.actor.global_batch_size == 64
    assert cfg.actor.optim.lr == 1e-5


def test_resume_allows_extending_budget_but_rejects_changed_recipe():
    from stage1.config import validate_resume_config
    saved = load_config()
    current = load_config(["runner.max_steps=12000"])
    validate_resume_config(current, saved)
    for override in ["actor.optim.lr=2.5e-5", "actor.optim.total_training_steps=8000",
                     "actor.global_batch_size=128", "actor.micro_batch_size=2",
                     "stage1.expected_world_size=2"]:
        with pytest.raises(ValueError, match="Resume configuration changed"):
            validate_resume_config(load_config([override]), saved)
