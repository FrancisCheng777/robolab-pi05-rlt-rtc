"""Compare queue-conditioned AC against the pinned upstream building blocks."""
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from pi05_rlt_rtc.learning.actor import QueueAwareRLTActor
from pi05_rlt_rtc.learning.critic import TwinQCritic
from pi05_rlt_rtc.learning.planning_state import PlanningStateSpec


def upstream_modules():
    # Load only the torch modules, without importing RLinf's distributed runtime.
    name = '_pinned_rlt_ac_modules'
    package = types.ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] /
        'stage1_integration/vendor/rlinf_stage1/rlinf/models/embodiment/modules')]
    sys.modules[name] = package
    return (importlib.import_module(name + '.utils'),
            importlib.import_module(name + '.q_head'))


def test_actor_matches_upstream_mlp_initialization_and_forward_with_queue_inputs():
    utils, _ = upstream_modules()
    spec = PlanningStateSpec(z_dim=16)
    state_dim = sum(d for _, d in spec.actor_state_layout())
    ref_dim = spec.chunk * spec.env_action_dim
    torch.manual_seed(81)
    actor = QueueAwareRLTActor(spec)
    torch.manual_seed(81)
    oracle = nn.Sequential(
        utils.layer_init(nn.Linear(state_dim + ref_dim, 256)), nn.Tanh(),
        utils.layer_init(nn.Linear(256, 256)), nn.Tanh(),
        utils.layer_init(nn.Linear(256, 256)), nn.Tanh(),
        utils.layer_init(nn.Linear(256, ref_dim), std=.01 * 2 ** .5))
    state, ref = torch.randn(3, state_dim), torch.randn(3, ref_dim)
    assert torch.equal(actor(state, ref).raw.flatten(1), oracle(torch.cat([state, ref], -1)))
    original = state.clone()
    dropped = actor(state, ref, apply_reference_dropout=True, reference_dropout_prob=1.)
    assert torch.equal(dropped.raw.flatten(1), oracle(torch.cat([state, torch.zeros_like(ref)], -1)))
    assert torch.equal(state, original)
    changed = state.clone()
    # Change pending prefix, leaving z_rl and proprio unchanged.
    changed[:, spec.z_dim + spec.proprio_dim] += 1
    assert not torch.equal(dropped.raw, actor(changed, torch.zeros_like(ref)).raw)


def test_critic_matches_upstream_initialization_and_forward():
    _, q = upstream_modules()
    torch.manual_seed(82)
    critic = TwinQCritic(35, 40)
    torch.manual_seed(82)
    heads = [q.QHead(hidden_size=35, action_feature_dim=40,
                    hidden_dims=[256, 256, 256], output_dim=1) for _ in range(2)]
    state, action = torch.randn(3, 35), torch.randn(3, 40)
    assert torch.equal(critic(state, action), torch.cat([h(state, action) for h in heads], -1))


def test_old_stage2_bundle_is_rejected_before_build(tmp_path):
    from pi05_rlt_rtc.learning.checkpoint import PolicyBundle, state_dict_sha256
    from pi05_rlt_rtc.contracts.errors import ContractError
    state = {'head.bias': torch.zeros(40)}
    path = tmp_path / 'old.pt'
    torch.save({'actor_state': state, 'actor_state_sha256': state_dict_sha256(state)}, path)
    with pytest.raises(ContractError, match='architecture'):
        PolicyBundle.load(path)
