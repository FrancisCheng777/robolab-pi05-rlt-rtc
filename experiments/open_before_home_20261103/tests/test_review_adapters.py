"""CPU boundary doubles matching the pinned upstream API; not simulator/GPU tests."""

import dataclasses
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from pi05_rlt_rtc import PLANNING_STATE_VERSION
from pi05_rlt_rtc.adapters.rlinf_feature import RLinfFeatureReferenceService
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.plan import PrevPlanContext
from pi05_rlt_rtc.rtc import RTCDispatcher
from tests.helpers import RecordingEnv, make_request


@pytest.fixture
def rlinf_boundary(monkeypatch, contract, codec):
    module = ModuleType("rlinf.models.embodiment.openpi.modules.model")
    module.preprocess_observation = lambda observation, train: observation
    monkeypatch.setitem(sys.modules, module.__name__, module)
    transforms = ModuleType("rlinf.models.embodiment.openpi.transforms.apply")
    transforms.apply_output_transform = lambda fn, outputs, action_chunk: {"actions": outputs["actions"][..., :8].detach().cpu()}
    monkeypatch.setitem(sys.modules, transforms.__name__, transforms)

    class Model(torch.nn.Module):
        action_dim = 32
        action_horizon = 10
        _output_transform_fn = None

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))
            self.prefix_calls = 0

        def env_obs_to_observation(self, obs):
            return SimpleNamespace(state=torch.tensor(obs["states"].copy()), tokenized_prompt=None)

        def build_prefix_cache(self, obs):
            self.prefix_calls += 1
            return torch.zeros(1, 2, 16), torch.ones(1, 2, dtype=torch.bool), ()

        def _select_rlt_prefix_embeddings(self, output, mask, prompt):
            return output, mask

        def _encode_rlt_flat(self, output, mask):
            return output[:, 0]

        def run_suffix(self, obs, x, t, cache, mask):
            assert t.device == x.device
            return x * self.weight

        def velocity_from_suffix(self, suffix):
            return suffix

    class Dispatcher(RTCDispatcher):
        def sample(self, *args, **kwargs):
            self.last_input = kwargs
            return super().sample(*args, **kwargs)

    model = Model()
    dispatcher = Dispatcher(num_steps=5)
    service = RLinfFeatureReferenceService(model=model, contract=contract, codec=codec, rtc_dispatcher=dispatcher, feature_version_id="feature-v1", norm_stats_sha256=codec.norm_stats.sha256, planning_state_version=PLANNING_STATE_VERSION)
    return service, model, dispatcher


@pytest.mark.parametrize("mode", ["none", "approx", "full_vjp"])
def test_first_real_adapter_request_has_no_previous_plan(rlinf_boundary, contract, mode):
    service, model, _ = rlinf_boundary
    request = dataclasses.replace(make_request(contract, request_id=0, capture_tick=0), sampler_mode=mode)
    result = service.extract(request, RecordingEnv().reset())
    assert result.reference_env_full.shape == (1, 10, 8)
    assert model.prefix_calls == 1


def test_feature_service_freezes_stage_two_model(rlinf_boundary):
    _, model, _ = rlinf_boundary
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_adapter_honors_previous_valid_mask(rlinf_boundary, contract):
    service, _, dispatcher = rlinf_boundary
    prev = PrevPlanContext(np.ones((5, 8), np.float32), np.array([True, False, True, True, False]), 0, 1, "feature-v1")
    request = make_request(contract, request_id=1, capture_tick=5, prev=prev)
    observation = dataclasses.replace(RecordingEnv().reset(), tick=5)
    service.extract(request, observation)
    mask = dispatcher.last_input["guidance_mask"]
    assert torch.count_nonzero(mask[:, [1, 4]]) == 0
    assert torch.count_nonzero(mask[..., 8:]) == 0


def test_request_cannot_misreport_sampler_step_count(rlinf_boundary, contract):
    service, _, _ = rlinf_boundary
    request = dataclasses.replace(make_request(contract, request_id=0, capture_tick=0), num_denoising_steps=10)
    with pytest.raises(ContractError):
        service.extract(request, RecordingEnv().reset())


def test_request_cannot_use_another_feature_version(rlinf_boundary, contract):
    service, _, _ = rlinf_boundary
    request = make_request(contract, request_id=0, capture_tick=0, feature_id="other-checkpoint")
    with pytest.raises(ContractError):
        service.extract(request, RecordingEnv().reset())


def test_approx_respects_padding_mask(contract):
    dispatcher = RTCDispatcher(num_steps=1)
    x = torch.ones(1, 10, 32)
    mask = torch.ones_like(x)
    mask[..., 8:] = 0
    result = dispatcher.sample("approx", x_init=x, velocity_native=lambda x, t: torch.zeros_like(x), prev_model_actions=torch.zeros_like(x), guidance_mask=mask, delay_steps=2)
    assert torch.equal(result.actions[..., 8:], x[..., 8:])
