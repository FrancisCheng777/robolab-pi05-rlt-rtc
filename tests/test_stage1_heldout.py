"""Exercise held-out loss aggregation and disabled augmentation without a VLA."""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage1_integration'))
from stage1.evaluate import evaluate_batches


@pytest.mark.parametrize("limit", [None, 2])
def test_heldout_weights_unequal_batches_and_never_augments(monkeypatch, limit):
    module = ModuleType('rlinf.models.embodiment.openpi.modules.model')
    modes = []
    module.preprocess_observation = lambda obs, train: modes.append(train) or obs
    monkeypatch.setitem(sys.modules, module.__name__, module)
    class Model:
        device = torch.device('cpu')
        def eval(self): return self
        def requires_grad_(self, value): assert value is False; return self
        def _unpack_sft_batch(self, batch): return SimpleNamespace(tokenized_prompt=None), batch
        def _observation_to_device(self, obs): return obs
        def _actions_to_device(self, actions): return actions
        def compute_loss(self, obs, actions, train):
            assert not torch.is_grad_enabled(); modes.append(train); return actions
        def build_prefix_cache(self, obs): return torch.zeros(1), None, None
        def _select_rlt_prefix_embeddings(self, prefix, mask, tokens): return prefix, mask
        def _rlt_forward(self, prefix, mask): return torch.tensor(2.), {}
    result = evaluate_batches(Model(), [torch.ones(2,10,32), torch.full((1,10,32), 4.)], max_batches=limit)
    assert result['examples'] == 3
    assert result['flow_mse_physical8'] == result['flow_mse_all32'] == 2.
    assert result['rlt_reconstruction_loss'] == 2.
    assert modes == [False] * 4
    with pytest.raises(ValueError, match='no complete'):
        evaluate_batches(Model(), [], max_batches=1)
