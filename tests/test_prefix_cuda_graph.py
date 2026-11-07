"""CUDA replay must preserve frozen features and own its returned tensors."""
import dataclasses
import pytest

torch = pytest.importorskip('torch')
from pi05_rlt_rtc.deployment.prefix_cuda_graph import FrozenPrefixGraph

@dataclasses.dataclass
class Observation:
    images: dict
    mask: object
    optional: object = None


def test_requires_cuda_inputs():
    fn = FrozenPrefixGraph(lambda obs: obs.images['wrist'] * 2)
    with pytest.raises(ValueError, match='CUDA'):
        fn(Observation({'wrist': torch.ones(2)}, torch.ones(2, dtype=torch.bool)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph requires GPU')
def test_replay_updates_inputs_exactly_and_does_not_mutate_previous_outputs():
    def prefix(obs):
        x = obs.images['wrist']
        return (x.sin() + x.square(), obs.mask.clone(), (x @ x.T,))
    fn = FrozenPrefixGraph(prefix)
    a = Observation({'wrist': torch.arange(16, device='cuda', dtype=torch.float32).reshape(4, 4)},
                    torch.ones(4, device='cuda', dtype=torch.bool))
    with torch.no_grad():
        expected = prefix(a)
        first = fn(a)
        saved = first[0].clone()
        b = Observation({'wrist': a.images['wrist'] + 2}, ~a.mask)
        second = fn(b)
        assert torch.equal(first[0], expected[0])
        assert torch.equal(first[0], saved)
        assert torch.equal(second[0], prefix(b)[0])
        assert torch.equal(second[1], b.mask)
        assert fn.validated
        with pytest.raises(ValueError, match='signature'):
            fn(Observation({'wrist': torch.ones(3, 4, device='cuda')}, b.mask))
        with pytest.raises(ValueError, match='signature'):
            fn(Observation({'different_camera': b.images['wrist']}, b.mask))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph requires GPU')
def test_tensor_camera_replay_runs_without_reentering_python_forward():
    calls = []
    def vision(image):
        calls.append(1)
        return image.square(), image.mean(dim=-1)
    fn = FrozenPrefixGraph(vision)
    a = torch.ones((1, 4, 4, 3), device='cuda')
    first = fn(a)
    captured_calls = len(calls)
    second = fn(a * 3)
    third = fn(a * 0)
    assert len(calls) == captured_calls
    assert torch.equal(first[0], a)
    assert torch.equal(second[0], a * 9)
    assert torch.equal(third[0], a * 0)
    with pytest.raises(ValueError, match='signature'):
        fn(a.to(torch.float64))
