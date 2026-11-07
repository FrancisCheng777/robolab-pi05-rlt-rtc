"""Replay the unchanged frozen prefix on CUDA, avoiding eager launch jitter.

Only used by the single-worker deployment service. Shapes and tensor layouts
are fixed; tensor values (including images, masks and prompt tokens) are copied
on every call. Outputs are cloned so a later replay cannot overwrite a caller's
KV cache. Capture is warmed and checked for exact equality before use.
"""
from __future__ import annotations

import dataclasses
import threading


class FrozenPrefixGraph:
    def __init__(self, forward):
        self.forward = forward
        self.validated = False
        self._lock = threading.Lock()
        self._graph = None

    @staticmethod
    def _flatten(observation):
        from torch.utils._pytree import tree_flatten
        if not dataclasses.is_dataclass(observation):
            return tree_flatten({'input': observation})
        return tree_flatten({f.name: getattr(observation, f.name)
                             for f in dataclasses.fields(observation)})

    @staticmethod
    def _signature(leaves, spec):
        import torch
        tensors = [x for x in leaves if torch.is_tensor(x)]
        if not tensors or any(x.device.type != 'cuda' for x in tensors):
            raise ValueError('prefix graph requires CUDA tensor inputs')
        if len({x.device for x in tensors}) != 1:
            raise ValueError('prefix graph requires one CUDA device')
        return (spec, tuple((tuple(x.shape), x.dtype, x.device, tuple(x.stride()))
                            if torch.is_tensor(x) else (type(x), x) for x in leaves))

    def __call__(self, observation):
        import torch
        from torch.utils._pytree import tree_map, tree_unflatten, tree_flatten
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('prefix CUDA graph cannot serve concurrent requests')
        try:
            leaves, spec = self._flatten(observation)
            signature = self._signature(leaves, spec)
            with torch.no_grad():
                if self._graph is None:
                    self._signature_value = signature
                    self._static_leaves = [x.clone() if torch.is_tensor(x) else x for x in leaves]
                    values = tree_unflatten(self._static_leaves, spec)
                    self._static = (dataclasses.replace(observation, **values)
                                    if dataclasses.is_dataclass(observation) else values['input'])
                    device = next(x.device for x in leaves if torch.is_tensor(x))
                    with torch.cuda.device(device):
                        current = torch.cuda.current_stream()
                        stream = torch.cuda.Stream()
                        stream.wait_stream(current)
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                expected = self.forward(self._static)
                        current.wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            self._output = self.forward(self._static)
                        graph.replay()
                        expected_leaves, expected_spec = tree_flatten(expected)
                        actual_leaves, actual_spec = tree_flatten(self._output)
                        if expected_spec != actual_spec or not all(
                            torch.equal(a, b) if torch.is_tensor(a) else a == b
                            for a, b in zip(expected_leaves, actual_leaves)
                        ):
                            raise RuntimeError('prefix CUDA graph changed frozen outputs')
                        self._graph = graph
                        self.validated = True
                else:
                    if signature != self._signature_value:
                        raise ValueError('prefix CUDA graph input signature changed; restart and warm the model')
                    for target, source in zip(self._static_leaves, leaves):
                        if torch.is_tensor(target):
                            target.copy_(source)
                    self._graph.replay()
                return tree_map(lambda x: x.clone() if torch.is_tensor(x) else x, self._output)
        finally:
            self._lock.release()
