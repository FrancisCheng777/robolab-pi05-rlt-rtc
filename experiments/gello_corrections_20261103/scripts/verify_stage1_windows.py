#!/usr/bin/env python3
"""CPU-only check of actual LeRobot loaders and Stage-1 worker batch construction.

The fixture must retain mock=true. Production loader rejection is tested before
explicit allow_mock diagnostic reads. No weights, CUDA workers or robot devices.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / 'stage1_integration'
sys.path[:0] = [str(ROOT), str(INTEGRATION), str(INTEGRATION / 'vendor/rlinf_stage1')]
import numpy as np
from pi05_rlt_rtc.data.physical_sft import PhysicalChunkDataset
from pi05_rlt_rtc.data.windows import WINDOW_POLICY
from pi05_rlt_rtc.adapters.panda_policy import PandaInputs
from pi05_demo.training_data import ChunkDataset
from stage1.data import PlugDataset
from stage1.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=INTEGRATION/'data/smoke_pipeline_update_fixture/dataset')
    parser.add_argument('--out', type=Path, default=ROOT/'reports/stage1_alignment/loader_integration.json')
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['mock'] is True, 'this verification entry accepts only labelled software fixtures'
    tokenizer = INTEGRATION/'stage1/assets/paligemma_tokenizer.model'
    stats = root/'norm_stats.json'
    report = {'kind':'cpu_mock_dataset_stage1_integration', 'mock':True,
              'hardware_opened':False, 'model_weights_loaded':False,
              'window_policy':WINDOW_POLICY, 'splits':{}}
    for split in ('train', 'validation'):
        factories = {
            'rlt': lambda allow: PhysicalChunkDataset(root/split, 10, norm_stats_path=stats, allow_mock=allow),
            'generic': lambda allow: ChunkDataset(root/split, 10, allow_mock=allow),
            'stage1': lambda allow: PlugDataset(str(root), str(tokenizer), 10, split=split, allow_mock=allow),
        }
        for name, factory in factories.items():
            try:
                factory(False)
            except ValueError:
                pass
            else:
                raise AssertionError(f'{name} accepted mock data for production')
        rlt, generic, stage1 = [factory(True) for factory in factories.values()]
        np.testing.assert_array_equal(rlt.indices, generic.starts)
        np.testing.assert_array_equal(rlt.indices, stage1.starts)
        assert rlt.window_policy == generic.window_policy == stage1.window_policy == WINDOW_POLICY
        for i in (0, len(rlt)//2, len(rlt)-1):
            a, b, c = rlt[i], generic[i], stage1[i]
            np.testing.assert_array_equal(a['action'], b['actions'])
            np.testing.assert_array_equal(c['actions'][:, :8], stage1.transform.normalize(a['action'], 'actions'))
            physical = PandaInputs()({'observation/state':a['observation.state'],
                'observation/image':a['observation.images.main'],
                'observation/wrist_image':a['observation.images.wrist'],
                'actions':a['action'], 'prompt':a['task']})
            expected = stage1.transform(physical)
            for key in ('state','tokenized_prompt','tokenized_prompt_mask'):
                np.testing.assert_array_equal(expected['observation'][key], c['observation'][key])
            for key in b['image']:
                np.testing.assert_array_equal(expected['observation']['image'][key], c['observation']['image'][key])
        starts = np.asarray(rlt.indices, dtype='<i8')
        report['splits'][split] = {'frames':len(rlt.ds), 'windows':len(rlt),
            'first_start':int(starts[0]), 'last_start':int(starts[-1]),
            'start_indices_sha256':hashlib.sha256(starts.tobytes()).hexdigest(),
            'all_three_loaders_agree':True, 'production_mock_gates_passed':True}
        if split == 'train':
            training_dataset = stage1
    # Corrupt only in-memory metadata on a marked fixture to verify all loader
    # constructors actually enforce continuity/interventions, not just helpers.
    base = training_dataset.ds
    class TableView:
        def __init__(self, overrides, missing=()):
            self.overrides = overrides
            self.column_names = [k for k in base.hf_dataset.column_names if k not in missing]
        def __getitem__(self, key):
            return self.overrides[key] if key in self.overrides else base.hf_dataset[key]
    class DatasetView:
        def __init__(self, table): self.hf_dataset = table
        def __len__(self): return len(base)
        def __getitem__(self, index): return base[index]
    flags = np.zeros(len(base), bool)
    flags[60] = True
    times = np.arange(len(base))/15
    gap = times.copy()
    gap[60:] += 1/15
    scenarios = {
        'intervention': ({'intervention': flags}, 101),
        'missing_sample_interval': ({'source_time': gap}, 102),
        'accumulated_drift': ({'source_time': times+np.arange(len(base))*.001}, 58),
    }
    constructors = [
        lambda: PhysicalChunkDataset(root/'train', 10, norm_stats_path=stats, allow_mock=True),
        lambda: ChunkDataset(root/'train', 10, allow_mock=True),
        lambda: PlugDataset(str(root), str(tokenizer), 10, split='train', allow_mock=True),
    ]
    report['adversarial_metadata'] = {}
    for name, (overrides, expected_count) in scenarios.items():
        with patch('lerobot.datasets.lerobot_dataset.LeRobotDataset', return_value=DatasetView(TableView(overrides))):
            datasets = [constructor() for constructor in constructors]
        indices = [dataset.indices if hasattr(dataset, 'indices') else dataset.starts for dataset in datasets]
        assert list(map(len, indices)) == [expected_count]*3
        for value in indices[1:]: np.testing.assert_array_equal(value, indices[0])
        report['adversarial_metadata'][name] = {'windows':expected_count, 'three_loaders_agree':True}
    with patch('lerobot.datasets.lerobot_dataset.LeRobotDataset', return_value=DatasetView(TableView({}, missing=('source_time',)))):
        for constructor in constructors:
            try:
                constructor()
            except ValueError as exc:
                assert 'source_time' in str(exc)
            else:
                raise AssertionError('missing receive timestamps were accepted')
    report['adversarial_metadata']['missing_source_time'] = {'rejected_by_all_three':True}
    # Use the real worker method, with only its dataset-construction boundary
    # explicitly opting into the already-marked mock fixture above.
    import stage1.worker as worker
    cfg = load_config([f'stage1.dataset_root={root}', f'stage1.tokenizer_path={tokenizer}'])
    loaders = []
    with patch.object(worker, 'PlugDataset', return_value=training_dataset) as factory:
        for rank in range(2):
            instance = SimpleNamespace(cfg=cfg, _world_size=2, _rank=rank)
            loader, _ = worker.PlugSftWorker.build_dataloader(instance, cfg.data.train_data_paths)
            batch = next(iter(loader))
            assert tuple(batch['actions'].shape) == (1,10,32)
            assert tuple(batch['observation']['state'].shape) == (1,32)
            assert batch['observation']['tokenized_prompt'].shape[-1] == 200
            loaders.append(loader)
        assert factory.call_count == 2
    ranks = [list(loader.sampler) for loader in loaders]
    assert set(ranks[0]).isdisjoint(ranks[1])
    assert len(ranks[0]) == len(ranks[1]) == len(training_dataset)//2
    report['worker'] = {'class':worker.PlugSftWorker.__name__,
        'module':worker.__file__, 'real_stateful_dataloader':True,
        'ranks_emulated_on_cpu':2, 'samples_per_rank':list(map(len,ranks)),
        'dropped_for_even_sharding':len(training_dataset)-sum(map(len,ranks)),
        'rank_indices_disjoint':True, 'batch_action_shape':[1,10,32],
        'fixture_boundary_override':True, 'gpu_training_executed':False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
