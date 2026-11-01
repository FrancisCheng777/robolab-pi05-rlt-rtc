#!/usr/bin/env python3
"""Compare actual OpenPI transforms with the current Stage-1 transform on CPU.

Uses synthetic arrays and existing train/smoke statistics, never model weights,
camera devices, GPU inference or a robot socket.
"""
import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-root', type=Path, default=ROOT/'stage1_integration', help='Stage-1 implementation root')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import openpi.transforms as transforms
    from openpi.training.config import AssetsConfig
    from pi05_rlt_rtc.adapters.panda_policy import panda_train_config

    other, dataset = args.collection_root.resolve(), args.dataset.resolve()
    sys.path.insert(0, str(other))
    spec = importlib.util.spec_from_file_location('reference_stage1_data', other / 'stage1/data.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    tokenizer = other / 'stage1/assets/paligemma_tokenizer.model'
    stats = json.loads((dataset / 'norm_stats.json').read_text())
    reference_transform = reference.Transform(stats, str(tokenizer), max_len=200)
    config = panda_train_config(tokenizer_path=tokenizer)
    factory = dataclasses.replace(config.data, assets=AssetsConfig(
        assets_dir=str(dataset.parent), asset_id=dataset.name))
    data = factory.create(config.assets_dirs, config.model)
    assert data.use_quantile_norm
    pipeline = transforms.compose([*data.data_transforms.inputs,
        transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
        *data.model_transforms.inputs])
    rng = np.random.default_rng(20261029)
    maximum = {'state': 0., 'actions': 0.}
    for _ in range(32):
        lo = np.asarray(stats['norm_stats']['state']['q01'])
        hi = np.asarray(stats['norm_stats']['state']['q99'])
        state = rng.uniform(lo, hi).astype(np.float32)
        action = rng.uniform(-1, 1, (10, 8)).astype(np.float32)
        from pi05_rlt_rtc.data.profile import model_images
        images, masks = model_images(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
        raw = {'state': state, 'actions': action, 'image': images,
               'image_mask': masks, 'prompt': 'insert the plug into the power strip'}
        expected = reference_transform(copy.deepcopy(raw))
        actual = pipeline({'observation/state': state, 'actions': action,
            'observation/wrist_image': images['base_0_rgb'],
            'prompt': raw['prompt']})
        for key, wanted in [('state', expected['observation']['state']), ('actions', expected['actions'])]:
            maximum[key] = max(maximum[key], float(np.max(np.abs(actual[key] - wanted))))
            np.testing.assert_allclose(actual[key], wanted, rtol=2e-5, atol=2e-5)
        for key in ('tokenized_prompt', 'tokenized_prompt_mask'):
            np.testing.assert_array_equal(actual[key], expected['observation'][key])
        for key in images:
            np.testing.assert_array_equal(actual['image'][key], expected['observation']['image'][key])
            np.testing.assert_array_equal(actual['image_mask'][key], expected['observation']['image_mask'][key])
    report = {'kind': 'cpu_synthetic_transform_equivalence', 'samples': 32, 'passed': True,
        'hardware_opened': False, 'model_weights_loaded': False, 'max_absolute_error': maximum,
        'tokens_masks_images_equal': True, 'dataset': str(dataset),
        'dataset_mock': json.loads((dataset / 'manifest.json').read_text())['mock'],
        'tokenizer_sha256': hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        'norm_stats_sha256': hashlib.sha256((dataset / 'norm_stats.json').read_bytes()).hexdigest(),
        'openpi_transforms': transforms.__file__}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
