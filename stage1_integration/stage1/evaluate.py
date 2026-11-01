"""Held-out Stage-1 flow loss and RLT reconstruction; no robot or optimizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def evaluate_batches(model, batches, *, max_batches, seed=0):
    from rlinf.models.embodiment.openpi.modules.model import preprocess_observation
    if max_batches <= 0:
        raise ValueError('max_batches must be positive')
    model.eval().requires_grad_(False)
    total_elements = env_elements = examples = count = 0
    loss_sum = env_sum = rlt_sum = 0.
    # Preserve the caller's RNG state. Held-out data is never augmented.
    devices = [model.device] if model.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(seed)
        for batch in batches:
            obs, actions = model._unpack_sft_batch(batch)
            obs = model._observation_to_device(obs)
            actions = model._actions_to_device(actions)
            per_element = model.compute_loss(obs, actions, train=False)
            prepared = preprocess_observation(obs, train=False)
            prefix, mask, _ = model.build_prefix_cache(prepared)
            prefix, mask = model._select_rlt_prefix_embeddings(prefix, mask, prepared.tokenized_prompt)
            rlt_loss, _ = model._rlt_forward(prefix, mask)
            if not torch.isfinite(per_element).all() or not torch.isfinite(rlt_loss).all():
                raise ValueError('nonfinite held-out loss')
            n = actions.shape[0]
            physical = per_element[..., :8]
            total_elements += per_element.numel(); env_elements += physical.numel(); examples += n
            loss_sum += per_element.double().sum().item(); env_sum += physical.double().sum().item()
            rlt_sum += rlt_loss.item() * n
            count += 1
            if count >= max_batches:
                break
    if not examples:
        raise ValueError('held-out dataset has no complete windows')
    return {'examples': examples, 'batches': count, 'flow_mse_all32': loss_sum / total_elements,
            'flow_mse_physical8': env_sum / env_elements, 'rlt_reconstruction_loss': rlt_sum / examples}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-batches', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.max_batches <= 0:
        parser.error('batch-size and max-batches must be positive')
    from pi05_rlt_rtc.config.runtime_config import load_runtime_config
    from pi05_rlt_rtc.deployment.assets import sha256, validate_assets
    from pi05_rlt_rtc.deployment.model_loader import load_planner
    from .data import PlugDataset, read_manifest
    cfg = load_runtime_config(args.runtime_config)
    assets = validate_assets(cfg)
    dataset_root = Path(args.dataset_root).resolve()
    manifest = read_manifest(dataset_root)
    if sha256(dataset_root / 'manifest.json') != sha256(Path(assets['checkpoint']) / 'manifest.json'):
        raise ValueError('held-out dataset manifest differs from the training export')
    if manifest['norm_stats_sha256'] != assets['norm_stats_sha256']:
        raise ValueError('held-out data must use exactly the Stage-1 training statistics')
    sources = manifest.get('sources', [])
    train_sessions = {s['session'] for s in sources if s['split'] == 'train'}
    validation_sessions = {s['session'] for s in sources if s['split'] == 'validation'}
    if not validation_sessions or train_sessions & validation_sessions:
        raise ValueError('require disjoint held-out collection sessions')
    # Avoid loading an unrelated Stage-2 actor during Stage-1 validation.
    import dataclasses
    cfg = dataclasses.replace(cfg, raw={**cfg.raw, 'policy': {'kind': 'teacher'}})
    ds = PlugDataset(str(dataset_root), assets['tokenizer'], horizon=cfg.contract.horizon,
                     max_len=cfg.raw['model'].get('architecture', {}).get('max_token_len', 200), split='validation')
    batches = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    planner, metadata = load_planner(cfg)
    report = evaluate_batches(planner.service.model, batches, max_batches=args.max_batches, seed=args.seed)
    report.update(metadata)
    report.update(split='validation', seed=args.seed, dataset_manifest_sha256=sha256(dataset_root / 'manifest.json'),
                  available_windows=len(ds), scope='held-out offline losses; not task success rate',
                  robot_commands_sent=0, hardware_validated=False)
    target = Path(args.out); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
