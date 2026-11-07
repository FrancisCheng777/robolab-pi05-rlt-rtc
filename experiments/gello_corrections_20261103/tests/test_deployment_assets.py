import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from pi05_rlt_rtc.config.runtime_config import load_runtime_config
from pi05_rlt_rtc.contracts.errors import ContractError, MockPolicyError
from pi05_rlt_rtc.adapters.rlinf_feature import PINNED_RLINF_COMMIT


def fixture_assets(tmp_path):
    from pi05_rlt_rtc.deployment.assets import build_source_lock
    cfg = load_runtime_config('configs/runtime_franka.yaml')
    root = tmp_path / 'rlinf'; (root / 'rlinf').mkdir(parents=True)
    (root / 'rlinf/__init__.py').write_text('# test double\n')
    (root / 'pi05_source_lock.json').write_text(json.dumps(build_source_lock(root)))
    ckpt = tmp_path / 'checkpoint'; (ckpt / 'model_state_dict').mkdir(parents=True)
    (ckpt / 'model_state_dict/full_weights.pt').write_bytes(b'fixture is NOT a real model')
    stats = ckpt / 'norm_stats.json'
    stats.write_text(json.dumps({'norm_stats': {k: {'q01': [-1]*d, 'q99': [1]*d} for k,d in [('state',9),('actions',8)]}}))
    (ckpt / 'paligemma_tokenizer.model').write_bytes(b'fixture tokenizer')
    (ckpt / 'manifest.json').write_text(json.dumps({'action_contract_id': cfg.action_contract_id, 'mock': False,
        'camera_views': ['wrist'], 'policy_image_mapping': {'base_0_rgb':'wrist'}, 'action_codec': {'joint_delta_scale':[.05]*7},
        'normalization_source': 'train split only', 'norm_stats_sha256': hashlib.sha256(stats.read_bytes()).hexdigest()}))
    from pi05_rlt_rtc.data.windows import WINDOW_POLICY
    (ckpt / 'stage1_metadata.json').write_text(json.dumps({'action_contract_id': cfg.action_contract_id,
        'synthetic': False, 'upstream_commit': PINNED_RLINF_COMMIT, 'normalization': 'quantile_q01_q99_then_pad',
        'tokenize_state_before_padding': True, 'patched_files': {}, 'step': 1, 'window_policy': WINDOW_POLICY,
        'tokenizer_sha256': hashlib.sha256((ckpt / 'paligemma_tokenizer.model').read_bytes()).hexdigest()}))
    from pi05_rlt_rtc.deployment.model_loader import build_model_config
    arch = build_model_config(cfg, ckpt, stats)
    (ckpt / 'stage1_resolved.yaml').write_text(yaml.safe_dump({'actor': {'model': arch}}))
    (ckpt / 'training_contract.json').write_text('{}')
    (ckpt / 'action_codec.json').write_text(json.dumps({'joint_delta_scale':[.05]*7}))
    return dataclasses.replace(cfg, raw={**cfg.raw, 'model': {**cfg.raw['model'],
        'rlinf_root': str(root), 'checkpoint': str(ckpt), 'norm_stats': str(stats)}})


def test_stage1_assets_reject_synthetic_and_changed_statistics(tmp_path):
    from pi05_rlt_rtc.deployment.assets import validate_assets
    cfg = fixture_assets(tmp_path)
    assert validate_assets(cfg)['structural_check_only'] is True
    p = Path(cfg.raw['model']['checkpoint']) / 'stage1_metadata.json'
    data = json.loads(p.read_text());data['synthetic'] = True;p.write_text(json.dumps(data))
    with pytest.raises(MockPolicyError): validate_assets(cfg)
    data['synthetic'] = False;p.write_text(json.dumps(data))
    Path(cfg.raw['model']['norm_stats']).write_text('{}')
    with pytest.raises(ContractError, match='statistics'): validate_assets(cfg)


def test_source_lock_rejects_edits_and_unlisted_source(tmp_path):
    from pi05_rlt_rtc.deployment.assets import verify_source_lock
    cfg = fixture_assets(tmp_path); root = Path(cfg.raw['model']['rlinf_root'])
    verify_source_lock(root)
    p = root / 'rlinf/extra.py';p.write_text('# unreviewed\n')
    with pytest.raises(ContractError, match='source'): verify_source_lock(root)
    p.unlink();(root / 'rlinf/__init__.py').write_text('# changed\n')
    with pytest.raises(ContractError, match='source'): verify_source_lock(root)


def test_bundle_is_relocatable_and_tamper_evident(tmp_path):
    from pi05_rlt_rtc.deployment.assets import export_bundle, verify_bundle, validate_assets
    cfg = fixture_assets(tmp_path)
    bundle = export_bundle(cfg, tmp_path / 'bundle')
    moved = tmp_path / 'relocated';bundle.rename(moved)
    assert verify_bundle(moved)['integrity_verified']
    relocated = load_runtime_config(moved / 'runtime.yaml')
    assert relocated.require_model_assets()[1].parent == moved
    assert validate_assets(cfg)['source']['source_tree_sha256'] == validate_assets(relocated)['source']['source_tree_sha256']
    (moved / 'checkpoint/paligemma_tokenizer.model').write_bytes(b'changed')
    with pytest.raises(ContractError, match='integrity'): verify_bundle(moved)
