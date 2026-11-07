"""Device-free asset preflight and atomic, relocatable deployment bundles.

Hashes establish integrity, not model quality or authorship. Structural checks
never claim a successful strict tensor load, GPU forward or hardware rollout.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import yaml

from pi05_rlt_rtc.adapters.rlinf_feature import PINNED_RLINF_COMMIT
from pi05_rlt_rtc.contracts.errors import ContractError, MissingAssetError, MockPolicyError


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _source_files(root):
    return {p.relative_to(root).as_posix(): sha256(p)
            for p in sorted((Path(root) / 'rlinf').rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.yaml', '.yml', '.json', '.toml', '.sh', '.cpp', '.h', '.cu'}
            and '__pycache__' not in p.parts}


def build_source_lock(root):
    files = _source_files(root)
    if not files:
        raise MissingAssetError('RLinf source is empty')
    return {'format': 'pi05-source-lock-v1', 'upstream_commit': PINNED_RLINF_COMMIT,
            'scope': 'local integrated snapshot; upstream plus documented Stage-1 patches', 'files': files}


def verify_source_lock(root):
    root = Path(root)
    path = root / 'pi05_source_lock.json'
    if not path.is_file():
        raise MissingAssetError(f'missing source lock: {path}')
    lock = json.loads(path.read_text())
    if lock.get('format') != 'pi05-source-lock-v1' or lock.get('upstream_commit') != PINNED_RLINF_COMMIT:
        raise ContractError('RLinf source lock version/commit mismatch')
    if lock.get('files') != _source_files(root):
        raise ContractError('RLinf source integrity mismatch (changed, missing or added source)')
    tree_digest = hashlib.sha256(json.dumps(lock['files'], sort_keys=True).encode()).hexdigest()
    return {'source_lock_sha256': sha256(path), 'source_tree_sha256': tree_digest,
            'files': len(lock['files']), 'upstream_commit': PINNED_RLINF_COMMIT}


def verify_runtime_source(root):
    root = Path(root).resolve()
    if (root / 'pi05_source_lock.json').is_file():
        return verify_source_lock(root)
    try:
        top = Path(subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--show-toplevel'], text=True).strip())
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError('RLinf requires a verified source lock or pinned git checkout') from exc
    if top != root or commit != PINNED_RLINF_COMMIT:
        raise ContractError('RLinf checkout root/commit mismatch')
    from pi05_rlt_rtc.adapters.upstream_profile import verify_profile_checkout
    verify_profile_checkout(root)
    digest = hashlib.sha256(json.dumps(_source_files(root), sort_keys=True).encode()).hexdigest()
    return {'upstream_commit': commit, 'source_tree_sha256': digest}


def validate_assets(cfg):
    root, checkpoint, stats = cfg.require_model_assets()
    if not checkpoint.is_dir():
        raise MissingAssetError('model.checkpoint must be a complete Stage-1 export directory')
    required = ('stage1_metadata.json', 'stage1_resolved.yaml', 'manifest.json',
                'training_contract.json', 'action_codec.json', 'paligemma_tokenizer.model', 'model_state_dict/full_weights.pt')
    for name in required:
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise MissingAssetError(f'incomplete Stage-1 export: {name}')
    meta = json.loads((checkpoint / 'stage1_metadata.json').read_text())
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    if meta.get('synthetic') is not False or manifest.get('mock') is not False:
        raise MockPolicyError('synthetic/mock Stage-1 assets cannot become a robot policy')
    if any(x.get('action_contract_id') != cfg.action_contract_id for x in (meta, manifest)):
        raise ContractError('Stage-1/dataset action contract mismatch')
    from pi05_rlt_rtc.data.profile import validate_dataset_profile
    validate_dataset_profile(manifest)
    if json.loads((checkpoint / "action_codec.json").read_text()) != manifest["action_codec"]:
        raise ContractError("checkpoint action codec differs from dataset")
    if meta.get('upstream_commit') != PINNED_RLINF_COMMIT:
        raise ContractError('Stage-1 upstream commit mismatch')
    tokenizer_digest = sha256(checkpoint / 'paligemma_tokenizer.model')
    if meta.get('tokenizer_sha256') != tokenizer_digest:
        raise ContractError('tokenizer does not match the Stage-1 training metadata')
    from pi05_rlt_rtc.data.windows import WINDOW_POLICY
    if meta.get('window_policy') != WINDOW_POLICY or type(meta.get('step')) is not int or meta['step'] < 1:
        raise ContractError('Stage-1 needs a trained step and the current contiguous-window contract')
    if meta.get('normalization') != 'quantile_q01_q99_then_pad' or meta.get('tokenize_state_before_padding') is not True:
        raise ContractError('Stage-1 normalization/tokenization semantics mismatch')
    digest = sha256(stats)
    if manifest.get('norm_stats_sha256') != digest or manifest.get('normalization_source') != 'train split only':
        raise ContractError('normalization statistics must match the training-only manifest')
    norm = json.loads(stats.read_text()).get('norm_stats', {})
    for key, dim in (('state', 9), ('actions', 8)):
        lo = np.asarray(norm.get(key, {}).get('q01', []), np.float32)
        hi = np.asarray(norm.get(key, {}).get('q99', []), np.float32)
        if lo.shape != (dim,) or hi.shape != (dim,) or not np.isfinite([lo, hi]).all() or np.any(lo > hi):
            raise ContractError(f'invalid {key} quantile statistics')
    from .model_loader import build_model_config
    expected = build_model_config(cfg, checkpoint, stats)['openpi']
    trained = yaml.safe_load((checkpoint / 'stage1_resolved.yaml').read_text())['actor']['model']['openpi']
    shape_keys = ('paligemma_variant', 'action_expert_variant', 'max_token_len', 'discrete_state_input',
                  'rlt_input_dim', 'rlt_embed_dim', 'rlt_prefix_seq_len', 'rlt_num_layers', 'rlt_num_heads',
                  'num_images_in_input', 'rlt_mlp_ratio', 'rlt_image_only', 'rlt_use_mask', 'action_horizon', 'action_env_dim', 'model_action_dim', 'use_rlt')
    for key in shape_keys:
        if trained.get(key) != expected[key]:
            raise ContractError(f'Stage-1/runtime architecture mismatch: {key}')
    for name, digest_value in meta.get('patched_files', {}).items():
        p = (root / name).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file() or sha256(p) != digest_value:
            raise ContractError(f'Stage-1 patched source mismatch: {name}')
    source = verify_runtime_source(root)
    return {'structural_check_only': True, 'strict_tensor_load': False, 'gpu_forward_tested': False,
            'hardware_validated': False, 'checkpoint': str(checkpoint), 'tokenizer': str(checkpoint / 'paligemma_tokenizer.model'),
            'tokenizer_sha256': sha256(checkpoint / 'paligemma_tokenizer.model'),
            'norm_stats_sha256': sha256(stats), 'source': source}


def preflight(args):
    from pi05_rlt_rtc.config.runtime_config import load_runtime_config
    cfg = load_runtime_config(args.runtime_config)
    checks, failures = {}, []
    try:
        checks['assets'] = validate_assets(cfg)
    except (ContractError, MissingAssetError, MockPolicyError, OSError, ValueError, KeyError) as exc:
        failures.append(f'{type(exc).__name__}: {exc}')
    # Do not enumerate devices, open sockets, import CUDA or read live sensors.
    report = {'format': 'pi05-preflight-v1', 'ready_for_model_load': not failures,
              'robot_commands_sent': 0, 'hardware_validated': False, 'checks': checks, 'failures': failures}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if not failures else 3


def export_bundle(cfg, destination):
    validate_assets(cfg)
    root, checkpoint, stats = cfg.require_model_assets()
    destination = Path(destination).resolve()
    if destination.exists():
        raise ContractError('bundle output already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.pi05-bundle-', dir=destination.parent))
    try:
        raw = copy.deepcopy(cfg.raw)
        target = staging / 'checkpoint'; (target / 'model_state_dict').mkdir(parents=True)
        for name in ('stage1_metadata.json', 'stage1_resolved.yaml', 'manifest.json', 'training_contract.json',
                     'action_codec.json', 'paligemma_tokenizer.model', 'model_state_dict/full_weights.pt'):
            shutil.copy2(checkpoint / name, target / name)
        shutil.copy2(stats, target / 'norm_stats.json')
        shutil.copytree(root / 'rlinf', staging / 'source/rlinf', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for notice in ('LICENSE', 'NOTICE'):
            if (root / notice).is_file():
                shutil.copy2(root / notice, staging / 'source' / notice)
        (staging / 'source/pi05_source_lock.json').write_text(json.dumps(build_source_lock(staging / 'source'), indent=2))
        raw['model'].update(rlinf_root='source', checkpoint='checkpoint', norm_stats='checkpoint/norm_stats.json')
        if raw['policy']['kind'] == 'actor':
            actor = cfg.asset(raw['policy'].get('checkpoint'), 'policy.checkpoint')
            from pi05_rlt_rtc.learning.checkpoint import PolicyBundle
            policy = PolicyBundle.load(actor)
            if policy.extra.get('mock') is not False:
                raise MockPolicyError('mock actor cannot enter a deployment bundle')
            if policy.contract != cfg.contract or policy.norm_stats_sha256 != sha256(stats):
                raise ContractError('actor contract/statistics mismatch')
            if policy.extra.get('runtime_contract') != cfg.wire_contract():
                raise ContractError('actor runtime contract mismatch')
            shutil.copy2(actor, staging / 'actor.pt'); raw['policy']['checkpoint'] = 'actor.pt'
        bridge = cfg.asset(raw['franka']['bridge_config'], 'franka.bridge_config')
        shutil.copy2(bridge, staging / 'bridge_config.json')
        raw['franka']['bridge_config'] = 'bridge_config.json'
        dependency = Path(__file__).resolve().parents[2] / 'vendor/robolab-collecting-pipeline'
        shutil.copytree(dependency, staging / 'vendor/robolab-collecting-pipeline',
                        ignore=shutil.ignore_patterns('.git', 'build', '__pycache__', '*.pyc', '.venv*'))
        if 'session' in raw:
            session = raw['session']
            shutil.copy2(cfg.asset(session['collection_config'], 'collection config'), staging/'collection_config.json')
            session['collection_config'] = 'collection_config.json'
            session['bridge_binary'] = 'vendor/robolab-collecting-pipeline/build/pi05_franka_bridge'
        (staging / 'runtime.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
        project = Path(__file__).resolve().parents[2]
        shutil.copytree(project / 'pi05_rlt_rtc', staging / 'pi05_rlt_rtc', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for subdir in ('pi05_demo', 'stage1'):
            if (project / 'stage1_integration' / subdir).is_dir():
                shutil.copytree(project / 'stage1_integration' / subdir, staging / 'stage1_integration' / subdir,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('pyproject.toml', 'README.md', 'NOTICE.md'):
            shutil.copy2(project / name, staging / name)
        manifest = {'format': 'pi05-deployment-bundle-v1', 'hardware_validated': False, 'gpu_forward_tested': False,
                    'strict_tensor_load': False, 'runtime_contract': cfg.wire_contract(),
                    'files': {p.relative_to(staging).as_posix(): sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}}
        (staging / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        verify_bundle(staging)
        staging.rename(destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'bundle_manifest.json').read_text())
    if manifest.get('format') != 'pi05-deployment-bundle-v1':
        raise ContractError('unsupported deployment bundle')
    present = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
               if p.is_file() and '__pycache__' not in p.parts and p.name != 'bundle_manifest.json'}
    if present != set(manifest['files']):
        raise ContractError('bundle integrity mismatch: missing or unlisted files')
    for name, expected in manifest['files'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ContractError(f'bundle integrity mismatch: {name}')
    from pi05_rlt_rtc.config.runtime_config import load_runtime_config
    validate_assets(load_runtime_config(directory / 'runtime.yaml'))
    return {'integrity_verified': True, 'files': len(manifest['files']), 'hardware_validated': False}


def bundle_command(args):
    from pi05_rlt_rtc.config.runtime_config import load_runtime_config
    if args.command == 'verify-bundle':
        print(json.dumps(verify_bundle(args.path), indent=2))
    else:
        print(export_bundle(load_runtime_config(args.runtime_config), args.out))
    return 0
