#!/usr/bin/env python3
"""Read-only comparison with the collection project. Never connects to hardware."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pi05_rlt_rtc.adapters.franka import map_joint_delta_action
from pi05_rlt_rtc.config.runtime_config import load_runtime_config
from pi05_rlt_rtc.contracts.physical import encode_action
from pi05_rlt_rtc.deployment.model_loader import build_model_config


def read_json(path):
    return json.loads(path.read_text())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def executable_ast(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-root', type=Path, required=True)
    parser.add_argument('--legacy-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    other, legacy = args.collection_root.resolve(), args.legacy_root.resolve()
    report = {'kind': 'static_and_cpu_contract_audit', 'hardware_opened': False,
              'collection_root': str(other), 'checks': [], 'differences': []}

    def check(name, condition, detail):
        report['checks'].append({'name': name, 'passed': bool(condition), 'detail': detail})

    local = read_json(ROOT / 'configs/training_contract.json')
    collected = read_json(other / 'configs/training_contract.json')
    fields = ('action_contract_id', 'control_hz', 'state_dim', 'action_dim', 'model_action_dim',
              'action_horizon', 'state_layout', 'action_layout', 'image_keys', 'image_mask',
              'image_shape', 'normalization', 'normalization_method', 'state_tokenization',
              'delta_actions_transform', 'droid_velocity_scaling')
    mismatches = [key for key in fields if local[key] != collected[key]]
    check('training_contract_fields', not mismatches, {'fields': fields, 'mismatches': mismatches})
    calibration = read_json(ROOT / 'configs/franka_bridge_panda.example.json')
    check('bridge_calibration', calibration == read_json(other / 'configs/bridge.real.json'),
          'Full JSON comparison, including unset hardware parameters; equality is not calibration approval.')
    cfg = load_runtime_config(ROOT / 'configs/runtime_franka.yaml')
    cc = read_json(other / 'configs/collection.json')
    check('controller_frequency_prompt_socket', cfg.control_hz == cc['control_hz'] and
          cfg.raw['prompt'] == cc['task'] and cfg.raw['franka']['socket'] == cc['socket'],
          {'hz': cc['control_hz'], 'socket': cc['socket'], 'concurrent_ownership_allowed': False})
    camera = cfg.raw['franka']['cameras']['kwargs']
    check('camera_configuration', all(camera[k] == v for k, v in cc['cameras'].items()), cc['cameras'])
    check('camera_device_implementation', executable_ast(ROOT / 'pi05_rlt_rtc/adapters/camera_devices.py') ==
          executable_ast(other / 'vendor/legacy_cameras.py'),
          'Executable AST matches; D435i-to-D405 changes in this file are documentation only. Both providers check the D405 model.')
    contract = module(other / 'pi05_demo/contract.py', 'collection_contract')
    random = np.random.default_rng(20261029)
    q = (np.asarray(calibration['joint_lower']) + np.asarray(calibration['joint_upper'])) / 2
    max_error = 0.
    for _ in range(1024):
        action = random.uniform(-1, 1, 8).astype(np.float32)
        target, width = contract.decode(q, action)
        np.testing.assert_array_equal(encode_action(q, target, width), contract.encode(q, target, width))
        got_q, got_width = map_joint_delta_action(action, q, delta_scale=calibration['joint_delta_scale'],
            joint_lower=calibration['joint_lower'], joint_upper=calibration['joint_upper'],
            finger_range=calibration['finger_range'], max_width=calibration['gripper_max_width'])
        max_error = max(max_error, float(np.max(np.abs(got_q-target))), abs(got_width-width))
        np.testing.assert_allclose(np.r_[got_q, got_width], np.r_[target, width], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(contract.state_vector(q, width), np.r_[q, width/2, width/2].astype(np.float32))
    check('physical_action_state_roundtrip', True, {'samples': 1024, 'max_absolute_error': max_error})
    stage1 = yaml.safe_load((other / 'configs/stage1_2xa800.yaml').read_text())['actor']['model']
    inference = build_model_config(cfg, '/not_loaded/checkpoint', '/not_loaded/norm_stats.json')
    arch = ('action_horizon', 'model_action_dim', 'action_env_dim', 'paligemma_variant',
            'action_expert_variant', 'max_token_len', 'discrete_state_input', 'use_rlt',
            'rlt_input_dim', 'rlt_embed_dim', 'rlt_prefix_seq_len', 'rlt_num_layers',
            'rlt_num_heads', 'rlt_mlp_ratio', 'rlt_use_mask', 'rlt_image_only')
    mismatch = [key for key in arch if stage1['openpi'][key] != inference['openpi'][key]]
    check('stage1_inference_architecture', not mismatch, {'fields': arch, 'mismatches': mismatch})
    old = read_json(legacy / 'franka_bridge_panda.example.json')
    report['differences'] += [
        {'name': 'legacy_control_incompatible', 'legacy_path': str(legacy),
         'legacy_delta_scale': old['joint_delta_scale'], 'legacy_finger_range': old['finger_range'],
         'legacy_max_width': old['gripper_max_width'],
         'legacy_uses_franka_hand': 'franka::Gripper' in (legacy / 'bridge.cpp').read_text(),
         'resolution': 'Keep this independent Robotiq bridge; do not use the legacy binary/config.'},
        {'name': 'stage1_window_selection',
         'collection': 'Updated Stage-1 uses complete episode windows without tail padding. The isolated snapshot additionally shares runtime timestamp/intervention filtering.',
         'this_project': 'PhysicalChunkDataset uses complete, contiguous, non-intervened H-step windows.',
         'resolution': 'stage1_integration uses pi05_rlt_rtc.data.windows for all three loaders; see reports/stage1_alignment/loader_integration.json. Original training code was not edited.'},
        {'name': 'stage1_loader_and_checkout', 'collection_profile': stage1['openpi']['config_name'],
         'runtime_profile': cfg.raw['model']['config_name'],
         'resolution': 'Architecture matches, but registry and optimized source checkout differ. Use a separately verified inference checkout; no automatic reuse of the training vendor tree.'},
    ]
    report['hardware_ready'] = bool(calibration.get('hardware_verified') and camera.get('wrist_serial'))
    report['fully_identical'] = False
    paths = [other / 'configs/training_contract.json', other / 'configs/collection.json',
             other / 'configs/bridge.real.json', other / 'configs/stage1_2xa800.yaml',
             other / 'stage1/data.py', other / 'stage1/worker.py', other / 'stage1/checkpoint.py',
             other / 'pi05_demo/contract.py', other / 'pi05_demo/export.py', other / 'pi05_demo/transport.py',
             other / 'pi05_demo/cameras.py', other / 'vendor/legacy_cameras.py',
             *sorted((other / 'cpp/bridge').glob('*')), legacy / 'bridge.cpp',
             legacy / 'franka_bridge_panda.example.json', legacy / 'build/pi05_franka_bridge']
    report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'checks': len(report['checks']), 'passed': sum(c['passed'] for c in report['checks']),
                      'fully_identical': False, 'hardware_ready': report['hardware_ready'], 'report': str(args.out)}))
    return int(not all(c['passed'] for c in report['checks']))


if __name__ == '__main__':
    raise SystemExit(main())
