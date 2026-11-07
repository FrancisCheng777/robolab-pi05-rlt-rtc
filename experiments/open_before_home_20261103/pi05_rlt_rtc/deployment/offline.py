"""Replay saved inputs through a planner, with no robot/camera construction."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.runtime.recording import load_request


def replay_recordings(cfg, backend, metadata, recording, output, *, allow_mock=False):
    source = Path(recording)
    files = sorted(source.glob('request-*.json')) if source.is_dir() else [source]
    if not files or any(not f.is_file() for f in files):
        raise ContractError('recording has no saved planner requests')
    if metadata.get('mock') is not False and not allow_mock:
        raise ContractError('mock planner cannot validate production inference')
    output = Path(output)
    if output.exists():
        raise ContractError('offline output must be a new directory')
    # Validate the entire input manifest before inference; reject mixed identities.
    identities = set()
    for path in files:
        data = load_request(path)
        recorded = data['metadata']
        if recorded.get('mock') is not False and not allow_mock:
            raise ContractError('mock recording cannot validate production inference')
        if recorded.get('runtime_contract') != cfg.wire_contract():
            raise ContractError('recording/runtime contract mismatch')
        request = data['request']
        for field in ('feature_version_id', 'policy_version_id'):
            if getattr(request, field) != metadata.get(field) or recorded.get(field) != metadata.get(field):
                raise ContractError(f'recording/planner {field} mismatch')
        identity = (request.env_id, request.episode_id, request.request_id)
        if identity in identities:
            raise ContractError('duplicate recorded planner request')
        identities.add(identity)
    output.mkdir(parents=True)
    timings, submitted, references, rows = [], [], [], []
    for path in files:
        data = load_request(path)
        req = data['request']
        start = time.perf_counter()
        result = backend.plan(req, data['observation'], data['pending_prefix'], req.rtc_context)
        elapsed = time.perf_counter() - start
        result.validate(cfg.contract)
        if np.any(np.abs(result.submitted) > 1) or np.any(np.abs(result.reference_env_full) > 1):
            raise ContractError('planner actions outside normalized physical command bounds')
        timings.append(elapsed)
        submitted.append(result.submitted)
        references.append(result.reference_env_full)
        rows.append({'file': str(path.resolve()), 'request_id': req.request_id,
                     'capture_tick': req.capture_tick, 'rng_seed': req.rng_seed, 'latency_s': elapsed})
    np.savez(output / 'plans.npz', submitted=np.stack(submitted), reference=np.stack(references))
    # First call may include lazy compilation. Report cold latency separately.
    steady = timings[1:]
    budget = cfg.contract.delay / cfg.control_hz
    report = {**metadata, 'format': 'pi05-offline-report-v1', 'samples': len(rows),
              'mock': metadata.get('mock') is not False or any(load_request(p)['metadata'].get('mock') is not False for p in files),
              'robot_commands_sent': 0, 'hardware_validated': False,
              'scope': 'recorded planner inputs only; excludes sensor, network and actuator latency',
              'cold_s': timings[0], 'p50_s': float(np.percentile(steady, 50)) if steady else None,
              'p95_s': float(np.percentile(steady, 95)) if steady else None,
              'delay_budget_s': budget,
              'within_measured_budget': max(steady) < budget if steady else None,
              'requests': rows}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def infer_recording(args):
    from .runtime import config_from_args, make_backend
    cfg = config_from_args(args)
    if not getattr(args, "out", None):
        raise ContractError("offline inference requires --out new_output_directory")
    if not Path(args.recording).exists():
        raise ContractError('recording path does not exist')
    backend, metadata = make_backend(cfg, args)
    report = replay_recordings(cfg, backend, metadata, args.recording, args.out)
    print(json.dumps(report, indent=2))
    return 0


def infer_demo(args):
    """Cold inference on one existing demonstration frame; no playback actions."""
    import sys
    from pi05_rlt_rtc import SCHEMA_VERSION, PLANNING_STATE_VERSION
    from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
    from pi05_rlt_rtc.contracts.plan import PlanRequest, PrevPlanContext
    from pi05_rlt_rtc.runtime.recording import write_request
    from .runtime import config_from_args, make_backend
    from .assets import sha256
    import h5py
    from pi05_rlt_rtc.adapters.robolab import activate
    activate()
    from pi05_demo.episode import inspect_episode
    cfg = config_from_args(args)
    path = Path(args.episode).resolve()
    info = inspect_episode(path, allow_mock=False)
    if info['outcome'] == 'aborted' or not 0 <= args.frame < info['frames']:
        raise ContractError('choose a valid frame in a complete non-aborted episode')
    out = Path(args.out)
    if out.exists():
        raise ContractError('offline output must be a new directory')
    with h5py.File(path, 'r') as f:
        i = args.frame
        if bool(f['intervention'][i]):
            raise ContractError('selected demonstration frame contains an intervention')
        if str(f.attrs['task']) != cfg.raw['prompt']:
            raise ContractError('demonstration/runtime prompt mismatch')
        obs = ObservationSnapshot('franka', path.stem, i, f['state'][i],
            {k: f[f'images/{k}'][i] for k in ('wrist',)}, {'wrist': True},
            cfg.raw['prompt'], float(f['timestamp'][i]),
            {k: float(f[f'camera_time/{k}'][i]) for k in ('wrist',)})
    backend, metadata = make_backend(cfg, args)
    request = PlanRequest(request_id=-1, env_id=obs.env_id, episode_id=obs.episode_id,
        episode_generation=1, capture_tick=obs.tick, apply_start_tick=obs.tick + cfg.contract.delay,
        queue_generation=1, policy_version_id=metadata['policy_version_id'], feature_version_id=metadata['feature_version_id'],
        schema_version=SCHEMA_VERSION, planning_state_version=PLANNING_STATE_VERSION, rtc_context=PrevPlanContext.empty(),
        rng_seed=args.seed, sampler_mode=cfg.rtc_mode, num_denoising_steps=cfg.num_steps)
    out.mkdir(parents=True)
    write_request(out / 'request-000000.json', request, obs, np.zeros((cfg.contract.delay,8), np.float32),
                  {**metadata, 'input_origin': 'demonstration frame; not historical policy-inference acceptance',
                   'episode_sha256': sha256(path), 'frame': args.frame})
    report = replay_recordings(cfg, backend, metadata, out / 'request-000000.json', out / 'inference')
    print(json.dumps(report, indent=2))
    return 0
