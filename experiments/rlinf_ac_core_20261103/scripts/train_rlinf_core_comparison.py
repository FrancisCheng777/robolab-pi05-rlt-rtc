"""Offline chronological RLinf-core comparison. Never imports robot control.

An existing recorded episode stands in for one online rollout. This is an
explicit offline approximation, not RLinf online training or a causal ablation.
"""
import argparse
from collections import Counter, deque
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from pi05_rlt_rtc.config.runtime_config import load_runtime_config
from pi05_rlt_rtc.deployment.model_loader import file_sha256
from pi05_rlt_rtc.deployment.reward_override import relabel_terminal_transition
from pi05_rlt_rtc.deployment.runtime import write_json
from pi05_rlt_rtc.deployment.training import QueueLearner
from pi05_rlt_rtc.learning.checkpoint import PolicyBundle, state_dict_sha256
from pi05_rlt_rtc.learning.losses import build_transition_batch
from pi05_rlt_rtc.learning.planning_state import PlanningStateSpec
from pi05_rlt_rtc.learning.replay import ReplayBuffer


def ordered_sources(sources):
    """The source manifest stores sorted IDs separately from directory order."""
    rows = []
    for directory in sources['replay_directories']:
        label = json.loads((Path(directory) / 'annotation.json').read_text())
        rows.append((Path(label['rollout']).stat().st_mtime_ns, label['trial_id'], directory))
    ids = [row[1] for row in rows]
    expected = sources['trial_ids']
    if (len(set(ids)) != len(ids) or len(set(expected)) != len(expected)
            or set(ids) != set(expected)):
        raise RuntimeError('source IDs and replay directories disagree')
    # Raw rollout save time orders independent collection sessions; repaired
    # label/replay mtimes would incorrectly move historical episodes later.
    rows.sort()
    return [row[2] for row in rows], [row[1] for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', required=True)
    parser.add_argument('--runtime-config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit-episodes', type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    def status(state, **fields):
        write_json(out / 'status.json', {'state': state, 'hardware_opened': False,
                   'automatic_policy_switch': False, **fields})
    # Serialize with all real inference/collection sessions, including CPU runs.
    lock_path = Path.home() / '.cache/pi05-stage1-8000/session.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    status('waiting_for_session_exit')
    fcntl.flock(lock, fcntl.LOCK_EX)
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        source_file = Path(args.sources).resolve()
        sources = json.loads(source_file.read_text())
        directories, expected_ids = ordered_sources(sources)
        if args.limit_episodes:
            directories = directories[:args.limit_episodes]
            expected_ids = expected_ids[:args.limit_episodes]
        if len(directories) < 2 or len(set(expected_ids)) != len(directories):
            raise RuntimeError('need at least two distinct labelled episodes')
        raw = yaml.safe_load(Path(args.runtime_config).read_text())
        raw['training'].update(gamma_per_tick=.96,
            terminal_rewards={'success': 1., 'failure': -1.},
            batch_size=256, actor_lr=3e-4, critic_lr=3e-4,
            critic_actor_ratio=4, lambda_q=.1, lambda_bc=5.,
            reference_dropout=.5, polyak_tau=.005, exploration_std=.002,
            actor_clip_grad=10., critic_clip_grad=10.)
        raw['policy'] = {'kind': 'teacher', 'checkpoint': None, 'exploration': False}
        config = out / 'runtime_rlinf_core.yaml'
        config.write_text(yaml.safe_dump(raw, sort_keys=False))
        cfg = load_runtime_config(config)
        hashes = {str(source_file): file_sha256(source_file)}
        metadata = None
        episodes, audit = [], []
        for expected, directory in zip(expected_ids, directories):
            directory = Path(directory)
            for name in ('annotation.json', 'runtime_manifest.json', 'replay.pt'):
                p = directory / name
                hashes[str(p)] = file_sha256(p)
            annotation = json.loads((directory / 'annotation.json').read_text())
            current = json.loads((directory / 'runtime_manifest.json').read_text())
            if (annotation['trial_id'] != expected or annotation.get('eligible') is not True
                    or annotation.get('mock') is not False or current.get('mock') is not False
                    or annotation.get('outcome') not in ('success', 'failure')):
                raise RuntimeError('invalid or misordered physical annotation')
            if hashes[str(directory / 'replay.pt')] != annotation['replay_sha256']:
                raise RuntimeError('source replay hash mismatch')
            if current['runtime_contract'] != cfg.wire_contract():
                raise RuntimeError('incompatible RTC runtime contract')
            replay = ReplayBuffer.load(directory / 'replay.pt')
            if (replay.feature_version_id != current['feature_version_id'] or
                    replay.norm_stats_sha256 != current['norm_stats_sha256']):
                raise RuntimeError('source feature identity mismatch')
            if metadata is not None and any(current[k] != metadata[k] for k in
                                           ('feature_version_id', 'norm_stats_sha256')):
                raise RuntimeError('mixed frozen feature versions')
            metadata = current
            transitions = [relabel_terminal_transition(t, annotation,
                           {'success': 1., 'failure': -1.}, gamma_per_tick=.96)
                           for t in replay.learnable_transitions()]
            terminals = [t for t in transitions if t.terminated]
            if len(terminals) != 1:
                raise RuntimeError(f'{expected}: requires one complete terminal')
            if any(t.gamma_per_tick != .96 for t in transitions):
                raise RuntimeError('discount migration incomplete')
            audit.append({'trial_id': expected, 'outcome': annotation['outcome'],
                'transitions': len(transitions), 'terminal_count': 1,
                'terminal_reward_sum': terminals[0].reward_sum,
                'terminal_primitive_reward': float(terminals[0].primitive_rewards.sum())})
            episodes.append(transitions)
        spec = PlanningStateSpec.from_contract(cfg.contract, z_dim=cfg.raw['model']['z_dim'])
        learner = QueueLearner(cfg, spec, device=args.device, seed=1234)
        initial_actor_hash = state_dict_sha256(learner.actor.state_dict())
        run = {'upstream_commit': 'c70606f08cdca259b8dec03d4430926b5b8fac9d',
            'source_round': str(source_file.parent), 'episodes': len(episodes),
            'episode_order': expected_ids, 'episode_order_basis': 'original rollout save mtime_ns',
            'outcomes': dict(Counter(row['outcome'] for row in audit)),
            'initialization': 'scratch, seed 1234; no inherited actor/critic/optimizer',
            'bc_warmup_updates': 0, 'updates_per_recorded_episode': 8,
            'minimum_episodes': 2, 'replay_window_episodes': 200,
            'training_mode': 'offline chronological replay, one recorded episode per update call',
            'expected_critic_updates': 8 * (len(episodes) - 1),
            'remaining_differences': ['9D joint proprio, not 19D TCP/wrench',
                '8D joint action x 5 ticks, not 7D TCP action x 10',
                'existing H/C/D=10/5/5 full-VJP RTC and 15Hz',
                'frozen feature checkpoint and normalization from existing Stage 1',
                'offline historical policies, no new online collection or human interventions',
                'terminal partial chunks retain valid masks and actual duration'],
            'source_hashes': hashes, 'reward_audit': audit}
        write_json(out / 'comparison.json', run)
        window = deque(maxlen=200)
        for number, transitions in enumerate(episodes, 1):
            window.append(transitions)
            replay = ReplayBuffer(max(1, sum(len(e) for e in window)),
                feature_version_id=metadata['feature_version_id'],
                norm_stats_sha256=metadata['norm_stats_sha256'])
            for episode in window:
                for tr in episode:
                    replay.push(tr)
            if number >= 2:
                learner.update(replay, 8)
                learner.counts['rounds'] += 1
            if number % 10 == 0:
                status('training',episodes_seen=number, counts=dict(learner.counts))
                print(json.dumps({'episodes_seen': number, 'counts': learner.counts}), flush=True)
        if learner.counts['updates'] != run['expected_critic_updates']:
            raise RuntimeError('wrong update budget')
        bundle = learner.save(out / 'latest', metadata, replay)
        loaded = PolicyBundle.load(out / 'latest/policy.pt')
        actor = loaded.build_actor(device=args.device)
        critic = loaded.build_critic(device=args.device)
        all_transitions = [t for episode in episodes for t in episode]
        differences, maxima = [], []
        for begin in range(0, len(all_transitions), 256):
            batch = build_transition_batch(all_transitions[begin:begin+256],spec,
                                          contract=cfg.contract,device=args.device)
            with torch.no_grad():
                prediction = actor(batch.actor_state,batch.curr_reference_suffix).actions
                original = learner.actor(batch.actor_state,batch.curr_reference_suffix).actions
                torch.testing.assert_close(prediction,original,rtol=0,atol=0)
                q = critic.online(batch.critic_state,prediction)
                if not torch.isfinite(prediction).all() or not torch.isfinite(q).all():
                    raise RuntimeError('nonfinite reloaded prediction or Q')
                maxima.append(float(prediction.abs().max()))
                differences.extend((prediction.flatten(1)-batch.curr_reference_suffix).abs().mean(1).cpu().tolist())
        if max(maxima) > 1:
            raise RuntimeError('prediction outside action contract')
        if any(not math.isfinite(float(v)) for row in learner.metrics for v in row.values()):
            raise RuntimeError('nonfinite training metrics')
        if any(file_sha256(Path(p)) != digest for p,digest in hashes.items()):
            raise RuntimeError('a source changed during comparison')
        if initial_actor_hash == loaded.actor_state_sha256:
            raise RuntimeError('actor did not update')
        result = {'checkpoint': str(out/'latest/policy.pt'),
            'runtime_config': str(config), 'policy_version_id': bundle.policy_version_id,
            'counts': learner.counts, 'episodes': len(episodes), 'outcomes': run['outcomes'],
            'transitions': len(all_transitions), 'elapsed_s': time.monotonic()-started,
            'source_files_unchanged': True, 'all_metrics_finite': True,
            'checkpoint_reload_predictions_identical': True,
            'max_abs_action': max(maxima),
            'mean_abs_action_difference_from_teacher': float(np.mean(differences)),
            'real_robot_success_verified': False}
        status('complete', **result)
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}',elapsed_s=time.monotonic()-started)
        raise


if __name__ == '__main__':
    main()
