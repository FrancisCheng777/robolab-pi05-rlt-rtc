"""Human-labelled, critical-phase RLT with sequential episode/update rounds."""
import contextlib
import dataclasses
import json
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty
from types import SimpleNamespace
import uuid

from pi05_rlt_rtc.contracts.errors import ContractError
from .episode import EpisodeSignals
from .runtime import config_from_args, make_backend, scheduler_kwargs, write_json


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    write_json(temp, value)
    os.replace(temp, path)


class TerminalKeys:
    def __enter__(self):
        if not sys.stdin.isatty():
            raise ContractError('session requires an operator terminal')
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.escape = False
        return self

    def read(self):
        if not select.select([self.fd], [], [], 0)[0]:
            return []
        data = os.read(self.fd, 256)
        if not data:
            raise EOFError('operator terminal disconnected')
        keys = []
        for char in data.decode(errors='ignore'):
            if char == '\x1b':
                self.escape = True
            elif self.escape:
                if char.isalpha() or char == '~':
                    self.escape = False
            elif char.lower() in 'rbachqx':
                keys.append(char.lower())
        return keys

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


def run_episode(cfg, backend, metadata, owner, cameras, keys, out, *, seed):
    from pi05_rlt_rtc.runtime.controller import WallClockController
    from pi05_rlt_rtc.runtime.queue import ActionQueue
    from pi05_rlt_rtc.runtime.recording import AsyncRequestRecorder
    from .session_env import SessionEnv
    from .rollout import save_rollout, annotate
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    settings = cfg.raw['session']
    signals = EpisodeSignals(settings['max_approach_ticks'], settings['max_critical_ticks'])
    trial_id = out.name
    metadata = {**metadata, 'platform':'franka', 'clock':'wall',
                'gamma_per_tick':cfg.raw['training'].get('gamma_per_tick', .99)}
    recorder = AsyncRequestRecorder(out/'observations', metadata)
    env = SessionEnv(cfg, owner, cameras, trial_id)
    queue = ActionQueue(cfg.contract, fallback_policy='stop')
    def hook(tick, obs):
        signals.poll(keys.read(), tick)
        env.rlt_active = signals.critical_start_tick is not None
        return obs, signals.outcome is not None
    controller = WallClockController(queue=queue, backend=backend, env=env,
        tick_hz=cfg.control_hz, seed=seed, episode_id=trial_id, record_events=True,
        observation_recorder=recorder, tick_hook=hook, **scheduler_kwargs(cfg, metadata))
    report = {'safety_stops':1, 'stop_request_acknowledged':False}
    error = None
    try:
        report.update(controller.run(settings['max_approach_ticks']+settings['max_critical_ticks']+1).summary())
    except BaseException as exc:
        error = exc
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            env.stop()
            controller.drain_planner()
            report["planner_errors"] = controller._worker.jobs_failed
        except BaseException as exc:
            error = error or exc
            report['error'] = f'{type(exc).__name__}: {exc}'
        try:
            recorder.close()
        except BaseException as exc:
            error = error or exc
            report['error'] = f'{type(exc).__name__}: {exc}'
        report['stop_request_acknowledged'] = env.stop_request_acknowledged
        metadata['critical_start_tick'] = signals.critical_start_tick
        metadata['terminal_reason'] = signals.reason
        queue.ledger.write_jsonl(out/'ledger.jsonl')
        write_json(out/'controller_report.json', report)
        write_json(out/'runtime_manifest.json', metadata)
        save_rollout(out/'rollout.pt', events=controller.events, metadata=metadata,
                     report=report, contract=cfg.contract, trial_id=trial_id, hardware_commands=env.command_log)
    fault = error or any(report.get(k,0) for k in ('safety_stops','planner_errors','deadline_misses'))
    outcome = 'aborted' if fault or not env.stop_request_acknowledged else signals.outcome or 'aborted'
    annotate(SimpleNamespace(rollout=str(out/'rollout.pt'),out=str(out/'label'),
        outcome=outcome,annotator=metadata['annotator'], terminal_tick=signals.terminal_tick,
        allow_mock=metadata['mock']))
    if error:
        raise error
    return signals


def update_round(cfg, root, *, seed, device, mock=False):
    """Recover replay from committed labels; publish a round only after all saves.

    Round names are bound to the complete ordered trial list. Re-entering after
    a crash cannot apply the same episode's update twice to committed weights.
    """
    from pi05_rlt_rtc.learning.replay import ReplayBuffer
    from .training import train
    root = Path(root)
    labels = []
    total = 0
    for directory in sorted((root/'episodes').glob('*/label')):
        annotation = json.loads((directory/'annotation.json').read_text())
        if annotation.get('eligible'):
            labels.append(str(directory.resolve()))
            total += len(ReplayBuffer.load(directory/'replay.pt').learnable_transitions())
    settings = cfg.raw['session']
    rounds = sorted((root/'rounds').glob('round-*'))
    previous = rounds[-1] if rounds else None
    if previous and json.loads((previous/'training_sources.json').read_text())['replay_directories'] == labels:
        return previous/'latest'
    if len(labels) < settings['warmup_episodes'] or total < settings['min_replay_transitions']:
        return previous/'latest' if previous else None
    index = len(rounds)+1
    destination = root/'rounds'/f'round-{index:06d}'
    temporary = root/'rounds'/f'.pending-{uuid.uuid4().hex}'
    args = SimpleNamespace(runtime_config=str(cfg.path), policy=None, replay=labels,
        warmup_updates=0 if previous else settings['warmup_updates'], updates=settings['updates_per_episode'],
        seed=seed, learner_device=device, resume=str(previous/'latest') if previous else None,
        out=str(temporary), allow_mock=mock)
    train(args)
    os.replace(temporary,destination)
    return destination/'latest'


def bind_policy(backend, metadata, checkpoint, cfg, *, mode, exploration=None):
    if checkpoint is not None:
        from pi05_rlt_rtc.learning.checkpoint import PolicyBundle
        bundle = PolicyBundle.load(Path(checkpoint)/'policy.pt')
        if (bundle.feature_version_id != metadata['feature_version_id'] or
            bundle.norm_stats_sha256 != metadata['norm_stats_sha256'] or
            bundle.extra.get('mock') != metadata['mock'] or
            bundle.extra.get('runtime_contract') != cfg.wire_contract()):
            raise ContractError('session checkpoint differs from frozen features/runtime')
        backend.actor = bundle.build_actor(device=cfg.raw['model']['device'])
        backend.policy_version_id = bundle.policy_version_id
        metadata.update(policy_kind='actor',policy_version_id=bundle.policy_version_id)
    backend.critical_only = True
    enabled = mode == 'train' and (exploration is not False)
    backend.deterministic = not enabled
    metadata['exploration'] = enabled


def _run_session(args):
    cfg = config_from_args(args)
    if not args.execute:
        print('No devices opened. Use --execute after setting trained model assets. Keys: r start, b critical, a failure, c success, h home/open, x abort, q quit.')
        return 0
    if not args.annotator.strip():
        raise ContractError('--annotator is required')
    if args.mode == "train" and args.policy:
        raise ContractError("training resumes from session rounds; use --policy only for fixed-policy evaluation")
    root = Path(args.out).resolve()
    root.mkdir(parents=True,exist_ok=True)
    backend, metadata = make_backend(cfg,args)  # validate/load assets before opening devices
    identity = {k:metadata[k] for k in ('feature_version_id','norm_stats_sha256','runtime_contract','mock')}
    identity['training'] = cfg.raw['training']
    identity['mode'] = args.mode
    identity['exploration'] = args.mode == 'train' and args.exploration is not False
    manifest = root/'session.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ContractError('session directory belongs to different model/config/mode')
    atomic_json(manifest,identity)
    metadata.update(annotator=args.annotator,mode=args.mode)
    checkpoint = update_round(cfg,root,seed=args.seed,device=cfg.raw['model']['device']) if args.mode=='train' else None
    bind_policy(backend,metadata,checkpoint,cfg,mode=args.mode,exploration=args.exploration)
    from .session_env import OwnerClient
    from pi05_rlt_rtc.adapters.cameras import create_cameras
    with contextlib.ExitStack() as stack:
        keys = stack.enter_context(TerminalKeys())
        owner = OwnerClient(cfg)
        stack.callback(owner.close)
        cameras = create_cameras(cfg.raw['franka']['cameras'])
        stack.callback(cameras.close)
        print('Held. r=start; b=critical; a=fail; c=success; h=home/open; x=abort; q=quit.',flush=True)
        while True:
            commands = keys.read()
            if 'q' in commands:
                break
            if 'h' in commands:
                if not owner.rpc('status')['ready']:
                    print('Wait for confirmed hold before home.',flush=True)
                    continue
                owner.rpc('home')
                print('Accepted home/open in progress; r is disabled until held.',flush=True)
            elif 'r' in commands:
                if not owner.rpc('status')['ready']:
                    print('Wait for confirmed hold.',flush=True)
                    continue
                number = len(list((root/'episodes').glob('*')))+1
                signals = run_episode(cfg,backend,metadata,owner,cameras,keys,
                    root/'episodes'/f'{number:06d}-{uuid.uuid4().hex[:8]}',seed=args.seed+number)
                if signals.reason == 'operator_quit':
                    break
                if args.mode == 'train':
                    owner.wait_ready()
                    checkpoint = update_round(cfg,root,seed=args.seed,device=cfg.raw['model']['device'])
                    bind_policy(backend,metadata,checkpoint,cfg,mode=args.mode,exploration=args.exploration)
                    # Discard keys typed during updates; never buffer a motion request.
                    keys.read()
                print('Held. h=explicit home/open; r=next episode; q=quit.',flush=True)
            time.sleep(.02)
    return 0


def run_session(args):
    if not args.execute:
        return _run_session(args)
    import fcntl
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.session.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError('another process owns this session directory') from exc
        return _run_session(args)
