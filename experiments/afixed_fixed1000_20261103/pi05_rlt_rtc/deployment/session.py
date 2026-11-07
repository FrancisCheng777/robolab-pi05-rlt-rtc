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
import threading
import signal
from collections import deque
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
    def __init__(self, emergency=None):
        self.emergency = emergency
        self.pending = deque()
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.error = None

    def __enter__(self):
        if not sys.stdin.isatty():
            raise ContractError('session requires an operator terminal')
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        attrs = termios.tcgetattr(self.fd)
        attrs[0] &= ~(termios.IXON | termios.IXOFF)
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        self.escape = False
        self.thread = threading.Thread(target=self._listen, daemon=True, name='operator-keys')
        self.thread.start()
        return self

    def _read_terminal(self):
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
            elif char.lower() in 'rbacheqx':
                keys.append(char.lower())
        return keys

    def _listen(self):
        try:
            while not self.stopping.wait(.01):
                keys = self._read_terminal()
                with self.lock:
                    self.pending.extend(keys)
                    if any(k in keys for k in ('e', 'x', 'q')) and self.emergency:
                        self.emergency()
        except BaseException as exc:
            self.error = exc
            if self.emergency:
                with contextlib.suppress(Exception):
                    self.emergency()

    def read(self):
        if self.error:
            raise self.error
        with self.lock:
            keys = list(self.pending)
            self.pending.clear()
        return keys

    def __exit__(self, *_):
        self.stopping.set()
        self.thread.join(timeout=3.)
        termios.tcsetattr(self.fd, termios.TCSANOW, self.saved)


def operator_start(keys, owner, *, home=False):
    """A queued stop wins over a start; a subsequent stop interrupts the start."""
    with keys.lock:
        if keys.error:
            raise keys.error
        if any(k in keys.pending for k in ('q', 'x', 'e')):
            return False
        if not owner.rpc('status')['ready']:
            return False
        if home:
            owner.rpc('home')
        else:
            owner.arm_episode()
        return True


def run_episode(cfg, backend, metadata, owner, cameras, keys, out, *, seed):
    from pi05_rlt_rtc.runtime.controller import WallClockController
    from pi05_rlt_rtc.runtime.queue import ActionQueue
    from pi05_rlt_rtc.runtime.recording import AsyncRequestRecorder
    from .session_env import SessionEnv, OperatorInterrupt
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
        was_critical = signals.critical_start_tick is not None
        signals.poll(keys.read(), tick)
        env.rlt_active = signals.critical_start_tick is not None
        if signals.outcome is None:
            if tick == 0:
                print(f'[运行中] 第 {trial_id} 轮已开始。b=进入关键阶段；a=失败；c=成功；e=停止并回位/张爪；x=保持；q=退出。', flush=True)
            if env.rlt_active and not was_critical:
                print('[关键阶段] 已收到 b。a=失败；c=成功；e=停止并回位/张爪；x=保持；q=退出。', flush=True)
        return obs, signals.outcome is not None
    controller = WallClockController(queue=queue, backend=backend, env=env,
        tick_hz=cfg.control_hz, seed=seed, episode_id=trial_id, record_events=True,
        observation_recorder=recorder, tick_hook=hook, **scheduler_kwargs(cfg, metadata))
    report = {'safety_stops':1, 'stop_request_acknowledged':False}
    error = None
    try:
        print(f'[准备本轮] 第 {trial_id} 轮：正在读取观测、生成首段动作；e/x/q 可中止。就绪后会显示「运行中」。', flush=True)
        report.update(controller.run(settings['max_approach_ticks']+settings['max_critical_ticks']+1).summary())
    except OperatorInterrupt:
        signals.poll(keys.read(), env._tick)
        report['safety_stops'] = 0
    except BaseException as exc:
        error = exc
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            env.stop()
        except BaseException as exc:
            error = error or exc
            report.setdefault('error', f'{type(exc).__name__}: {exc}')
            report.setdefault('cleanup_errors', []).append(f'hold: {type(exc).__name__}: {exc}')
        try:
            controller.drain_planner()
            report["planner_errors"] = controller._worker.jobs_failed
            timings = controller._worker.timings
            write_json(out/'planner_timings.json', timings)
            report['last_planner_request'] = timings[-1] if timings else None
        except BaseException as exc:
            error = error or exc
            report.setdefault('error', f'{type(exc).__name__}: {exc}')
            report.setdefault('cleanup_errors', []).append(f'planner: {type(exc).__name__}: {exc}')
        try:
            recorder.close()
        except BaseException as exc:
            error = error or exc
            report.setdefault('error', f'{type(exc).__name__}: {exc}')
            report.setdefault('cleanup_errors', []).append(f'recorder: {type(exc).__name__}: {exc}')
        report['stop_request_acknowledged'] = env.stop_request_acknowledged
        stop_events = [e for e in queue.ledger.events if e.get('event') in
                       ('trial_stopped_wallclock', 'plan_rejected', 'deadline_miss', 'planner_error')]
        if stop_events:
            report['controller_stop_reason'] = stop_events[-1].get('reason', stop_events[-1]['event'])
        metadata['critical_start_tick'] = signals.critical_start_tick
        metadata['terminal_reason'] = signals.reason
        queue.ledger.write_jsonl(out/'ledger.jsonl')
        write_json(out/'controller_report.json', report)
        write_json(out/'runtime_manifest.json', metadata)
        save_rollout(out/'rollout.pt', events=controller.events, metadata=metadata,
                     report=report, contract=cfg.contract, trial_id=trial_id, hardware_commands=env.command_log)
    fault = error or any(report.get(k,0) for k in ('safety_stops','planner_errors','deadline_misses'))
    if fault:
        if report.get('deadline_misses'):
            print('[本轮已停止：规划超时] 下一段动作未及时返回，已保持。此轮中断不计入训练；需要重新按 r 开始新一轮。', flush=True)
        print('Controller stopped: '+json.dumps({key: report.get(key) for key in
            ('ticks_run', 'controller_stop_reason', 'deadline_misses', 'underruns', 'planner_errors', 'error')}), flush=True)
    outcome = 'aborted' if fault or not env.stop_request_acknowledged else signals.outcome or 'aborted'
    annotate(SimpleNamespace(rollout=str(out/'rollout.pt'),out=str(out/'label'),
        outcome=outcome,annotator=metadata['annotator'], terminal_tick=signals.terminal_tick,
        allow_mock=metadata['mock']))
    if error:
        raise error
    return signals


def training_batch(cfg, root):
    """Count eligible episodes not yet consumed by a published learner round."""
    root = Path(root)
    labels = []
    for directory in sorted((root/'episodes').glob('*/label')):
        annotation = json.loads((directory/'annotation.json').read_text())
        if annotation.get('eligible') and annotation.get('outcome') in ('success', 'failure'):
            labels.append(str(directory.resolve()))
    rounds = sorted((root/'rounds').glob('round-*'))
    previous = rounds[-1] if rounds else None
    used = json.loads((previous/'training_sources.json').read_text())['replay_directories'] if previous else []
    if not set(used).issubset(labels):
        raise ContractError('previously trained episodes are missing or no longer eligible')
    pending = [label for label in labels if label not in set(used)]
    every = cfg.raw['session'].get('update_every_episodes', 1)
    required = max(every, cfg.raw['session']['warmup_episodes'] if previous is None else every)
    # Include a startup backlog (or extra short episodes needed by the transition
    # gate). Truncating to the first ten could otherwise leave that gate stuck.
    return previous, used, pending, required


def update_round(cfg, root, *, seed, device, mock=False):
    """Recover replay from committed labels; publish a round only after all saves.

    Round names are bound to the complete ordered trial list. Re-entering after
    a crash cannot apply the same episode's update twice to committed weights.
    """
    from pi05_rlt_rtc.learning.replay import ReplayBuffer
    from .training import train
    root = Path(root)
    settings = cfg.raw['session']
    previous, used, pending, required = training_batch(cfg, root)
    if len(pending) < required:
        print(f'[采集进度] 本批新增有效轨迹 {len(pending)}/{required}；成功和失败计入，中断不计入。', flush=True)
        return previous/'latest' if previous else None
    labels = used + pending
    total = sum(len(ReplayBuffer.load(Path(directory)/'replay.pt').learnable_transitions()) for directory in labels)
    if total < settings['min_replay_transitions']:
        print(f'[等待有效数据] 可学习 transition {total}/{settings["min_replay_transitions"]}，尚未训练。', flush=True)
        return previous/'latest' if previous else None
    index = len(list((root/'rounds').glob('round-*')))+1
    destination = root/'rounds'/f'round-{index:06d}'
    temporary = root/'rounds'/f'.pending-{uuid.uuid4().hex}'
    updates = settings.get('updates_per_batch')
    if updates is None:
        updates = settings['updates_per_episode'] * (len(pending) if settings.get('update_every_episodes', 1) > 1 else 1)
    status = {'round': index, 'new_episodes': len(pending), 'total_episodes': len(labels),
              'critic_updates': updates, 'started_at_unix': time.time(), 'state': 'training'}
    atomic_json(root/'batch_status.json', status)
    print(f'[本批已满，可以训练] 新增 {len(pending)} 条有效轨迹；开始第 {index} 批训练，'
          f'{updates} 次 critic 更新。暂停采集，完成后使用新权重；训练期间按 r 不会排队启动。', flush=True)
    started = time.monotonic()
    args = SimpleNamespace(runtime_config=str(cfg.path), policy=None, replay=labels,
        warmup_updates=0 if previous else settings['warmup_updates'], updates=updates,
        seed=seed, learner_device=device, resume=str(previous/'latest') if previous else None,
        out=str(temporary), allow_mock=mock)
    try:
        train(args)
        os.replace(temporary,destination)
    except BaseException:
        atomic_json(root/'batch_status.json', {**status, 'state': 'failed', 'elapsed_s': time.monotonic()-started})
        raise
    elapsed = time.monotonic()-started
    atomic_json(root/'batch_status.json', {**status, 'state': 'complete', 'elapsed_s': elapsed,
                'checkpoint': str(destination/'latest')})
    print(f'[训练完成] 第 {index} 批耗时 {elapsed:.1f} 秒；权重已保存。', flush=True)
    return destination/'latest'


def bind_policy(backend, metadata, checkpoint, cfg, *, mode, exploration=None):
    if checkpoint is not None:
        # Published round directories are immutable. Reusing a round must not
        # rebuild its actor (or pretend a new version was trained) after a trial.
        actor_device = cfg.raw['model'].get('actor_inference_device', cfg.raw['model']['device'])
        binding = (str(Path(checkpoint).resolve()), str(actor_device),
                   metadata['feature_version_id'], metadata['norm_stats_sha256'],
                   metadata['mock'], json.dumps(cfg.wire_contract(), sort_keys=True))
        if getattr(backend, '_bound_policy_checkpoint', None) != binding or backend.actor is None:
            from pi05_rlt_rtc.learning.checkpoint import PolicyBundle
            bundle = PolicyBundle.load(Path(checkpoint)/'policy.pt')
            if (bundle.feature_version_id != metadata['feature_version_id'] or
                bundle.norm_stats_sha256 != metadata['norm_stats_sha256'] or
                bundle.extra.get('mock') != metadata['mock'] or
                bundle.extra.get('runtime_contract') != cfg.wire_contract()):
                raise ContractError('session checkpoint differs from frozen features/runtime')
            backend.actor = bundle.build_actor(device=actor_device)
            backend.policy_version_id = bundle.policy_version_id
            backend._bound_policy_checkpoint = binding
            print(f'[新权重已就绪] {bundle.policy_version_id[:26]}；下一轮按 r 开始，b 后由此 Actor 接管。', flush=True)
        metadata.update(policy_kind='actor',policy_version_id=backend.policy_version_id)
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
    if getattr(args, 'planner_url', None) and not getattr(args, 'resident_teacher', False) and (args.mode != 'eval' or cfg.raw['policy']['kind'] != 'teacher'):
        raise ContractError('resident planner sessions currently support teacher eval only')
    from .bridge_preflight import probe_bridge
    probe_bridge(cfg.raw['franka']['socket'], json.loads(
        cfg.asset(cfg.raw['franka']['bridge_config'], 'bridge config').read_text()))
    print('Bridge connection verified. Loading model once; keep this session open for subsequent trials.', flush=True)
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
    if getattr(args, 'collect_only', False) and list((root/'rounds').glob('round-*')):
        raise ContractError('collection-only is for initial teacher data; use a new session after actor rounds exist')
    checkpoint = update_round(cfg,root,seed=args.seed,device=cfg.raw['model']['device']) if args.mode=='train' and not getattr(args, 'collect_only', False) else None
    if args.mode == 'eval' and getattr(args, 'resident_teacher', False) and args.policy:
        checkpoint = Path(args.policy).parent
    bind_policy(backend,metadata,checkpoint,cfg,mode=args.mode,exploration=args.exploration)
    from .session_env import OwnerClient
    from pi05_rlt_rtc.adapters.cameras import create_cameras
    with contextlib.ExitStack() as stack:
        owner = OwnerClient(cfg)
        stack.callback(owner.close)
        keys = stack.enter_context(TerminalKeys(emergency=owner.interrupt))
        cameras = create_cameras(cfg.raw['franka']['cameras'])
        stack.callback(cameras.close)
        print('Held. r=start; b=critical; a=fail; c=success; e=stop/home/open; x=hold; q=quit.',flush=True)
        pending_home = False
        home_in_progress = False
        while True:
            commands = keys.read()
            if 'q' in commands:
                break
            if 'x' in commands:
                pending_home = home_in_progress = False
                owner.hold()
                print('Held. e=home/open; r=next episode; q=quit.', flush=True)
                continue
            if 'e' in commands or 'h' in commands:
                pending_home = True
                if 'h' in commands:
                    owner.interrupt()
                print('[复位请求] 等待保持后先张爪，再回初始位姿；x 取消，q 退出。', flush=True)
            if pending_home:
                if 'r' in commands:
                    print('[尚未开始] 正在等待回位，r 未执行；看到「回位完成」后请重新按 r。', flush=True)
                if operator_start(keys, owner, home=True):
                    pending_home = False
                    home_in_progress = True
                    print('[复位中] 先在当前位置张开夹爪，确认完成后回初始位姿；x=保持，q=退出。', flush=True)
                time.sleep(.02)
                continue
            if home_in_progress:
                if owner.rpc('status')['ready']:
                    home_in_progress = False
                    print('[回位完成｜等待指令] 已回到采集初始位姿并张爪；请按 r 开始新一轮，q 退出。', flush=True)
                if 'r' in commands:
                    print('[尚未开始] 回位阶段收到的 r 不会排队执行；请现在重新按 r。' if not home_in_progress else '[尚未开始] 仍在回位，r 未执行；回位完成后请重新按 r。', flush=True)
                # Never buffer r pressed while home is still underway.
                time.sleep(.02)
                continue
            if 'r' in commands:
                print('[收到 r] 正在检查就绪状态；e/x/q 可中止。', flush=True)
                if not operator_start(keys, owner):
                    print('[尚未开始] 未确认保持就绪，或有停止指令待处理；就绪后请重新按 r。',flush=True)
                    continue
                number = len(list((root/'episodes').glob('*')))+1
                signals = run_episode(cfg,backend,metadata,owner,cameras,keys,
                    root/'episodes'/f'{number:06d}-{uuid.uuid4().hex[:8]}',seed=args.seed+number)
                if signals.reason == 'operator_quit':
                    break
                if signals.reason == 'operator_home':
                    pending_home = True
                if args.mode == 'train' and not getattr(args, 'collect_only', False):
                    owner.wait_ready()
                    checkpoint = update_round(cfg,root,seed=args.seed,device=cfg.raw['model']['device'])
                    bind_policy(backend,metadata,checkpoint,cfg,mode=args.mode,exploration=args.exploration)
                    # Discard buffered start, but preserve stop/home/quit intent.
                    with keys.lock:
                        keys.pending = deque(k for k in keys.pending if k in ('q','x','e'))
                print('[本轮结束｜等待回位] e=回位/张爪；x=保持；q=退出。' if pending_home else
                      '[本轮结束｜已保持] e=回位/张爪；r=开始新一轮；q=退出。', flush=True)
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
        def terminate(signum, frame):
            raise KeyboardInterrupt(f'session signal {signum}')
        saved = {sig: signal.signal(sig, terminate) for sig in (signal.SIGHUP, signal.SIGTERM)}
        try:
            return _run_session(args)
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
