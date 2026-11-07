"""Teacher approach -> held queue handover -> relative GELLO demonstrations."""
import dataclasses
import time
from pathlib import Path
import numpy as np
from .episode import EpisodeSignals
from .runtime import write_json,scheduler_kwargs
from .takeover import TakeoverProtocol
from .correction_data import RelativeGello,TriggerReleased
from .correction_io import FrameWriter


def wait_for_trigger(owner,leader,keys,signals,*,timeout=30.):
    deadline=time.monotonic()+timeout;since=None
    print('[等待 GELLO] 机械臂已保持。按住扳机到底，保持 GELLO 姿态；就绪后自动接手。e/x/q 可中止。',flush=True)
    while time.monotonic()<deadline:
        signals.poll([k for k in keys.read() if k not in ('b','g')],signals.critical_start_tick+1)
        if signals.outcome:return None
        q,width=leader.read()
        now=time.monotonic()
        if 0<=width<=.005:
            if since is None:since=now
            if now-since>=.15 and owner.rpc('status')['ready']:
                return np.asarray(q),width
        else:since=None
        time.sleep(.02)
    raise TimeoutError('30 秒内未确认 GELLO 扳机按到底')


def run_correction_episode(cfg,backend,metadata,owner,cameras,keys,out,*,seed):
    from .session import operator_start,atomic_json
    from .session_env import SessionEnv,OperatorInterrupt
    from pi05_rlt_rtc.runtime.controller import WallClockController
    from pi05_rlt_rtc.runtime.queue import ActionQueue
    from pi05_rlt_rtc.runtime.recording import AsyncRequestRecorder
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    settings=cfg.raw['session']
    if getattr(backend,'actor',None) is not None:
        raise ValueError('correction collection must use the frozen teacher, not an old actor')
    signals=EpisodeSignals(settings['max_approach_ticks'],settings['max_critical_ticks'])
    protocol=TakeoverProtocol(settings['takeover_gate'])
    if protocol.gate.config['mode']!='auto':raise ValueError('GELLO handover needs auto gate')
    env=SessionEnv(cfg,owner,cameras,out.name)
    recorder=AsyncRequestRecorder(out/'teacher_observations',metadata)
    queue=ActionQueue(cfg.contract,fallback_policy='stop')
    manifest=dict(format='gello-correction-episode-v1',trial_id=out.name,metadata=metadata,
                  runtime_contract=cfg.wire_contract(),takeover_gate=settings['takeover_gate'],
                  training_kind='human_BC_only',rl_eligible=False,complete=False,eligible_bc=False)
    atomic_json(out/'correction.json',manifest)
    def keys_hook(tick,obs):
        pressed=keys.read()
        signals.poll([k for k in pressed if k not in ('b','g')],tick)
        if 'g' in pressed and signals.outcome is None:
            protocol.confirm_grasp(tick)
            print('[抓稳已确认] 进入区域后自动切换 GELLO；请保持扳机按到底。',flush=True)
        return obs,signals.outcome is not None or signals.critical_start_tick is not None
    def pose_hook(tick,obs):
        event=protocol.poll(signals,[],tick,obs.sensor_metadata.get('takeover_sample',{}),time.monotonic())
        if event and event['trigger']:
            print('[区域已触发] 正在保持机械臂并结束 teacher 队列，然后对齐 GELLO 相对起点。',flush=True)
        return obs  # env.rlt_active stays False; there is never an AC during demonstration.
    controller=WallClockController(queue=queue,backend=backend,env=env,tick_hz=cfg.control_hz,
        seed=seed,episode_id=out.name,record_events=True,observation_recorder=recorder,
        tick_hook=keys_hook,observation_hook=pose_hook,**scheduler_kwargs(cfg,metadata))
    writer=None;error=None;report={};handover=None
    try:
        print('[示范轮开始] teacher 抓取；抓稳按 g；区域内自动接 GELLO。c=成功，a=失败，e=回位，x=保持，q=退出。',flush=True)
        try:
            report=controller.run(settings['max_approach_ticks']+1).summary()
        finally:
            env.stop();controller.drain_planner();recorder.close()
        if any(report.get(k,0) for k in ('safety_stops','planner_errors','deadline_misses','underruns')):
            raise RuntimeError('teacher approach stopped with a runtime fault')
        if signals.outcome is None and signals.critical_start_tick is not None:
            ready=wait_for_trigger(owner,owner.correction_leader,keys,signals)
            if ready is not None and signals.outcome is None:
                if not operator_start(keys,owner):raise OperatorInterrupt('handover interrupted')
                obs=env.observe()
                leader_q,width=owner.correction_leader.read()
                mapper=RelativeGello(obs.states[:7],leader_q,hz=cfg.control_hz)
                handover=dict(robot_q=obs.states[:7].tolist(),leader_q=leader_q.tolist(),
                              width_command_m=0.,observation_tick=env._tick,
                              monotonic=time.monotonic(),joint_mapping='relative_anchor',
                              max_reference_speed_rad_s=mapper.speed)
                writer=FrameWriter(out/'human_frames.jsonl')
                print('[GELLO 已接管] 相对起点已对齐。持续按住扳机并演示插入；成功 c，失败 a。松扳机将保持，不会张爪。',flush=True)
                started=time.monotonic();step=0
                while signals.outcome is None:
                    target=started+step/cfg.control_hz
                    if target>time.monotonic():time.sleep(target-time.monotonic())
                    tick=env._tick
                    signals.poll([k for k in keys.read() if k not in ('b','g')],tick)
                    if signals.outcome:break
                    if time.monotonic()-target>=1/cfg.control_hz:raise TimeoutError('GELLO observation loop overrun')
                    obs=env.observe();leader_q,width=owner.correction_leader.read()
                    action=mapper.action(leader_q,width,obs.states[:7])
                    writer.check()
                    if time.monotonic()-target>=1/cfg.control_hz:raise TimeoutError('GELLO capture exceeded tick deadline')
                    audit=owner.rpc('action',action=action.tolist())
                    mapper.reference=np.asarray(audit['q_target'])
                    writer.submit(obs,audit,leader_q)
                    env._tick+=1;step+=1
    except TriggerReleased as exc:
        signals.outcome,signals.reason='aborted','gello_trigger_released'
        print('[示范已保持] '+str(exc),flush=True)
    except OperatorInterrupt:
        signals.poll(keys.read(),env._tick)
        if signals.outcome is None:signals.outcome,signals.reason='aborted','operator_interrupt'
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        signals.outcome,signals.reason='aborted','correction_fault'
        print('[示范已停止] '+error,flush=True)
        if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
    finally:
        try:env.stop()
        except BaseException as exc:error=error or f'hold: {exc}'
        try:controller.drain_planner();recorder.close()
        except BaseException as exc:error=error or f'planner/recorder: {exc}'
        if writer is not None:
            try:writer.close()
            except BaseException as exc:error=error or f'storage: {exc}'
        if error:signals.outcome,signals.reason='aborted','correction_fault'
        count=writer.count if writer else 0
        manifest.update(complete=True,outcome=signals.outcome or 'aborted',reason=signals.reason,
            error=error,grasp_confirmed=protocol.armed,grasp_confirmation_tick=protocol.confirmation_tick,
            gate_trigger_tick=signals.critical_start_tick,handover=handover,frames=count,
            frames_sha256=writer.digest.hexdigest() if writer else None,
            eligible_bc=bool(not error and protocol.armed and handover and signals.outcome=='success' and
                             count>=cfg.contract.horizon and env.stop_request_acknowledged),
            stop_acknowledged=env.stop_request_acknowledged)
        atomic_json(out/'correction.json',manifest)
        write_json(out/'takeover_events.json',protocol.events)
        write_json(out/'teacher_report.json',report)
        write_json(out/'teacher_hardware_commands.json',env.command_log)
        queue.ledger.write_jsonl(out/'teacher_ledger.jsonl')
        print(f"[示范已保存] {out.name}：{manifest['outcome']}，{count} 帧，BC 可用={manifest['eligible_bc']}。",flush=True)
    if error:raise RuntimeError(error)
    return signals
