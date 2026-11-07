"""Serial open-loop DROID chunks, using the accepted independent CPU owner."""
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import numpy as np
import websockets.sync.client
from openpi_client import msgpack_numpy
from pi05_rlt_rtc.adapters.cameras_panda import create_provider
from pi05_rlt_rtc.deployment.bridge_preflight import prepare_socket, probe_bridge
from pi05_rlt_rtc.deployment.managed_eval import lock_file, stop_child
from pi05_rlt_rtc.deployment.session import TerminalKeys, operator_start
from pi05_rlt_rtc.deployment.session_env import OwnerClient, SessionEnv
from pi05_rlt_rtc.deployment.stage2_session import require_robot_idle, station_status
from action_adapter import convert, gripper_state

ROOT = Path(__file__).resolve().parent
CODE = Path('/home/heinrich/robolab-pi05-afixed-collect100-20261103')
PROMPT = 'insert the plug into the power strip'
SOCKET = '/tmp/robolab-rlt.sock'
END_KEYS = set('qexac')

class StopEpisode(Exception):
    def __init__(self, key): self.key = key

def check_keys(keys):
    found = keys.read()
    for k in 'qexac':
        if k in found: raise StopEpisode(k)

def request_policy(request, keys=None):
    with websockets.sync.client.connect('ws://127.0.0.1:18766', compression=None,
                                        max_size=8*1024*1024, open_timeout=3) as conn:
        meta = msgpack_numpy.unpackb(conn.recv(timeout=3))
        assert meta['policy']=='pi05_droid' and meta['rtc'] is False and meta['actor'] is False
        assert meta['camera_profile']=='wrist_only_missing_exterior_masked'
        if request is None: return meta
        conn.send(msgpack_numpy.packb(request))
        deadline = time.monotonic()+20
        while time.monotonic()<deadline:
            if keys is not None: check_keys(keys)
            try: response=conn.recv(timeout=.04)
            except TimeoutError: continue
            if isinstance(response,str): raise RuntimeError(response)
            actions=np.asarray(msgpack_numpy.unpackb(response)['actions'])
            if actions.shape!=(15,8) or not np.isfinite(actions).all():
                raise ValueError('invalid DROID action chunk')
            return actions
        raise TimeoutError('DROID inference exceeded 20 seconds; robot remains held')

def configuration():
    import yaml
    raw=yaml.safe_load((CODE/'configs/runtime_franka_stage2.yaml').read_text())
    # Reuse only hardware profiles; do not construct an RTC planner/scheduler.
    raw['franka']['joint_command_mode']='measured_delta'
    raw['franka']['socket']=SOCKET
    raw['prompt']=PROMPT
    return SimpleNamespace(raw=raw, asset=lambda value,label:Path(value))

def make_request(obs):
    return {'observation/exterior_image_1_left':np.zeros((224,224,3),np.uint8),
            'observation/wrist_image_left':obs.images['wrist'],
            'observation/joint_position':obs.states[:7],
            'observation/gripper_position':gripper_state(float(obs.states[7]+obs.states[8])),
            'prompt':PROMPT}

def episode(cfg, owner, cameras, keys, directory):
    directory.mkdir(parents=True, exist_ok=False)
    env=SessionEnv(cfg,owner,cameras,directory.name)
    tick=0
    outcome='aborted'; end_key='x'
    print('[运行中] 原始 pi05_droid；无 RTC / AC，不需要 b；c成功/a失败/e回位/x保持/q退出。',flush=True)
    try:
        with (directory/'actions.jsonl').open('w') as log:
            for chunk in range(113):  # bounded to at most 904 actions at 15 Hz
                check_keys(keys)
                owner.hold()  # serial inference: no actions execute while planning
                obs=env.observe()
                np.savez_compressed(directory/f'input-{chunk:04d}.npz',q=obs.states[:7],
                    width=obs.states[7]+obs.states[8],wrist=obs.images['wrist'])
                before=time.monotonic()
                actions=request_policy(make_request(obs),keys)
                inference_s=time.monotonic()-before
                np.save(directory/f'predicted-{chunk:04d}.npy',actions)
                check_keys(keys)
                if not operator_start(keys,owner): raise StopEpisode('x')
                for row in range(8):
                    check_keys(keys)
                    started=time.monotonic()
                    command,conversion=convert(actions[row])
                    audit=owner.rpc('action',action=command.tolist())
                    log.write(json.dumps(dict(tick=tick,chunk=chunk,row=row,
                        inference_s=inference_s,raw_action=actions[row].tolist(),
                        conversion=conversion,audit=audit))+'\n');log.flush()
                    tick+=1
                    while time.monotonic()-started < 1/15:
                        check_keys(keys);time.sleep(.005)
            outcome='timeout'
    except StopEpisode as stop:
        end_key=stop.key
        outcome={'c':'success','a':'failure'}.get(end_key,'aborted')
    except BaseException:
        outcome='error';raise
    finally:
        try: owner.hold()
        finally:
            (directory/'annotation.json').write_text(json.dumps(dict(outcome=outcome,
                executed_actions=tick,policy='pi05_droid',rtc=False,actor=False),indent=2))
    print(f'[本轮结束] {outcome}；已保存 {directory}；e回位，r再试，q退出。',flush=True)
    return end_key

def main():
    os.environ['PI05_CONTROL_CPU']='12'
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,os.sched_getaffinity(0)-{12})
    def interrupt(signum,frame): raise KeyboardInterrupt()
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupt)
    cfg=configuration()
    with lock_file(Path.home()/'.cache/pi05-stage1-8000/session.lock'):
        assert station_status()['accepted_settings_restored'],'Restore station settings first'
        metadata=request_policy(None)
        require_robot_idle()
        prepare_socket(SOCKET)
        out=ROOT/'evaluations'/time.strftime('%Y%m%d_%H%M%S')
        out.mkdir(parents=True,exist_ok=False)
        (out/'manifest.json').write_text(json.dumps(dict(metadata,control_hz=15,
            execute_rows=8,planning='hold_then_infer_then_execute',
            joint_delta_cap_rad=.05,gripper='DROID threshold .5, 0=open / 1=closed',
            prompt=PROMPT),indent=2))
        bridge=owner=cameras=None
        try:
            with (out/'bridge.log').open('a') as log:
                bridge=subprocess.Popen(['bash',str(CODE/'scripts/start_franka_bridge.sh'),'--execute'],
                    cwd=CODE,env={**os.environ,'PI05_BRIDGE_SOCKET':SOCKET},
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            expected=json.loads(Path(cfg.raw['franka']['bridge_config']).read_text())
            deadline=time.monotonic()+25
            while True:
                if bridge.poll() is not None:raise RuntimeError('bridge exited; see '+str(out/'bridge.log'))
                try:probe_bridge(SOCKET,expected);break
                except Exception:
                    if time.monotonic()>deadline:raise
                    time.sleep(.1)
            owner=OwnerClient(cfg)
            cameras=create_provider(**cfg.raw['franka']['cameras']['kwargs'])
            # Validate a live frame and state while held, before offering r.
            obs=SessionEnv(cfg,owner,cameras,'preflight').observe()
            np.savez_compressed(out/'preflight.npz',q=obs.states[:7],wrist=obs.images['wrist'])
            with TerminalKeys(emergency=owner.interrupt) as keys:
                print('[DROID 已就绪｜保持中] r开始；e先张爪再回位；x保持；q退出。无需按 b。',flush=True)
                while True:
                    pressed=keys.read()
                    if 'q' in pressed:break
                    if 'e' in pressed:
                        owner.hold();owner.rpc('home')
                        print('[回位中] 等待张爪、回位完成后再按 r。',flush=True)
                    elif 'x' in pressed:owner.hold()
                    elif 'r' in pressed:
                        if not operator_start(keys,owner):
                            print('尚未停稳或回位未完成，请稍后再按 r。',flush=True);continue
                        key=episode(cfg,owner,cameras,keys,out/('episode-'+uuid.uuid4().hex[:10]))
                        if key=='q':break
                        if key=='e':owner.rpc('home')
                    time.sleep(.02)
        finally:
            if owner is not None:owner.close()
            if cameras is not None:cameras.close()
            stop_child(bridge)
            if bridge is not None:prepare_socket(SOCKET)

if __name__=='__main__':
    main()
