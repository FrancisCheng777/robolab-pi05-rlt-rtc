"""Prepared Stage-2 entry point. Default check is read-only and opens no hardware."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .managed_eval import lock_file, owned_process, run_eval
from .rpc import RemotePlanner
from pi05_rlt_rtc.config.runtime_config import load_runtime_config


def replay_inventory(root, cfg):
    rows=[]
    for path in sorted(Path(root).glob('episodes/*/label/annotation.json')):
        label=json.loads(path.read_text())
        manifest=json.loads((path.parent/'runtime_manifest.json').read_text())
        rows.append({'trial_id':label['trial_id'],'outcome':label['outcome'],
            'eligible':bool(label.get('eligible') and label.get('outcome') in ('success','failure')
                and label.get('mock') is False and manifest.get('runtime_contract')==cfg.wire_contract()),
            'learnable':label.get('replay_counts',{}).get('learnable',0),
            'directory':str(path.parent)})
    return rows



def station_status():
    """Read existing accepted station settings; no USB/network/robot commands."""
    status = {'realtime_kernel': Path('/sys/kernel/realtime').exists() and
              Path('/sys/kernel/realtime').read_text().strip() == '1'}
    try:
        serial = Path('/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAAL8Y3V-if00-port0').resolve(strict=True)
        status['gripper_usb_latency_ms'] = int((Path('/sys/bus/usb-serial/devices')/serial.name/'latency_timer').read_text())
        irqs = list(Path('/sys/class/net/enp5s0/device/msi_irqs').iterdir())
        if len(irqs) != 1:
            raise RuntimeError('NIC IRQ topology changed')
        cpu = int((Path('/proc/irq')/irqs[0].name/'effective_affinity_list').read_text())
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                if (proc/'comm').read_text().strip() == f'ksoftirqd/{cpu}':
                    pid=int(proc.name)
                    status['network_softirq_policy'] = os.sched_getscheduler(pid)
                    status['network_softirq_priority'] = os.sched_getparam(pid).sched_priority
                    break
            except (OSError,ValueError):
                continue
    except (OSError,ValueError,RuntimeError) as exc:
        status['error'] = str(exc)
    status['accepted_settings_restored'] = bool(status.get('realtime_kernel') and
        status.get('gripper_usb_latency_ms') == 1 and
        status.get('network_softirq_policy') == os.SCHED_FIFO and
        status.get('network_softirq_priority') == 50)
    return status


def require_robot_idle(ip='172.16.0.40'):
    """Read libfranka state only. Never clear errors, initialize control or move."""
    try:
        reply = subprocess.run(['echo_robot_state', ip], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f'无法只读检查 Panda 状态：{exc}') from exc
    if reply.returncode:
        raise RuntimeError('Panda 状态读取失败：' + reply.stderr[-1500:])
    samples = [json.loads(line) for line in reply.stdout.splitlines() if line.startswith('{')]
    if not samples:
        raise RuntimeError('Panda 状态检查没有返回有效数据')
    state = samples[-1]
    mode, errors = state.get('robot_mode'), state.get('current_errors')
    if mode != 'Idle' or errors != []:
        raise RuntimeError(f"Panda 当前为 {mode}，当前错误={errors}，上次运动错误={state.get('last_motion_errors')}。"
            '请在 Franka Desk 按提示恢复到无故障 Idle 后重试。电脑通信设置不会自动清除机器人 Reflex；未启动 bridge 或运动。')
    print('Panda 状态检查通过：Idle，无当前错误。', flush=True)
    return state

def require_output_storage(root):
    """Check the real mount and durable writes before starting any hardware."""
    import tempfile
    root = Path(root).absolute()
    parts = root.parts
    if len(parts) >= 4 and parts[1] == 'media':
        mount = Path(*parts[:4])
        if not mount.is_mount():
            raise RuntimeError(f'数据盘未挂载：{mount}；请重新连接硬盘。未启动硬件。')
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix='.storage-check-', dir=root) as f:
            f.write(b'pi05 storage preflight\n');f.flush();os.fsync(f.fileno())
    except OSError as exc:
        raise RuntimeError(f'数据目录不可写：{root}：{exc}；未启动硬件。') from exc


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('check','collect','train','eval'),nargs='?',default='check')
    parser.add_argument('--runtime-config',required=True)
    parser.add_argument('--session-root',required=True)
    parser.add_argument('--policy',help='policy.pt for fixed actor eval')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    cfg=load_runtime_config(args.runtime_config)
    station=station_status()
    root=Path(args.session_root).resolve()
    state_path=Path.home()/'.cache/pi05-stage1-8000/model.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else None
    if not owned_process(state):
        raise RuntimeError('先用现有 Stage-1 prewarm 启动冻结 teacher；本命令不会另起一份 VLA。')
    os.environ['PI05_PLANNER_TOKEN']=state['token']
    teacher=RemotePlanner(state['url'],cfg,timeout=2,token=state['token'])
    rows=replay_inventory(root,cfg) if root.exists() else []
    valid=[r for r in rows if r['eligible']]
    print(json.dumps({'mode':args.mode,'teacher_pid':state['pid'],
        'feature_version_id':teacher.metadata['feature_version_id'],
        'runtime_contract':cfg.wire_contract(),'training':cfg.raw['training'],
        'session_root':str(root),'eligible_episodes':len(valid),
        'successes':sum(r['outcome']=='success' for r in valid),
        'failures':sum(r['outcome']=='failure' for r in valid),
        'learnable_transitions':sum(r['learnable'] for r in valid),
        'update_gate':{'episodes':cfg.raw['session']['warmup_episodes'],
                       'new_episodes_per_batch':cfg.raw['session'].get('update_every_episodes',1),
                       'transitions':cfg.raw['session']['min_replay_transitions']},
        'station':station, 'hardware_opened':False},ensure_ascii=False,indent=2),flush=True)
    if args.mode=='check' or not args.execute:
        if args.mode!='check':print('仅检查。操作员准备好后显式加 --execute；此时才会连接机械臂。')
        return 0
    require_output_storage(root)
    if args.policy and not Path(args.policy).is_file():
        raise RuntimeError(f'策略权重不可读：{args.policy}；检查数据盘连接。未启动硬件。')
    if not station['accepted_settings_restored']:
        raise RuntimeError('已观察到 FCI 丢包；先在无 bridge 运行时执行 ~/prepare_pi05_stage2_station.sh 恢复既有站点设置，再做回位验收。')
    if args.policy and args.mode!='eval':
        raise RuntimeError('--policy only applies to fixed actor eval')
    # CPU 12 is the accepted collection-control core on this specific station.
    control_cpu=int(os.environ.get('PI05_CONTROL_CPU','12'))
    allowed=os.sched_getaffinity(0)
    if control_cpu not in allowed or len(allowed)<2:
        raise RuntimeError('accepted control CPU is unavailable; inspect station affinity')
    os.environ['PI05_CONTROL_CPU']=str(control_cpu)
    os.sched_setaffinity(0,allowed-{control_cpu})
    directory=Path.home()/'.cache/pi05-stage1-8000'
    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    for sig in (signal.SIGHUP,signal.SIGTERM):signal.signal(sig,interrupted)
    with lock_file(directory/'session.lock'):
        from .bridge_preflight import prepare_socket
        prepare_socket(cfg.raw['franka']['socket'])
        require_robot_idle(os.environ.get('PI05_ROBOT_IP', '172.16.0.40'))
        # Guard against the user starting another inference client while updating AC.
        # Both Stage-1 and Stage-2 launchers use this same ownership lock.
        mode='train' if args.mode in ('collect','train') else 'eval'
        output=root if args.mode in ('collect','train') else root/'eval'/time.strftime('%Y%m%d_%H%M%S')
        return run_eval(args.runtime_config,directory,state['url'],mode=mode,
            session_out=output,resident_teacher=True,policy=args.policy,collect_only=args.mode=='collect')


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as exc:
        print(f'启动未完成：{exc}', file=__import__('sys').stderr, flush=True)
        raise SystemExit(2)
