"""Single-command teacher evaluation; resident GPU service owns NO hardware."""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def process_start(pid):
    try:
        text = Path(f'/proc/{int(pid)}/stat').read_text()
        fields = text[text.rfind(')')+2:].split()
        if fields[0] == 'Z':
            return None
        return fields[19]
    except (OSError, ValueError):
        return None


def owned_process(state):
    return bool(state and state.get('start') and process_start(state['pid']) == state['start'])


@contextlib.contextmanager
def lock_file(path):
    with Path(path).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('已有 eval/模型管理进程在运行，请使用原终端。') from exc
        yield


def fingerprint(config):
    digest = hashlib.sha256()
    files = sorted((ROOT/'pi05_rlt_rtc').rglob('*.py')) + [Path(config)]
    for path in files:
        digest.update(str(path).encode()); digest.update(path.read_bytes())
    from pi05_rlt_rtc.config.runtime_config import load_runtime_config
    cfg = load_runtime_config(config)
    # Bind reuse to asset metadata too, without rereading multi-GB weights on every entry.
    for value in cfg.raw['model'].values():
        if isinstance(value, str):
            path = Path(value).expanduser()
            if path.is_file():
                stat = path.stat()
                digest.update(f'{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
            elif path.is_dir():
                for asset in sorted(path.rglob('*')):
                    if asset.is_file() and '__pycache__' not in asset.parts and '.git' not in asset.parts and asset.suffix != '.pyc':
                        stat = asset.stat()
                        digest.update(f'{asset}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    return digest.hexdigest()


def stop_owned(state, timeout=8.):
    if not owned_process(state):
        return
    os.kill(state['pid'], signal.SIGTERM)
    deadline = time.monotonic()+timeout
    while owned_process(state) and time.monotonic() < deadline:
        time.sleep(.05)
    if owned_process(state):
        os.kill(state['pid'], signal.SIGKILL)


def metadata(url, token):
    req = urllib.request.Request(url+'/metadata', headers={'Authorization':'Bearer '+token})
    with urllib.request.urlopen(req, timeout=.5) as response:
        return json.load(response)


def ensure_model(config, directory, *, port=18765):
    state_path = directory/'model.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    stamp = fingerprint(config)
    url = f'http://127.0.0.1:{port}'
    if owned_process(state):
        if state.get('fingerprint') != stamp or state.get('url') != url:
            raise RuntimeError('模型/代码已变化，先运行 stop-model 再 eval，不能复用旧模型。')
        token = state['token']
        print(f"复用常驻模型 PID {state['pid']}，无需重新加载/预热。", flush=True)
    else:
        # Refuse an unmanaged service; never kill a process merely by port/name.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
        token = secrets.token_hex(32)
        env = {**os.environ, 'PI05_PLANNER_TOKEN':token, 'PYTHONUNBUFFERED':'1'}
        with (directory/'model.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-m', 'pi05_rlt_rtc.tools.cli',
                'serve-planner', '--runtime-config', str(config), '--host', '127.0.0.1', '--port', str(port)],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        state = {'pid':process.pid, 'start':process_start(process.pid), 'fingerprint':stamp,
                 'url':url, 'token':token}
        state_path.write_text(json.dumps(state)); state_path.chmod(0o600)
        print(f"首次启动模型并预热；进度日志：{directory/'model.log'}", flush=True)
    os.environ['PI05_PLANNER_TOKEN'] = token
    deadline = time.monotonic()+900
    last_print = 0
    while time.monotonic() < deadline:
        if not owned_process(state):
            raise RuntimeError(f"模型服务已退出，查看 {directory/'model.log'}")
        try:
            meta = metadata(url, token)
            if meta.get('mock') is not False:
                raise RuntimeError('拒绝复用 mock 模型')
            print('模型已就绪。', flush=True)
            return url
        except (OSError, ValueError):
            pass
        if time.monotonic()-last_print >= 15:
            print('等待模型加载/预热，未连接机械臂…', flush=True)
            last_print = time.monotonic()
        time.sleep(.25)
    raise RuntimeError('预热超过 15 分钟；保留日志，请检查 model.log')


def stop_child(child, timeout=8.):
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.kill(); child.wait(timeout=3.)


def run_eval(config, directory, url, *, mode='eval', session_out=None, resident_teacher=False, policy=None, collect_only=False):
    from pi05_rlt_rtc.config.runtime_config import load_runtime_config
    from .bridge_preflight import prepare_socket, probe_bridge
    cfg = load_runtime_config(config)
    sock = cfg.raw['franka']['socket']
    expected = json.loads(cfg.asset(cfg.raw['franka']['bridge_config'], 'bridge config').read_text())
    prepare_socket(sock)  # rejects any existing live bridge; never steals ownership
    bridge = session = None
    try:
        with (directory/'bridge.log').open('a') as log:
            bridge = subprocess.Popen(['bash', str(ROOT/'scripts/start_franka_bridge.sh'), '--execute'],
                cwd=ROOT, env={**os.environ, 'PI05_BRIDGE_SOCKET':sock}, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic()+25
        while True:
            if bridge.poll() is not None:
                raise RuntimeError(f"bridge 启动失败，查看 {directory/'bridge.log'}")
            try:
                probe_bridge(sock, expected); break
            except Exception:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"bridge 未就绪，查看 {directory/'bridge.log'}")
                time.sleep(.1)
        out = Path(session_out) if session_out else ROOT/'outputs/stage1_8000_eval'/time.strftime('%Y%m%d_%H%M%S')
        extra = ['--resident-teacher'] if resident_teacher else []
        if collect_only:
            extra += ['--collect-only']
        if policy:
            extra += ['--policy', str(policy)]
        session = subprocess.Popen([sys.executable, '-m', 'pi05_rlt_rtc.tools.cli', 'session',
            '--runtime-config', str(config), '--planner-url', url, '--rpc-timeout-s', '2',
            '--mode', mode, '--exploration' if mode == 'train' and cfg.raw.get('session', {}).get('exploration', True) else '--no-exploration',
            '--annotator', 'heinrich', '--out', str(out), '--execute', *extra], cwd=ROOT)
        return session.wait()
    finally:
        # Session EOF/SIGTERM closes owner and holds first; only then stop our bridge.
        stop_child(session)
        stop_child(bridge)
        if bridge is not None and bridge.poll() is not None:
            with contextlib.suppress(Exception):
                prepare_socket(sock)
        print('硬件会话已关闭。模型保留供下次 eval 复用；stop-model 可释放显存。', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('eval', 'prewarm', 'stop-model', 'status'))
    parser.add_argument('--runtime-config', required=True)
    args = parser.parse_args()
    directory = Path.home()/'.cache/pi05-stage1-8000'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        with lock_file(directory/'session.lock'):
            if args.mode in ('status', 'stop-model'):
                path = directory/'model.json'
                state = json.loads(path.read_text()) if path.exists() else None
                if args.mode == 'stop-model':
                    stop_owned(state)
                    path.unlink(missing_ok=True)
                    print('常驻模型已停止。')
                else:
                    print('模型进程运行中' if owned_process(state) else '模型未运行')
                return 0
            url = ensure_model(Path(args.runtime_config).resolve(), directory)
            return run_eval(args.runtime_config, directory, url) if args.mode == 'eval' else 0
    except KeyboardInterrupt:
        print('操作已中断。', flush=True)
        return 130
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
