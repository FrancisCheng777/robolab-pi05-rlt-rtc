"""Read-only bridge probe and conservative cleanup of abandoned Unix sockets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import stat

from pi05_rlt_rtc.contracts.errors import EnvironmentNotReadyError


def prepare_socket(path):
    """Remove only an owned socket with ECONNREFUSED and no kernel endpoint."""
    path = Path(path)
    try:
        before = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISSOCK(before.st_mode) or before.st_uid != os.getuid():
        raise EnvironmentNotReadyError(f'refusing to remove non-socket or foreign path: {path}')
    try:
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(.2)
            client.connect(str(path))
    except ConnectionRefusedError:
        # A bound-but-not-yet-listening endpoint is NOT an abandoned file.
        registry = Path('/proc/net/unix')
        if not registry.exists() or any(
                len(fields := line.split(maxsplit=7)) >= 8 and fields[7] == str(path)
                for line in registry.read_text().splitlines()):
            raise EnvironmentNotReadyError(f'socket still belongs to a process: {path}')
        now = path.lstat()
        if (now.st_dev, now.st_ino) != (before.st_dev, before.st_ino):
            raise EnvironmentNotReadyError('socket changed during cleanup; retry explicitly')
        path.unlink()
        return True
    except OSError as exc:
        raise EnvironmentNotReadyError(f'cannot confirm stale bridge socket: {exc}') from exc
    raise EnvironmentNotReadyError('bridge is already listening; keep its existing terminal and start eval')


def probe_bridge(path, expected_calibration=None):
    """Request state only; never claim, arm, home, stop, or command a robot."""
    try:
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(.3)
            client.connect(str(path))
            client.sendall(b'{"protocol":1,"op":"state"}\n')
            data = bytearray()
            while b'\n' not in data:
                block = client.recv(16384)
                if not block or len(data)+len(block) > 131072:
                    raise ValueError('invalid bridge reply')
                data.extend(block)
        state = json.loads(data.split(b'\n', 1)[0])
        if (state.get('protocol') != 1 or not state.get('ok') or not state.get('ready')
                or state.get('mock') is not False or state.get('fault')
                or state.get('stopped') or state.get('stopping')):
            raise ValueError('bridge is not ready for real inference')
        if state.get('claimed'):
            raise ValueError('bridge already belongs to a session; use that session instead of starting another')
        if expected_calibration is not None and state.get('calibration') != expected_calibration:
            raise ValueError('bridge calibration does not match runtime config')
        return state
    except (OSError, ValueError) as exc:
        raise EnvironmentNotReadyError(
            f'bridge 未就绪，尚未加载模型：{exc}。先在终端1启动 bridge 并确认没有退出，再运行 eval。'
        ) from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'check'))
    parser.add_argument('socket')
    args = parser.parse_args()
    try:
        if args.mode == 'prepare':
            if prepare_socket(args.socket):
                print('Removed confirmed abandoned bridge socket.', flush=True)
        else:
            probe_bridge(args.socket)
            print('Bridge connection verified (read-only).', flush=True)
    except EnvironmentNotReadyError as exc:
        parser.exit(2, str(exc)+'\n')


if __name__ == '__main__':
    main()
