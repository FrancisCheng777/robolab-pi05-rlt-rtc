#!/usr/bin/env python3
"""Cross-check both projects' clients against both C++ bridges using --mock only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pi05_rlt_rtc.adapters.franka import UnixFrankaTransport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    other = args.collection_root.resolve()
    sys.path.insert(0, str(other))
    from pi05_demo.transport import Bridge
    rows = []
    for binary in (ROOT / 'build/franka/pi05_franka_bridge', other / 'build/pi05_franka_bridge'):
        for client_name in ('rlt_runtime', 'collection'):
            with tempfile.TemporaryDirectory(prefix='pi05-interop-', dir='/tmp') as tmp:
                socket = str(Path(tmp) / 'mock.sock')
                with (Path(tmp) / 'bridge.log').open('w') as log:
                    process = subprocess.Popen([str(binary), '--mock', '--config',
                        str(other / 'configs/bridge.mock.json'), '--socket', socket], stdout=log, stderr=log)
                    try:
                        transport = UnixFrankaTransport(socket, timeout=.2)
                        deadline = time.monotonic() + 5
                        while True:
                            try:
                                state = transport.request({'op': 'state', 'protocol': 1})
                                if state['ready']:
                                    break
                            except Exception:
                                if process.poll() is not None:
                                    raise RuntimeError('mock bridge exited')
                            if time.monotonic() > deadline:
                                raise RuntimeError('mock bridge startup timed out')
                            time.sleep(.01)
                        assert state['mock'] is True
                        if client_name == 'collection':
                            client = Bridge(socket, mock=True)
                            client.claim()
                            ack = client.command(client.state(), np.array(state['q']), state['gripper_width'])
                            client.close()
                        else:
                            owner = {'protocol': 1, 'session': state['session'], 'controller_id': uuid.uuid4().hex}
                            transport.request({**owner, 'op': 'claim'})
                            current = transport.request({'op': 'state', 'protocol': 1})
                            ack = transport.request({**owner, 'op': 'command', 'sequence': 1,
                                'state_sequence': current['state_sequence'], 'q_target': current['q'],
                                'gripper_width': current['gripper_width']})
                            transport.request({**owner, 'op': 'stop'})
                        assert ack['sequence'] == 1
                        rows.append({'binary': str(binary), 'client': client_name, 'passed': True,
                                     'mode': '--mock', 'sequence_ack': ack['sequence']})
                    finally:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=3)
    report = {'kind': 'mock_socket_interoperability', 'hardware_opened': False, 'checks': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': len(rows), 'hardware_opened': False}))


if __name__ == '__main__':
    main()
