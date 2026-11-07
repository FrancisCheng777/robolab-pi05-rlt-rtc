"""Pipe fault injection only; no robot or GPU connections."""
import io
import json
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.deployment.session_env import OwnerClient


@pytest.fixture
def pipe_owner():
    read_fd, write_fd = os.pipe()
    owner = OwnerClient.__new__(OwnerClient)
    owner._reply_buffer = b''
    owner._request_id = 0
    owner._failure = None
    owner.interrupted = threading.Event()
    owner.log_path = 'mock-owner.log'
    owner.process = SimpleNamespace(stdout=os.fdopen(read_fd, 'rb', buffering=0),
        stdin=io.BytesIO(), poll=lambda: None, send_signal=Mock())
    try:
        yield owner, write_fd
    finally:
        owner.process.stdout.close()
        os.close(write_fd)


def test_coalesced_replies_remain_readable_without_new_pipe_bytes(pipe_owner):
    owner, fd = pipe_owner
    os.write(fd, b'{"ok": true}\n{"ok": false, "error": "heartbeat failed"}\n')
    assert owner._read_reply(time.monotonic()+.1)['ok']
    assert owner._read_reply(time.monotonic()+.1)['error'] == 'heartbeat failed'


def test_partial_line_obeys_deadline(pipe_owner):
    owner, fd = pipe_owner
    os.write(fd, b'{"ok":')
    started = time.monotonic()
    with pytest.raises(TimeoutError, match='complete reply'):
        owner._read_reply(started+.05)
    assert time.monotonic()-started < .5


@pytest.mark.parametrize('reply', [
    {'ok': True, 'request_id': 99, 'result': {}},
    {'ok': False, 'error': 'heartbeat: TimeoutError: timed out'},
])
def test_uncertain_channel_latches_and_cannot_accept_late_ack(pipe_owner, reply):
    owner, fd = pipe_owner
    os.write(fd, (json.dumps(reply)+'\n').encode())
    with pytest.raises(ContractError):
        owner._rpc('status')
    assert owner.interrupted.is_set()
    assert owner.process.stdin.closed  # EOF, not another command
    assert owner.process.send_signal.call_count == 1
    with pytest.raises(ContractError, match='fault latched'):
        owner._rpc('hold')
    assert owner._request_id == 1


def test_matching_reply_is_returned(pipe_owner):
    owner, fd = pipe_owner
    os.write(fd, b'{"ok": true, "request_id": 1, "result": {"ready": true}}\n')
    assert owner._rpc('status') == {'ready': True}
    assert owner._failure is None
