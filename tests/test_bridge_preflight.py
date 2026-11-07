"""Socket lifecycle checks; no real bridge or hardware is opened."""
import json
from pathlib import Path
import socket
import threading

import pytest

from pi05_rlt_rtc.contracts.errors import EnvironmentNotReadyError
from pi05_rlt_rtc.deployment.bridge_preflight import prepare_socket, probe_bridge


def test_removes_only_closed_owned_socket(tmp_path):
    if not Path('/proc/net/unix').exists(): pytest.skip('Linux endpoint registry required')
    p = tmp_path/'stale.sock'
    with socket.socket(socket.AF_UNIX) as s: s.bind(str(p))
    assert prepare_socket(p) and not p.exists()
    assert not prepare_socket(p)


@pytest.mark.parametrize('listen', [False, True])
def test_preserves_live_or_not_yet_listening_endpoint(tmp_path, listen):
    p = tmp_path/'bound socket.sock'
    with socket.socket(socket.AF_UNIX) as s:
        s.bind(str(p))
        if listen: s.listen(1)
        with pytest.raises(EnvironmentNotReadyError): prepare_socket(p)
        assert p.exists()


def test_preserves_non_socket_and_symlink(tmp_path):
    f=tmp_path/'data';f.write_text('keep')
    link=tmp_path/'alias';link.symlink_to(f)
    for p in (f, link):
        with pytest.raises(EnvironmentNotReadyError): prepare_socket(p)
    assert f.read_text()=='keep' and link.is_symlink()


@pytest.mark.parametrize('claimed', [False, True])
def test_probe_is_state_only_and_rejects_active_owner(tmp_path, claimed):
    p=tmp_path/'server.sock';seen=[]
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(p));server.listen(1)
        def respond():
            with server.accept()[0] as client:
                seen.append(json.loads(client.recv(1024)))
                client.sendall(json.dumps({'protocol':1,'ok':True,'ready':True,'mock':False,
                                          'claimed':claimed,'calibration':{'test':1}}).encode()+b'\n')
        thread=threading.Thread(target=respond);thread.start()
        if claimed:
            with pytest.raises(EnvironmentNotReadyError,match='already belongs'): probe_bridge(p)
        else:
            assert probe_bridge(p,{'test':1})['ready']
        thread.join(timeout=1)
    assert seen==[{'protocol':1,'op':'state'}]


def test_stale_socket_fails_before_model_loading(tmp_path):
    p=tmp_path/'stale.sock'
    with socket.socket(socket.AF_UNIX) as s:s.bind(str(p))
    with pytest.raises(EnvironmentNotReadyError,match='尚未加载模型'):probe_bridge(p)
