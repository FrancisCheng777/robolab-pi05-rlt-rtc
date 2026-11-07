import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from pi05_rlt_rtc.deployment import managed_eval as m


@pytest.mark.skipif(not Path('/proc/self/stat').exists(), reason='Linux process identity registry required')
def test_model_identity_prevents_pid_reuse_and_unowned_kill():
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    try:
        state={'pid':child.pid,'start':m.process_start(child.pid)}
        assert m.owned_process(state)
        m.stop_owned({**state,'start':'invalid'})
        assert child.poll() is None
        m.stop_owned(state,timeout=1)
        child.wait(timeout=2)
        assert not m.owned_process(state)
    finally:
        if child.poll() is None:child.kill();child.wait()


def test_session_lock_rejects_duplicate(tmp_path):
    with m.lock_file(tmp_path/'lock'):
        with pytest.raises(RuntimeError,match='已有'):
            with m.lock_file(tmp_path/'lock'):pass


def test_ready_model_reused_without_starting_another_process(tmp_path,monkeypatch):
    monkeypatch.setattr(m, 'process_start', lambda pid: 'test-process-start')
    state={'pid':os.getpid(),'start':m.process_start(os.getpid()),'fingerprint':'test','url':'http://127.0.0.1:18765','token':'test-token'}
    (tmp_path/'model.json').write_text(json.dumps(state))
    monkeypatch.setattr(m,'fingerprint',lambda _: 'test')
    monkeypatch.setattr(m,'metadata',lambda *a: {'mock':False})
    spawn=Mock(side_effect=AssertionError('must not reload'))
    monkeypatch.setattr(m.subprocess,'Popen',spawn)
    assert m.ensure_model('test',tmp_path)==state['url']
    assert m.ensure_model('test',tmp_path)==state['url']
    spawn.assert_not_called()
    monkeypatch.setattr(m,'fingerprint',lambda _: 'changed')
    with pytest.raises(RuntimeError,match='代码已变化'):m.ensure_model('test',tmp_path)


def test_session_failure_still_closes_owned_bridge(tmp_path,monkeypatch):
    from pi05_rlt_rtc.deployment import bridge_preflight
    from pi05_rlt_rtc.config import runtime_config
    expected=tmp_path/'config.json';expected.write_text('{}')
    cfg=SimpleNamespace(raw={'franka':{'socket':str(tmp_path/'socket'),'bridge_config':str(expected)}},asset=lambda *a:expected)
    monkeypatch.setattr(runtime_config,'load_runtime_config',lambda _:cfg)
    monkeypatch.setattr(bridge_preflight,'prepare_socket',Mock())
    monkeypatch.setattr(bridge_preflight,'probe_bridge',Mock())
    bridge=Mock();bridge.poll.side_effect=[None,None,0]
    session=Mock();session.wait.side_effect=[KeyboardInterrupt,0];session.poll.return_value=None
    monkeypatch.setattr(m.subprocess,'Popen',Mock(side_effect=[bridge,session]))
    with pytest.raises(KeyboardInterrupt):m.run_eval('cfg',tmp_path,'http://127.0.0.1:18765')
    session.terminate.assert_called_once();bridge.terminate.assert_called_once()
    assert bridge_preflight.prepare_socket.call_count==2
