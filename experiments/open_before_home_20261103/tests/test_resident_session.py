import dataclasses
import os
import pty
import threading
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from pi05_rlt_rtc.deployment.episode import EpisodeSignals
from pi05_rlt_rtc.deployment.session import TerminalKeys, operator_start
from pi05_rlt_rtc.deployment.session_env import OwnerClient, OperatorInterrupt

@pytest.mark.parametrize('keys,reason', [(['c','e'],'operator_home'),(['e','x'],'operator_abort'),(['e','q'],'operator_quit')])
@pytest.mark.parametrize('tick',[0,10])
def test_urgent_keys_override_scoring_and_preserve_intent_at_tick_zero(keys, reason, tick):
    signals=EpisodeSignals();signals.poll(keys,tick)
    assert signals.outcome=='aborted' and signals.reason==reason


def test_keyboard_interrupts_without_main_loop_polling(monkeypatch):
    master, slave=pty.openpty()
    with os.fdopen(slave, 'r') as terminal:
        monkeypatch.setattr('sys.stdin', terminal)
        held=threading.Event()
        with TerminalKeys(emergency=held.set) as keys:
            os.write(master,b'\x13x')  # Ctrl-S must not suspend terminal output
            assert held.wait(.5)
            assert keys.read()==['x']
    os.close(master)


def test_queued_stop_prevents_home_or_start():
    keys=SimpleNamespace(lock=threading.Lock(),pending=deque(['q']),error=None)
    owner=Mock()
    assert operator_start(keys,owner,home=True) is False
    assert operator_start(keys,owner) is False
    owner.rpc.assert_not_called()


def test_hold_latches_before_inflight_rpc_finishes():
    owner=OwnerClient.__new__(OwnerClient)
    owner._rpc_lock=threading.RLock();owner.interrupted=threading.Event()
    owner._rpc=Mock()
    with owner._rpc_lock:
        thread=threading.Thread(target=owner.interrupt);thread.start()
        assert owner.interrupted.wait(.5)
        with pytest.raises(OperatorInterrupt):owner.rpc('action',action=[0]*8)
    thread.join(1)
    owner._rpc.assert_called_once_with('hold')


def test_resident_service_uses_same_executor_and_keeps_planning_state():
    from concurrent.futures import ThreadPoolExecutor
    from http.server import ThreadingHTTPServer
    from tests.test_platform_integration import backend_double
    from pi05_rlt_rtc.tools.planner_server import PlannerService, make_handler
    from pi05_rlt_rtc.deployment.rpc import RemotePlanner
    from pi05_rlt_rtc.tools.cli import _toy_request
    cfg,toy,backend,meta=backend_double()
    threads=[];original=backend.plan
    def plan(*args):
        threads.append(threading.get_ident());return original(*args)
    backend.plan=plan
    with ThreadPoolExecutor(max_workers=1) as executor:
        expected=executor.submit(threading.get_ident).result()
        service=PlannerService(backend,toy.contract,metadata=meta,executor=executor)
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(service))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_address[1]}'
            obs=dataclasses.replace(toy.env.reset(),env_id='toy0',episode_id='toy0-ep1')
            request=_toy_request(toy)
            for _ in range(2):
                remote=RemotePlanner(url,cfg)
                result=remote.plan(request,obs,np.zeros((2,8)),request.rtc_context)
                assert result.planning_state is not None
                assert result.planning_state.plan_id is None
                np.testing.assert_array_equal(result.planning_state.reference_full[0],result.reference_env_full)
            assert threads==[expected,expected]
        finally:
            server.shutdown();server.server_close();thread.join(2)
