"""Actual C++ mock bridge + isolated CPU owner; never connects to hardware."""
import dataclasses
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest
from pi05_rlt_rtc.config.runtime_config import load_runtime_config
from pi05_rlt_rtc.deployment.session_env import OwnerClient


@pytest.mark.parametrize('joint_mode', ['measured_delta', 'bounded_reference'])
def test_persistent_owner_two_trials_home_and_hold_heartbeat(tmp_path, joint_mode):
    binary=Path('build/bridge-mock/pi05_franka_bridge').resolve()
    if not binary.exists():pytest.skip('build with scripts/build_bridge.sh --mock')
    cfg=load_runtime_config('configs/runtime_franka.yaml')
    cfg=dataclasses.replace(cfg,raw={**cfg.raw,'franka':{**cfg.raw['franka'],'socket':str(tmp_path/'mock.sock'),
                                                     'joint_command_mode':joint_mode},
        'session':{**cfg.raw['session'],'control_python':sys.executable}})
    calibration=cfg.asset(cfg.raw['franka']['bridge_config'],'bridge config')
    process=subprocess.Popen([str(binary),'--mock','--config',str(calibration),'--socket',cfg.raw['franka']['socket']],
                             stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+5
        while not Path(cfg.raw['franka']['socket']).exists() and time.monotonic()<deadline:time.sleep(.02)
        owner=OwnerClient(cfg,mock=True)
        try:
            for _ in range(2):
                initial=owner.rpc('status')['state']
                for tick in range(4):
                    audit=owner.rpc('action',action=[0]*8)
                    assert audit['bridge_acknowledged']
                    assert audit['joint_command_mode'] == joint_mode
                    time.sleep(.04)
                assert owner.hold()['ready']
                # Learner can block longer than the bridge watchdog while held.
                time.sleep(.6)
                assert owner.rpc('status')['ready']
                owner.rpc('home')
                status=owner.wait_ready(timeout=30)
                assert status['ready']
                assert abs(status['state']['gripper_width']-.085)<.005
        finally:owner.close()
    finally:
        process.terminate();process.wait(timeout=5)


def test_owner_exits_on_policy_timeout(tmp_path):
    binary=Path('build/bridge-mock/pi05_franka_bridge').resolve()
    if not binary.exists():pytest.skip('build with scripts/build_bridge.sh --mock')
    cfg=load_runtime_config('configs/runtime_franka.yaml')
    cfg=dataclasses.replace(cfg,raw={**cfg.raw,'franka':{**cfg.raw['franka'],'socket':str(tmp_path/'mock.sock')},
        'session':{**cfg.raw['session'],'control_python':sys.executable}})
    process=subprocess.Popen([str(binary),'--mock','--config',str(cfg.asset(cfg.raw['franka']['bridge_config'],'config')),
                              '--socket',cfg.raw['franka']['socket']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+5
        while not Path(cfg.raw['franka']['socket']).exists() and time.monotonic()<deadline:time.sleep(.02)
        owner=OwnerClient(cfg,mock=True)
        try:
            owner.rpc('action',action=[0]*8)
            assert owner.process.wait(timeout=3)!=0
        finally:owner.close()
    finally:process.terminate();process.wait(timeout=5)


def test_background_keyboard_holds_during_busy_main_and_home(tmp_path, monkeypatch):
    import pty
    from pi05_rlt_rtc.deployment.session import TerminalKeys, operator_start
    from pi05_rlt_rtc.deployment.session_env import OperatorInterrupt
    binary=Path('build/bridge-mock/pi05_franka_bridge').resolve()
    if not binary.exists():pytest.skip('requires mock bridge')
    cfg=load_runtime_config('configs/runtime_franka.yaml')
    cfg=dataclasses.replace(cfg,raw={**cfg.raw,'franka':{**cfg.raw['franka'],'socket':str(tmp_path/'keys.sock')},
        'session':{**cfg.raw['session'],'control_python':sys.executable}})
    bridge=subprocess.Popen([str(binary),'--mock','--config',str(cfg.asset(cfg.raw['franka']['bridge_config'],'config')),
        '--socket',cfg.raw['franka']['socket']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+5
        while not Path(cfg.raw['franka']['socket']).exists() and time.monotonic()<deadline:time.sleep(.01)
        owner=OwnerClient(cfg,mock=True)
        master,slave=pty.openpty()
        try:
            with os.fdopen(slave,'r') as terminal:
                monkeypatch.setattr('sys.stdin',terminal)
                with TerminalKeys(emergency=owner.interrupt) as keys:
                    owner.rpc('action',action=[0]*8)
                    os.write(master,b'e')
                    # Main thread does not poll keys or advance controller for 0.5s.
                    # Without independent hold this exceeds the 250ms action watchdog.
                    time.sleep(.5)
                    assert owner.process.poll() is None
                    assert owner.wait_ready()['ready']
                    assert keys.read()==['e']
                    with pytest.raises(OperatorInterrupt):owner.rpc('action',action=[0]*8)
                    assert operator_start(keys,owner,home=True)
                    os.write(master,b'x');time.sleep(.3)
                    assert keys.read()==['x']
                    assert owner.wait_ready()['ready']
                    assert operator_start(keys,owner)
                    owner.rpc('action',action=[0]*8)
                    os.write(master,b'q');time.sleep(.5)
                    assert owner.process.poll() is None
                    assert owner.wait_ready()['ready']
                    assert keys.read()==['q']
        finally:
            os.close(master);owner.close()
        assert owner.process.poll() is not None
    finally:
        bridge.terminate();bridge.wait(timeout=5)
