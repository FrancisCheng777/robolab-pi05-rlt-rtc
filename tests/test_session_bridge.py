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


def test_persistent_owner_two_trials_home_and_hold_heartbeat(tmp_path):
    binary=Path('build/bridge-mock/pi05_franka_bridge').resolve()
    if not binary.exists():pytest.skip('build with scripts/build_bridge.sh --mock')
    cfg=load_runtime_config('configs/runtime_franka.yaml')
    cfg=dataclasses.replace(cfg,raw={**cfg.raw,'franka':{**cfg.raw['franka'],'socket':str(tmp_path/'mock.sock')},
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
