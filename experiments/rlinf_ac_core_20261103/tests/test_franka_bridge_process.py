"""Cross-process wire/watchdog tests, explicitly using C++ --mock, never a robot."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import pytest

from pi05_rlt_rtc.adapters.franka import UnixFrankaTransport
from pi05_rlt_rtc.contracts.errors import ContractError, EnvironmentNotReadyError
from tests.test_platform_integration import robot_fixture


@pytest.fixture
def cpp_bridge(tmp_path):
    binary = Path(os.environ.get("PI05_FRANKA_BRIDGE", "runs/platform_implementation/cpp_build/pi05_franka_bridge")).resolve()
    if not binary.is_file():
        pytest.skip("build the C++ mock bridge and set PI05_FRANKA_BRIDGE to test the process boundary")
    cfg, _, _ = robot_fixture(tmp_path)
    calibration = cfg.raw["franka"]["bridge_config"]
    with tempfile.TemporaryDirectory(prefix="pi05-", dir="/tmp") as directory:
        path = str(Path(directory) / "bridge.sock")
        log = (tmp_path / "cpp.log").open("w")
        process = subprocess.Popen([str(binary), "--mock", "--config", calibration, "--socket", path], stdout=log, stderr=log)
        transport = UnixFrankaTransport(path, timeout=.1)
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    state = transport.request({"op": "state", "protocol": 1})
                    if state["ready"]: break
                except EnvironmentNotReadyError: pass
                if process.poll() is not None: pytest.fail("bridge exited during startup")
                time.sleep(.01)
            else: pytest.fail("bridge never became ready")
            assert state["mock"] is True
            owner = {"protocol": 1, "session": state["session"], "controller_id": "a" * 32}
            transport.request({**owner, "op": "claim"})
            yield transport, owner
        finally:
            process.terminate()
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=3)
            log.close()


def command(transport, owner, sequence=1):
    state = transport.request({"op": "state", "protocol": 1})
    return {**owner, "op": "command", "sequence": sequence, "state_sequence": state["state_sequence"],
            "q_target": [q + .03 for q in state["q"]], "gripper_width": .04}


def test_cpp_lease_order_and_stop_are_enforced(cpp_bridge):
    transport, owner = cpp_bridge
    request = command(transport, owner)
    assert transport.request(request)["sequence"] == 1
    with pytest.raises(ContractError, match="duplicate/out-of-order"):
        transport.request(request)
    with pytest.raises(ContractError, match="does not own"):
        transport.request({**owner, "op": "stop", "controller_id": "b" * 32})
    transport.request({**owner, "op": "stop"})
    with pytest.raises(ContractError, match="cannot re-arm"):
        transport.request(command(transport, owner, 2))


def test_cpp_watchdog_brakes_after_python_stops_sending(cpp_bridge):
    transport, owner = cpp_bridge
    transport.request(command(transport, owner))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = transport.request({"op": "state", "protocol": 1})
        if state["stopped"]: break
        time.sleep(.01)
    assert state["stopped"] and "watchdog" in state["fault"]
    assert max(abs(v) for v in state["dq"]) < 1e-6
    with pytest.raises(ContractError, match="cannot re-arm"):
        transport.request(command(transport, owner, 2))


def test_cpp_stale_state_and_limits_are_rejected(cpp_bridge):
    transport, owner = cpp_bridge
    stale = command(transport, owner)
    stale["state_sequence"] = 10**12
    with pytest.raises(ContractError, match="stale"):
        transport.request(stale)
    bad = command(transport, owner)
    bad["q_target"][0] = 100
    with pytest.raises(ContractError, match="limit"):
        transport.request(bad)


def test_cpp_collection_event_timestamps_match_the_new_pipeline(cpp_bridge):
    transport, owner = cpp_bridge
    state = transport.request({"op": "state", "protocol": 1})
    assert state["timestamp_clock"] == "host_monotonic_seconds"
    assert state["state_bridge_receive_s"] <= time.monotonic()
    assert state["gripper_bridge_read_s"] <= time.monotonic()
    request = command(transport, owner)
    sent = time.monotonic()
    reply = transport.request(request)
    assert sent <= reply["command_enqueue_s"] <= time.monotonic()
