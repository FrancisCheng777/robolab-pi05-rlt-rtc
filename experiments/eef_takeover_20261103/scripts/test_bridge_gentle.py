#!/usr/bin/env python3
"""Read the Panda/Robotiq state; --execute explicitly runs one bounded joint test."""
import argparse
from pathlib import Path
import sys
import time
import uuid
import numpy as np

# Also support direct use from a source checkout before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi05_rlt_rtc.adapters.franka import UnixFrankaTransport
from pi05_rlt_rtc.contracts.physical import validate_calibration_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default="/tmp/pi05-plug-demo.sock")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    transport = UnixFrankaTransport(args.socket, timeout=.2)
    state = transport.request({"op": "state", "protocol": 1})
    validate_calibration_contract(state["calibration"])
    print({k: state.get(k) for k in ("ready", "mock", "fault", "stopped", "claimed", "q", "gripper_width")})
    if not args.execute:
        return 0
    if not state["ready"] or state["fault"] or state["stopped"] or state["claimed"]:
        raise RuntimeError("start a fresh, ready bridge for this trial")
    owner = {"protocol": 1, "session": state["session"], "controller_id": uuid.uuid4().hex}
    target = np.array(state["q"], float)
    target[0] += .003  # one 0.003 rad displacement; never cumulative
    width = state["gripper_width"]  # preserve the measured aperture
    transport.request({**owner, "op": "claim"})
    try:
        for sequence in range(1, 21):
            current = transport.request({"op": "state", "protocol": 1})
            if current["fault"] or current["stopped"]:
                raise RuntimeError(current["fault"] or "bridge stopped")
            transport.request({**owner, "op": "command", "sequence": sequence,
                "state_sequence": current["state_sequence"], "q_target": target.tolist(), "gripper_width": width})
            time.sleep(.05)
        measured = transport.request({"op": "state", "protocol": 1})
        print("measured displacement (rad):", np.array(measured["q"]) - np.array(state["q"]))
    finally:
        transport.request({**owner, "op": "stop"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
