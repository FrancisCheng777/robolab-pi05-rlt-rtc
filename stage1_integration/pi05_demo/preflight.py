"""Inspect configuration and dependencies without connecting to the robot."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
from .contract import CONTRACT_ID
ROOT=Path(__file__).resolve().parents[1]
def check(hardware=False):
    c=json.loads((ROOT/"configs/collection.json").read_text())
    b=json.loads((ROOT/"configs/bridge.real.json").read_text())
    errors=[]
    if b["action_contract_id"]!=CONTRACT_ID:errors.append("bridge action contract mismatch")
    if c["control_hz"]!=15:errors.append("this dataset version requires 15 Hz")
    if hardware:
        for key in ("load_mass","load_center","load_inertia","gripper_device","robotiq_open_position","robotiq_closed_position"):
            if b.get(key) is None:errors.append(f"bridge.real.json: fill {key}")
        if not c["hardware_verified"] or not b["hardware_verified"]:errors.append("hardware verification flags are false")
        if not c["cameras"]["wrist_serial"]:errors.append("fill D405 wrist_serial")
        if not Path(c["gello_port"]).exists():errors.append("GELLO serial device missing")
        if b.get("gripper_device") and not Path(b["gripper_device"]).exists():errors.append("Robotiq serial device missing")
    return {"action_contract_id":CONTRACT_ID,"hardware_ready":hardware and not errors,
        "errors":errors,"dependencies":{n:importlib.util.find_spec(n) is not None
        for n in ("numpy","h5py","pyrealsense2","pylibfreenect2","dynamixel_sdk")}}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--hardware-config",action="store_true")
    a=p.parse_args();result=check(a.hardware_config);print(json.dumps(result,indent=2))
    if result["errors"]:raise SystemExit(2)
if __name__=="__main__":main()
