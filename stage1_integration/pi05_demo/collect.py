"""One manually reset demonstration trial; never auto-recovers or auto-homes."""
import argparse,importlib.util,json,select,sys,termios,time,tty
from pathlib import Path
import numpy as np
from .contract import MAX_WIDTH,encode,validate_target,validate_frames
from .cameras import Pair,SyntheticPair
from .episode import EpisodeWriter
from .transport import Bridge

ROOT=Path(__file__).resolve().parents[1]
class Gello:
    def __init__(self,port):
        sys.path.insert(0,str(ROOT/"vendor"))
        from gello.agents.gello_agent import GelloAgent
        spec=importlib.util.spec_from_file_location("demo_gello_normalizer",ROOT/"vendor/gello_publish_zenoh.py")
        mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
        self.agent=GelloAgent(port=port);self.normalizer=mod.PandaJointPositionNormalizer()
    def read(self):
        raw=self.agent.act({})
        action,clipped=self.normalizer.normalize(raw)
        if clipped:raise ValueError(f"GELLO outside joint limits: {clipped}")
        if action.shape!=(8,) or not 0<=action[7]<=1:raise ValueError("GELLO requires 7 joints + closure in [0,1]")
        return action[:7],float((1-action[7])*MAX_WIDTH)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",default=str(ROOT/"configs/collection.json"))
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock",action="store_true");mode.add_argument("--execute",action="store_true")
    ap.add_argument("--socket");ap.add_argument("--output")
    ap.add_argument("--frames",type=int,default=0,help="mock-only length")
    ap.add_argument("--session",required=True,help="physical collection session ID; used for train/val split")
    ap.add_argument("--mock-outcome",choices=["success","failure"],default="success")
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text())
    if args.execute and (args.frames or not cfg["hardware_verified"]):
        ap.error("real collection needs hardware_verified=true and interactive outcome labels")
    if args.execute and not sys.stdin.isatty():ap.error("real collection requires an interactive terminal")
    if args.mock and args.frames<1:ap.error("--mock requires --frames > 0")
    bridge=None;cameras=None;writer=None;terminal=None
    outcome="aborted";reason="";error=None;critical=False;intervention=False
    try:
        cameras=SyntheticPair() if args.mock else Pair(cfg["cameras"])
        gello=None if args.mock else Gello(cfg["gello_port"])
        bridge=Bridge(args.socket or cfg["socket"],mock=args.mock)
        st=bridge.state()
        if args.execute:
            target,width=gello.read();validate_target(target)
            encode(st["q"],target,width)  # refuse misaligned leader before claim
            input("Align GELLO with Panda. Enter to start; s=success f=failure a=abort c=critical i=intervention: ")
        # Refresh after waiting for the operator, not before.
        st=bridge.state();bridge.claim()
        writer=EpisodeWriter(args.output or str(ROOT/"data"/("mock_raw" if args.mock else "raw")),
            mock=args.mock,task=cfg["task"],session=args.session,config={**cfg,"bridge_calibration":st["calibration"]})
        if args.execute:
            terminal=termios.tcgetattr(sys.stdin);tty.setcbreak(sys.stdin.fileno())
        period=1/cfg["control_hz"];deadline=time.monotonic()
        while True:
            if args.execute and select.select([sys.stdin],[],[],0)[0]:
                key=sys.stdin.read(1).lower()
                if key in ("s","f","a"):
                    outcome={"s":"success","f":"failure","a":"aborted"}[key];break
                if key=="c":critical=not critical;print(f"critical={critical}",flush=True)
                if key=="i":intervention=not intervention;print(f"intervention={intervention}",flush=True)
            st=bridge.state();received=time.monotonic();q=np.asarray(st["q"],float)
            frames,camera_metadata=cameras.capture_with_metadata()
            if args.mock:target=q+np.full(7,.0002);width=.05
            else:target,width=gello.read()
            gello_read=time.monotonic()
            validate_target(target);encode(q,target,width)
            stamp=time.monotonic();validate_frames(frames,stamp)
            state_age=stamp-st["state_bridge_receive_s"]
            gripper_age=stamp-st["gripper_bridge_read_s"]
            if state_age>.05 or gripper_age>.5:raise RuntimeError("feedback expired during input collection")
            # Account for camera/teleop work since the robot state was received.
            # Sequence-age rejection is independently enforced by the C++ bridge.
            reply=bridge.command(st,target,width)
            timing={"state_bridge_receive_s":st["state_bridge_receive_s"],
                    "state_client_receive_s":received,"gripper_bridge_read_s":st["gripper_bridge_read_s"],
                    "gello_read_s":gello_read,**{k:reply[k] for k in
                    ("command_send_s","command_enqueue_s","command_ack_s")}}
            writer.append(q=q,target=target,width=st["gripper_width"],command_width=width,
                frames=frames,timestamp=stamp,state_sequence=st["state_sequence"],
                command_sequence=bridge.seq,ack=reply.get("ok"),critical=critical,
                intervention=intervention,state_age=state_age,gripper_age=gripper_age,
                timing=timing,camera_metadata=camera_metadata)
            if args.mock and writer.count>=args.frames:
                outcome=args.mock_outcome;break
            deadline+=period
            lag=time.monotonic()-deadline
            if lag>period:raise RuntimeError(f"missed control deadline by {lag:.3f}s")
            time.sleep(max(0,deadline-time.monotonic()))
    except (Exception,KeyboardInterrupt) as exc:
        error=exc;reason=f"{type(exc).__name__}: {exc}"
    finally:
        # Stop robot before filesystem finalization, camera shutdown or any blocking cleanup.
        if bridge is not None:
            try:bridge.close()
            except Exception as exc:
                outcome="aborted";reason+=f" stop failed: {exc}";error=error or exc
        if terminal is not None:termios.tcsetattr(sys.stdin,termios.TCSADRAIN,terminal)
        if writer is not None:
            if error is not None:outcome="aborted"
            path=writer.finish(outcome,reason);print(json.dumps({"episode":str(path),"outcome":outcome,"frames":writer.count}))
        if cameras is not None:cameras.close()
    if error is not None:raise SystemExit(reason)
if __name__=="__main__":main()
