"""End-to-end C++ mock bridge -> collector -> HDF5. Never connects to hardware."""
import json,os,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pi05_demo.episode import inspect_episode
run=ROOT/"data"/f"smoke_{uuid.uuid4().hex[:8]}"
run.mkdir()
sock=f"/tmp/pi05-demo-smoke-{uuid.uuid4().hex[:8]}.sock"
env={**os.environ,"PYTHONPATH":str(ROOT)}
reports=[]
for session in ("smoke_train","smoke_validation"):
    log=(run/f"{session}.bridge.log").open("w")
    bridge=subprocess.Popen([str(ROOT/"build/pi05_franka_bridge"),"--mock","--config",
        str(ROOT/"configs/bridge.mock.json"),"--socket",sock],stdout=log,stderr=log)
    try:
        deadline=time.monotonic()+5
        while not Path(sock).exists() and bridge.poll() is None and time.monotonic()<deadline:time.sleep(.01)
        time.sleep(.05)
        result=subprocess.run([str(ROOT/".venv/bin/python"),"-m","pi05_demo.collect","--mock",
            "--frames","12","--session",session,"--socket",sock,"--output",str(run/"raw")],
            cwd=ROOT,env=env,text=True,capture_output=True,timeout=15)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        info=json.loads(result.stdout.strip().splitlines()[-1])
        audited=inspect_episode(info["episode"],allow_mock=True)
        assert audited["frames"]==12 and audited["outcome"]=="success"
        reports.append(audited)
    finally:
        bridge.terminate()
        try:bridge.wait(timeout=5)
        except subprocess.TimeoutExpired:bridge.kill();bridge.wait()
        log.close()
report={"mock":True,"run":str(run),"episodes":reports}
(ROOT/"reports/latest_smoke.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
