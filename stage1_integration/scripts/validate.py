"""Reproducible hardware-free validation; never starts a real robot backend."""
import hashlib,importlib.metadata,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
env={**os.environ,"PYTHONPATH":f"{ROOT}:{ROOT/'runtime'}"}
def run(name,command,expected=0,cwd=ROOT):
    with (ROOT/"reports"/f"{name}.log").open("w") as log:
        result=subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    if result.returncode!=expected:
        raise RuntimeError(f"{name}: expected exit {expected}, got {result.returncode}; see reports/{name}.log")
    print(f"PASS {name}",flush=True)
    return {"exit_code":result.returncode,"log":f"reports/{name}.log"}
summary={"hardware_motion":False,"checks":{}}
summary["checks"]["unit"]=run("unit",[str(ROOT/".venv/bin/python"),"-m","unittest","discover","-s","tests","-q"])
summary["checks"]["cpp"]=run("cpp",["ctest","--output-on-failure"],cwd=ROOT/"build")
summary["checks"]["collection"]=run("collection",[str(ROOT/".venv/bin/python"),"scripts/smoke.py"])
summary["checks"]["runtime_config"]=run("runtime_config",[str(ROOT/".venv/bin/python"),"-m","pi05_rlt_rtc.tools.cli","runtime-validate","--runtime-config","configs/runtime_franka.yaml"])
summary["checks"]["hardware_gate"]=run("hardware_gate",[str(ROOT/".venv/bin/python"),"-m","pi05_demo.preflight","--hardware-config"],expected=2)
smoke=json.loads((ROOT/"reports/latest_smoke.json").read_text())
output=Path(smoke["run"])/"dataset"
code="""
import json,sys
from pathlib import Path
from pi05_demo.export import export
from pi05_demo.training_data import ChunkDataset
from pi05_demo.episode import inspect_episode
run=Path(sys.argv[1])
manifest=export(run/"raw",run/"dataset",validation_sessions=["smoke_validation"],allow_mock=True)
for split in ("train","validation"):
    ds=ChunkDataset(run/"dataset"/split,10,allow_mock=True)
    for i in (0,len(ds)-1):
        row=ds[i]
        assert row["actions"].shape==(10,8) and row["state"].shape==(9,)
        assert row["image"]["base_0_rgb"].shape==(224,224,3)
        assert not row["image_mask"]["right_wrist_0_rgb"]
try:ChunkDataset(run/"dataset"/"train",10)
except ValueError:pass
else:raise AssertionError("mock data entered real loader")
print(json.dumps({"sft_chunk_loader":True,"mock_rejection":True,"manifest":manifest["validation"]},indent=2))
"""
summary["checks"]["export"]=run("export",["/home/heinrich/roboLab/.venv-lerobot/bin/python","-c",code,smoke["run"]])
manifest=json.loads((ROOT/"reports/source_manifest.json").read_text())
changed=[p for p,digest in manifest.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest]
integrity={"checked_original_files":len(manifest),"changed_original_files":changed}
(ROOT/"reports/source_integrity.json").write_text(json.dumps(integrity,indent=2))
if changed:raise RuntimeError(f"original source fingerprint changed: {changed}")
summary["source_integrity"]=integrity
summary["mock_dataset"]=str(output)
summary["runtime_assets_verified"]=False
(ROOT/"reports/validation.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
