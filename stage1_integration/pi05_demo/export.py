"""Export successful demonstrations, split by session, with train-only norm stats."""
import argparse,json,hashlib,shutil,uuid
from pathlib import Path
import h5py,numpy as np
from .episode import inspect_episode
from .contract import CONTRACT_ID
from .timing import audit_timing
from .chunks import WINDOW_POLICY
ROOT=Path(__file__).resolve().parents[1]

def stats(arr):
    a=np.asarray(arr,np.float64)
    return {k:v.tolist() for k,v in {
        "mean":a.mean(0),"std":a.std(0),"q01":np.quantile(a,.01,axis=0),
        "q99":np.quantile(a,.99,axis=0)}.items()}

def select_sources(paths,validation_sessions,allow_mock):
    selected=[];skipped=[]
    for p in paths:
        info=inspect_episode(p,allow_mock=allow_mock)
        if info["outcome"]!="success" or not info["frames"]:
            skipped.append(info);continue
        info["split"]="validation" if info["session"] in validation_sessions else "train"
        selected.append(info)
    if not selected:raise ValueError("no successful demonstrations")
    if {x["mock"] for x in selected}=={True,False}:raise ValueError("cannot mix synthetic and real data")
    if len({x["config_sha256"] for x in selected})!=1:raise ValueError("mixed collection/calibration configs; export separate dataset versions")
    known={x["session"] for x in selected}
    if set(validation_sessions)-known:raise ValueError("unknown validation session")
    if not any(x["split"]=="train" for x in selected):raise ValueError("no training episodes")
    if not allow_mock and not any(x["split"]=="validation" for x in selected):
        raise ValueError("reserve a separate collection session with --validation-session")
    return selected,skipped

def export(input_dir,output,*,validation_sessions=(),allow_mock=False,fps=15):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    output=Path(output).absolute()
    if output.exists():raise ValueError("output already exists; choose a new dataset version")
    paths=sorted(Path(input_dir).glob("*.h5"))
    sources,skipped=select_sources(paths,set(validation_sessions),allow_mock)
    temp=output.parent/f".{output.name}.inprogress-{uuid.uuid4().hex[:8]}"
    temp.mkdir(parents=True)
    all_train_states=[];all_train_actions=[]
    manifest={"schema":1,"mock":any(s["mock"] for s in sources),"action_contract_id":CONTRACT_ID,
        "fps":fps,"state_layout":["q1","q2","q3","q4","q5","q6","q7","width/2","width/2"],
        "action_layout":["dq1/.03","dq2/.03","dq3/.03","dq4/.03","dq5/.03","dq6/.03","dq7/.03","2*width/.085-1"],
        "camera_processing":"RGB; main rotated clockwise and center-cropped per collection config; both letterboxed 224x224",
        "success_label":"operator; synthetic when mock=true","sources":sources,"skipped":skipped,
        "normalization_source":"train split only",
        "chunk_sampling":"complete_episode_windows_no_tail_padding",
        "window_policy":dict(WINDOW_POLICY),
        "timing_policy":{"max_interval_error_fraction":.5,"max_grid_error_frames":1.,
                         "device_clocks_aligned":False},"validation":{}}
    features={
        "observation.state":{"dtype":"float32","shape":(9,),"names":manifest["state_layout"]},
        "action":{"dtype":"float32","shape":(8,),"names":manifest["action_layout"]},
        "observation.images.main":{"dtype":"video","shape":(224,224,3),"names":["height","width","channels"]},
        "observation.images.wrist":{"dtype":"video","shape":(224,224,3),"names":["height","width","channels"]},
        "critical":{"dtype":"bool","shape":(1,),"names":["critical"]},
        "intervention":{"dtype":"bool","shape":(1,),"names":["intervention"]},
        "source_time":{"dtype":"float64","shape":(1,),"names":["relative_monotonic_seconds"]},
    }
    try:
        for split in ("train","validation"):
            entries=[x for x in sources if x["split"]==split]
            if not entries:continue
            ds=LeRobotDataset.create(repo_id=f"local/pi05_plug_{split}",root=temp/split,
                fps=fps,robot_type="panda_robotiq2f85",features=features,use_videos=True)
            count=0;boundaries=[]
            for entry in entries:
                p=Path(entry["path"]);entry["sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
                with h5py.File(p,"r") as f:
                    config=json.loads(f.attrs["config_json"])
                    if config.get("control_hz")!=fps:raise ValueError("export FPS must match collection FPS")
                    times=f["timestamp"][:]
                    entry["timing_quality"]=audit_timing(f,fps)
                    boundaries.extend((count,count+entry["frames"]-1))
                    for i in range(entry["frames"]):
                        row={"observation.state":f["state"][i].astype(np.float32),
                            "action":f["action"][i].astype(np.float32),
                            "task":entry["task"],
                            "critical":np.array([f["critical"][i]],bool),
                            "intervention":np.array([f["intervention"][i]],bool),
                            "source_time":np.array([times[i]-times[0]],np.float64)}
                        for camera in ("main","wrist"):row[f"observation.images.{camera}"]=f[f"images/{camera}"][i]
                        ds.add_frame(row)
                    ds.save_episode();count+=entry["frames"]
                    if split=="train":
                        all_train_states.append(f["state"][:]);all_train_actions.append(f["action"][:])
            ds.finalize()
            loaded=LeRobotDataset(f"local/pi05_plug_{split}",root=temp/split,video_backend="pyav")
            if len(loaded)!=count:raise ValueError("official loader count mismatch")
            for i in boundaries:
                row=loaded[i]
                for camera in ("main","wrist"):
                    if tuple(row[f"observation.images.{camera}"].shape)!=(3,224,224):
                        raise ValueError("official loader image decode failed")
            manifest["validation"][split]={"frames":count,"episodes":len(entries),"official_loader":True,"video_boundaries":len(boundaries)}
        norm={"norm_stats":{"state":stats(np.concatenate(all_train_states)),
                            "actions":stats(np.concatenate(all_train_actions))}}
        (temp/"norm_stats.json").write_text(json.dumps(norm,indent=2,allow_nan=False))
        manifest["norm_stats_sha256"]=hashlib.sha256((temp/"norm_stats.json").read_bytes()).hexdigest()
        (temp/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
        temp.rename(output)
        return manifest
    except BaseException:
        # Keep partial output for diagnosis, never publish it as complete.
        (temp/"FAILED").write_text("Export failed; this directory is not a training dataset.\n")
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",default=str(ROOT/"data/raw"));p.add_argument("--output",required=True)
    p.add_argument("--validation-session",action="append",default=[])
    p.add_argument("--allow-mock",action="store_true")
    a=p.parse_args()
    result=export(a.input,a.output,validation_sessions=a.validation_session,allow_mock=a.allow_mock)
    print(json.dumps({"output":a.output,"mock":result["mock"],"validated":result["validation"]},indent=2))
if __name__=="__main__":main()
