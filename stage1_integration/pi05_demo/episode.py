"""Append-only HDF5 episodes. Incomplete files are never eligible for export."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import h5py
import numpy as np
from .contract import CONTRACT_ID, encode, state_vector, validate_frames
from .timing import TIMING_FIELDS, validate_timing, validate_camera_metadata

SCHEMA="pi05_plug_demo/1"
class EpisodeWriter:
    def __init__(self,root,*,mock,task,session,config):
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        self.final=root/f"episode_{time.time_ns()}_{uuid.uuid4().hex[:8]}.h5"
        self.partial=self.final.with_suffix(".h5.inprogress")
        self.f=h5py.File(self.partial,"x")
        self.f.attrs.update(schema=SCHEMA,action_contract_id=CONTRACT_ID,mock=bool(mock),
            task=task,session=session,config_json=json.dumps(config,sort_keys=True),
            config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
            timestamp_clock="host_monotonic_seconds",camera_timestamp_kind="host_receive_time",
            timing_version=1,exposure_alignment_verified=False,
            success=-1,complete=False)
        self.count=0;self.last_time=None
    def append(self,*,q,target,width,command_width,frames,timestamp,state_sequence,
               command_sequence,ack,critical=False,intervention=False,state_age=0.,gripper_age=0.,
               timing=None,camera_metadata=None):
        if ack is not True:
            raise ValueError("unacknowledged command cannot become demonstration")
        validate_frames(frames,timestamp)
        if self.last_time is not None and timestamp<=self.last_time:
            raise ValueError("non-monotonic episode timestamps")
        if not np.isfinite(state_age) or not 0<=state_age<=.05 or not np.isfinite(gripper_age) or not 0<=gripper_age<=.5:
            raise ValueError("stale robot/gripper feedback")
        version=2 if timing is not None else 1
        if self.count and self.f.attrs["timing_version"]!=version:
            raise ValueError("cannot change timing schema inside an episode")
        if timing is not None:validate_timing(timing,timestamp)
        camera_metadata=camera_metadata or {k:{"receive_monotonic_s":stamp,
            "ready_monotonic_s":stamp,"source_time_s":None,"source_clock":"unavailable","frame_id":-1}
            for k,(_,stamp) in frames.items()}
        for name,(_,stamp) in frames.items():
            meta=camera_metadata[name]
            validate_camera_metadata(meta,stamp,timestamp)
            clock_key=f"camera_source_clock/{name}"
            if clock_key in self.f.attrs and self.f.attrs[clock_key]!=meta["source_clock"]:
                raise ValueError("camera clock domain changed during episode")
        values=dict(q=np.asarray(q),q_target=np.asarray(target),gripper_width=width,
            gripper_command_width=command_width,state=state_vector(q,width),
            action=encode(q,target,command_width),timestamp=timestamp,
            state_age=state_age,gripper_age=gripper_age,
            state_sequence=state_sequence,command_sequence=command_sequence,
            command_ack=True,critical=bool(critical),intervention=bool(intervention))
        if timing is not None:values.update({f"timing/{k}":float(timing[k]) for k in TIMING_FIELDS})
        self.f.attrs["timing_version"]=version
        for name,(rgb,stamp) in frames.items():
            values[f"images/{name}"]=rgb
            values[f"camera_time/{name}"]=stamp
            meta=camera_metadata[name];source=meta.get("source_time_s")
            values[f"camera_ready_time/{name}"]=meta["ready_monotonic_s"]
            values[f"camera_source_time/{name}"]=0. if source is None else float(source)
            values[f"camera_source_time_valid/{name}"]=source is not None
            values[f"camera_frame_id/{name}"]=meta.get("frame_id",-1)
            self.f.attrs[f"camera_source_clock/{name}"]=meta["source_clock"]
        for name,value in values.items():
            a=np.asarray(value)
            if name not in self.f:
                options={"compression":"lzf"} if a.ndim else {}
                self.f.create_dataset(name,shape=(0,*a.shape),maxshape=(None,*a.shape),
                    chunks=(1,*a.shape),dtype=a.dtype,**options)
            ds=self.f[name]
            ds.resize(self.count+1,axis=0);ds[self.count]=a
        self.count+=1;self.last_time=timestamp
        self.f.flush()
    def finish(self,outcome,reason=""):
        if outcome not in ("success","failure","aborted"):
            raise ValueError("explicit episode outcome required")
        if self.count==0 and outcome!="aborted":
            raise ValueError("empty episode cannot succeed/fail")
        self.f.attrs.update(outcome=outcome,success=int(outcome=="success"),
            outcome_source="synthetic" if self.f.attrs["mock"] else "operator",
            reason=reason,complete=True,frames=self.count)
        self.f.flush();self.f.close()
        os.replace(self.partial,self.final)
        return self.final
    def close_incomplete(self):
        if self.f.id.valid:self.f.close()

def inspect_episode(path,*,allow_mock=False):
    with h5py.File(path,"r") as f:
        if f.attrs.get("schema")!=SCHEMA or not f.attrs.get("complete"):
            raise ValueError("incomplete or unknown episode")
        if f.attrs.get("action_contract_id")!=CONTRACT_ID:
            raise ValueError("action contract mismatch")
        if bool(f.attrs["mock"]) and not allow_mock:
            raise ValueError("mock episodes cannot enter real training")
        n=int(f.attrs["frames"])
        if f.attrs.get("outcome")=="aborted":
            return dict(path=str(path),frames=n,outcome="aborted",mock=bool(f.attrs["mock"]),session=str(f.attrs["session"]),task=str(f.attrs["task"]),action_contract_id=CONTRACT_ID,config_sha256=str(f.attrs["config_sha256"]))
        for key in ("q","q_target","gripper_width","gripper_command_width","state","action","timestamp",
                    "state_age","gripper_age","command_ack","images/main","images/wrist",
                    "camera_time/main","camera_time/wrist"):
            if key not in f or len(f[key])!=n:raise ValueError(f"invalid {key}")
        if f.attrs.get("timing_version",1)==2:
            for key in TIMING_FIELDS:
                if f"timing/{key}" not in f or len(f[f"timing/{key}"])!=n:
                    raise ValueError(f"invalid timing/{key}")
        if "camera_source_time" in f:
            for name in ("main","wrist"):
                for field in ("camera_ready_time","camera_source_time","camera_source_time_valid","camera_frame_id"):
                    if f"{field}/{name}" not in f or len(f[f"{field}/{name}"])!=n:
                        raise ValueError(f"invalid {field}/{name}")
        if n and (not np.isfinite(f["timestamp"][:]).all() or np.any(np.diff(f["timestamp"][:])<=0)):
            raise ValueError("invalid timestamps")
        # Check every physical/action correspondence; labels must not drift.
        for i in range(n):
            if f.attrs.get("timing_version",1)==2:
                validate_timing({k:float(f[f"timing/{k}"][i]) for k in TIMING_FIELDS},float(f["timestamp"][i]))
            if "camera_source_time" in f:
                for name in ("main","wrist"):
                    validate_camera_metadata({
                        "receive_monotonic_s":float(f[f"camera_time/{name}"][i]),
                        "ready_monotonic_s":float(f[f"camera_ready_time/{name}"][i]),
                        "source_time_s":float(f[f"camera_source_time/{name}"][i]) if f[f"camera_source_time_valid/{name}"][i] else None,
                        "source_clock":str(f.attrs[f"camera_source_clock/{name}"]),
                        "frame_id":int(f[f"camera_frame_id/{name}"][i]),
                    },float(f[f"camera_time/{name}"][i]),float(f["timestamp"][i]))
            validate_frames({k:(f[f"images/{k}"][i],float(f[f"camera_time/{k}"][i])) for k in ("main","wrist")},float(f["timestamp"][i]))
            if not 0<=f["state_age"][i]<=.05 or not 0<=f["gripper_age"][i]<=.5:raise ValueError("stale feedback in recording")
            expected=encode(f["q"][i],f["q_target"][i],f["gripper_command_width"][i])
            if not np.allclose(expected,f["action"][i],atol=1e-6):raise ValueError("action mismatch")
            if not np.allclose(state_vector(f["q"][i],f["gripper_width"][i]),f["state"][i],atol=1e-6):
                raise ValueError("state mismatch")
            if not f["command_ack"][i]:raise ValueError("missing command ACK")
        return dict(path=str(path),frames=n,outcome=str(f.attrs["outcome"]),mock=bool(f.attrs["mock"]),
                    session=str(f.attrs["session"]),task=str(f.attrs["task"]),
                    action_contract_id=CONTRACT_ID,config_sha256=str(f.attrs["config_sha256"]))
