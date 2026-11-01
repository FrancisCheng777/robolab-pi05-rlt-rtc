"""Shared capture provider for demonstration collection and RLT runtime."""
import sys,threading,time
from pathlib import Path
import numpy as np
from .contract import validate_frames
VENDOR=Path(__file__).resolve().parents[1]/"vendor"

class Pair:
    def __init__(self,config):
        sys.path.insert(0,str(VENDOR))
        from legacy_cameras import KinectV2,RealSenseRGB,resize_with_pad
        self.resize=resize_with_pad;self.sources={};self.frames={};self.metadata={};self.error=None
        self.stop=threading.Event();self.lock=threading.Lock();self.threads=[]
        try:
            import pyrealsense2 as rs
            serial=config.get("wrist_serial")
            if not serial or not config.get("main_serial"):
                raise ValueError("set both camera serial numbers")
            devices={d.get_info(rs.camera_info.serial_number):d.get_info(rs.camera_info.name)
                     for d in rs.context().query_devices()}
            if serial not in devices or "D405" not in devices[serial]:
                raise ValueError(f"configured wrist camera is not a connected D405: {devices}")
            self.sources["main"]=KinectV2(serial=config["main_serial"],
                rotate_cw=config["main_rotate_cw"],crop_square=config["main_crop_square"])
            self.sources["wrist"]=RealSenseRGB(serial=serial,width=640,height=480,fps=30)
            for key,source in self.sources.items():
                th=threading.Thread(target=self._read,args=(key,source),daemon=True)
                self.threads.append(th);th.start()
            deadline=time.monotonic()+8
            while time.monotonic()<deadline:
                try:self.capture();break
                except ValueError:time.sleep(.02)
            else:raise RuntimeError(f"camera warmup failed: {self.error}")
        except BaseException:self.close();raise
    def _read(self,key,source):
        while not self.stop.is_set():
            try:
                sample=source.read_with_metadata()
                if sample is None:
                    self.stop.wait(.01);continue
                rgb,metadata=sample
                stamp=metadata["receive_monotonic_s"]
                rgb=self.resize(rgb,224,224).astype(np.uint8)
                metadata["ready_monotonic_s"]=time.monotonic()
                with self.lock:
                    self.frames[key]=(rgb,stamp)
                    self.metadata[key]=metadata
            except Exception as exc:
                self.error=str(exc);self.stop.wait(.02)
    def capture(self):
        return self.capture_with_metadata()[0]
    def capture_with_metadata(self):
        with self.lock:
            frames={k:(v[0].copy(),v[1]) for k,v in self.frames.items()}
            metadata={k:dict(v) for k,v in self.metadata.items()}
        validate_frames(frames,time.monotonic())
        return frames,metadata
    def close(self):
        self.stop.set()
        for th in self.threads:th.join(timeout=2.5)
        for source in self.sources.values():source.close()

class SyntheticPair:
    def __init__(self):self.sequence=0
    def capture(self):
        return self.capture_with_metadata()[0]
    def capture_with_metadata(self):
        self.sequence+=1
        t=time.monotonic();image=np.zeros((224,224,3),np.uint8)
        image[::8,:,0]=255;image[:,::8,2]=255
        frames={k:(image.copy(),t) for k in ("main","wrist")}
        metadata={k:{"receive_monotonic_s":t,"ready_monotonic_s":t,
                     "source_time_s":t,"source_clock":"synthetic_host_monotonic",
                     "frame_id":self.sequence} for k in frames}
        return frames,metadata
    def close(self):pass

def create_provider(config=None,**kwargs):
    """Factory consumed by the copied pi05-rlt-rtc Franka adapter."""
    from pi05_rlt_rtc.adapters.cameras import CameraFrame
    pair=Pair(config or kwargs)
    class Adapter:
        def capture(self):return {k:CameraFrame(rgb,stamp) for k,(rgb,stamp) in pair.capture().items()}
        def close(self):pair.close()
    return Adapter()
