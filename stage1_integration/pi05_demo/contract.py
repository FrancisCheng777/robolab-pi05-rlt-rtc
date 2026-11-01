"""One physical action contract for collection, SFT and RLT execution."""
import numpy as np
CONTRACT_ID = "panda_delta_q0.03_robotiq_width0.085_v1"
SCALE = np.full(7, 0.03)
MAX_WIDTH = 0.085
LOWER=np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-.0175,-2.8973])
UPPER=np.array([2.8973,1.7628,2.8973,-.0698,2.8973,3.7525,2.8973])

def vector(value,n,name):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=(n,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain {n} finite values")
    return a

def encode(q,target,width):
    q,target=vector(q,7,"q"),vector(target,7,"target")
    if not np.isfinite(width) or not 0<=width<=MAX_WIDTH:
        raise ValueError("invalid measured/commanded gripper width")
    a=np.r_[(target-q)/SCALE,2*width/MAX_WIDTH-1]
    if np.any(np.abs(a)>1+1e-7):
        raise ValueError("command exceeds action contract; align GELLO, do not clip labels")
    return np.clip(a,-1,1).astype(np.float32)

def decode(q,action):
    q,action=vector(q,7,"q"),vector(action,8,"action")
    if np.any(np.abs(action)>1+1e-7):
        raise ValueError("action outside [-1,1]")
    return q+action[:7]*SCALE,float((action[7]+1)*MAX_WIDTH/2)

def state_vector(q,width):
    q=vector(q,7,"q")
    if not np.isfinite(width) or not 0<=width<=MAX_WIDTH:
        raise ValueError("invalid width")
    # Two half-width scalars preserve the existing project's 9-D state interface.
    # They represent one Robotiq aperture, not two independently measured joints.
    return np.r_[q,width/2,width/2].astype(np.float32)

def validate_target(q):
    q=vector(q,7,"target")
    if np.any(q<LOWER) or np.any(q>UPPER):
        raise ValueError("Panda joint limits exceeded")

def validate_frames(frames,now,max_age=.2,max_skew=.08):
    if set(frames)!={"main","wrist"}:
        raise ValueError("both Kinect and D405 are required")
    stamps=[]
    for key,(rgb,stamp) in frames.items():
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3:
            raise ValueError(f"{key}: expected HWC uint8 RGB")
        age=now-stamp
        if not np.isfinite(age) or not 0<=age<=max_age:
            raise ValueError(f"{key}: stale or future image ({age:.3f}s)")
        stamps.append(stamp)
    if max(stamps)-min(stamps)>max_skew:
        raise ValueError("camera timestamp skew exceeded")
