"""One shared v2 single-wrist profile for SFT, deployment and checkpoint checks."""
import numpy as np
from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.physical import ACTION_CONTRACT_ID

CAMERAS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')

def validate_dataset_profile(manifest):
    if (manifest.get('action_contract_id') != ACTION_CONTRACT_ID or
        manifest.get('camera_views') != ['wrist'] or
        manifest.get('policy_image_mapping') != {'base_0_rgb': 'wrist'}):
        raise ContractError('Expected v2 0.05 actions and one wrist camera mapped to base_0_rgb; re-export original HDF5')
    codec = manifest.get('action_codec', {})
    scale = np.asarray(codec.get('joint_delta_scale', []), dtype=float)
    if scale.shape != (7,) or not np.allclose(scale, [.05]*7, rtol=0, atol=1e-10):
        raise ContractError('Dataset action_codec must explicitly encode seven 0.05 rad scales')

def lerobot_rgb(value):
    value = np.asarray(value)
    if value.shape == (3,224,224):
        value = value.transpose(1,2,0)
    if np.issubdtype(value.dtype,np.floating):
        if not np.isfinite(value).all() or value.min()<0 or value.max()>1:
            raise ContractError('LeRobot RGB must be in [0,1]')
        value = np.rint(value*255).astype(np.uint8)
    return value

def model_images(wrist):
    wrist=np.asarray(wrist)
    if wrist.shape != (224,224,3) or wrist.dtype != np.uint8:
        raise ContractError('Expected a single RGB224 uint8 wrist image')
    return ({CAMERAS[0]:wrist.copy(),CAMERAS[1]:np.zeros_like(wrist),CAMERAS[2]:np.zeros_like(wrist)},
            dict(zip(CAMERAS,(True,False,False))))
