"""Unmodified released DROID weights; explicitly missing exterior view, no RTC."""
import json
import logging
import os
from pathlib import Path
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.80')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from openpi.policies import policy_config
from openpi.training import config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path.home()/'.cache/openpi/openpi-assets/checkpoints/pi05_droid'
METADATA = dict(policy='pi05_droid', checkpoint=str(CHECKPOINT), rtc=False,
                actor=False, camera_profile='wrist_only_missing_exterior_masked',
                action_horizon=15, num_steps=10, action_space='droid_joint_velocity_gripper_position')

def main():
    logging.basicConfig(level=logging.INFO)
    started = time.monotonic()
    print('Loading local released pi05_droid; no task finetuning, no RTC, no AC.', flush=True)
    policy = policy_config.create_trained_policy(config.get_config('pi05_droid'), CHECKPOINT,
                                                sample_kwargs={'num_steps': 10})
    original_transform = policy._input_transform
    def wrist_only(data):
        result = original_transform(data)
        result['image_mask']['base_0_rgb'] = np.False_
        return result
    policy._input_transform = wrist_only
    dummy = {'observation/exterior_image_1_left': np.zeros((224,224,3), np.uint8),
             'observation/wrist_image_left': np.zeros((224,224,3), np.uint8),
             'observation/joint_position': np.array([0,-.628,0,-2.513,0,1.885,0],np.float32),
             'observation/gripper_position': np.array([0.],np.float32),
             'prompt': 'insert the plug into the power strip'}
    print('Compiling ordinary DROID inference; no robot connected.', flush=True)
    times = []
    for _ in range(2):
        before = time.monotonic()
        out = policy.infer(dummy)['actions']
        assert out.shape == (15,8) and np.isfinite(out).all(), (out.shape, out)
        times.append(time.monotonic()-before)
    report = dict(METADATA, warmup_seconds=times, ready_elapsed_s=time.monotonic()-started,
                  image_masks={k:bool(v) for k,v in policy._input_transform(dummy)['image_mask'].items()},
                  smoke_action_shape=list(out.shape), smoke_actions_finite=True)
    (ROOT/'server_verified.json').write_text(json.dumps(report,indent=2))
    print('DROID READY '+json.dumps(report), flush=True)
    WebsocketPolicyServer(policy, host='127.0.0.1', port=18766, metadata=METADATA).serve_forever()

if __name__ == '__main__':
    main()
