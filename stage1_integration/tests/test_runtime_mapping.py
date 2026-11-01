import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
from pi05_demo.contract import encode,decode,LOWER,UPPER
from pi05_rlt_rtc.adapters.franka import map_joint_delta_action
class RuntimeMappingTests(unittest.TestCase):
    def test_collection_actions_reproduce_runtime_hardware_targets(self):
        rng=np.random.default_rng(42)
        for _ in range(100):
            q=(LOWER+UPPER)/2;target=q+rng.uniform(-.029,.029,7);width=float(rng.uniform(0,.085))
            action=encode(q,target,width)
            got,got_width=map_joint_delta_action(action,q,delta_scale=[.03]*7,
                joint_lower=LOWER,joint_upper=UPPER,finger_range=[0,.0425],max_width=.085)
            np.testing.assert_allclose(got,target,atol=1e-8)
            self.assertAlmostEqual(got_width,width,places=7)
