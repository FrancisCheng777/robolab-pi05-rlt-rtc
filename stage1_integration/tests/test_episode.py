import tempfile,time,unittest
from pathlib import Path
import numpy as np
from pi05_demo.episode import EpisodeWriter,inspect_episode
from pi05_demo.contract import validate_frames
class EpisodeTests(unittest.TestCase):
    def sample(self):
        t=time.monotonic()
        return dict(q=np.zeros(7),target=np.full(7,.01),width=.04,command_width=.03,
            frames={k:(np.zeros((16,16,3),np.uint8),t-.01) for k in ("main","wrist")},
            timestamp=t,state_sequence=1,command_sequence=1,ack=True)
    def test_mock_separation_and_complete_episode(self):
        with tempfile.TemporaryDirectory() as d:
            w=EpisodeWriter(d,mock=True,task="insert",session="test",config={})
            self.assertEqual(list(Path(d).glob("*.h5")),[])
            w.append(**self.sample());p=w.finish("success")
            with self.assertRaises(ValueError):inspect_episode(p)
            self.assertEqual(inspect_episode(p,allow_mock=True)["frames"],1)
    def test_rejected_command_does_not_become_training_sample(self):
        with tempfile.TemporaryDirectory() as d:
            w=EpisodeWriter(d,mock=True,task="insert",session="test",config={})
            s=self.sample();s["ack"]=False
            with self.assertRaises(ValueError):w.append(**s)
            self.assertEqual(w.count,0);w.close_incomplete()
    def test_stale_camera_and_skew(self):
        s=self.sample();f=s["frames"];t=s["timestamp"]
        f["wrist"]=(f["wrist"][0],t-.5)
        with self.assertRaises(ValueError):validate_frames(f,t)
        f["wrist"]=(f["wrist"][0],t-.15)
        with self.assertRaises(ValueError):validate_frames(f,t)
    def test_corrupted_action_rejected(self):
        import h5py
        with tempfile.TemporaryDirectory() as d:
            w=EpisodeWriter(d,mock=True,task="insert",session="test",config={})
            w.append(**self.sample());p=w.finish("success")
            with h5py.File(p,"r+") as f:f["action"][0]=0
            with self.assertRaises(ValueError):inspect_episode(p,allow_mock=True)
