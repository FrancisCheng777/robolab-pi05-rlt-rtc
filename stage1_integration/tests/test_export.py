import tempfile,time,unittest
from pathlib import Path
import numpy as np
from pi05_demo.episode import EpisodeWriter
from pi05_demo.export import select_sources
class SplitTests(unittest.TestCase):
    def make(self,d,session,outcome="success"):
        w=EpisodeWriter(d,mock=False,task="insert",session=session,config={"control_hz":15})
        t=time.monotonic()
        w.append(q=np.zeros(7),target=np.full(7,.01),width=.04,command_width=.03,
            frames={k:(np.zeros((16,16,3),np.uint8),t) for k in ("main","wrist")},
            timestamp=t,state_sequence=1,command_sequence=1,ack=True)
        return w.finish(outcome)
    def test_session_split_and_failure_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            paths=[self.make(d,"day1"),self.make(d,"day1"),self.make(d,"day2"),self.make(d,"day2","failure")]
            selected,skipped=select_sources(paths,{"day2"},False)
            self.assertEqual([x["split"] for x in selected],["train","train","validation"])
            self.assertEqual(len(skipped),1)
    def test_real_export_requires_holdout(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):select_sources([self.make(d,"day1")],set(),False)
