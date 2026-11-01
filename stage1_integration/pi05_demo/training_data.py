"""Framework-neutral offline SFT adapter using only complete action chunks.

The caller's VLA training config must use this joint-delta contract without
DROID velocity conversion or a second DeltaActions transform.
"""
import json
from pathlib import Path
import numpy as np
from pi05_rlt_rtc.contracts.physical import ACTION_CONTRACT_ID as CONTRACT_ID
from .chunks import table_window_starts, WINDOW_POLICY
class ChunkDataset:
    def __init__(self,root,horizon=10,*,allow_mock=False):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        root=Path(root);manifest=json.loads((root.parent/"manifest.json").read_text())
        if manifest["action_contract_id"]!=CONTRACT_ID:raise ValueError("wrong action contract")
        if manifest["mock"] and not allow_mock:raise ValueError("mock dataset prohibited for real SFT")
        from pi05_rlt_rtc.data.profile import validate_dataset_profile
        validate_dataset_profile(manifest)
        if horizon<1:raise ValueError("positive horizon required")
        self.ds=LeRobotDataset(f"local/pi05_plug_{root.name}",root=root,video_backend="pyav")
        self.horizon=horizon
        self.starts=table_window_starts(self.ds.hf_dataset,horizon)
        self.window_policy=dict(WINDOW_POLICY)
        self.actions=np.asarray(self.ds.hf_dataset["action"],np.float32)
    def __len__(self):return len(self.starts)
    @staticmethod
    def array(x):return x.detach().cpu().numpy() if hasattr(x,"detach") else np.asarray(x)
    def __getitem__(self,index):
        start=int(self.starts[index]);row=self.ds[start]
        actions=self.actions[start:start+self.horizon].copy()
        from pi05_rlt_rtc.data.profile import model_images, lerobot_rgb
        images,masks=model_images(lerobot_rgb(row["observation.images.wrist"]))
        return {"state":self.array(row["observation.state"]).astype(np.float32),
            "actions":actions,"image":images,"image_mask":masks,"prompt":row["task"]}
