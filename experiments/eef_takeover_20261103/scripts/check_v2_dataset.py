#!/usr/bin/env python3
"""Read-only Stage-1 loader check on an existing v2 LeRobot export."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'stage1_integration')]
from stage1.data import PlugDataset
from pi05_demo.training_data import ChunkDataset

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path)
    p.add_argument('--allow-mock',action='store_true')
    args=p.parse_args()
    ds=PlugDataset(str(args.dataset),str(ROOT/'stage1_integration/stage1/assets/paligemma_tokenizer.model'),allow_mock=args.allow_mock)
    item=ds[0]
    generic=ChunkDataset(args.dataset/'train',allow_mock=args.allow_mock)
    raw=generic[0]
    np.testing.assert_allclose(item['actions'][:,:8],ds.transform.normalize(raw['actions'],'actions'),atol=1e-6)
    obs=item['observation']
    assert [bool(x) for x in obs['image_mask'].values()]==[True,False,False]
    assert np.all(item['actions'][:,8:]==0)
    print(json.dumps({'mock':json.loads((args.dataset/'manifest.json').read_text())['mock'],'training_windows':len(ds),'actions':list(item['actions'].shape),
        'state':list(obs['state'].shape),'image_masks':{k:bool(v) for k,v in obs['image_mask'].items()},
        'dataset_modified':False,'hardware_opened':False},indent=2))
