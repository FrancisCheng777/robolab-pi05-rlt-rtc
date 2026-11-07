#!/usr/bin/env python3
"""Generate a labelled synthetic HDF5 -> LeRobot fixture. No bridge/devices."""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor/robolab-collecting-pipeline')]
from pi05_demo.episode import EpisodeWriter
from pi05_demo.export import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    for session in ('synthetic_train','synthetic_validation'):
        writer = EpisodeWriter(args.out / 'raw', mock=True, task='insert the plug into the power strip',
                               session=session, config={'control_hz':15, 'fixture':True, 'cameras':{'views':['wrist']}, 'teleop_controller':'robolab_torque'})
        for i in range(120):
            t = 100 + i / 15
            q = np.array([0., -.6, 0., -2.5, 0., 1.9, 0.]) + i*.0001
            writer.append(q=q, target=q + .003, width=.04, command_width=.04,
                frames={k:(np.full((224,224,3), i, np.uint8), t) for k in ('wrist',)},
                timestamp=t, state_sequence=i+1, command_sequence=i+1, ack=True)
        writer.finish('success', 'synthetic fixture only, no physical task')
    manifest = export(args.out / 'raw', args.out / 'dataset',
                      validation_sessions=('synthetic_validation',), allow_mock=True)
    print({'mock': manifest['mock'], 'dataset':str(args.out / 'dataset')})


if __name__ == '__main__': main()
