#!/usr/bin/env python3
"""Reproducible CPU/data/mock validation. Does not launch physical services."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-python', required=True, help='Python with LeRobot 0.4.4 and pytest')
    p.add_argument('--model-python', required=True, help='Python with OpenPI/RLinf inference dependencies')
    p.add_argument('--bridge', type=Path, required=True, help='newly built mock-only C++ bridge')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    bridge = args.bridge.resolve()
    cache = bridge.parent / 'CMakeCache.txt'
    if not cache.is_file() or 'PI05_WITH_FRANKA:BOOL=OFF' not in cache.read_text():
        raise ValueError('validation requires a bridge built with PI05_WITH_FRANKA=OFF')
    data_python = str(Path(args.data_python).absolute())
    model_python = str(Path(args.model_python).absolute())
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'JAX_PLATFORMS':'cpu',
           'PYTHONDONTWRITEBYTECODE':'1', 'OMP_NUM_THREADS':'2',
           'PYTHONPATH':str(ROOT)+':'+str(ROOT/'stage1_integration'),
           'PI05_FRANKA_BRIDGE':str(bridge), 'PI05_TEST_DATASET':str(out/'fixture/dataset')}
    checks = [
        ('fixture', [data_python,'scripts/generate_test_dataset.py','--out',str(out/'fixture')]),
        ('cpp', ['ctest','--test-dir',str(bridge.parent),'--output-on-failure']),
        ('runtime', [data_python,'-m','pytest','tests','-q','-ra']),
        ('collection', [data_python,'-m','unittest','discover','-s','stage1_integration/tests','-p','test_*.py','-v']),
        ('stage1_data', [data_python,'-m','unittest','discover','-s','stage1_integration/tests/stage1','-p','test_data.py','-v']),
        ('stage1_gradients', [data_python,'-m','unittest','discover','-s','stage1_integration/tests/stage1','-p','test_model.py','-v']),
        ('stage1_inference', [model_python,'-m','unittest','discover','-s','stage1_integration/tests/stage1','-p','test_inference.py','-v']),
        ('transforms', [model_python,'scripts/audit_model_transforms.py','--collection-root',str(ROOT/'stage1_integration'),
                        '--dataset',str(out/'fixture/dataset'),'--out',str(out/'transforms.json')]),
        ('windows', [data_python,'scripts/verify_stage1_windows.py','--dataset',str(out/'fixture/dataset'),
                     '--out',str(out/'windows.json')]),
    ]
    rows = []
    for name, cmd in checks:
        with (out/f'{name}.log').open('w') as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        rows.append({'check':name,'exit_code':result.returncode,'log':str(out/f'{name}.log')})
        print(json.dumps(rows[-1]),flush=True)
    report = {'scope':'CPU synthetic data, tiny random VLA and mock-only bridge', 'mock':True,
              'hardware_opened':False,'gpu_model_validated':False,'passed':all(r['exit_code']==0 for r in rows),
              'checks':rows,'interpreters':{'data':data_python,'model':model_python}}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__': sys.exit(main())
