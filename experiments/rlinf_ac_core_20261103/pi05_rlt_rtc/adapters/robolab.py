"""Read-only access to the pinned, operator-accepted RoboLab dependency."""
import importlib
import os
from pathlib import Path
import sys
from pi05_rlt_rtc.contracts.errors import ContractError

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / 'vendor/robolab-collecting-pipeline'

def activate(root=None):
    root = Path(root or os.environ.get('ROBOLAB_ROOT', DEFAULT_ROOT)).expanduser().resolve()
    if not (root/'pi05_demo/session.py').is_file():
        raise ContractError('Initialize vendor submodules or set ROBOLAB_ROOT to the validated collection checkout')
    loaded=sys.modules.get('pi05_demo')
    if loaded and Path(loaded.__file__).resolve().parent != root/'pi05_demo':
        raise ContractError('A different pi05_demo is imported; use separate training/control processes')
    sys.path.insert(0,str(root))
    return root
