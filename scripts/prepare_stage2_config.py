"""Derive Stage-2 from the accepted inference contract, keeping frozen assets."""
import argparse
from pathlib import Path
import yaml
from pi05_rlt_rtc.config.runtime_config import load_runtime_config
from pi05_rlt_rtc.learning.defaults import RLINF_STAGE2_DEFAULTS

p=argparse.ArgumentParser();p.add_argument('--accepted-runtime',required=True);p.add_argument('--out',required=True);p.add_argument('--control-python',required=True);a=p.parse_args()
cfg=load_runtime_config(a.accepted_runtime)
if cfg.rtc_mode!='full_vjp' or cfg.raw['policy']['kind']!='teacher':raise SystemExit('requires accepted full-VJP teacher')
raw=yaml.safe_load(Path(a.accepted_runtime).read_text())
# Resolve path-valued assets before relocating the YAML.
for group,key in [('model','rlinf_root'),('model','checkpoint'),('model','norm_stats'),('franka','bridge_config'),('session','collection_config'),('session','bridge_binary')]:
 raw[group][key]=str(cfg.asset(raw[group][key],key))
raw['session']['control_python']=str(Path(a.control_python).expanduser().absolute())
raw['training'].update(RLINF_STAGE2_DEFAULTS)
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():raise SystemExit('refusing to overwrite an existing Stage-2 config')
out.write_text(yaml.safe_dump(raw,sort_keys=False))
assert load_runtime_config(out).wire_contract()==cfg.wire_contract()
print(out)
