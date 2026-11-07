"""Synthetic CPU-only AC wiring check at production dimensions; NOT task training."""
import argparse,dataclasses,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from pi05_rlt_rtc.config.runtime_config import load_runtime_config
from pi05_rlt_rtc.deployment.training import QueueLearner
from pi05_rlt_rtc.learning.checkpoint import PolicyBundle
from pi05_rlt_rtc.learning.planning_state import PlanningStateSpec
from pi05_rlt_rtc.learning.replay import ReplayBuffer
from tests.test_learner import _collect

p=argparse.ArgumentParser();p.add_argument('--runtime-config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(2)
cfg=load_runtime_config(a.runtime_config);spec=PlanningStateSpec.from_contract(cfg.contract,z_dim=2048)
_,_,_,transitions=_collect(cfg.contract,num_ticks=210)
def resize(state):
 if state is None:return None
 return dataclasses.replace(state,spec=spec,z_rl=np.zeros((1,2048),np.float32))
for tr in transitions:
 tr.planning_state=resize(tr.planning_state)
 tr.next_planning_state=resize(tr.next_planning_state)
r=ReplayBuffer(10000,feature_version_id='feature-v1',norm_stats_sha256=transitions[0].norm_stats_sha256)
for tr in transitions:r.push(tr)
start=time.monotonic();learner=QueueLearner(cfg,spec,device='cpu',seed=42)
learner.warmup(r,4);learner.update(r,8)
meta={'feature_version_id':'feature-v1','norm_stats_sha256':r.norm_stats_sha256,'mock':True,'fingerprint':{'test':'synthetic-cpu-only'}}
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
bundle=learner.save(out/'mock-latest',meta,r);loaded=PolicyBundle.load(out/'mock-latest/policy.pt')
assert bundle.actor_state_sha256==loaded.actor_state_sha256
assert learner.counts['actor_updates']==2 and learner.counts['updates']==8
assert loaded.extra['mock'] is True
report={'mock':True,'hardware_opened':False,'trained_real_policy':False,'device':'cpu','z_dim':2048,
 'batch_size':learner.settings['batch_size'],'elapsed_s':time.monotonic()-start,
 'actor_parameters':sum(x.numel() for x in learner.actor.parameters()),
 'critic_online_parameters':sum(x.numel() for x in learner.critic.online.parameters()),
 'counts':learner.counts,'checkpoint_bytes':sum(x.stat().st_size for x in (out/'mock-latest').glob('*')),
 'finite_metrics':all(np.isfinite(v) for m in learner.metrics for v in m.values()),'reload_matches':True}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
