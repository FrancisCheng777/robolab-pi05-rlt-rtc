#!/usr/bin/env python3
"""No-device CUDA numerical smoke: tiny RANDOM upstream model and mock replay.

Never opens a robot/camera and never claims trained full-model latency/quality.
"""
import dataclasses
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'stage1_integration'),str(ROOT/'stage1_integration/tests/stage1')]
from model_fixture import tiny_model
from pi05_rlt_rtc import SCHEMA_VERSION, PLANNING_STATE_VERSION
from pi05_rlt_rtc.action_codec.codec import ActionCodec, NormStats
from pi05_rlt_rtc.adapters.rlinf_feature import RLinfFeatureReferenceService
from pi05_rlt_rtc.adapters.panda_policy import PandaInputs, PandaOutputs
from pi05_rlt_rtc.contracts.spaces import HorizonContract
from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
from pi05_rlt_rtc.contracts.plan import PlanRequest, PrevPlanContext
from pi05_rlt_rtc.rtc.dispatcher import RTCDispatcher


def main():
    assert torch.cuda.is_available(), 'CUDA required; do not report CPU results as GPU validation'
    torch.set_num_threads(4)
    model=tiny_model(eval_mode=True).cuda().eval().requires_grad_(False)
    def tokens(data):
        data={k:v for k,v in data.items() if k!='prompt'}
        return {**data,'state':np.pad(data['state'],(0,23)),
                'tokenized_prompt':np.arange(8,dtype=np.int64),'tokenized_prompt_mask':np.ones(8,bool)}
    model.setup_transforms([PandaInputs(),tokens],[PandaOutputs()])
    codec=ActionCodec(NormStats.symmetric(),name='MOCK-CUDA')
    contract=HorizonContract(3,1,1)
    service=RLinfFeatureReferenceService(model=model,contract=contract,codec=codec,
        rtc_dispatcher=RTCDispatcher(num_steps=2),feature_version_id='MOCK-CUDA',
        norm_stats_sha256=codec.norm_stats.sha256,planning_state_version=PLANNING_STATE_VERSION)
    obs=ObservationSnapshot('fixture','fixture',0,np.zeros(9,np.float32),
        {'wrist':np.zeros((224,224,3),np.uint8)},{'wrist':True},'synthetic CUDA fixture')
    prev=PrevPlanContext(np.zeros((3,8),np.float32),np.ones(3,bool),0,0,'MOCK-CUDA')
    req=PlanRequest(0,'fixture','fixture',1,0,1,1,'MOCK-CUDA','MOCK-CUDA',
                   SCHEMA_VERSION,PLANNING_STATE_VERSION,prev,42,'none',2)
    modes={}
    for mode in ('none','approx','full_vjp'):
        start=time.monotonic()
        a=service.extract(dataclasses.replace(req,sampler_mode=mode),obs)
        b=service.extract(dataclasses.replace(req,sampler_mode=mode),obs)
        torch.cuda.synchronize()
        np.testing.assert_allclose(a.reference_env_full,b.reference_env_full,atol=1e-5)
        assert np.isfinite(a.z_rl).all()
        modes[mode]={'finite':True,'seeded_repeatable':True,'two_calls_s':time.monotonic()-start}
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    # Exercise full-size z=2048 queue actor/critic on cached mock transitions.
    from pi05_rlt_rtc.evaluation.toy_pipeline import build_toy, collect_toy
    from pi05_rlt_rtc.deployment.training import QueueLearner
    from pi05_rlt_rtc.config.runtime_config import load_runtime_config
    toy=build_toy(z_dim=2048)
    # Public toy collector returns replay; this data is explicitly marked mock.
    replay=collect_toy(toy,num_ticks=40).replay
    cfg=load_runtime_config(ROOT/'configs/runtime_franka.yaml')
    learner=QueueLearner(cfg,toy.spec,device='cuda:0',seed=3)
    learner.warmup(replay,2);learner.update(replay,2)
    report={'mock':True,'random_tiny_upstream_weights':True,'hardware_opened':False,
        'trained_full_model_tested':False,'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0),'total_memory_mib':torch.cuda.get_device_properties(0).total_memory//2**20,
        'peak_allocated_mib':torch.cuda.max_memory_allocated()//2**20,'rtc':modes,
        'actor_critic_z_dim':2048,'learner_counts':learner.counts,'learner_metrics':learner.metrics}
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
