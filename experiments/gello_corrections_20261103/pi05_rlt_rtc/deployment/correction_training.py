"""Reconstruct queue-conditioned human BC examples; no invented Bellman labels."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
import torch
from .correction_data import demo_windows
from .runtime import write_json
from .rpc import RemotePlanner
from .managed_eval import lock_file,owned_process
from pi05_rlt_rtc import SCHEMA_VERSION,PLANNING_STATE_VERSION
from pi05_rlt_rtc.contracts.plan import PlanRequest,PrevPlanContext
from pi05_rlt_rtc.config.runtime_config import load_runtime_config
from pi05_rlt_rtc.learning.actor import QueueAwareRLTActor
from pi05_rlt_rtc.learning.planning_state import PlanningStateSpec
from pi05_rlt_rtc.learning.checkpoint import PolicyBundle,state_dict_sha256
from pi05_rlt_rtc.learning.serialization import atomic_torch_save


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare_episode(directory,cfg,backend,metadata,*,allow_mock=False):
    from pi05_rlt_rtc.tools.planner_server import _observation_from_json
    directory=Path(directory);manifest=json.loads((directory/'correction.json').read_text())
    if (manifest.get('format')!='gello-correction-episode-v1' or not manifest.get('complete') or
        not manifest.get('eligible_bc') or manifest.get('outcome')!='success' or
        manifest.get('grasp_confirmed') is not True or manifest.get('gate_trigger_tick') is None or
        manifest.get('handover') is None or manifest.get('error') or not manifest.get('stop_acknowledged')):
        raise ValueError('only complete, confirmed, successful correction demonstrations may train BC')
    source=directory/'human_frames.jsonl'
    if sha256(source)!=manifest['frames_sha256']:raise ValueError('human frame checksum mismatch')
    original=manifest['metadata']
    if original.get('mock') is not False and not (allow_mock and original.get('mock') is True):
        raise ValueError('mock corrections cannot train a real policy')
    for k in ('feature_version_id','norm_stats_sha256','runtime_contract'):
        if original.get(k)!=metadata.get(k):raise ValueError('frozen teacher metadata changed: '+k)
    if manifest['runtime_contract']!=cfg.wire_contract():raise ValueError('demo runtime contract mismatch')
    frames=[json.loads(line) for line in source.open()]
    if len(frames)!=manifest['frames']:raise ValueError('frame count mismatch')
    states=[];references=[];targets=[];ticks=[]
    for i,prefix,target in demo_windows(frames,cfg.contract,hz=cfg.control_hz):
        raw=frames[i]['observation'];obs=_observation_from_json(raw)
        obs=dataclasses.replace(obs,episode_id='bc-'+manifest['trial_id'])
        prev=PrevPlanContext.empty()
        if i>=cfg.contract.chunk:
            # H-C=D here: the shifted earlier human chunk is exactly this prefix.
            # This is hindsight supervision, not a claim of an actually committed policy queue.
            prev=PrevPlanContext(prefix,np.ones_like(prefix,bool),obs.tick-cfg.contract.chunk,
                                 i//cfg.contract.chunk-1,metadata['feature_version_id'])
        request=PlanRequest(request_id=i//cfg.contract.chunk,env_id=obs.env_id,episode_id=obs.episode_id,
            episode_generation=1,capture_tick=obs.tick,apply_start_tick=obs.tick+cfg.contract.delay,
            queue_generation=1,policy_version_id=metadata['policy_version_id'],
            feature_version_id=metadata['feature_version_id'],schema_version=SCHEMA_VERSION,
            planning_state_version=PLANNING_STATE_VERSION,rtc_context=prev,rng_seed=i+1234,
            sampler_mode=cfg.rtc_mode,num_denoising_steps=cfg.num_steps,
            last_command=np.asarray(frames[i-1]['audit']['normalized_action'],np.float32) if i else np.r_[np.zeros(7),-1.].astype(np.float32))
        result=backend.plan(request,obs,prefix,prev)
        ps=result.planning_state
        if ps is None:raise ValueError('teacher did not return RL features')
        states.append(ps.actor_state_vector()[0]);references.append(ps.reference_suffix_vector(cfg.contract)[0])
        targets.append(target.reshape(-1));ticks.append(obs.tick)
        if len(states)%20==0:print(f"[特征] {manifest['trial_id']}：{len(states)} 个纠正窗口",flush=True)
    if not states:raise ValueError('no continuous H-step human window')
    return dict(format='human-bc-cache-v1',mock=original.get('mock'),
        source=str(directory),source_sha256=manifest['frames_sha256'],manifest_sha256=sha256(directory/'correction.json'),trial_id=manifest['trial_id'],
        metadata=metadata,queue_context='hindsight_human_prefix_not_rl_replay',
        actor_state=torch.tensor(np.stack(states),dtype=torch.float32),
        reference=torch.tensor(np.stack(references),dtype=torch.float32),
        target=torch.tensor(np.stack(targets),dtype=torch.float32),ticks=ticks)


def fit_corrections(caches,cfg,metadata,out,*,steps=500,seed=1234):
    if not caches or steps<1:raise ValueError('need nonempty BC data and positive steps')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    # Split complete episodes, never adjacent windows from the same demonstration.
    ordered=sorted(caches,key=lambda x:x['trial_id']);heldout=ordered[-1:] if len(ordered)>=3 else []
    train=ordered[:-1] if heldout else ordered
    def tensors(parts):
        return [torch.cat([x[k] for x in parts]) for k in ('actor_state','reference','target')]
    x,r,y=tensors(train);val=tensors(heldout) if heldout else None
    torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed)
    spec=PlanningStateSpec.from_contract(cfg.contract,z_dim=cfg.raw['model']['z_dim'])
    actor=QueueAwareRLTActor(spec,fixed_std=.002)
    optimizer=torch.optim.Adam(actor.parameters(),lr=cfg.raw['training']['actor_lr'])
    metrics=[];best=None;best_loss=float('inf')
    def score(data):
        with torch.no_grad():return float(((actor(data[0],data[1]).actions.flatten(1)-data[2])**2).mean())
    initial=score((x,r,y))
    for step in range(steps):
        index=torch.randint(len(x),(min(256,len(x)),),generator=rng)
        prediction=actor(x[index],r[index]).actions.flatten(1)
        loss=((prediction-y[index])**2).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite correction BC loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),10.,error_if_nonfinite=True);optimizer.step()
        if (step+1)%25==0 or step+1==steps:
            train_loss=score((x,r,y));selection_loss=score(val) if val else train_loss
            metrics.append(dict(step=step+1,train_mse=train_loss,validation_mse=selection_loss if val else None))
            if selection_loss<best_loss:
                best_loss=selection_loss;best={k:v.detach().clone() for k,v in actor.state_dict().items()}
    actor.load_state_dict(best)
    with torch.no_grad():
        evaldata=val if val else (x,r,y)
        pred=actor(evaldata[0],evaldata[1]).actions
        grip_open_fraction=float((pred[:,:,7]>-.88).float().mean())
    policy=PolicyBundle(spec=spec,contract=cfg.contract,actor_state=actor.state_dict(),critic_state=None,
        feature_version_id=metadata['feature_version_id'],norm_stats_sha256=metadata['norm_stats_sha256'],
        policy_version_id='queue-rlt:'+state_dict_sha256(actor.state_dict()),
        training_counts={'warmup_updates':steps,'updates':0,'actor_updates':0,'rounds':0},
        actor_hidden_dims=actor.hidden_dims,codec_name='rlinf_openpi_joint_codec',
        extra=dict(mock=metadata.get('mock',False),runtime_contract=cfg.wire_contract(),
            fingerprint=metadata['fingerprint'],training_kind='human_correction_BC',critic_initialized=False,
            queue_context='hindsight_human_prefix_not_rl_replay',exploration_std=.002))
    policy.save(out/'policy.pt')
    report=dict(policy_version_id=policy.policy_version_id,initial_train_mse=initial,
        final_train_mse=score((x,r,y)),best_selection_mse=best_loss,
        validation_is_heldout=bool(val),gripper_open_fraction=grip_open_fraction,
        train_episodes=[x['trial_id'] for x in train],heldout_episodes=[x['trial_id'] for x in heldout],
        source_files=[dict(path=x['source'],sha256=x['source_sha256']) for x in ordered],
        critic_trained=False,physical_evaluation=False,metrics=metrics)
    write_json(out/'training_report.json',report)
    print('[BC 完成] '+str(out/'policy.pt'),flush=True)
    print(json.dumps({k:report[k] for k in ('initial_train_mse','final_train_mse','validation_is_heldout','gripper_open_fraction','critic_trained')}),flush=True)
    return policy,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime-config',required=True);p.add_argument('--root',required=True)
    p.add_argument('--steps',type=int,default=500);p.add_argument('--seed',type=int,default=1234)
    args=p.parse_args();cfg=load_runtime_config(args.runtime_config);root=Path(args.root).resolve()
    with lock_file(Path.home()/'.cache/pi05-stage1-8000/session.lock'):
        state=json.loads((Path.home()/'.cache/pi05-stage1-8000/model.json').read_text())
        if not owned_process(state):raise RuntimeError('先启动 Stage1 prewarm')
        backend=RemotePlanner(state['url'],cfg,timeout=30,token=state['token']);metadata=backend.metadata
        paths=[]
        for f in sorted(root.glob('eval/*/episodes/*/correction.json')):
            m=json.loads(f.read_text())
            if m.get('eligible_bc'):paths.append(f.parent)
        if not paths:raise RuntimeError('没有成功且完整的 GELLO 纠正示范；请先采集并按 c 标注成功')
        cache_root=root/'bc_cache';cache_root.mkdir(exist_ok=True)
        caches=[]
        for directory in paths:
            m=json.loads((directory/'correction.json').read_text())
            key=hashlib.sha256((sha256(directory/'correction.json')+m['frames_sha256']+metadata['feature_version_id']+metadata['norm_stats_sha256']+'human-bc-cache-v1').encode()).hexdigest()[:20]
            path=cache_root/(directory.name+'-'+key+'.pt')
            if path.exists():
                data=torch.load(path,map_location='cpu',weights_only=True)
                if sha256(directory/'human_frames.jsonl')!=data['source_sha256']:raise ValueError('source changed after caching')
                if sha256(directory/'correction.json')!=data['manifest_sha256']:raise ValueError('label changed after caching')
                if data['metadata']['runtime_contract']!=cfg.wire_contract():raise ValueError('cache runtime contract mismatch')
                if data.get('mock') is not False:raise ValueError('mock cache cannot train a real policy')
            else:
                data=prepare_episode(directory,cfg,backend,metadata);atomic_torch_save(data,path)
            caches.append(data)
        torch.set_num_threads(2)
        output=root/'bc_runs'/(time.strftime('%Y%m%d_%H%M%S')+'-'+uuid.uuid4().hex[:6])
        fit_corrections(caches,cfg,metadata,output,steps=args.steps,seed=args.seed)

if __name__=='__main__':main()
