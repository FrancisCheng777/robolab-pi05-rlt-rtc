"""Train an isolated terminal-reward comparison after the robot session exits."""
import argparse
import fcntl
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--session-root',required=True)
    parser.add_argument('--runtime-config',required=True)
    parser.add_argument('--comparison-root',required=True)
    parser.add_argument('--reward-mode',choices=['signed','old'],default='signed')
    args=parser.parse_args()
    root=Path(args.session_root).resolve();out=Path(args.comparison_root).resolve()
    out.mkdir(parents=True,exist_ok=True)
    job_lock=(out/'job.lock').open('a');fcntl.flock(job_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=out/'status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('state')=='complete':
        print('Comparison already complete; no duplicate training.',flush=True);return
    def status(state,**kw):
        temp=out/'status.tmp';temp.write_text(json.dumps({'state':state,'updated_at_unix':time.time(),
            'hardware_opened':False,'automatic_policy_switch':False,**kw},indent=2));temp.replace(status_path)
    status('waiting_for_session_exit')
    hardware_lock=(Path.home()/'.cache/pi05-stage1-8000/session.lock').open('a')
    fcntl.flock(hardware_lock,fcntl.LOCK_EX)
    started=time.monotonic()
    try:
        import yaml
        from pi05_rlt_rtc.config.runtime_config import load_runtime_config
        from pi05_rlt_rtc.deployment.training import train
        from pi05_rlt_rtc.deployment.model_loader import file_sha256
        from pi05_rlt_rtc.learning.checkpoint import PolicyBundle
        cfg=load_runtime_config(args.runtime_config)
        parent=root/'rounds/round-000003/latest'
        reference=root/'rounds/round-000004/latest'
        sources=json.loads((reference.parent/'training_sources.json').read_text())
        manifest=root/'comparisons/round4_repaired_labels/manifest.json'
        repaired=json.loads(manifest.read_text())
        if repaired['trial_ids']!=sources['trial_ids']:raise RuntimeError('Reconstructed trial IDs differ')
        labels=repaired['replay_directories']
        rewards={'success':1.,'failure':-1. if args.reward_mode=='signed' else 0.}
        if len(labels)!=40:raise RuntimeError('Expected the fixed 40-episode fourth-round dataset')
        p=PolicyBundle.load(parent/'policy.pt');a=PolicyBundle.load(reference/'policy.pt')
        updates=a.training_counts['updates']-p.training_counts['updates']
        if updates!=1000 or a.training_counts['warmup_updates']!=p.training_counts['warmup_updates']:
            raise RuntimeError('Reference round is not the expected 1000-update comparison')
        raw=cfg.raw.copy();raw['training']={**raw['training'],'terminal_rewards':rewards}
        config=out/'runtime_new_rewards.yaml';config.write_text(yaml.safe_dump(raw,sort_keys=False))
        hashes={str(Path(d)/'replay.pt'):file_sha256(Path(d)/'replay.pt') for d in labels}
        hashes.update({str(Path(d)/'replay.pt'):file_sha256(Path(d)/'replay.pt') for d in sources['replay_directories']})
        hashes.update({str(parent/'policy.pt'):file_sha256(parent/'policy.pt'),
                       str(reference/'policy.pt'):file_sha256(reference/'policy.pt')})
        comparison={'parent_checkpoint':str(parent),'reference_checkpoint':str(reference),
            'reconstruction_manifest':str(manifest),'source_trial_ids':sources['trial_ids'],'terminal_rewards':rewards,
            'critic_updates':updates,'actor_updates':250,'source_hashes':hashes}
        (out/'comparison.json').write_text(json.dumps(comparison,indent=2))
        status('training',**comparison)
        training_started=time.monotonic()
        train(SimpleNamespace(runtime_config=str(config),policy=None,replay=labels,
            warmup_updates=0,updates=updates,seed=0,learner_device='cuda:0',resume=str(parent),
            out=str(out/'training'),allow_mock=False,allow_reward_change=True))
        elapsed=time.monotonic()-training_started
        b=PolicyBundle.load(out/'training/latest/policy.pt')
        if b.training_counts!=a.training_counts:raise RuntimeError('A/B update counts differ')
        metrics=json.loads((out/'training/latest/metrics.json').read_text())
        if any(isinstance(v,(float,int)) and not math.isfinite(v) for row in metrics['updates'] for v in row.values()):
            raise RuntimeError('Nonfinite comparison metric')
        if any(file_sha256(Path(name))!=value for name,value in hashes.items()):
            raise RuntimeError('Source replay or historical checkpoint changed')
        status('complete',**comparison,elapsed_s=elapsed,total_elapsed_s=time.monotonic()-started,
            checkpoint=str(out/'training/latest'),policy_version_id=b.policy_version_id,
            counts=b.training_counts,all_metrics_finite=True,source_files_unchanged=True)
        print(json.dumps({'state':'complete','elapsed_s':elapsed,'policy_version_id':b.policy_version_id}),flush=True)
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}',elapsed_s=time.monotonic()-started)
        raise


if __name__=='__main__':main()
