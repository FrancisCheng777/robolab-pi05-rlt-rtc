"""Rebuild derived terminal replay, preserving original rollout and labels."""
import argparse,json
from pathlib import Path
from pi05_rlt_rtc.deployment.rollout import load_rollout,label_rollout
from pi05_rlt_rtc.deployment.model_loader import file_sha256
from pi05_rlt_rtc.deployment.runtime import write_json
from pi05_rlt_rtc.learning.replay import ReplayBuffer

p=argparse.ArgumentParser();p.add_argument('--session-root',required=True);args=p.parse_args()
root=Path(args.session_root).resolve();out=root/'comparisons/round4_repaired_labels'
if out.exists():raise RuntimeError('Use a new derived replay directory; never overwrite a prior reconstruction')
sources=json.loads((root/'rounds/round-000004/training_sources.json').read_text())
rows=[];directories=[]
for source in sources['replay_directories']:
 source=Path(source);a=json.loads((source/'annotation.json').read_text())
 raw=Path(a['rollout']);assert file_sha256(raw)==a['rollout_sha256']
 original=ReplayBuffer.load(source/'replay.pt')
 data=load_rollout(raw);replay,label=label_rollout(data,outcome=a['outcome'],annotator=a['annotator'],terminal_tick=a['terminal_tick'])
 terminal=[t for t in replay.learnable_transitions() if t.terminated]
 if not label['eligible'] or len(terminal)!=1:raise RuntimeError(f'Cannot reconstruct unique terminal for {a["trial_id"]}')
 dest=out/'labels'/a['trial_id'];replay.save(dest/'replay.pt')
 label.update(rollout=str(raw),rollout_sha256=a['rollout_sha256'],replay_counts=replay.counts(),replay_sha256=file_sha256(dest/'replay.pt'),
    derived_from_annotation=str(source/'annotation.json'),source_annotation_sha256=file_sha256(source/'annotation.json'),source_replay_sha256=file_sha256(source/'replay.pt'))
 write_json(dest/'annotation.json',label);write_json(dest/'runtime_manifest.json',data['metadata'])
 rows.append({'trial_id':a['trial_id'],'outcome':a['outcome'],'terminal_tick':a['terminal_tick'],
    'original_terminal_count':sum(t.terminated and t.is_learnable() for t in original.all_transitions()),
    'reconstructed_terminal_count':1,'terminal_duration_ticks':terminal[0].duration_ticks,
    'original_learnable_count':len(original.learnable_transitions()),'reconstructed_learnable_count':len(replay.learnable_transitions()),
    'source_replay_sha256':label['source_replay_sha256']})
 directories.append(str(dest))
write_json(out/'manifest.json',{'trial_ids':sources['trial_ids'],'replay_directories':directories,'episodes':rows,'raw_rollouts_modified':False})
print(json.dumps({'episodes':len(rows),'recovered_terminals':sum(r['original_terminal_count']==0 for r in rows),'outcomes':{v:sum(r['outcome']==v for r in rows) for v in ('success','failure')},'manifest':str(out/'manifest.json')}))
