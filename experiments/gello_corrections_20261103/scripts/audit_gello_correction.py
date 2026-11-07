from pathlib import Path
import json,hashlib,numpy as np
from pi05_rlt_rtc.config.runtime_config import load_runtime_config
from pi05_rlt_rtc.deployment.correction_data import demo_windows
root=Path('/home/heinrich/pi05-corrections-20261103')
paths=sorted(root.glob('eval/*/episodes/*/correction.json'),key=lambda p:p.stat().st_mtime)
if not paths:
 print(json.dumps({'status':'waiting_for_first_episode'}));raise SystemExit
p=paths[-1];m=json.loads(p.read_text());report={'episode':str(p.parent),**{k:m.get(k) for k in ['complete','outcome','reason','error','eligible_bc','grasp_confirmed','grasp_confirmation_tick','gate_trigger_tick','frames','stop_acknowledged']}}
if not m.get('complete'):
 print(json.dumps(report));raise SystemExit
f=p.parent/'human_frames.jsonl'
if f.exists():
 rows=[];h=hashlib.sha256()
 with f.open('rb') as stream:
  for line in stream:h.update(line);rows.append(json.loads(line))
 report['checksum_matches']=h.hexdigest()==m.get('frames_sha256');report['frame_count_matches']=len(rows)==m.get('frames')
 if rows:
  t=np.array([r['observation']['captured_monotonic'] for r in rows]);q=np.array([r['observation']['states'][:7] for r in rows]);target=np.array([r['audit']['q_target'] for r in rows]);leader=np.array([r['leader_q'] for r in rows]);actions=np.array([r['audit']['normalized_action'] for r in rows])
  n=max(1,int(np.sum(t-t[0]<=1.0)))
  measured=np.array(rows[0]['audit'].get('measured_q',q[0]))
  report.update(first_action=actions[0].tolist(),first_target_minus_measured_rad=(target[0]-measured).tolist(),first_target_minus_handover_rad=(target[0]-np.array(m['handover']['robot_q'])).tolist(),
     first_second_samples=n,first_second_robot_range_rad=np.ptp(q[:n],axis=0).tolist(),first_second_target_range_rad=np.ptp(target[:n],axis=0).tolist(),first_second_leader_range_rad=np.ptp(leader[:n],axis=0).tolist(),
     max_normalized_action=float(np.max(np.abs(actions))),gripper_action_range=[float(actions[:,7].min()),float(actions[:,7].max())],
     all_acks=all(r['audit'].get('bridge_acknowledged') is True for r in rows),reference_limited_count=sum(bool(r['audit'].get('reference_limited')) for r in rows),
     frame_gap_ms_percentiles=(np.percentile(np.diff(t)*1000,[0,50,95,100]).tolist() if len(t)>1 else []))
  c=load_runtime_config('configs/runtime_gello_corrections.yaml');windows=list(demo_windows(rows,c.contract,hz=c.control_hz));report['valid_bc_windows']=len(windows)
  report['possible_bc_windows']=max(0,(len(rows)-c.contract.horizon)//c.contract.chunk+1)
  teacher=p.parent/'teacher_hardware_commands.json'
  if teacher.exists():
   commands=json.loads(teacher.read_text())
   if commands:report['last_teacher_target_to_first_human_target_rad']=(target[0]-np.array(commands[-1]['q_target'])).tolist()
(root/'latest_capture_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,ensure_ascii=False))
