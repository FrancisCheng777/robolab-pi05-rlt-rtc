"""Link completed eligible trials from the agreed evaluation session; never relabel."""
from pathlib import Path
import hashlib
import json

from pi05_rlt_rtc.config.runtime_config import load_runtime_config


def import_completed(root):
    root = Path(root)
    origin = json.loads((root/'branch_origin.json').read_text())
    cfg = load_runtime_config(root/'runtime_afixed.yaml')
    imported = []
    for source in origin['evaluation_sources']:
        for label_path in sorted(Path(source).glob('episodes/*/label/annotation.json')):
            label = json.loads(label_path.read_text())
            if not (label.get('eligible') and label.get('outcome') in ('success', 'failure')
                    and label.get('mock') is False and label.get('critical_start_tick') is not None
                    and label.get('replay_counts', {}).get('learnable', 0) > 0):
                continue
            manifest = json.loads((label_path.parent/'runtime_manifest.json').read_text())
            if (manifest['policy_version_id'] != origin['policy_version_id']
                    or manifest.get('exploration') is not False
                    or manifest['runtime_contract'] != cfg.wire_contract()):
                raise RuntimeError(f'evaluation does not match fixed round10 collection: {label_path}')
            for path, expected in [(Path(label['rollout']), label['rollout_sha256']),
                                   (label_path.parent/'replay.pt', label['replay_sha256'])]:
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f'completed evaluation checksum mismatch: {path}')
            episode = label_path.parent.parent.resolve()
            target = root/'episodes'/episode.name
            if target.exists() or target.is_symlink():
                if target.resolve() != episode:
                    raise RuntimeError(f'episode destination collision: {target}')
            else:
                target.symlink_to(episode, target_is_directory=True)
            imported.append({'trial_id': label['trial_id'], 'source': str(episode), 'outcome': label['outcome']})
    temp = root/'imported_evaluations.json.tmp'
    temp.write_text(json.dumps(imported, indent=2)+'\n')
    temp.replace(root/'imported_evaluations.json')
    print(f'[已保存评测转入采集] {len(imported)} 条有效轨迹；原始记录和标签保持不变。', flush=True)
    return imported


if __name__ == '__main__':
    import sys
    import_completed(sys.argv[1])
