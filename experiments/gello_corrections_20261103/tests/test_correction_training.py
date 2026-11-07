import dataclasses
import json
import numpy as np
import torch
from tests.test_platform_integration import backend_double
from pi05_rlt_rtc.deployment.correction_io import FrameWriter
from pi05_rlt_rtc.deployment.correction_training import prepare_episode,fit_corrections
from pi05_rlt_rtc.learning.checkpoint import PolicyBundle
import pytest


def demo(tmp_path):
    cfg,_,_,metadata=backend_double()
    from pi05_rlt_rtc.evaluation.toy_pipeline import build_toy
    from pi05_rlt_rtc.deployment.planner import FeaturePlanner
    toy=build_toy(z_dim=16,delay=5)
    toy.feature_service.is_mock=False;toy.feature_service.enforce_single_prefix=False
    backend=FeaturePlanner(service=toy.feature_service,spec=toy.spec,contract=toy.contract,
                           control_hz=15,policy_version_id='toy-untrained-v1')
    cfg=dataclasses.replace(cfg,contract=toy.contract,raw={**cfg.raw,'model':{**cfg.raw['model'],'z_dim':16,'device':'cpu'}})
    metadata={**metadata,'mock':True,'runtime_contract':cfg.wire_contract()}
    path=tmp_path/'demo';path.mkdir()
    writer=FrameWriter(path/'human_frames.jsonl')
    for i in range(25):
        obs=dataclasses.replace(toy.env.reset(),env_id='franka',episode_id='demo',tick=200+i,
                                captured_monotonic=10+i/cfg.control_hz)
        audit=dict(normalized_action=[.08]*7+[-1.],q_target=[.004]*7,bridge_acknowledged=True)
        writer.submit(obs,audit,np.zeros(7))
    writer.close()
    m=dict(format='gello-correction-episode-v1',complete=True,eligible_bc=True,outcome='success',
           grasp_confirmed=True,gate_trigger_tick=195,handover={'robot_q':[0]*7},error=None,
           stop_acknowledged=True,metadata=metadata,runtime_contract=cfg.wire_contract(),
           frames=25,frames_sha256=writer.digest.hexdigest(),trial_id='demo')
    (path/'correction.json').write_text(json.dumps(m))
    return cfg,backend,metadata,path


def test_end_to_end_human_recording_feature_cache_bc_training_and_policy_load(tmp_path):
    cfg,backend,metadata,path=demo(tmp_path)
    data=prepare_episode(path,cfg,backend,metadata,allow_mock=True)
    assert len(data['ticks'])==4
    np.testing.assert_allclose(data['target'][:,7::8],-1.)
    assert not torch.equal(data['target'],data['reference'])
    policy,report=fit_corrections([data],cfg,metadata,tmp_path/'weights',steps=50)
    loaded=PolicyBundle.load(tmp_path/'weights/policy.pt')
    assert loaded.policy_version_id==policy.policy_version_id
    assert loaded.critic_state is None and not report['critic_trained']
    assert report['final_train_mse']<report['initial_train_mse']
    actor=loaded.build_actor(device='cpu')
    assert torch.isfinite(actor(data['actor_state'],data['reference']).actions).all()


def test_mock_or_changed_data_cannot_enter_physical_training(tmp_path):
    cfg,backend,metadata,path=demo(tmp_path)
    with pytest.raises(ValueError,match='mock'):
        prepare_episode(path,cfg,backend,metadata)
    with (path/'human_frames.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='checksum'):
        prepare_episode(path,cfg,backend,metadata,allow_mock=True)


def test_unsuccessful_or_unconfirmed_episode_excluded(tmp_path):
    cfg,backend,metadata,path=demo(tmp_path)
    m=json.loads((path/'correction.json').read_text());m['grasp_confirmed']=False
    (path/'correction.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='confirmed'):
        prepare_episode(path,cfg,backend,metadata,allow_mock=True)
