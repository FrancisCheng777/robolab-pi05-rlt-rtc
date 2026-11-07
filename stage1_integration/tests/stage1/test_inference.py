"""Real upstream numerical layers at tiny dimensions, RANDOM weights, CPU only.

The prompt IDs are a 256-token synthetic fixture. This is not model acceptance.
"""
import dataclasses
import unittest
from pathlib import Path

import numpy as np
import torch
from model_fixture import tiny_model, batch


class NumericalInferenceTest(unittest.TestCase):
    def test_three_rtc_modes_and_heldout_loss(self):
        from pi05_rlt_rtc import SCHEMA_VERSION, PLANNING_STATE_VERSION
        from pi05_rlt_rtc.action_codec.codec import ActionCodec, NormStats
        from pi05_rlt_rtc.adapters.rlinf_feature import RLinfFeatureReferenceService
        from pi05_rlt_rtc.adapters.panda_policy import PandaInputs, PandaOutputs
        from pi05_rlt_rtc.contracts.spaces import HorizonContract
        from pi05_rlt_rtc.contracts.observation import ObservationSnapshot
        from pi05_rlt_rtc.contracts.plan import PlanRequest, PrevPlanContext
        from pi05_rlt_rtc.rtc.dispatcher import RTCDispatcher
        from stage1.evaluate import evaluate_batches
        torch.manual_seed(19)
        model = tiny_model(eval_mode=True).eval().requires_grad_(False)
        def synthetic_tokens(data):
            data = {k: v for k, v in data.items() if k != 'prompt'}
            return {**data, 'state': np.pad(data['state'], (0,23)),
                    'tokenized_prompt': np.arange(8, dtype=np.int64),
                    'tokenized_prompt_mask': np.ones(8, bool)}
        model.setup_transforms([PandaInputs(), synthetic_tokens], [PandaOutputs()])
        codec = ActionCodec(NormStats.symmetric(), name='SYNTHETIC-CPU')
        contract = HorizonContract(3,1,1)
        service = RLinfFeatureReferenceService(model=model, contract=contract, codec=codec,
            rtc_dispatcher=RTCDispatcher(num_steps=2), feature_version_id='SYNTHETIC-CPU',
            norm_stats_sha256=codec.norm_stats.sha256, planning_state_version=PLANNING_STATE_VERSION)
        obs = ObservationSnapshot('fixture','fixture',0,np.zeros(9,np.float32),
            {k: np.zeros((224,224,3),np.uint8) for k in ('main','wrist')},
            {'main':True,'wrist':True},'synthetic CPU fixture')
        prev = PrevPlanContext(np.zeros((3,8),np.float32),np.ones(3,bool),0,0,'SYNTHETIC-CPU')
        req = PlanRequest(0,'fixture','fixture',1,0,1,1,'SYNTHETIC-CPU','SYNTHETIC-CPU',
                          SCHEMA_VERSION,PLANNING_STATE_VERSION,prev,42,'none',2)
        for mode in ('none','approx','full_vjp'):
            request = dataclasses.replace(req, sampler_mode=mode)
            first = service.extract(request, obs)
            second = service.extract(request, obs)
            first.validate(contract, expected_batch=1)
            np.testing.assert_allclose(first.reference_env_full, second.reference_env_full, atol=1e-6)
            self.assertTrue(np.isfinite(first.z_rl).all())
        # Trace the complete cached suffix and its input-gradient path. The
        # eager compiler backend checks graph compatibility without CUDA.
        model.run_suffix = torch.compile(model.run_suffix, backend='eager', fullgraph=True)
        compiled = service.extract(request, obs)
        np.testing.assert_allclose(compiled.reference_env_full, second.reference_env_full, atol=1e-6)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        loss = evaluate_batches(model, [batch()], max_batches=1)
        self.assertTrue(np.isfinite(loss['flow_mse_physical8']))
        self.assertTrue(np.isfinite(loss['rlt_reconstruction_loss']))

    def test_local_tokenizer_matches_training(self):
        from pi05_rlt_rtc.adapters.tokenizer import LocalPaliGemmaTokenizer
        from stage1.data import Transform
        path = Path(__file__).resolve().parents[2] / 'stage1/assets/paligemma_tokenizer.model'
        stats = {'norm_stats': {k: {'q01':[-1.]*d,'q99':[1.]*d} for k,d in [('state',9),('actions',8)]}}
        train = Transform(stats, str(path))
        state = np.linspace(-.8,.8,9,dtype=np.float32)
        prompt = 'insert_the plug\ninto the power strip'
        transformed = train({'state':state,'actions':np.zeros((3,8)),
            'image':{k:np.zeros((224,224,3),np.uint8) for k in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},
            'image_mask':{'base_0_rgb':True,'left_wrist_0_rgb':True,'right_wrist_0_rgb':False},'prompt':prompt})
        tokens, mask = LocalPaliGemmaTokenizer(path).tokenize(prompt, train.normalize(state,'state'))
        np.testing.assert_array_equal(tokens, transformed['observation']['tokenized_prompt'])
        np.testing.assert_array_equal(mask, transformed['observation']['tokenized_prompt_mask'])
