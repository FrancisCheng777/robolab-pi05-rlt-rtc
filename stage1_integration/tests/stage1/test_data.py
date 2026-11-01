import tempfile
import os
import unittest
from pathlib import Path

import numpy as np

from stage1.config import ROOT, load_config, validate_contract
from stage1.data import PlugDataset, SyntheticDataset, read_manifest

TOKENIZER = ROOT / "stage1/assets/paligemma_tokenizer.model"

def fixture_root():
    explicit = os.environ.get("PI05_TEST_DATASET")
    if explicit:
        return Path(explicit)
    roots = list((ROOT / "data").glob("smoke_*/dataset/manifest.json"))
    if not roots:
        raise RuntimeError("Set PI05_TEST_DATASET to an explicitly synthetic exported dataset")
    return roots[0].parent


class DataContractTest(unittest.TestCase):
    def test_synthetic_batch_and_tokenize_before_padding(self):
        ds = SyntheticDataset(str(TOKENIZER), 10, 200)
        item = ds[0]
        self.assertEqual(item["actions"].shape, (10, 32))
        self.assertEqual(item["observation"]["state"].shape, (32,))
        self.assertTrue(np.all(item["actions"][:, 8:] == 0))
        self.assertTrue(np.all(item["observation"]["state"][9:] == 0))
        mask = item["observation"]["tokenized_prompt_mask"]
        text = ds.transform.tokenizer.decode(
            item["observation"]["tokenized_prompt"][mask].tolist()
        )
        values = text.split("State: ")[1].split(";")[0].split()
        self.assertEqual(len(values), 9)
        self.assertFalse(item["observation"]["image_mask"]["right_wrist_0_rgb"])

    def test_complete_action_windows_and_mock_gate(self):
        root = fixture_root()
        with self.assertRaisesRegex(ValueError, "Mock"):
            PlugDataset(str(root), str(TOKENIZER))
        ds = PlugDataset(str(root), str(TOKENIZER), allow_mock=True)
        episodes = np.asarray(ds.ds.hf_dataset["episode_index"])
        expected = sum(max(0, int(np.sum(episodes == ep)) - 9) for ep in np.unique(episodes))
        self.assertEqual(len(ds), expected)
        item = ds[len(ds) - 1]
        last_actions = np.asarray(ds.ds.hf_dataset["action"])[-10:]
        norm = ds.transform.normalize(last_actions, "actions")
        np.testing.assert_allclose(item["actions"][:, :8], norm, atol=1e-6)
        from pi05_demo.training_data import ChunkDataset
        generic = ChunkDataset(root / "train", allow_mock=True)
        self.assertEqual(len(generic), expected)
        np.testing.assert_allclose(generic[len(generic) - 1]["actions"], last_actions)
        self.assertEqual(
            item["observation"]["image"]["base_0_rgb"].shape, (224, 224, 3)
        )

    def test_modified_statistics_rejected(self):
        source = fixture_root()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_bytes(
                (source / "manifest.json").read_bytes()
            )
            (root / "norm_stats.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_manifest(root, allow_mock=True)

    def test_config_keeps_effective_batch_and_rejects_no_shard(self):
        cfg = load_config()
        result = validate_contract(cfg)
        self.assertEqual(result["global_batch_size"], 256)
        self.assertEqual(result["accumulation_steps"], 128)
        cfg.actor.fsdp_config.sharding_strategy = "no_shard"
        with self.assertRaisesRegex(ValueError, "full sharding"):
            validate_contract(cfg)

    def test_config_rejects_lora_and_nosync(self):
        for override in (
            "actor.model.is_lora=true",
            "actor.fsdp_config.enable_gradient_accumulation=true",
        ):
            with self.assertRaises(ValueError):
                validate_contract(load_config([override]))


if __name__ == "__main__":
    unittest.main()
