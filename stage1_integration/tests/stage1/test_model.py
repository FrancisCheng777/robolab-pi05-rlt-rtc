import copy
import tempfile
import unittest
from pathlib import Path

import torch
from model_fixture import batch, tiny_model

torch.set_num_threads(2)


class CheckpointingTest(unittest.TestCase):
    def test_sft_recomputes_every_backbone_and_rlt_branch(self):
        torch.manual_seed(7)
        model = tiny_model().train()
        model.gradient_checkpointing_enable({"use_reentrant": False})
        counts = {"vision": 0, "language": 0, "rlt_encoder": 0, "rlt_decoder": 0}
        blocks = [
            model.img.encoder.layers[0],
            model.llm.layers[0],
            model.rlt_module.encoder.layers[0],
            model.rlt_module.decoder.layers[0],
        ]
        handles = []
        for key, block in zip(counts, blocks):

            def hook(module, args, key=key):
                counts[key] += 1

            handles.append(block.register_forward_pre_hook(hook))
        output = model(data=batch())
        self.assertEqual(set(output), {"loss", "vla_loss", "rlt_loss"})
        forward_counts = counts.copy()
        output["loss"].backward()
        for key, count in counts.items():
            self.assertGreater(
                count, forward_counts[key], f"{key} did not recompute in backward"
            )
        for handle in handles:
            handle.remove()
        for prefix in (
            "img.",
            "llm.",
            "rlt_module.encoder.",
            "rlt_module.decoder.",
            "action_in_proj.",
            "action_out_proj.",
        ):
            grads = [
                p.grad
                for n, p in model.named_parameters()
                if n.startswith(prefix) and p.grad is not None
            ]
            self.assertTrue(grads, prefix)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads), prefix)
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.0, prefix)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_recompute_preserves_losses_gradients_and_reload(self):
        torch.manual_seed(21)
        plain = tiny_model().train()
        checked = copy.deepcopy(plain)
        checked.gradient_checkpointing_enable({"use_reentrant": False})
        losses = []
        for model in (plain, checked):
            torch.manual_seed(100)
            out = model(data=batch())
            out["loss"].backward()
            losses.append(out["loss"].detach())
        torch.testing.assert_close(*losses)
        for (name, p), (_, q) in zip(
            plain.named_parameters(), checked.named_parameters()
        ):
            self.assertEqual(p.grad is None, q.grad is None, name)
            if p.grad is not None:
                torch.testing.assert_close(
                    p.grad, q.grad, rtol=1e-4, atol=1e-6, msg=name
                )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full_weights.pt"
            torch.save(checked.state_dict(), path)
            restored = tiny_model()
            restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
            for n, p in checked.state_dict().items():
                torch.testing.assert_close(p, restored.state_dict()[n], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
