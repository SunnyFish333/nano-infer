"""Small CPU backbone tests; requires the project's PyTorch/Transformers dependencies."""
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class Qwen35Tests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.patches = [
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.compile", side_effect=lambda fn: fn),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        from nanovllm.models import qwen3_5
        from nanovllm.utils.context import reset_context
        self.module = qwen3_5
        self.addCleanup(reset_context)
        torch.manual_seed(42)

    def config(self):
        return SimpleNamespace(
            model_type="qwen3_5_text", hidden_size=8, vocab_size=16,
            num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
            intermediate_size=12, hidden_act="silu", rms_norm_eps=1e-6,
            num_attention_heads=2, num_key_value_heads=1, head_dim=4,
            linear_num_key_heads=1, linear_num_value_heads=2,
            linear_key_head_dim=2, linear_value_head_dim=3,
            linear_conv_kernel_dim=2, tie_word_embeddings=True,
            rope_parameters={"rope_theta": 10000, "partial_rotary_factor": 1.0,
                             "mrope_section": [1, 1, 0], "mrope_interleaved": True},
        )

    def test_layer_selection_and_weight_tying(self):
        from nanovllm.layers.gdn_attention import GDNAttention
        model = self.module.Qwen3_5ForCausalLM(self.config())
        self.assertIsInstance(model.model.layers[0].linear_attn, GDNAttention)
        self.assertIsInstance(model.model.layers[1].self_attn, self.module.Qwen3_5Attention)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        with self.assertRaises(ValueError):
            cfg = self.config()
            cfg.layer_types[0] = "unknown"
            self.module.Qwen3_5ForCausalLM(cfg)

    def test_per_head_gate_layout_and_decode_shape(self):
        torch = self.torch
        attn = self.module.Qwen3_5Attention(self.config(), 1)
        with torch.no_grad():
            for param in attn.parameters():
                param.normal_(std=0.02)
        projected = torch.arange(32, dtype=torch.float32).reshape(2, 16)
        attn.q_proj.forward = lambda x: projected
        attn.q_norm = torch.nn.Identity()
        attn.k_norm = torch.nn.Identity()
        attn.o_proj = torch.nn.Identity()
        seen = {}
        def rotary(pos, q, k):
            seen["q"] = q
            return q, k
        attn.rotary_emb.forward = rotary
        inputs = torch.zeros(2, 8)
        for decode in (False, True):
            attn.attn.forward = lambda q, k, v: torch.ones_like(q).unsqueeze(1) if decode else torch.ones_like(q)
            output = attn(torch.zeros(3, 2, dtype=torch.long), inputs)
            expected_q, gate = projected.reshape(2, 2, 8).chunk(2, dim=-1)
            torch.testing.assert_close(seen["q"], expected_q)
            torch.testing.assert_close(output, gate.reshape(2, 8).sigmoid())

    def test_text_positions_and_precomputed_image_embeddings(self):
        torch = self.torch
        from nanovllm.utils.context import set_context
        model = self.module.Qwen3_5ForCausalLM(self.config())
        with torch.no_grad():
            for param in model.parameters():
                param.normal_(std=0.02)
        # Isolate embedding/position plumbing from the attention kernels.
        model.model.layers = torch.nn.ModuleList([])
        ids = torch.tensor([1, 2, 3])
        positions = torch.arange(3)
        set_context(True, cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32))
        with torch.no_grad():
            output = model(ids, positions)
            output_3d = model(ids, positions[None].expand(3, -1))
            torch.testing.assert_close(output, output_3d)
            self.assertEqual(tuple(model.compute_logits(output).shape), (1, 16))
            mask = torch.tensor([False, True, False])
            image = torch.randn(1, 8)
            injected = model(ids, positions, image, mask)
            expected = model.model.embed_tokens(ids).clone()
            expected[mask] = image
            torch.testing.assert_close(injected, model.model.norm(expected))
            with self.assertRaises(ValueError):
                model(ids, positions, image, None)

    def test_complete_dense_backbone_prefill_with_cpu_attention(self):
        torch = self.torch
        from nanovllm.utils.context import set_context
        model = self.module.Qwen3_5ForCausalLM(self.config())
        def cpu_attention(q, k, v):
            repeats = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                v.transpose(0, 1)[None], is_causal=True)
            return out[0].transpose(0, 1)
        model.model.layers[1].self_attn.attn.forward = cpu_attention
        set_context(True, cu_seqlens_q=torch.tensor([0, 5], dtype=torch.int32))
        with torch.no_grad():
            for param in model.parameters():
                param.normal_(std=0.02)
            output = model(torch.tensor([1, 2, 3, 4, 5]), torch.arange(5))
            self.assertEqual(tuple(output.shape), (5, 8))
            logits = model.compute_logits(output)
            self.assertEqual(tuple(logits.shape), (1, 16))
            self.assertTrue(torch.isfinite(logits).all().item())

    def test_gdn_prefill_decode_continuity_with_grouped_heads(self):
        torch = self.torch
        from nanovllm.layers.gdn_attention import GDNAttention
        for kernel in (1, 3):
            layer = GDNAttention(8, 1, 2, 2, 3, kernel, 1e-6)
            layer.conv_pool = torch.zeros(1, layer.conv_dim, kernel - 1)
            layer.ssm_pool = torch.zeros(1, 2, 2, 3)
            mixed = torch.randn(5, layer.conv_dim)
            g = -torch.rand(5, 2)
            beta = torch.rand(5, 2)
            with torch.no_grad():
                expected = layer._prefill_segment(mixed, g, beta, 0, False)
                expected_state = layer.ssm_pool.clone()
                first = layer._prefill_segment(mixed[:2], g[:2], beta[:2], 0, False)
                outputs = [first]
                for t in range(2, 5):
                    outputs.append(layer._decode_batch(mixed[t:t+1], g[t:t+1], beta[t:t+1], torch.tensor([0])))
                torch.testing.assert_close(torch.cat(outputs), expected, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(layer.ssm_pool, expected_state, rtol=2e-4, atol=2e-5)

    def test_language_weight_mapping_and_missing_shard_detection(self):
        torch = self.torch
        from nanovllm.utils.loader import load_model
        model = self.module.Qwen3_5ForCausalLM(self.config())
        checkpoint = {}
        expected = {}
        for name, param in model.named_parameters():
            value = torch.randn_like(param)
            expected[name] = value
            source = "model.language_model." + name[len("model."):] if name.startswith("model.") else name
            if ".gate_up_proj." in source:
                gate, up = value.chunk(2, dim=0)
                checkpoint[source.replace(".gate_up_proj.", ".gate_proj.")] = gate
                checkpoint[source.replace(".gate_up_proj.", ".up_proj.")] = up
            else:
                checkpoint[source] = value
        checkpoint["model.visual.unused.weight"] = torch.zeros(1)
        checkpoint["mtp.unused.weight"] = torch.zeros(1)
        class Reader:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def keys(self): return checkpoint.keys()
            def get_tensor(self, name): return checkpoint[name]
        with patch("nanovllm.utils.loader.glob", return_value=["mock.safetensors"]), \
                patch("nanovllm.utils.loader.safe_open", return_value=Reader()), torch.no_grad():
            load_model(model, "unused")
            for name, param in model.named_parameters():
                torch.testing.assert_close(param, expected[name])
            del checkpoint["model.language_model.layers.0.mlp.up_proj.weight"]
            with self.assertRaisesRegex(ValueError, "Missing Qwen3.5"):
                load_model(model, "unused")


if __name__ == "__main__":
    unittest.main()
