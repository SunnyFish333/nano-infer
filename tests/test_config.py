import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.config import Config


class ConfigTests(unittest.TestCase):
    def load(self, hf_config, **kwargs):
        path = str(Path(__file__).resolve().parent)
        with patch("nanovllm.config.AutoConfig.from_pretrained", return_value=hf_config) as loader:
            config = Config(path, **kwargs)
            loader.assert_called_once_with(path)
            return config

    def test_dense_config_keeps_original_model_config(self):
        hf_config = SimpleNamespace(model_type="qwen3", max_position_embeddings=2048)
        config = self.load(hf_config)
        self.assertIs(config.hf_config, hf_config)
        self.assertIs(config.text_config, hf_config)
        self.assertIsNone(config.vision_config)
        self.assertEqual(config.max_model_len, 2048)
        self.assertEqual(config.image_token_id, -1)
        self.assertEqual(config.video_token_id, -1)

    def test_multimodal_config_preserves_both_backbones(self):
        text = SimpleNamespace(model_type="qwen3_5_text", max_position_embeddings=262144)
        vision = SimpleNamespace(hidden_size=768)
        hf_config = SimpleNamespace(
            model_type="qwen3_5", text_config=text, vision_config=vision,
            image_token_id=10, video_token_id=11,
            vision_start_token_id=12, vision_end_token_id=13,
        )
        config = self.load(hf_config, max_model_len=8192, max_num_batched_tokens=16384)
        self.assertIs(config.hf_config, hf_config)
        self.assertIs(config.text_config, text)
        self.assertIs(config.vision_config, vision)
        self.assertEqual(config.max_model_len, 8192)
        self.assertEqual(config.max_num_batched_tokens, 16384)
        self.assertEqual((config.image_token_id, config.video_token_id,
                          config.vision_start_token_id, config.vision_end_token_id), (10, 11, 12, 13))

    def test_derived_fields_are_not_constructor_arguments(self):
        accepted = {f.name for f in fields(Config) if f.init}
        self.assertIn("num_gdn_slots", accepted)
        for name in ("hf_config", "text_config", "vision_config", "has_gdn", "image_token_id"):
            self.assertNotIn(name, accepted)


if __name__ == "__main__":
    unittest.main()
