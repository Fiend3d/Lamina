"""Pinned GGUF header regression tests; no model weights or GPU required."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.gguf_reader import GGML_TYPES, TensorInfo
from tools.lamina_model import expected_tensors, validate_header

FIXTURE = Path(__file__).with_name("qwen36_header.json")


def header():
    saved = json.loads(FIXTURE.read_text(encoding="utf-8"))
    tensors = [TensorInfo(item["name"], item["shape"], item["type_id"],
                          GGML_TYPES[item["type_id"]], item["offset"])
               for item in saved["tensors"]]
    return SimpleNamespace(version=3, metadata=saved["metadata"], tensors=tensors,
                           data_start=saved["data_start"], alignment=saved["alignment"])


class ModelContractTest(unittest.TestCase):
    def test_pinned_header(self):
        model = header()
        self.assertEqual(len(model.tensors), 733)
        self.assertEqual(model.metadata["general.architecture"], "qwen35moe")
        validate_header(model)

    def test_layer_families(self):
        expected = expected_tensors()
        self.assertIn("blk.3.attn_q.weight", expected)
        self.assertNotIn("blk.3.attn_qkv.weight", expected)
        self.assertIn("blk.4.attn_qkv.weight", expected)
        self.assertNotIn("blk.4.indexer.q_proj.weight", expected)

    def test_rejects_wrong_architecture(self):
        model = header()
        model.metadata["general.architecture"] = "qwen4exp"
        with self.assertRaisesRegex(ValueError, "general.architecture"):
            validate_header(model)

    def test_rejects_wrong_tensor_shape(self):
        model = header()
        tensor = next(t for t in model.tensors if t.name == "blk.0.attn_qkv.weight")
        tensor.shape = [2048, 4096]
        with self.assertRaisesRegex(ValueError, "attn_qkv.weight shape"):
            validate_header(model)


if __name__ == "__main__":
    unittest.main()
