import unittest
from unittest.mock import patch

from tools import lamina_mtp


class FakeResponse:
    def __init__(self, status, content_range):
        self.status = status
        self.headers = {"Content-Range": content_range}

    def read(self, *args):
        return b""


class MtpTest(unittest.TestCase):
    def test_pack_covers_every_checkpoint_tensor_once(self):
        sources = [source for source, _ in lamina_mtp.PACK.values()]
        kinds = {kind for _, kind in lamina_mtp.PACK.values()}
        # gate_up_proj is split into two GGUF tensors; every other tensor maps once.
        self.assertEqual(len(set(sources)), 19)
        self.assertEqual(sources.count("mtp.layers.0.mlp.experts.gate_up_proj"), 2)
        self.assertLessEqual(kinds, {"norm", "f32", "q8", "gate", "up", "experts"})
        self.assertTrue(all(name.startswith("mtp.") for name in lamina_mtp.PACK))
        self.assertTrue(all(kind == "norm" for _, (source, kind) in lamina_mtp.PACK.items() if "norm" in source))

    def test_range_reader_rejects_ignored_ranges(self):
        with patch("urllib.request.urlopen", return_value=FakeResponse(200, None)):
            with self.assertRaises(RuntimeError):
                lamina_mtp.read_range("https://example.invalid/shard", 0, 7)
        with patch("urllib.request.urlopen", return_value=FakeResponse(206, "bytes 0-15/100")):
            with self.assertRaises(RuntimeError):
                lamina_mtp.read_range("https://example.invalid/shard", 0, 7)
        with patch("urllib.request.urlopen", return_value=FakeResponse(206, "bytes 0-7/100")):
            self.assertIsNotNone(lamina_mtp.read_range("https://example.invalid/shard", 0, 7))


if __name__ == "__main__":
    unittest.main()
