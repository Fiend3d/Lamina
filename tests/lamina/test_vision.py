"""Image setup and metadata transport checks without model weights or a GPU."""
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.gguf_reader import GGUFFile
from tools.lamina_vision import vocabulary_model


def string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


class VisionTest(unittest.TestCase):
    def test_vocab_copy_preserves_metadata_and_drops_noncontiguous_weights(self):
        metadata = (string("general.architecture") + struct.pack("<I", 8) + string("qwen35moe") +
                    string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) +
                    string("<|image_pad|>") + string("hello"))
        tensor = string("appended.weight") + struct.pack("<IQIQ", 1, 1, 0, 4096)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.gguf"
            source.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 1, 2) + metadata + tensor)
            destination = vocabulary_model(source, root / "vocab.gguf")
            original, copied = GGUFFile(source), GGUFFile(destination)
            self.assertEqual(original.metadata, copied.metadata)
            self.assertEqual(copied.tensors, [])
            self.assertEqual(destination.read_bytes()[24:copied.metadata_end], metadata)
            self.assertEqual(destination.stat().st_size % 32, 0)
            self.assertEqual(source.read_bytes()[-len(tensor):], tensor)


if __name__ == "__main__":
    unittest.main()
