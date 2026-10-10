"""CPU/CUDA mtmd image encoder transport for each model's pinned projector."""
import base64
import json
import struct
import tempfile
import urllib.request
from pathlib import Path
from tools.lamina_protocol import NativeProcess
from tools.gguf_reader import GGUFFile


def image_payload(item):
    """Read bounded image bytes; HTTP URLs are fetched again to detect changes."""
    url = item.get("image_url", item.get("image"))
    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str):
        raise ValueError("image_url requires a URL or base64 data URL")
    limit = 20 * 1024 * 1024
    if url.startswith("data:image/"):
        prefix, sep, encoded = url.partition(",")
        if not sep or not prefix.endswith(";base64") or len(encoded) > limit * 4 // 3 + 4:
            raise ValueError("invalid or oversized image data URL")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise ValueError("invalid image base64") from error
    elif url.startswith(("https://", "http://")):
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read(limit + 1)
    else:
        raise ValueError("images require http(s) or base64 data URLs")
    if len(payload) > limit:
        raise ValueError("image exceeds 20 MiB")
    return payload


def vocabulary_model(model, destination):
    """Copy exact GGUF metadata without weights for mtmd's vocab-only loader.

    Lamina's Ornith preparation appends F32 SSM tensors. llama.cpp requires a
    contiguous tensor directory even for vocab-only loading; the image encoder
    only needs the vocabulary and metadata, never those text-model weights.
    """
    header = GGUFFile(model)
    with model.open("rb") as source:
        magic, version, _, count = struct.unpack("<IIQQ", source.read(24))
        metadata = source.read(header.metadata_end - 24)
    with destination.open("wb") as output:
        output.write(struct.pack("<IIQQ", magic, version, 0, count))
        output.write(metadata)
        output.write(b"\0" * ((-output.tell()) % header.alignment))
    return destination


class Vision:
    def __init__(self, executable, model, projector, data, max_tokens=2048, threads=8, gpu=False):
        if not executable.is_file() or not projector.is_file():
            raise FileNotFoundError("vision encoder/projector missing; run START-HERE.bat setup for the selected model")
        self.data = data
        self.data.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.vocabulary = tempfile.TemporaryDirectory(prefix="vision-vocab-", dir=self.data)
        try:
            vocab = vocabulary_model(model, Path(self.vocabulary.name) / "vocab.gguf")
            self.native = NativeProcess([str(executable.resolve()), "--mmproj", str(projector.resolve()),
                                         "--model", str(vocab.resolve()), "--threads", str(threads),
                                         "--max-tokens", str(max_tokens), "--flash-attn", "off",
                                         *(["--gpu"] if gpu else [])], timeout=300)
        except BaseException:
            self.vocabulary.cleanup()
            raise
        try:
            ready = self.native.read()
            if ready != "READY 2048":
                detail = "\n".join(self.native.errors)
                raise RuntimeError(f"vision encoder did not become ready: {ready}\n{detail}")
        except BaseException:
            self.native.close()
            self.vocabulary.cleanup()
            raise

    def encode(self, item, directory, index):
        return self.encode_payload(image_payload(item), directory, index)

    def encode_payload(self, payload, directory, index):
        from PIL import Image
        source = directory / f"image-{index}.input"
        source.write_bytes(payload)
        with Image.open(source) as picture:
            picture.load()
            image = directory / f"image-{index}.png"
            picture.convert("RGB").save(image)
        destination = directory / f"image-{index}.sve"
        self.native.process.stdin.write("ENC " + json.dumps(str(image), ensure_ascii=False) + " " +
                                        json.dumps(str(destination), ensure_ascii=False) + "\n")
        self.native.process.stdin.flush()
        reply = self.native.read().split()
        if len(reply) != 5 or reply[0] != "OK":
            raise RuntimeError("vision encoder: " + " ".join(reply))
        with destination.open("rb") as output: header = output.read(20)
        magic, count, nx, ny, width = struct.unpack("<5i", header)
        if magic != 0x31455653 or width != 2048 or count != nx * ny or count != int(reply[1]) or count > self.max_tokens:
            raise RuntimeError(f"invalid vision output header: tokens={count}, grid={nx}x{ny}, width={width}, limit={self.max_tokens}")
        return destination, count

    def close(self):
        self.native.close()
        self.vocabulary.cleanup()
