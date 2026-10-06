"""CPU mtmd image encoder transport for the pinned Qwen3.6 projector."""
import base64
import struct
import tempfile
import urllib.request
from pathlib import Path
from tools.lamina_protocol import NativeProcess


class Vision:
    def __init__(self, executable, model, projector, data, max_tokens=1024, threads=8):
        if not executable.is_file() or not projector.is_file():
            raise FileNotFoundError("vision encoder/projector missing; build tools/vision and run python -m tools.lamina_assets")
        self.data = data
        self.data.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.native = NativeProcess([str(executable.resolve()), "--mmproj", str(projector.resolve()),
                                     "--model", str(model.resolve()), "--threads", str(threads),
                                     "--max-tokens", str(max_tokens), "--flash-attn", "off"], timeout=300)
        try:
            if self.native.read() != "READY 2048":
                raise RuntimeError("vision encoder output width differs from Qwen3.6")
        except BaseException:
            self.native.close()
            raise

    def encode(self, item, directory, index):
        from PIL import Image
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
        source = directory / f"image-{index}.input"
        source.write_bytes(payload)
        with Image.open(source) as picture:
            picture.load()
            image = directory / f"image-{index}.png"
            picture.convert("RGB").save(image)
        destination = directory / f"image-{index}.sve"
        self.native.process.stdin.write(f"ENC {image} {destination}\n")
        self.native.process.stdin.flush()
        reply = self.native.read().split()
        if len(reply) != 5 or reply[0] != "OK":
            raise RuntimeError("vision encoder: " + " ".join(reply))
        with destination.open("rb") as output: header = output.read(20)
        magic, count, nx, ny, width = struct.unpack("<5i", header)
        if magic != 0x31455653 or width != 2048 or count != nx * ny or count != int(reply[1]) or count > self.max_tokens:
            raise RuntimeError("invalid vision output header")
        return destination, count

    def close(self):
        self.native.close()
