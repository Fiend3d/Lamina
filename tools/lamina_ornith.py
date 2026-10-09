"""Set up the Ornith-1.5-35B-A3B text model for Lamina.

Ornith shares Lamina's qwen35moe graph, so it needs only its weights, tokenizer
and chat template, a one-off fix to two SSM tensors, and a packed MTP head:

    python -m tools.lamina_ornith setup

Steps (each skipped when already done):
  * download Ornith-1.5-35B-Q4_K_M.gguf (resumable, ~20 GiB) and its tokenizer
    and chat template into ../Lamina-data;
  * re-encode blk.*.ssm_alpha/beta from Q4_K to F32 (the CUDA DeltaNet path
    needs F32; the tensor bytes are appended and the directory is patched, so no
    second copy of the model is written);
  * pack the in-file nextn/MTP layer (blk.40.*) into an mtp.* side GGUF.

The portable release includes the compatible engine; use
`START-HERE.bat --model ornith` to select and prepare this model.
"""
from __future__ import annotations

import struct
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.gguf_reader import GGUFFile  # noqa: E402

BASE = "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B"
GGUF = "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF"
FILENAME = "Ornith-1.5-35B-Q4_K_M.gguf"
SIZE = 21713463040
SMALL = ("tokenizer.json", "chat_template.jinja", "config.json")


def download(url: str, dest: Path, size: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pos = dest.stat().st_size if dest.exists() else 0
    if size is None:
        pos = 0
    elif pos >= size:      # complete, or already patched (larger): leave it
        return
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    print(f"   downloading {dest.name} ({pos / 2**30:.1f} GiB done)", flush=True)
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("ab" if pos else "wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


def requantize_ssm_f32(path: Path) -> int:
    """Re-encode blk.*.ssm_alpha/beta to F32 in place. Returns the number changed."""
    import numpy as np
    import gguf
    from gguf.quants import dequantize

    SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with path.open("r+b") as f:
        assert f.read(4) == b"GGUF"
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        def rstr():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n)
        def skip():
            (t,) = struct.unpack("<I", f.read(4))
            if t == 8:
                rstr()
            elif t == 9:
                (et,) = struct.unpack("<I", f.read(4)); (cnt,) = struct.unpack("<Q", f.read(8))
                if et == 8:
                    for _ in range(cnt): rstr()
                else:
                    f.seek(SCALAR[et] * cnt, 1)
            else:
                f.seek(SCALAR[t], 1)
        for _ in range(n_kv):
            rstr(); skip()
        entries = []
        for _ in range(n_tensors):
            name = rstr().decode("utf-8")
            (nd,) = struct.unpack("<I", f.read(4))
            dims = list(struct.unpack(f"<{nd}Q", f.read(8 * nd)))
            field = f.tell()
            type_id, offset = struct.unpack("<IQ", f.read(12))
            entries.append((name, dims, type_id, offset, field))
        data_start = (f.tell() + 31) // 32 * 32
        changed = 0
        for name, dims, type_id, offset, field in entries:
            if not name.endswith(("ssm_alpha.weight", "ssm_beta.weight")) or type_id == 0:
                continue
            qtype = gguf.GGMLQuantizationType(type_id)
            elems = int(np.prod(dims))
            block_elems, block_bytes = gguf.quants.GGML_QUANT_SIZES[qtype]
            f.seek(data_start + offset)
            arr = np.asarray(dequantize(np.frombuffer(f.read(elems // block_elems * block_bytes), dtype=np.uint8), qtype),
                             dtype=np.float32)
            payload = arr.tobytes()
            f.seek(0, 2)
            f.write(b"\0" * ((-f.tell()) % 32))
            new_off = f.tell() - data_start
            f.write(payload)
            f.seek(field)
            f.write(struct.pack("<IQ", 0, new_off))
            changed += 1
    return changed


def setup(data: Path) -> None:
    models = data / "models"
    tokdir = data / "ornith"
    model = models / FILENAME
    for name in SMALL:
        if not (tokdir / name).is_file():
            download(f"{BASE}/resolve/main/{name}", tokdir / name)
    if not model.is_file() or model.stat().st_size < SIZE:
        download(f"{GGUF}/resolve/main/{FILENAME}", model, SIZE)
    changed = requantize_ssm_f32(model)
    if changed:
        print(f"   re-encoded {changed} ssm tensors to F32")
    mtp = data / "mtp" / "ornith-mtp.gguf"
    if not mtp.is_file():
        import tools.lamina_inline_mtp as inline
        inline.main(["--model", str(model), "--out", str(mtp), "--norms", "raw"])
    if not model.is_file():
        raise SystemExit("Ornith model missing after setup")


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent.parent
    data = root.parent / "Lamina-data"
    if argv and argv[0] != "setup":
        raise SystemExit("usage: python -m tools.lamina_ornith setup")
    setup(data)
    print("Ornith setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
