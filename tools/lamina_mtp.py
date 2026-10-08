"""Fetch the Qwen3.6-35B-A3B multi-token-prediction (MTP) head.

The pinned UD-Q4_K_M GGUF has no MTP tensors, but the official BF16
checkpoint does: 19 `mtp.*` tensors (1.57 GiB) in its last two of 26 shards.
A safetensors shard starts with a JSON header that gives every tensor's byte
range, so HTTP range requests read only those tensors, not the 67 GB checkpoint.

    python -m tools.lamina_mtp fetch     # resumable; writes ../Lamina-data/mtp/bf16
    python -m tools.lamina_mtp verify    # recomputes SHA-256 against the manifest
    python -m tools.lamina_mtp pack      # writes ../Lamina-data/mtp/qwen36-mtp-q8_0.gguf

`pack` writes Lamina names and the main GGUF's orientation (ne0 = input
width): matrices and experts as Q8_0, norms (as 1 + w) and the router as F32.
The fused gate_up_proj is gate-first (measured: tools.mtp_acceptance), split
into ffn_gate_exps and ffn_up_exps. The head consumes the main model's hidden
state after the final norm (metadata lamina.mtp.hidden = "post_final_norm").

A range is accepted only with HTTP 206 and the exact Content-Range asked for:
a mirror or proxy that ignores Range would otherwise return the shard's start.
HF_ENDPOINT selects a mirror of the same revision. Nothing here runs a model.

Checkpoint RMSNorm weights are zero-centred (the model applies 1 + w); the
GGUF conversion stores 1 + w. Consumers of these raw files must add 1 to every
`*norm*` weight (checked against blk.0.attn_norm, blk.3.attn_q_norm and
output_norm of the pinned GGUF).
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "Lamina-data" / "mtp" / "bf16"
REPO = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"  # the config pinned in docs/DEVELOPER_HANDOFF.md
SHARDS = ("model-00025-of-00026.safetensors", "model-00026-of-00026.safetensors")
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}
CHUNK = 8 << 20


def endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").strip().rstrip("/")


def shard_url(shard: str) -> str:
    return f"{endpoint()}/{REPO}/resolve/{REVISION}/{shard}"


def read_range(url: str, start: int, end: int):
    """Opens bytes [start, end] (inclusive) and checks the server honoured the range."""
    response = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"}), timeout=120)
    expected = f"bytes {start}-{end}/"
    if response.status != 206 or not (response.headers.get("Content-Range") or "").startswith(expected):
        raise RuntimeError(f"{url}: range {start}-{end} not honoured "
                           f"(status {response.status}, Content-Range {response.headers.get('Content-Range')!r})")
    return response


def header(shard: str):
    url = shard_url(shard)
    size = struct.unpack("<Q", read_range(url, 0, 7).read())[0]
    if size > 100 << 20:
        raise RuntimeError(f"{shard}: implausible safetensors header size {size}")
    meta = json.loads(read_range(url, 8, 7 + size).read())
    return 8 + size, meta


def inventory():
    """Every mtp.* tensor with its shard, dtype, shape and absolute byte range."""
    tensors = {}
    for shard in SHARDS:
        base, meta = header(shard)
        for name, info in meta.items():
            if not name.startswith("mtp."):
                continue
            begin, end = info["data_offsets"]
            elements = 1
            for dim in info["shape"]:
                elements *= dim
            if info["dtype"] not in DTYPE_BYTES or end - begin != elements * DTYPE_BYTES[info["dtype"]]:
                raise RuntimeError(f"{name}: unexpected dtype or size in {shard}")
            tensors[name] = {"shard": shard, "dtype": info["dtype"], "shape": info["shape"],
                             "start": base + begin, "end": base + end}
    if len(tensors) != 19:
        raise RuntimeError(f"expected 19 MTP tensors at {REVISION}, found {len(tensors)}")
    return tensors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    tensors = inventory()
    for name, info in sorted(tensors.items(), key=lambda kv: kv[1]["end"] - kv[1]["start"]):
        path = out / (name + ".bin")
        length = info["end"] - info["start"]
        have = path.stat().st_size if path.exists() else 0
        if have > length:
            path.unlink(); have = 0
        if have < length:
            response = read_range(shard_url(info["shard"]), info["start"] + have, info["end"] - 1)
            with path.open("ab") as handle:
                for block in iter(lambda: response.read(CHUNK), b""):
                    handle.write(block)
        if path.stat().st_size != length:
            raise RuntimeError(f"{name}: received {path.stat().st_size} of {length} bytes")
        info["sha256"] = sha256(path)
        print(f"{name:55s} {info['dtype']} {info['shape']} {length / 2**20:8.1f} MiB", flush=True)
    manifest = {"repo": REPO, "revision": REVISION, "tensors": tensors}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'manifest.json'} ({sum(t['end'] - t['start'] for t in tensors.values()) / 2**30:.2f} GiB)")


def verify(out: Path) -> int:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    bad = [name for name, info in manifest["tensors"].items()
           if not (out / (name + ".bin")).exists() or sha256(out / (name + ".bin")) != info["sha256"]]
    for name in bad:
        print("bad or missing:", name)
    print("verified" if not bad else f"{len(bad)} tensor(s) failed")
    return 3 if bad else 0


# Lamina GGUF name -> (checkpoint name, kind). Kinds: "norm" (F32, stored 1 + w),
# "f32", "q8" (matrix), "gate"/"up" (halves of the fused expert gate_up_proj), "experts".
PACK = {
    "mtp.fc.weight": ("mtp.fc.weight", "q8"),
    "mtp.pre_fc_norm_embedding.weight": ("mtp.pre_fc_norm_embedding.weight", "norm"),
    "mtp.pre_fc_norm_hidden.weight": ("mtp.pre_fc_norm_hidden.weight", "norm"),
    "mtp.attn_norm.weight": ("mtp.layers.0.input_layernorm.weight", "norm"),
    "mtp.attn_q.weight": ("mtp.layers.0.self_attn.q_proj.weight", "q8"),
    "mtp.attn_k.weight": ("mtp.layers.0.self_attn.k_proj.weight", "q8"),
    "mtp.attn_v.weight": ("mtp.layers.0.self_attn.v_proj.weight", "q8"),
    "mtp.attn_output.weight": ("mtp.layers.0.self_attn.o_proj.weight", "q8"),
    "mtp.attn_q_norm.weight": ("mtp.layers.0.self_attn.q_norm.weight", "norm"),
    "mtp.attn_k_norm.weight": ("mtp.layers.0.self_attn.k_norm.weight", "norm"),
    "mtp.post_attention_norm.weight": ("mtp.layers.0.post_attention_layernorm.weight", "norm"),
    "mtp.ffn_gate_inp.weight": ("mtp.layers.0.mlp.gate.weight", "f32"),
    "mtp.ffn_gate_exps.weight": ("mtp.layers.0.mlp.experts.gate_up_proj", "gate"),
    "mtp.ffn_up_exps.weight": ("mtp.layers.0.mlp.experts.gate_up_proj", "up"),
    "mtp.ffn_down_exps.weight": ("mtp.layers.0.mlp.experts.down_proj", "experts"),
    "mtp.ffn_gate_shexp.weight": ("mtp.layers.0.mlp.shared_expert.gate_proj.weight", "q8"),
    "mtp.ffn_up_shexp.weight": ("mtp.layers.0.mlp.shared_expert.up_proj.weight", "q8"),
    "mtp.ffn_down_shexp.weight": ("mtp.layers.0.mlp.shared_expert.down_proj.weight", "q8"),
    "mtp.ffn_gate_inp_shexp.weight": ("mtp.layers.0.mlp.shared_expert_gate.weight", "f32"),
    "mtp.output_norm.weight": ("mtp.norm.weight", "norm"),
}


def load_raw(src: Path, name: str, info) -> "np.ndarray":
    import numpy as np
    raw = np.fromfile(src / (name + ".bin"), dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])


def pack(src: Path, out: Path):
    import gguf
    import numpy as np
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    if manifest["revision"] != REVISION or set(manifest["tensors"]) != {source for source, _ in PACK.values()}:
        raise RuntimeError("fetched tensors do not match the pinned MTP head")
    writer = gguf.GGUFWriter(str(out), "lamina-mtp")
    writer.add_string("lamina.mtp.source_repo", REPO)
    writer.add_string("lamina.mtp.source_revision", REVISION)
    writer.add_string("lamina.mtp.hidden", "post_final_norm")
    writer.add_uint32("lamina.mtp.experts", 256)
    writer.add_uint32("lamina.mtp.experts_used", 8)
    for name, (source, kind) in PACK.items():
        value = load_raw(src, source, manifest["tensors"][source])
        if kind in ("gate", "up"):
            half = value.shape[1] // 2  # gate_up_proj [256, 2 * ff, hidden], gate rows first
            value = value[:, :half, :] if kind == "gate" else value[:, half:, :]
        if kind == "norm":
            writer.add_tensor(name, np.ascontiguousarray(value + 1, dtype=np.float32))
        elif kind == "f32":
            writer.add_tensor(name, np.ascontiguousarray(value.reshape(-1) if value.shape[0] == 1 else value, dtype=np.float32))
        else:
            q = gguf.quants.quantize(np.ascontiguousarray(value, dtype=np.float32), gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(name, q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        print(f"{name:34s} {kind:7s} {list(value.shape)}", flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {out} ({out.stat().st_size / 2**20:.0f} MiB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("inventory", "fetch", "verify", "pack"))
    parser.add_argument("--out", type=Path, default=DATA)
    parser.add_argument("--gguf", type=Path, default=DATA.parent / "qwen36-mtp-q8_0.gguf")
    args = parser.parse_args()
    if args.command == "inventory":
        for name, info in sorted(inventory().items()):
            print(name, info["dtype"], info["shape"], info["shard"], info["start"], info["end"])
    elif args.command == "fetch":
        fetch(args.out)
    elif args.command == "pack":
        pack(args.out, args.gguf)
    else:
        sys.exit(verify(args.out))


if __name__ == "__main__":
    main()
