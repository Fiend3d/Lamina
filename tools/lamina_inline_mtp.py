"""Pack an in-file nextn/MTP layer (blk.<N>.*) into a Lamina MTP GGUF (mtp.*).

Some Qwen3.x MoE checkpoints (for example Ornith-1.5-35B-A3B) carry the MTP head
as an extra in-file layer at the text layer index instead of a separate file.
`lamina-infer --mtp` reads a small side GGUF with `mtp.*` names; this rewrites the
in-file layer into that shape, copying the quantized matrices verbatim.

    python -m tools.lamina_inline_mtp --model <model.gguf> --out <mtp.gguf> [--norms raw]

`--norms` controls the head's RMS-norm convention: `raw` (default; the in-file
GGUF already stores the effective gamma, so `w` is used as-is) or `plus1` (store
w + 1, GemmaRMSNorm style, matching tools/lamina_mtp.py for raw checkpoints).
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.gguf_reader import GGUFFile  # noqa: E402

NAMES = {
    "mtp.fc.weight": "nextn.eh_proj.weight",
    "mtp.pre_fc_norm_embedding.weight": "nextn.enorm.weight",
    "mtp.pre_fc_norm_hidden.weight": "nextn.hnorm.weight",
    "mtp.output_norm.weight": "nextn.shared_head_norm.weight",
    "mtp.attn_norm.weight": "attn_norm.weight",
    "mtp.attn_q.weight": "attn_q.weight",
    "mtp.attn_k.weight": "attn_k.weight",
    "mtp.attn_v.weight": "attn_v.weight",
    "mtp.attn_output.weight": "attn_output.weight",
    "mtp.attn_q_norm.weight": "attn_q_norm.weight",
    "mtp.attn_k_norm.weight": "attn_k_norm.weight",
    "mtp.post_attention_norm.weight": "post_attention_norm.weight",
    "mtp.ffn_gate_inp.weight": "ffn_gate_inp.weight",
    "mtp.ffn_gate_exps.weight": "ffn_gate_exps.weight",
    "mtp.ffn_up_exps.weight": "ffn_up_exps.weight",
    "mtp.ffn_down_exps.weight": "ffn_down_exps.weight",
    "mtp.ffn_gate_shexp.weight": "ffn_gate_shexp.weight",
    "mtp.ffn_up_shexp.weight": "ffn_up_shexp.weight",
    "mtp.ffn_down_shexp.weight": "ffn_down_shexp.weight",
    "mtp.ffn_gate_inp_shexp.weight": "ffn_gate_inp_shexp.weight",
}
NORMS = {"mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
         "mtp.attn_norm.weight", "mtp.attn_q_norm.weight", "mtp.attn_k_norm.weight",
         "mtp.post_attention_norm.weight", "mtp.output_norm.weight"}
ALIGN = 32


def _str(out, value: str):
    raw = value.encode("utf-8")
    out.write(struct.pack("<Q", len(raw)))
    out.write(raw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--norms", choices=("raw", "plus1"), default="raw")
    a = ap.parse_args(argv)

    model = GGUFFile(a.model)
    by_name = {t.name: t for t in model.tensors}
    layers = [t.name.split(".")[1] for t in model.tensors
              if t.name.startswith("blk.") and ".nextn.eh_proj." in t.name]
    if not layers:
        raise SystemExit("no in-file nextn/MTP layer found (blk.<N>.nextn.eh_proj.weight)")
    layer = max(layers, key=lambda s: int(s))
    tensors = []
    with a.model.open("rb") as fh:
        for name, suffix in NAMES.items():
            t = by_name.get(f"blk.{layer}.{suffix}")
            if t is None:
                raise SystemExit(f"missing in-file MTP tensor blk.{layer}.{suffix}")
            fh.seek(model.data_start + t.offset)
            data = fh.read(t.expected_bytes())
            type_id = t.type_id
            if name in NORMS and a.norms == "plus1":
                import numpy as np
                data = (np.frombuffer(data, dtype=np.float32) + 1.0).tobytes()
            tensors.append([name, list(t.shape), type_id, data])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("wb") as out:
        out.write(struct.pack("<I", 0x46554747))
        out.write(struct.pack("<I", 3))
        out.write(struct.pack("<Q", len(tensors)))
        out.write(struct.pack("<Q", 4))
        for key, value in (("general.architecture", "lamina-mtp"),
                           ("general.name", "Lamina inline MTP"),
                           ("lamina.mtp.hidden", "post_final_norm"),
                           ("lamina.mtp.source", a.model.name)):
            _str(out, key)
            out.write(struct.pack("<I", 8))
            _str(out, value)
        offsets, offset = [], 0
        for name, shape, type_id, data in tensors:
            _str(out, name)
            out.write(struct.pack("<I", len(shape)))
            out.write(struct.pack(f"<{len(shape)}Q", *shape))
            out.write(struct.pack("<I", type_id))
            out.write(struct.pack("<Q", offset))
            offsets.append(offset)
            offset += (-offset) % ALIGN
            offset += len(data)
        out.write(b"\0" * ((-out.tell()) % ALIGN))
        data_start = out.tell()
        for (name, shape, type_id, data), off in zip(tensors, offsets):
            assert out.tell() == data_start + off, (out.tell(), data_start + off)
            out.write(data)
            out.write(b"\0" * ((-out.tell()) % ALIGN))
    print(f"wrote {a.out} ({a.out.stat().st_size / 2**20:.0f} MiB), layer {layer}, norms {a.norms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
