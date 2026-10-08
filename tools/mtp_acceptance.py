"""Measure greedy draft acceptance of the Qwen3.6 MTP head on real generations.

The engine's `--trace` mode decodes each benchmark prompt and generates
greedy tokens x, recording the residual h_t after every stepped token. For
each position t this independent NumPy implementation of the MTP head
(following vLLM's qwen3_5_mtp.py) fuses norm(embed(x_{t+1})) with norm(h_t),
runs the single MTP decoder layer causally over the sequence, applies the
final norm and the main model's LM head, and checks whether the draft equals
x_{t+2}, the main model's own greedy token. That fraction is the acceptance of
one-token greedy speculation.

Two details the published code does not pin for this checkpoint are measured
instead of assumed: whether h_t is the residual before or after the main
model's final norm, and the gate/up order inside the fused gate_up_proj. Only
the right combination reaches high acceptance.

    python -m tools.mtp_acceptance --engine build-cuda/lamina-infer.exe

Requires tools.lamina_mtp fetch. Checkpoint norm weights are stored as w and
applied as 1 + w (see tools/lamina_mtp.py).
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import gguf
import numpy as np

from tools.gguf_reader import GGUFFile
from tools.lamina_model import FILENAME
from tools.performance_check import PROMPTS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "Lamina-data"


def rms(x, w):
    return x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + 1e-6) * w


def silu(x):
    return x / (1 + np.exp(-x))


def sig(x):
    return 1 / (1 + np.exp(-x))


def rope(x, pos):
    """The reference's partial rotary embedding (first 64 of 256 dims), pos as a column vector."""
    freq = pos[:, None] / 10000000.0 ** (2 * np.arange(32) / 64)
    co, si = np.cos(freq)[:, None, :], np.sin(freq)[:, None, :]
    a, b = x[..., :32].copy(), x[..., 32:64].copy()
    x[..., :32] = a * co - b * si
    x[..., 32:64] = a * si + b * co
    return x


def load_packed(path: Path):
    """The packed MTP GGUF, mapped back to checkpoint names (norms already 1 + w)."""
    from tools.lamina_mtp import PACK
    reader = gguf.GGUFReader(str(path))
    tensors = {t.name: t for t in reader.tensors}
    w, halves = {}, {}
    for name, (source, kind) in PACK.items():
        t = tensors[name]
        shape = [int(d) for d in reversed(t.shape)]
        value = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(shape)
        if kind in ("gate", "up"):
            halves[kind] = value
        else:
            w[source] = value.reshape(1, -1) if source.endswith("shared_expert_gate.weight") else value
    w["mtp.layers.0.mlp.experts.gate_up_proj"] = np.concatenate([halves["gate"], halves["up"]], axis=1)
    return w


class Model:
    def __init__(self, gguf_path: Path, mtp_dir: Path, mtp_gguf: Path = None):
        self.gg = GGUFFile(gguf_path)
        self.by_name = {t.name: t for t in self.gg.tensors}
        self.mm = np.memmap(gguf_path, dtype=np.uint8, mode="r")
        self.w = {}
        if mtp_gguf:
            self.w = load_packed(mtp_gguf)
        else:
            manifest = json.loads((mtp_dir / "manifest.json").read_text(encoding="utf-8"))
            for name, info in manifest["tensors"].items():
                raw = np.fromfile(mtp_dir / (name + ".bin"), dtype=np.uint16)
                if info["dtype"] != "BF16":
                    raise RuntimeError(f"{name}: expected BF16")
                value = (raw.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])
                self.w[name] = value + 1 if "norm" in name else value  # zero-centred RMSNorm weights
        self.output_norm = self.values("output_norm.weight")
        head = self.by_name["output.weight"]
        self.head = self.values("output.weight").reshape(head.shape[1], head.shape[0])

    def values(self, name, rows=None):
        t = self.by_name[name]
        qtype = gguf.GGMLQuantizationType[t.type_name]
        block, size = gguf.GGML_QUANT_SIZES[qtype]
        if rows is None:
            start, count = self.gg.data_start + t.offset, t.elements
        else:
            start, count = self.gg.data_start + t.offset + rows * (t.shape[0] // block) * size, t.shape[0]
        data = self.mm[start:start + count // block * size]
        return np.frombuffer(data, dtype="<f4").copy() if t.type_name == "F32" else gguf.dequantize(data, qtype)

    def embed(self, token):
        return self.values("token_embd.weight", rows=int(token))

    def mtp_logits(self, hidden, next_tokens, post_norm: bool, gate_first: bool, context_start: int = 0):
        """Draft logits for positions 0..L-1 from h_t (rows of hidden) and x_{t+1}."""
        w = self.w
        L = len(next_tokens)
        h = rms(hidden, self.output_norm) if post_norm else hidden
        e = np.stack([self.embed(t) for t in next_tokens])
        fused = np.concatenate([rms(e, w["mtp.pre_fc_norm_embedding.weight"]),
                                rms(h, w["mtp.pre_fc_norm_hidden.weight"])], axis=1)
        x = (fused @ w["mtp.fc.weight"].T).astype(np.float32)
        p = "mtp.layers.0."
        n = rms(x, w[p + "input_layernorm.weight"])
        qfull = (n @ w[p + "self_attn.q_proj.weight"].T).reshape(L, 16, 2, 256)
        q = rms(qfull[:, :, 0, :].copy(), w[p + "self_attn.q_norm.weight"])
        gate = qfull[:, :, 1, :]
        k = rms((n @ w[p + "self_attn.k_proj.weight"].T).reshape(L, 2, 256), w[p + "self_attn.k_norm.weight"])
        v = (n @ w[p + "self_attn.v_proj.weight"].T).reshape(L, 2, 256)
        pos = np.arange(L, dtype=np.float64)  # relative positions are all attention sees
        q, k = rope(q, pos), rope(k, pos)
        attn = np.zeros((L, 16, 256), dtype=np.float32)
        mask = np.triu(np.full((L, L), -np.inf), 1)
        mask[context_start:, :context_start] = -np.inf  # optional: no MTP history before context_start
        for head in range(16):
            kv = head // 8
            scores = q[:, head, :] @ k[:, kv, :].T / 16 + mask
            probs = np.exp(scores - scores.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            attn[:, head, :] = probs @ v[:, kv, :]
        attn *= sig(gate)
        x = x + attn.reshape(L, -1) @ w[p + "self_attn.o_proj.weight"].T
        n = rms(x, w[p + "post_attention_layernorm.weight"])
        router = n @ w[p + "mlp.gate.weight"].T
        order = np.argsort(-router, axis=1)[:, :8]
        picked = np.take_along_axis(router, order, axis=1)
        weights = np.exp(picked - picked.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        moe = np.zeros_like(x)
        gate_up, down = w[p + "mlp.experts.gate_up_proj"], w[p + "mlp.experts.down_proj"]
        for expert in np.unique(order):
            rows, slots = np.nonzero(order == expert)
            gu = n[rows] @ gate_up[expert].T
            g, u = (gu[:, :512], gu[:, 512:]) if gate_first else (gu[:, 512:], gu[:, :512])
            moe[rows] += weights[rows, slots][:, None] * ((silu(g) * u) @ down[expert].T)
        shared = (silu(n @ w[p + "mlp.shared_expert.gate_proj.weight"].T) * (n @ w[p + "mlp.shared_expert.up_proj.weight"].T)) \
            @ w[p + "mlp.shared_expert.down_proj.weight"].T
        moe += sig(n @ w[p + "mlp.shared_expert_gate.weight"].T) * shared
        x = x + moe
        return rms(x, w["mtp.norm.weight"]) @ self.head.T


def trace(engine: Path, model: Path, tokens, generate: int, compute_mode: str, mtp: Path = None):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "trace.bin"
        extra = ["--mtp", str(mtp.resolve())] if mtp else []
        run = subprocess.run([str(engine.resolve()), str(model.resolve()), "--cuda", "--compute-mode", compute_mode, *extra,
                              "--trace", str(out), str(generate), *map(str, tokens)], check=True, capture_output=True, text=True)
        raw = out.read_bytes()
    count, width = np.frombuffer(raw[:8], dtype="<i4")
    ids = np.frombuffer(raw[8:8 + 4 * count], dtype="<i4")
    offset = 8 + 4 * count
    hidden = np.frombuffer(raw[offset:offset + 4 * (count - 1) * width], dtype="<f4").reshape(count - 1, width)
    offset += 4 * (count - 1) * width
    drafts = None
    if offset < len(raw):  # trailer written with --mtp: int32 count, then drafts
        n = int(np.frombuffer(raw[offset:offset + 4], dtype="<i4")[0])
        drafts = np.frombuffer(raw[offset + 4:offset + 4 + 4 * n], dtype="<i4")
    return ids, hidden, drafts, run.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", type=Path, default=ROOT / "build-cuda" / "lamina-infer.exe")
    parser.add_argument("--model", type=Path, default=DATA / "models" / FILENAME)
    parser.add_argument("--mtp", type=Path, default=DATA / "mtp" / "bf16")
    parser.add_argument("--mtp-gguf", type=Path, help="use the packed head (tools.lamina_mtp pack) instead of BF16")
    parser.add_argument("--engine-mtp", type=Path, help="also let the engine draft with this packed head and compare")
    parser.add_argument("--mtp-context", choices=("full", "generation"), default="full",
                        help="generation: the MTP layer attends only to positions from the first generated token")
    parser.add_argument("--variants", choices=("all", "chosen"), default="all",
                        help="chosen: post-norm hidden, gate-first only")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--compute-mode", choices=("f32", "fast"), default="fast")
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPTS), default=list(PROMPTS))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    from lamina import chat_prompt
    tokenizer = Tokenizer.from_file(str(DATA / "tokenizer" / "tokenizer.json"))
    model = Model(args.model, args.mtp, args.mtp_gguf)
    variants = [(post, gate_first) for post in (False, True) for gate_first in (True, False)]
    if args.variants == "chosen" or args.mtp_gguf:
        variants = [(True, True)]  # the packed file is gate-first by construction
    totals = {v: [0, 0] for v in variants}
    report = {"compute_mode": args.compute_mode, "tokens": args.tokens, "prompts": {}}
    for name in args.prompts:
        prompt = tokenizer.encode(chat_prompt([{"role": "user", "content": PROMPTS[name]}]), add_special_tokens=False).ids
        ids, hidden, gpu_drafts, engine_line = trace(args.engine, args.model, prompt, args.tokens, args.compute_mode, args.engine_mtp)
        P, T = len(prompt), len(ids)
        # Position t pairs h_t with x_{t+1} and drafts x_{t+2}; score the generated region.
        L = T - 2
        report["prompts"][name] = {"prompt_tokens": P}
        for post, gate_first in variants:
            start = P - 1 if args.mtp_context == "generation" else 0
            drafts = np.argmax(model.mtp_logits(hidden[:L], ids[1:L + 1], post, gate_first, start), axis=1)
            scored = range(P - 1, L)
            hits = sum(int(drafts[t] == ids[t + 2]) for t in scored)
            totals[(post, gate_first)][0] += hits
            totals[(post, gate_first)][1] += len(scored)
            label = f"{'post' if post else 'pre'}-norm hidden, {'gate' if gate_first else 'up'}-first"
            report["prompts"][name][label] = hits / len(scored)
            print(f"{name:6s} {label:32s} acceptance {hits}/{len(scored)} = {hits / len(scored):.3f}", flush=True)
            if gpu_drafts is not None and (post, gate_first) == (True, True):
                # Engine draft g guesses ids[P + g + 1], made at main position t = P - 1 + g.
                pairs = [(int(gpu_drafts[g]), int(drafts[P - 1 + g]), int(ids[P + g + 1]))
                         for g in range(len(gpu_drafts)) if P - 1 + g < L]
                agree = sum(a == b for a, b, _ in pairs) / len(pairs)
                accept = sum(a == c for a, _, c in pairs) / len(pairs)
                report["prompts"][name]["engine"] = {"agreement_with_numpy": agree, "acceptance": accept, "trace": engine_line}
                print(f"{name:6s} engine drafts: acceptance {accept:.3f}, agreement with NumPy {agree:.3f} ({engine_line})", flush=True)
    report["overall"] = {}
    for (post, gate_first), (hits, total) in totals.items():
        label = f"{'post' if post else 'pre'}-norm hidden, {'gate' if gate_first else 'up'}-first"
        report["overall"][label] = hits / total
        print(f"overall {label:32s} {hits}/{total} = {hits / total:.3f}")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
