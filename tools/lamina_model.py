"""The pinned Qwen3.6 GGUF contract for Lamina's first model.

Header checks run without loading the tensor data into memory. The model file
itself is kept beside the repository in ../Lamina-data.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from tools.gguf_reader import GGUFFile

REPOSITORY = "unsloth/Qwen3.6-35B-A3B-GGUF"
REVISION = "a483e9e6cbd595906af30beda3187c2663a1118c"
FILENAME = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
SIZE = 22_134_528_992
SHA256 = "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"
URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}"

METADATA = {
    "general.architecture": "qwen35moe",
    "qwen35moe.block_count": 40,
    "qwen35moe.embedding_length": 2048,
    "qwen35moe.expert_count": 256,
    "qwen35moe.expert_used_count": 8,
    "qwen35moe.expert_feed_forward_length": 512,
    "qwen35moe.expert_shared_feed_forward_length": 512,
    "qwen35moe.attention.head_count": 16,
    "qwen35moe.attention.head_count_kv": 2,
    "qwen35moe.full_attention_interval": 4,
    "qwen35moe.ssm.state_size": 128,
    "qwen35moe.ssm.group_count": 16,
    "qwen35moe.ssm.time_step_rank": 32,
    "qwen35moe.ssm.inner_size": 4096,
    "qwen35moe.ssm.conv_kernel": 4,
    "tokenizer.ggml.bos_token_id": 248044,
    "tokenizer.ggml.eos_token_id": 248046,
}


def expected_tensors() -> dict[str, list[int]]:
    shared = {
        "attn_norm.weight": [2048],
        "post_attention_norm.weight": [2048],
        "ffn_gate_inp.weight": [2048, 256],
        "ffn_gate_inp_shexp.weight": [2048],
        "ffn_gate_shexp.weight": [2048, 512],
        "ffn_up_shexp.weight": [2048, 512],
        "ffn_down_shexp.weight": [512, 2048],
        "ffn_gate_exps.weight": [2048, 512, 256],
        "ffn_up_exps.weight": [2048, 512, 256],
        "ffn_down_exps.weight": [512, 2048, 256],
    }
    linear = {
        "attn_qkv.weight": [2048, 8192],
        "attn_gate.weight": [2048, 4096],
        "ssm_out.weight": [4096, 2048],
        "ssm_conv1d.weight": [4, 8192],
        "ssm_alpha.weight": [2048, 32],
        "ssm_beta.weight": [2048, 32],
        "ssm_a": [32],
        "ssm_dt.bias": [32],
        "ssm_norm.weight": [128],
    }
    attention = {
        "attn_q.weight": [2048, 8192],
        "attn_k.weight": [2048, 512],
        "attn_v.weight": [2048, 512],
        "attn_output.weight": [4096, 2048],
        "attn_q_norm.weight": [256],
        "attn_k_norm.weight": [256],
    }
    out = {
        "token_embd.weight": [2048, 248320],
        "output_norm.weight": [2048],
        "output.weight": [2048, 248320],
    }
    for layer in range(40):
        fields = shared | (attention if layer % 4 == 3 else linear)
        out.update({f"blk.{layer}.{name}": shape for name, shape in fields.items()})
    return out


def validate_header(model: GGUFFile) -> None:
    if model.version != 3:
        raise ValueError(f"GGUF version {model.version}, expected 3")
    for key, expected in METADATA.items():
        actual = model.metadata.get(key)
        if actual != expected:
            raise ValueError(f"{key} is {actual!r}, expected {expected!r}")
    expected = expected_tensors()
    found = {tensor.name: tensor for tensor in model.tensors}
    if len(found) != len(model.tensors):
        raise ValueError("duplicate tensor names")
    missing = expected.keys() - found.keys()
    extra = found.keys() - expected.keys()
    if missing or extra:
        raise ValueError(f"tensor inventory differs: missing {sorted(missing)[:5]}, extra {sorted(extra)[:5]}")
    for name, shape in expected.items():
        tensor = found[name]
        if tensor.shape != shape:
            raise ValueError(f"{name} shape {tensor.shape}, expected {shape}")
        if tensor.expected_bytes() is None:
            raise ValueError(f"{name} has unsupported or unaligned type {tensor.type_name}")
    ordered = sorted(model.tensors, key=lambda tensor: tensor.offset)
    for index, tensor in enumerate(ordered):
        end = ordered[index + 1].offset if index + 1 < len(ordered) else SIZE - model.data_start
        size = tensor.expected_bytes()
        if tensor.offset + size > end or end - tensor.offset - size >= model.alignment:
            raise ValueError(f"{tensor.name} tensor bytes do not match GGUF offsets")


def validate_file(path: Path) -> None:
    if path.stat().st_size != SIZE:
        raise ValueError(f"{path.name} is {path.stat().st_size} bytes, expected {SIZE}")
    validate_header(GGUFFile(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise ValueError(f"{path.name} SHA-256 differs from the pinned model")
