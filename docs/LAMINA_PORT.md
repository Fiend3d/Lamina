# Qwen3.6 port status

For the file map, exact validation sequence and GPU performance work plan, see
[`DEVELOPER_HANDOFF.md`](DEVELOPER_HANDOFF.md).

The input contract is pinned to Unsloth `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` revision
`a483e9e6cbd595906af30beda3187c2663a1118c`. It has 40 layers, with full attention every fourth layer,
256 routed experts and 8 selected per token. The GGUF header contains 733 tensors and uses `qwen35moe` metadata.

Completed: a C++ architecture guard and GGUF inspector; a Python tensor inventory and shape validator; a resumable,
opt-in downloader with pinned checksum; a sibling `Lamina-data` tensor index; a scalar C++ Qwen3.6 single-token
execution graph that reads quantized GGUF weights directly; and a tokenizer-backed text client with a limited
OpenAI-compatible chat endpoint.

The Strata-derived CUDA MMVQ kernels compile on Windows with CUDA 13.4. Lamina uses a direct FP32-activation
matvec path for Q8_0, Q4_K, Q5_K, and Q6_K weights, avoiding Q8_1 activation quantization. An isolated projection
suite matches scalar dequantized matvecs to approximately `1e-7` relative L2. The independent 40-layer reference
check passes on an RTX 4060 Ti: maximum hidden-state difference `1.14e-6`, next token 20 and logit 9.544323.
The whole token chain runs on the device: the hidden state, RMS norms, residuals, DeltaNet (`native_gdn` conv/L2/gate
kernels plus `native_gdn_step` with the `(i*h_v+h)*S+j` state layout and a SiLU closing norm), attention (device KV
cache, `attn_norm_rope`, `attn_decode`) and the router+MoE (dense FP32 GEMV, `router_topk`, `dot_sigmoid`, grouped
expert MMVQ, SwiGLU, `moe_combine`). Only the router's eight ids/weights and the final hidden state touch the host.
Dense/shared weights are pinned resident, the DeltaNet pre-MoE sequence is a captured CUDA graph per layer, the router
uses a mapped-pinned doorbell, and a block-hoisted Q4_K grouped kernel accelerates the gate/up projections. Weight
uploads use `cudaMemcpyAsync` on the kernel stream; a synchronous pageable copy on the legacy stream had raced the
projections and corrupted one expert's output. On an i5-12400F + RTX 4060 Ti a 120-token token-ID benchmark measured a 
median 0.043 s (19.6 later tokens/s) with ~26 ms of GPU-busy time per step. The accurate direct-FP32-activation MMVQ
decode is the remaining wall (~16 ms in the MoE); the Q8_1 path is ~4x cheaper but too inaccurate (0.127 hidden error),
so the next work is an accurate FP32-decode GEMV and graph-captured expert streaming like Strata.

The inherited Strata CUDA execution engine is Qwen3.8-specific. It expects a 48-layer gated-residual model and an
attention indexer, neither of which exists in this Qwen3.6 artifact. The scalar path implements Qwen3.6's ordinary
residuals, 30 recurrent DeltaNet layers, 10 gated full-attention layers with partial RoPE, top-eight softmax routing,
and shared experts. The published GGUF has only F32, Q8_0, Q4_K, Q5_K and Q6_K tensors; all five types are decoded.

All forty layers were compared on tokens 42 and 43 against an independent NumPy implementation using `gguf`'s
dequantizers and the pinned GGUF bytes. The largest difference among 2,048 hidden values after layer 39 was
`1.01e-6`. Both implementations selected next-token ID 20 with logit `9.544323` (native logit shown to six decimal
places). This covers nonzero recurrent state, attention KV caches and RoPE, routed experts, shared experts, final
normalization, and the output projection. Reproduce with
`python -m tools.reference_prefix --layers 40 --model PATH --engine PATH` after installing
`requirements-reference.txt`. A shorter prefix can be checked against a cropped GGUF containing the needed tensors.

The remaining native port requires:

1. Write an accurate, high-throughput FP32-activation quantized GEMV. The MoE's direct-decode kernels are the wall
   (~16 ms of a ~26 ms GPU-busy step); they decode one element at a time. Also finish the grouped kernels (Q5_K/Q6_K
   down, Q8_0 DeltaNet) and capture the MoE with a fixed resident expert set (Strata's doorbell + CPU pool). Prompt
   prefill is still unbatched.
2. Fuse compatible weight projections further and optimize transfers without regressing the verified 40-layer parity gate.
3. Broader prompt and generation comparison with the official Transformers model on capable hardware.
4. CUDA runtime checks on Linux as well as the now-verified Windows path.
5. Streaming API responses, sampling, multimodal inputs, and complete OpenAI compatibility if needed.

`lamina-infer` and `lamina.py` expose the scalar path. The two-token numerical check validates its model equations;
broader generation quality and useful 35B throughput still need the work above. The old Strata binaries and scripts
are retained only as source material for the port.
