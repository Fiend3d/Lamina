# Qwen3.6 port status

Lamina uses the pinned Unsloth Qwen3.6-35B-A3B UD-Q4_K_M GGUF at revision
`a483e9e6cbd595906af30beda3187c2663a1118c`, engine architecture `qwen35moe`.
Its 40-layer graph has 30 recurrent DeltaNet and ten gated attention layers,
256 routed experts, top-eight routing and a shared expert. Strata source
baseline: `6f32ec070f23ced9f50e704d854d775da52591ab`.

Implemented in the native Lamina path:

- GGUF architecture/tensor guards, checksum-pinned downloads and scalar reference path.
- CUDA layer math, FP32 activations, accurate quantized projections, persistent
  recurrent/KV state, partial interleaved mRoPE and original-slot MoE reduction.
- Bounded expert streaming, pinned uploads, LRU residency, dynamic pointer-table
  CUDA MoE graphs, DeltaNet graphs and an optional persistent CPU miss pool.
- Causal layer-major batched text/image prefill, GPU routing/gather/SwiGLU/scatter,
  strict FP32 GEMM for larger batches, native kernels for small groups, and
  prefill scratch reclamation before decode.
- Fused shared-KV FP32 attention without the batch partial matrix, all-column
  history reuse and event-protected double-buffered host transfers.
- 32K default context and 128K complete FP32 RAM-backed KV with bounded staging;
  optional lossy FP16 storage in host or device memory, selected with --kv-type.
- Resident CLI/chat API, SSE UTF-8 streaming/usage, seeded sampling and stops,
  thinking, tool/schema validation, CPU image encoder/multiple images,
  request reset, context preflight and disconnect cancellation/recovery.
- Reproducible Release Windows build and verified sibling-directory model,
  tokenizer/template, mmproj and portable CUDA/cuBLAS assets.

The inherited Qwen3.8 graph remains disabled for Lamina. FP32 activation
accuracy is retained; the faster Q8 activation path is not enabled. Structured
JSON is prompted/validated, not guaranteed by a grammar decoder. Video, a web
UI and full OpenAI API feature parity are not included.

On this RTX 4060 8 GB/Ryzen 1700X/64 GB Windows machine, the independent
four-token 40-layer reference passes at maximum hidden difference
`3.59125275e-6`, next token369 and matching logit within `3e-6`.
The state suite passes full-layer causal prefill/decode/reset and synthetic
image mRoPE with both host/device KV, maximum difference `8.85501504e-6`,
below the unchanged `1e-5` gate. Isolated attention checks exercise all
131,072 positions including a distant-first-token signal. Seventeen Python
checks pass. These numerical tests cover specific inputs, not all generation.

Measured on this machine: matched short decode reaches a three-run median
14.129 tokens/s versus 13.260 for the original Lamina Release build (6.55%
faster), with identical predictions on all 120 teacher-forced inputs.
A 4096-token host-KV prompt reaches its first token in 26.783 seconds with
2048-token chunks, now the client default, versus 35.923 seconds with 1024.

Before the attention/KV update, the real 32K capacity run (32736 prompt + 16 generated tokens) completed at
351.465 seconds to first token, 10.870 later tokens/s and 7661.67 MiB peak
total GPU memory. The real 128K run (131040 prompt + 16 generated tokens)
completed with full FP32 host KV: 3115.672 seconds to first token, 0.689 later
tokens/s, 6980.10 MiB peak total GPU memory and 17693.29 MiB process RAM.
Both used explicit 1024-token chunks. Peaks include desktop memory. Repeated
token capacity stress does not establish semantic retrieval quality.
The 128K latency is substantial; these results do not establish Strata-class
performance. Previous RTX 4060 Ti measurements describe different hardware.

The optional CPU expert-miss policy passes an independent four-token,
40-layer reference check at maximum difference 3.33945929e-6. It measured
6.459 tokens/s with the normal cache and 3.908 with a forced 2200 MiB cache;
GPU expert streaming remains the default because it was faster.
The real API suite covers 11 checks, including images and cancellation
recovery. MoE graph profiling records 4760 replays and 40 initial misses.

The attention/KV update retains the FP32 numerical gates. Its single matched
16K host-KV run improves first-token time from 125.814 to 104.682 seconds and
later throughput from 4.657 to 6.560 tokens/s at the same sampled VRAM peak.
Short decode is unchanged at 13.93 tokens/s. Optional FP16 host/device KV reaches
9.675/16.673 tokens/s at 16K. Both precision modes pass 11 real API checks.
FP16 is experimental and lossy: a 64-token comparison against FP32 measured
1.37% relative hidden-state drift; host/device/reset/image behavior matches.
[Current measurements](../bench/results/2026-10-06-attention/VALIDATION.md)
record exact commands, hashes, precision differences and validation boundaries.

The updated FP16 device capacity profile also completed the full 128K run:
131040 prompt + 16 generated tokens, 2048-token chunks, first token 1736.531 s,
later 7.212 tokens/s, peak total GPU 7304.03 MiB and process RAM 12591.10 MiB.
Prefill remains slow (28 min 57 s). This changes precision and chunk size from
the earlier FP32 host run; it is capacity/performance evidence, not an equal-
precision speed comparison or a semantic long-context quality test.

See [the developer handoff](DEVELOPER_HANDOFF.md) for active files, exact build
and validation commands, measurement records and remaining validation boundaries.
Official Transformers generation comparison, semantic long-context retrieval
and Linux GPU runtime remain unverified; compilation does not establish
runtime performance.
