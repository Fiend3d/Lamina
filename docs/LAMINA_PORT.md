# Qwen3.6 port status

The input contract is pinned to Unsloth `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` revision
`a483e9e6cbd595906af30beda3187c2663a1118c`. It has 40 layers, with full attention every fourth layer,
256 routed experts and 8 selected per token. The GGUF header contains 733 tensors and uses `qwen35moe` metadata.

Completed: a C++ architecture guard and GGUF inspector; a Python tensor inventory and shape validator; a resumable,
opt-in downloader with pinned checksum; a sibling `Lamina-data` tensor index; and a CPU-only CMake build path.

The inherited Strata execution engine is Qwen3.8-specific. It expects a 48-layer gated-residual model and an attention
indexer, neither of which exists in this Qwen3.6 artifact. The remaining native port requires:

1. A Qwen3.6 weight loader and expert layout for the published Q4_K/Q5_K/Q6_K/Q8_0 mix.
2. A 40-layer execution graph with Qwen3.6 DeltaNet state, full attention with its gate and partial RoPE, and the
   correct residual and normalization order.
3. Eight-way softmax routing, routed and shared expert math, cache placement, and prompt prefill.
4. Text tokenizer and chat template integration, a terminal client, and `/v1/chat/completions`.
5. Windows and Linux CUDA compile checks and reference-logit validation on capable NVIDIA hardware.

Until those tasks are done, no Lamina executable can generate Qwen3.6 text. The old Strata binaries and scripts are
retained only as source material for the port.
