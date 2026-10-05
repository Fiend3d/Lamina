# Qwen3.6 port status

The input contract is pinned to Unsloth `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` revision
`a483e9e6cbd595906af30beda3187c2663a1118c`. It has 40 layers, with full attention every fourth layer,
256 routed experts and 8 selected per token. The GGUF header contains 733 tensors and uses `qwen35moe` metadata.

Completed: a C++ architecture guard and GGUF inspector; a Python tensor inventory and shape validator; a resumable,
opt-in downloader with pinned checksum; a sibling `Lamina-data` tensor index; a scalar C++ Qwen3.6 single-token
execution graph that reads quantized GGUF weights directly; and a tokenizer-backed text client with a limited
OpenAI-compatible chat endpoint.

The inherited Strata CUDA execution engine is Qwen3.8-specific. It expects a 48-layer gated-residual model and an
attention indexer, neither of which exists in this Qwen3.6 artifact. The scalar path implements Qwen3.6's ordinary
residuals, 30 recurrent DeltaNet layers, 10 gated full-attention layers with partial RoPE, top-eight softmax routing,
and shared experts. The published GGUF has only F32, Q8_0, Q4_K, Q5_K and Q6_K tensors; all five types are decoded.

The first twelve layers were compared on tokens 42 and 43 against an independent NumPy implementation using
`gguf`'s dequantizers and the published GGUF bytes. The largest difference among 2,048 hidden values after layer 11
was `3.17e-7`. This covers nonzero recurrent state, three attention KV caches and RoPE, routed experts, and shared
experts. Reproduce with `python -m tools.reference_prefix --layers 12 --model PATH --engine PATH` after installing
`requirements-reference.txt`. A cropped GGUF through layer 11 also works.

The remaining native port requires:

1. GPU projection kernels, efficient quantized expert staging, and batched prompt prefill.
2. Full-model logit and generation comparison with the official reference on capable hardware.
3. CUDA compilation and runtime checks on Windows and Linux.
4. Streaming API responses, sampling, multimodal inputs, and complete OpenAI compatibility if needed.

`lamina-infer` and `lamina.py` expose the scalar path for development. Their generated output remains unvalidated
until it is checked against reference logits; useful 35B throughput also needs the CUDA work. The old Strata binaries
and scripts are retained only as source material for the port.
