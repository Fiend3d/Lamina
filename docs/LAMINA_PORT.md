# Qwen3.6 port status

The input contract is pinned to Unsloth `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` revision
`a483e9e6cbd595906af30beda3187c2663a1118c`. It has 40 layers, with full attention every fourth layer,
256 routed experts and 8 selected per token. The GGUF header contains 733 tensors and uses `qwen35moe` metadata.

Completed: a C++ architecture guard and GGUF inspector; a Python tensor inventory and shape validator; a resumable,
opt-in downloader with pinned checksum; a sibling `Lamina-data` tensor index; a scalar C++ Qwen3.6 single-token
execution graph that reads quantized GGUF weights directly; and a tokenizer-backed text client with a limited
OpenAI-compatible chat endpoint.

An opt-in hybrid CUDA projection path is wired into that scalar graph. It uses Strata's native Q8_1 activation
quantizer and MMVQ kernels for Q8_0, Q4_K, Q5_K and Q6_K matrices, uploading selected expert slices and caching
at most 512 MiB of weights on the device. F32 operations, DeltaNet state, attention, and MoE routing still run on
the CPU. The CUDA path has not been compiled or numerically checked on NVIDIA hardware in this workspace.
On a CUDA machine, run `python -m tools.reference_prefix --layers 40 --engine PATH --cuda --max-diff 0.01`
to compare its hidden state against the independent NumPy equations. The tolerance is a starting diagnostic value,
not an established parity bound; inspect any routing or next-token mismatch before relying on generation.

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

1. Compile and numerically compare the hybrid CUDA path on NVIDIA hardware; move DeltaNet, attention, routing,
   expert staging, and prompt prefill to the GPU for useful throughput.
2. Broader prompt and generation comparison with the official Transformers model on capable hardware.
3. CUDA compilation and runtime checks on Windows and Linux.
4. Streaming API responses, sampling, multimodal inputs, and complete OpenAI compatibility if needed.

`lamina-infer` and `lamina.py` expose the scalar path. The two-token numerical check validates its model equations;
broader generation quality and useful 35B throughput still need the work above. The old Strata binaries and scripts
are retained only as source material for the port.
