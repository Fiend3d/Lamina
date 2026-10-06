# Lamina developer handoff

Read this before changing inference code. Lamina targets the pinned Qwen3.6-35B-A3B
UD-Q4_K_M GGUF. It is a native Strata port in progress, with a correct but slow
CPU path and a hybrid CUDA path that runs the whole token chain (norms, residuals,
DeltaNet, attention, router and MoE) on the GPU. The hybrid passes the 40-layer
parity gate but is launch-bound, not yet Strata-class: do not report it
as a fast GPU engine until the remaining device-resident work below is done and
measured throughput supports that claim.

## What runs today

| Path | State | Evidence | Performance |
| --- | --- | --- | --- |
| Scalar C++ | Full 40-layer text graph runs | Two tokens checked against independent NumPy across all layers; same next token | About 0.395 later tokens/s in a four-token notebook sample |
| Hybrid CUDA | Whole token chain on the GPU: hidden state, RMS norms, residuals, DeltaNet, attention (device KV cache, partial RoPE, GQA), device router top-8 and MoE all stay in VRAM; only the router's 8 ids/weights and the final hidden state touch the host. Dense weights are pinned resident; each DeltaNet layer's pre-MoE sequence is a captured CUDA graph; the router uses a mapped-pinned doorbell; the MoE uses grouped MMVQ with a block-hoisted Q4_K kernel | Windows CUDA 13.4, RTX 4060 Ti; isolated projection and elementwise suites pass; independent 40-layer check max diff `1.14e-6`, next token 20, logit 9.544323 | i5-12400F + RTX 4060 Ti: 120-token token-ID run median 0.043 s (19.6 later tokens/s). `LAMINA_PROFILE=1` reports ~38 ms GPU-busy per step, of which the MoE is ~26 ms; the accurate FP32 quantized decode is the wall |
| Inherited Strata CUDA engine | Qwen3.8-specific source only | Different 48-layer graph and routing; not a Lamina target | Not applicable |

The scalar measurement used an AMD Ryzen 5 7520U with a warm file cache. The
hybrid numbers above used an Intel Core i5-12400F (64 GB RAM) with a warm file
cache and the whole 20.6 GiB GGUF memory-mapped from `../Lamina-data`.

A race fixed during this work: the weight cache uploaded quantized blocks with a
**synchronous pageable `cudaMemcpy` on the legacy stream** while the projections
ran on a **non-blocking stream**, so a queued projection could read a
partially-uploaded weight. Weight uploads now use `cudaMemcpyAsync` on the
kernel stream. Any new device-resident work must keep every host/device transfer
ordered with the stream the kernels use.
It is not a GPU performance estimate. The 22.1 GB GGUF is memory mapped from
`../Lamina-data`; an 8 GB GPU does not need to hold the whole file, but its
throughput and even successful execution still require testing.
The scalar path defaults to at most eight workers for large projections;
`LAMINA_CPU_THREADS=1..64` overrides it. A matched warm-cache 40-layer pass
took 4.46 seconds with one worker and 2.98 seconds with eight on this notebook.

## Repository map and sources of truth

- `tools/lamina_model.py`: pinned GGUF URL, revision, 22,134,528,992-byte size,
  SHA-256, metadata, tensor names, shapes and offset checks. This is the input
  contract. `include/lamina/model/qwen36.hpp` independently checks the same
  tensor layout in C++.
- `setup.py`: resumable model download, pinned tokenizer download and sibling
  `Lamina-data` index. Never commit the model or generated index to the source
  tree.
- `src/model/inference.cpp`: Qwen3.6 token graph, GGUF row decoding, CPU
  projections, 30 recurrent DeltaNet layers, 10 full-attention layers, top-eight
  experts, shared expert, residuals and final logits.
- `src/model/cuda_projection.cpp`: hybrid projection adapter. It uses a direct
  FP32-activation path in Strata's `src/kernels/cuda/native_mmvq.cu` for Q8_0,
  Q4_K, Q5_K and Q6_K, an LRU weight cache sized to 75% of free VRAM, and
  blocking activation/result transfers with explicit device synchronization.
  `LAMINA_CUDA_CACHE_MB` lowers the cache limit. The CUDA-only
  `lamina-cuda-projection-check` target compares selected tensors with scalar
  dequantized matvecs. `LAMINA_PROFILE=1` prints per-token total, CUDA-matvec,
  CPU-matvec and remaining graph time; a sample showed ~218 ms of CUDA matvecs
  in a ~270 ms token step.
- `src/model/infer_main.cpp`: token-ID CLI, interactive text-client protocol,
  and `--prefix` hidden-state diagnostic. `lamina.py` applies the tokenizer and
  text chat template, then calls this binary. `tools/benchmark.py` measures
  multiple tokens in one process.
- `tools/reference_prefix.py`: independent NumPy and gguf-py calculation for
  tokens 42 and 43. Use it before and after changing layer math. It checks the
  final hidden state and, with 40 layers, the selected next token and logit.
- `src/core/`, most of `src/kernels/`, `src/prefill/`, `ref/`: inherited Strata
  implementation and references. These still assume Qwen3.8 unless explicitly
  ported and validated. `LAMINA_BUILD_LEGACY` is off by default.

Model equations and GGUF conversion orientation: the [Qwen model config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json),
[Transformers Qwen3.5 MoE implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py),
and [llama.cpp Qwen converter](https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py).
Those upstream files may change; the pinned GGUF bytes and local checks define
this build's supported input.

## Build and validate

Use CMake 3.24+, a C++20 compiler, Python and sufficient disk space. On
Windows, executable paths below add `Release/` and `.exe` with a multi-config
generator. The CPU build needs no CUDA toolkit.

```sh
python -m pip install -r requirements.txt
python setup.py --download
python setup.py --tokenizer
python setup.py --index
cmake -S . -B build
cmake --build build --config Release --target lamina-gguf lamina-infer
python -m unittest discover -s tests/lamina
build/lamina-gguf ../Lamina-data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --check
python lamina.py chat --engine build/lamina-infer --prompt "Hi" --max-tokens 1
```

The download validates the entire model checksum and may take a long time. To
check exact layer math, install `requirements-reference.txt` and run:

```sh
python -m pip install -r requirements-reference.txt
python -m tools.reference_prefix --layers 4 --engine build/lamina-infer
python -m tools.reference_prefix --layers 40 --engine build/lamina-infer
```

The known full-graph check on tokens 42 and 43 had maximum hidden-value
difference `1.0023523324687034e-6`, next token ID `20` in both paths, and
native logit `9.544323`. This checks the scalar math for that input, not general
generation quality or GPU parity.

For CUDA kernel checks, configure `-DLAMINA_ENABLE_CUDA=ON` and an appropriate
`CMAKE_CUDA_ARCHITECTURES`, build `lamina-cuda-projection-check`, and run it with
the pinned model path. The toolkit-only job in
`.github/workflows/lamina-cuda-build.yml` [passed on Linux with CUDA 12.6](https://github.com/Fiend3d/Lamina/actions/runs/37366756905);
it cannot run inference without a GPU. On an NVIDIA machine, start with a
four-layer check, then all 40 layers:

```sh
cmake -S . -B build-cuda -DLAMINA_ENABLE_CUDA=ON
cmake --build build-cuda --config Release --target lamina-infer
python -m tools.reference_prefix --layers 4 --engine build-cuda/lamina-infer --cuda --max-diff 0.01
python -m tools.reference_prefix --layers 40 --engine build-cuda/lamina-infer --cuda --max-diff 0.01
```

`0.01` is a diagnostic starting tolerance, not a proven error bound. Inspect
per-layer output and expert choices if it fails. Recheck next-token IDs and
actual text prompts. Do not weaken the parity gate merely to make it pass.

## Performance work in order

1. **Keep the full-graph parity gate.** The device MoE and GPU router path
   passes on RTX 4060 Ti: maximum hidden difference `1.73e-6`, next token 20 and
   native logit 9.544323. Do not weaken this gate.
2. **Measured state (i5-12400F + RTX 4060 Ti).** `LAMINA_PROFILE=1` reports
   `device gpu_busy_ms` (CUDA-event time across the whole device chain) of
   ~26 ms/step, of which the MoE is ~16 ms and the DeltaNet+attention+norms are 
   ~12 ms. The expert cache hit rate is ~0.93 with ~5 GB resident of the 11 GB
   limit; dense and shared weights are pinned. `LAMINA_DEV_PROFILE=1` splits the
   MoE into router and kernels.
3. **The MoE's accurate decode is the wall (next, biggest win).** The direct
   FP32-activation MMVQ path decodes one quantized element at a time (~30 cycles
   per element measured) because it must not quantize the activation: the Q8_1
   `native_mmvq` path is ~4x cheaper but shifting the MoE to it moved the 40
   layer hidden-state error to `0.127` (same top-1 token here, logit 9.601 vs
   9.544). Speed therefore needs a hand-written accurate FP32-decode GEMV —
   e.g. a proper block/multi-row tiling with `__dp4a`-style integer weight codes
   and exact per-group scales, or a two-level scheme that keeps the activation
   in FP32 — rather than more launch or transfer tuning. The grouped Q4_K kernel
   in `src/kernels/cuda/native_mmvq.cu` (`native_mmvq_f32_q4k_grouped_kernel`) is
   the template; Q5_K/Q6_K down projections and the Q8_0 DeltaNet projections are
   still on the generic per-element path.
4. **Capture more graphs.** DeltaNet layers already replay a captured pre-MoE
   graph. The MoE cannot be captured while its expert weights change each token;
   capturing it needs the Strata design: a fixed resident expert set, graphs per
   layer, and a CPU pool for misses, with a mapped-pinned doorbell. Attention can
   be captured once its position and KV-append offset are read from device memory.
5. **Overlap CPU and GPU experts.** The cache holds ~93% of requested experts;
   split each layer at the router and compute the misses on a CPU pool
   concurrently, as Strata's `src/core/session.cpp` does, for a larger card or a
   smaller VRAM budget. Measure hit rate, bytes transferred and peak VRAM.
6. **Add prompt prefill and broader validation.** Decode-only token loops make
   prompt processing slow. Preserve causal DeltaNet/attention state while
   batching prefill. Compare several prompts and generated tokens with an
   official Transformers reference before promising useful output quality.

The next developer should report separately: build success, numerical parity,
text quality, first-token time, later tokens/s, and peak VRAM. A passing build
or a single matching token is not evidence of fast or broadly correct
inference.
