# Lamina developer handoff

Read this before changing inference code. Lamina targets the pinned Qwen3.6-35B-A3B
UD-Q4_K_M GGUF. It is a native Strata port in progress, with a correct but slow
CPU path and an unverified hybrid CUDA projection path. Do not report it as a
fast GPU engine until the hardware checks below pass and measured throughput
supports that claim.

## What runs today

| Path | State | Evidence | Performance |
| --- | --- | --- | --- |
| Scalar C++ | Full 40-layer text graph runs | Two tokens checked against independent NumPy across all layers; same next token | About 0.395 later tokens/s in a four-token notebook sample |
| Hybrid CUDA | CUDA quantized projections wired into scalar graph | Linux CUDA 12.6 toolkit compile passed; Windows CUDA compile and GPU runtime remain unverified | Unknown |
| Inherited Strata CUDA engine | Qwen3.8-specific source only | Different 48-layer graph and routing; not a Lamina target | Not applicable |

The scalar measurement used an AMD Ryzen 5 7520U with a warm file cache.
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
- `src/model/cuda_projection.cpp`: opt-in hybrid projection adapter. It uses
  Strata's `src/kernels/cuda/native_mmvq.cu` and `iq_kernels.cu`, an LRU weight
  cache sized to 75% of free VRAM, and synchronous activation/result transfers
  per projection. `LAMINA_CUDA_CACHE_MB` lowers the weight-cache limit.
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

For CUDA, configure `-DLAMINA_ENABLE_CUDA=ON` and an appropriate
`CMAKE_CUDA_ARCHITECTURES`. The toolkit-only job in
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

1. **Get a real CUDA result.** Compile on Windows and Linux with the toolkit.
   Run the prefix comparison and a short text prompt on an NVIDIA GPU. Record
   GPU model, CUDA version, available VRAM, system RAM and exact command.
2. **Measure before changing kernels.** Run `python -m tools.benchmark --engine
   build-cuda/lamina-infer --cuda --tokens 4`, then repeat with `--cache-mb
   6144` on a larger card. Record first-token time and later-token throughput.
   The cache cap approximates an 8 GB card's cache budget; it does not emulate
   PCIe bandwidth or memory pressure. Test a physical 8 GB card as well.
3. **Profile transfer and CPU time.** The hybrid graph copies input/output for
   each projection, synchronizes after every call, and performs recurrent
   state, attention, routing and residual work on the CPU. Identify measured
   time spent in each before deciding which kernels to port next.
4. **Keep state on GPU.** Port DeltaNet causal convolution and recurrent state,
   full-attention KV cache, RoPE, normalization, routing, and residuals. Use
   the Qwen3.6 shapes and equations above. Avoid routing every intermediate
   tensor through host memory.
5. **Stage experts within the VRAM budget.** The GGUF is larger than 8 or 16 GB.
   Upload selected quantized expert slices, reuse likely weights, and overlap
   transfer with compute where safe. Measure miss rate, bytes transferred and
   actual peak VRAM on both card sizes.
6. **Add prompt prefill and broader validation.** Decode-only token loops make
   prompt processing slow. Preserve causal DeltaNet/attention state while
   batching prefill. Compare several prompts and generated tokens with an
   official Transformers reference before promising useful output quality.

The next developer should report separately: build success, numerical parity,
text quality, first-token time, later tokens/s, and peak VRAM. A passing build
or a single matching token is not evidence of fast or broadly correct
inference.
