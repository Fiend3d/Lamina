# Lamina developer handoff

For the next optimization phase, follow the prioritized
[Strata performance roadmap](STRATA_PERFORMANCE_ROADMAP.md), including measured
gaps, profiling requirements, scheduling work and acceptance gates.


Current state, measured on an RTX 3050 8 GB / Ryzen 7 5700X / 64 GiB machine:
fast mode runs cache-missed experts on the CPU by default, and the matched
comparison measured **29.76 tokens/s for Lamina versus 28.56 for CUDA
llama.cpp b11474**, ahead in prose, code and math. The **40 tokens/s target is
not met**; prefill is still slower than llama.cpp; no Strata parity is claimed.
Exact commands, the stage timeline that drove the work, fixed bugs and limits
are in [the CPU-expert record](../bench/results/2026-10-07-rtx3050-cpu-experts/README.md).
The RTX 4060 machine of the earlier records (llama.cpp 25.47 versus Lamina
16.40 before this work, see [that comparison](../bench/results/2026-10-07-llama-cuda/README.md)
and [validation](../bench/results/2026-10-06-strata-plan/VALIDATION.md)) has not
been remeasured.

Start here. Lamina targets the pinned Qwen3.6-35B-A3B UD-Q4_K_M GGUF,
architecture `qwen35moe`. Strata baseline is
`6f32ec070f23ced9f50e704d854d775da52591ab`. Model, tokenizer, mmproj,
environment and toolchain assets belong in `../Lamina-data`.
The inherited Qwen3.8 graph in `src/core/` and `src/prefill/` is source material,
not Lamina inference. Do not feed it this GGUF. Preserve upstream notices.

The streaming policy (FP32 default, or `LAMINA_EXPERT_POLICY=stream`) prefetches
the next layer's predicted experts (`LAMINA_PREFETCH`, default on only with
`LAMINA_HOST_REGISTER=1`); see [prefetch measurements](../bench/results/2026-10-07-rtx3050-prefetch/README.md).
Build for a non-Ada GPU with `python -m tools.build_windows --cuda-arch 86`
(default 89 is the RTX 4060; a mismatched architecture silently gives wrong
kernels).

Profile a change with `LAMINA_TIMELINE=1` before choosing the next one. It
prints per-token averages of stream-ordered stage times (dense, router,
host_gap, moe_setup, resident, miss_wait, combine, tail and others), which
partition the stream's wall time, idle gaps included. Its markers cost about
2 ms per token; never use timeline runs as headline numbers.

## Active implementation

- `tools/lamina_model.py`, `include/lamina/model/qwen36.hpp`: pinned checksum,
  file size, metadata and tensor shape/offset contract; 40 layers, 30 DeltaNet,
  10 gated full-attention, 256 experts/top-eight, ordinary residuals.
- `src/model/inference.cpp`: native Qwen3.6 scalar and CUDA layer graph,
  layer-major causal text/image prefill, physical versus mRoPE positions,
  reset and output logits. Long text prefill traverses all tiles of one layer before advancing layers.
  A full FP32 residual (<=1 GiB) stays on GPU; work tiles stay <=2048 columns.
  The `PROMPT chunk tokens...` protocol selects this path. BATCH/PREFILL remain
  compatible chunk commands; image segments use their existing chunk path.
- `src/model/cuda_projection.cpp`: allocation-accounted VRAM budget, separate
  expert weight-cache target, constant-time LRU, worker-backed pinned upload
  ring, dense/shared weights pinned during decode, CUDA DeltaNet and dynamic MoE graphs.
  Long prefill pins only the current layer: completed layers enter LRU and
  are re-pinned on reuse. Captured dense graphs are retired and invalidated
  before their weights become evictable.
  Decode graphs read updated device pointer tables; graph keys do not depend
  on selected expert weight addresses. Never free/reuse a weight while a
  queued kernel can read it. A separate copy stream uses retirement/readiness events; resident experts
  execute while misses upload. Next-layer prefetch applies layer L+1's router to
  layer L's normalized input, copies the predicted missing experts behind layer
  L's own misses, and orders the compute stream after a `prefetch_ready` event
  (`drain_prefetch`) before any reader touches them. Optional `LAMINA_HOST_REGISTER=1` registers the
  mapped model for direct RAM DMA (21.1 GiB pinned on this machine; startup cost).
  The cache target is at most 75% of free memory, capped at 5000 MiB on <=8 GiB GPUs; whole-expert blocks and bounded frequency-aware
  eviction are opt-in via `LAMINA_ATOMIC_EXPERT_CACHE=1` and
  `LAMINA_CACHE_POLICY=lfu`. Enlarging the cache regressed and was reverted.
- `src/kernels/cuda/native_mmvq.cu`: inherited generic quantized kernels plus
  accurate FP32-activation grouped/device-table/column adapters. Optional `--compute-mode fast` uses Strata Q8 activation kernels.
  Default `f32` retains the independent 1e-5 gate; fast uses a separate
  held-out KL/perplexity gate, never FP32 equivalence. Q8 table rows whose dot
  product fits one warp (`n_in / DIV * T <= 32`, for example 512-wide Q4_K/Q5_K
  expert down projections) run one warp per row, bitwise equal to the
  four-warp kernel.
- Batched prefill uses native FP32 column reductions below 16 columns and,
  optionally, a single dequantized matrix scratch plus strict FP32 cuBLAS GEMM
  for larger groups. Fast mode uses BF16 inputs/weights with FP32 accumulation for quantized
  projections at eight or more columns. Router, norms, GDN recurrence and
  mixture accumulation remain FP32. TF32 is disabled. `LAMINA_PREFILL_BLAS` CMake option is
  off in default hardware-free builds and on in the Windows CUDA helper.
  `LAMINA_PREFILL_BLAS=0` runtime environment selects the native fallback.
  cuBLAS workspace is explicitly 8 MiB; matrix scratch is bounded/reused.
  Prefill hidden/norm/mixer buffers and position triples remain on GPU across layers;
  batched mRoPE uses two launches per attention layer instead of two per token.
  Only routing
  metadata and the final hidden row return to the CPU.
- `native_gdn_preprocess.cu`, `native_gdn.cu`: causal convolution and recurrent
  state retained across prefill chunks; recurrence traverses columns on GPU.
- `src/model/cuda_kernels.cu`: norms, routing, expert gathers/scatters and
  original-slot FP32 combination, interleaved [11,11,10] mRoPE, causal batched
  attention and history-tile softmax merge. Fused prefill reuses shared KV
  across eight query warps, preserves 128-key reduction order and updates
  the running accumulator directly. All up to 2048 query columns share one
  history sweep; the accumulator is at most 32.25 MiB. No batch partial
  matrix is allocated. Decode stages 2048 history tokens per tile; full tiles share KV loads
  across eight grouped-query heads while retaining the 128-key reduction order.
  Prefill at >=16 columns uses bounded matrix QK/PV and causal online softmax;
  FP32 uses pedantic SGEMM, fast uses BF16 tensor-core inputs and probabilities.
  `LAMINA_MATRIX_ATTN=0` selects the original fused path for comparisons.
  The matrix softmax needs a barrier between reading the row maximum and
  reusing its reduction buffer; without it the 40-layer prefill check drifted
  run to run on SM86 (fixed 2026-10-07). Router top-k is a block-wide argmax
  (formerly one thread), with bit-identical ids and weights.
- Host KV uses two bounded pinned/device slots and a separate nonblocking
  transfer stream. Per-slot copy/consumer events protect pinned-memory and
  device-buffer reuse; compute waits on copy completion. FP32 staging is
  16 MiB pinned plus 16 MiB device in total, halved for FP16.
- Optional `--kv-type f16` packs only KV into IEEE half; projections, queries,
  softmax and value accumulation remain FP32. Both host/device KV support it.
  FP32 remains the default and keeps the unchanged 1e-5 reference gate.
  FP16 is lossy and may change expert routing; do not apply the FP32 model
  parity claim to it. `lamina-kv-precision-check` tests host/device, reset,
  decode/image continuity and reports FP32 drift.
- GPU KV is default at up to 32K. Above 32K, auto selects complete host
  history (FP32 by default); 128K needs about 5 GiB for the ten attention layers, plus model
  pages and other buffers. GPU staging remains bounded. No sliding window or
  history truncation. Prefill scratch is freed before decode, preserving KV,
  convolution and recurrent state and restoring the expert-cache budget.
- `src/model/cpu_experts.cpp`: persistent spinning worker pool for the CPU
  miss policy, the default in fast mode (`LAMINA_EXPERT_POLICY` overrides; FP32
  defaults to `stream`). One token-layer's CPU experts form a batch whose rows
  are split across all workers; the calling thread helps in `wait()`. Each row
  is computed exactly as a whole-expert task would compute it. FP32 uses FP64
  dequantized reductions; fast mode uses ggml AVX2 quantized dots (build option
  `LAMINA_CPU_QUANT=ON`). Workers default to a quarter of the hardware threads,
  at most eight (`LAMINA_CPU_THREADS`). The decode router publishes the expert
  input to mapped host memory before the doorbell, CPU results return through
  mapped memory read by a kernel, and the batch starts before GPU setup.
- CPU-miss admission (`cuda_projection.cpp`, registered RAM only): after a
  layer's combine, up to `LAMINA_ADMIT_PER_LAYER` (default 1) CPU experts are
  copied into a request-local VRAM pool (`LAMINA_ADMIT_MB`, default 2048) with
  its own LRU. Admissions become visible only at the next decode token, after
  their copy event completes, and RESET drops the pool. Base-cache uploads never
  evict admissions unless only pinned entries remain. This keeps fast-mode output
  identical across repeated requests although CPU and GPU expert arithmetic
  differ in their last bits.
- `src/model/infer_main.cpp`, `sampling.hpp`: persistent native protocol,
  RESET/SAMPLE/BATCH/PREFILL/PROMPT/IMAGE, seeded sampling, prefix diagnostics.
- `tools/lamina_chat.py`, `lamina_protocol.py`, `lamina_vision.py`, `lamina.py`:
  official pinned chat template, native resident process, CPU mtmd image
  encoder, serialized HTTP API, SSE/usage/reasoning, tools, stop handling,
  JSON/schema validation, disconnect recovery and preflight context checks.
  Tool/JSON output is validated after generation, not grammar-constrained.
  Linux runtime, video and a web UI are outside this implementation.
- `tools/bootstrap_cuda.py`, `build_windows.py`, `lamina_assets.py`: verified
  portable CUDA 13.3 including cuBLAS, Release MSVC/Ninja build, runtime DLLs,
  pinned template/tokenizer config and F16 mmproj. Vision uses pinned llama.cpp
  revision from `third_party/ggml/VERSION.txt`, CPU only.

## Validation commands

Hardware-free defaults (Windows executable paths shown):

```powershell
cmake -S . -B build
cmake --build build --config Release --target lamina-gguf lamina-infer lamina-sampling-check
python -m unittest discover -s tests/lamina
build/Release/lamina-sampling-check.exe
build/Release/lamina-gguf.exe ../Lamina-data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --check
```

Install `requirements.txt`, `requirements-build.txt`, `requirements-reference.txt`
and `requirements-benchmark.txt` in `../Lamina-data/venv`. Build the CUDA path
with `python -m tools.build_windows --vision`. For other GPUs/toolkits configure
CMake with `LAMINA_ENABLE_CUDA=ON`, suitable `CMAKE_CUDA_ARCHITECTURES`, and
optionally `LAMINA_PREFILL_BLAS=ON` (requires cuBLAS development files).

```powershell
build-cuda/lamina-cuda-elementwise-check.exe
build-cuda/lamina-cuda-attention-check.exe
build-cuda/lamina-cuda-projection-check.exe ../Lamina-data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
build-cuda/lamina-prefill-check.exe ../Lamina-data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
build-cuda/lamina-kv-precision-check.exe ../Lamina-data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
build-cuda/lamina-cuda-attention-check.exe --benchmark
$env:OPENBLAS_NUM_THREADS='1'
python -m tools.reference_prefix --layers 40 --engine build-cuda/lamina-infer.exe --cuda --batched --tokens 42 43 44 45 --max-context 131072 --kv-cache host
python -m tools.runtime_check
```

Keep the independent reference's default `1e-5` tolerance. The 2026-10-06
four-token independent check passed at `3.59125275e-6`, next token 369 in both
implementations, native logit 12.448531 versus reference 12.448533724.
The batched state suite checks 4/40 layers, host/device caches, causal chunk
boundaries, decode after prefill, reset, image mRoPE and following text. All
pass; largest difference `8.85501504e-6`. Seventeen-column projection checks
cover native/GEMM adapters with relative L2 below `1e-6`. Attention tests
include 131,072 positions and a nonzero value only at the oldest position,
plus batched causal masking. The attention test also compares FP16 storage
against an independent scalar reference with explicitly rounded inputs.
These are independent scalar tests, not a full
128K model-generation test. Twenty-one Python tests and sampling checks pass.

## Measurements and validation boundaries

Machine: RTX 4060 8188 MiB, driver 610.88, Ryzen 7 1700X (8c/16t), 64 GiB RAM,
PCIe Gen3 x8, Windows, Release CUDA 13.3. ComfyUI was stopped before testing.
NVML peaks are sampled total GPU memory including desktop, not process-only
VRAM. Warm mapped-file cache is used. Do not compare the previous i5/4060 Ti
19.6 tokens/s number as though it were measured on this machine.

Exact commands, hardware, timings, peaks and executable hashes are in
`bench/results/2026-10-06-rtx4060-8gb/`. The baseline comparison uses the
original Lamina Release build at commit `81c0c41808442ba8a31c787f42a3b5a448b2d1c8`,
not the inherited Qwen3.8 Strata engine.

The following measurements precede the attention/KV update.

| Run | First token | Later tokens/s | Peak total GPU MiB |
| --- | ---: | ---: | ---: |
| Original Lamina, three-run matched median | 1.437-1.477 s | 13.260 | 6635.91 |
| New Lamina, three-run matched median | 1.542-1.574 s | 14.129 | 6641.79 |
| Final host-query 1024 binary, matched 120 | 3.665 s | 14.078 | 6771.33 |
| 4096 prompt, chunk 1024, host KV | 35.923 s | 12.062 | 6982.00 |
| 4096 prompt, chunk 2048, host KV | 26.783 s | 11.080 | 6982.00 |
| 32736 prompt, chunk 1024, device KV | 351.465 s | 10.870 | 7661.67 |
| 131040 prompt, chunk 1024, host KV | 3115.672 s | 0.689 | 6980.10 |

The matched 120 outputs are identical. Median short decode improved 6.55%;
MoE graphs recorded 4760 replays/40 initial misses. The final binary's different
startup measurement is retained with its own hash; the three-run comparison
predates only host-KV query grouping and an expert-count guard. The client now
defaults to 2048-token chunks, selected from the measured 4K comparison. Both
full-context tests used explicit 1024-token chunks. Context stress repeats token 42;
it establishes capacity and timing, not semantic retrieval or general quality.
The 128K run used 17.28 GiB peak process RAM and retained complete FP32 KV history.
Its 51 min 56 s first-token wait and 0.689 tokens/s are substantial limitations,
not evidence of Strata-class performance. The original Debug result is excluded.

```powershell
python -m tools.benchmark --engine build-cuda/lamina-infer.exe --cuda --input-tokens bench/results/2026-10-06-rtx4060-8gb/teacher-forced-120.txt --tokens 120 --quiet --json bench/results/2026-10-06-rtx4060-8gb/final-current-matched.json
python -m tools.benchmark --engine build-cuda/lamina-infer.exe --cuda --prompt-tokens 4096 --chunk 2048 --tokens 16 --max-context 131072 --kv-cache host --quiet --json bench/results/2026-10-06-rtx4060-8gb/final-prefill-4096-chunk2048.json
python -m tools.benchmark --engine build-cuda/lamina-infer.exe --cuda --prompt-tokens 32736 --chunk 1024 --tokens 16 --max-context 32768 --kv-cache auto --quiet --json bench/results/2026-10-06-rtx4060-8gb/context-32k.json
python -m tools.benchmark --engine build-cuda/lamina-infer.exe --cuda --prompt-tokens 131040 --chunk 1024 --tokens 16 --max-context 131072 --kv-cache host --quiet --json bench/results/2026-10-06-rtx4060-8gb/context-128k.json
```

Run GPU measurements one at a time. The real API suite passes 11 requests,
including SSE UTF-8/usage, seeded repeat, reset, reasoning, JSON schema,
forced tools, OCR, multiple images and recovery after disconnect.
The isolated largest-angle rotary check covers positions 0/32767/65536/131071
against analytic double-precision sin/cos, max error9.13e-7. This isolates the
first frequency; it is not an all-frequency/full-model 128K numerical reference.

The optional CPU-miss policy also passed the independent 40-layer check
(maximum difference `3.33945929e-6`, next token 369). Set
`LAMINA_EXPERT_POLICY=cpu-miss`, `LAMINA_CPU_THREADS=8`,
`LAMINA_CUDA_CACHE_MB=2200` and `OPENBLAS_NUM_THREADS=1`, then run:

```powershell
python -m tools.reference_prefix --layers 40 --engine build-cuda/lamina-infer.exe --cuda --tokens 42 43 44 45 --max-context 131072 --kv-cache host
```

On the RTX 3050 machine with the current binary this check measures
`2.16125275e-6`, next token 369. The 6.459 and 3.908 tokens/s CPU-miss results
recorded for the RTX 4060 machine predate the row-split CPU pool and admission;
on the RTX 3050 machine the CPU policy is now the faster fast-mode default.

### Attention/KV update

`bench/results/2026-10-06-attention/VALIDATION.md` records the current update.
The saved pre-update binary is SHA256 `e2623cb10cd09d4e9580863c4e4c62ec93633518632bd0b1a24ec170864c613d`;
the fused/FP16 binary is `152b597382051d3a3888228f6157fd9496ceb14718f41fd85c2ecd2a1a672b89`.
Comparison is to the completed first port, not original Lamina or Strata.
Single matched 16K/2048-chunk host-KV runs: FP32 first-token time 125.814 ->
104.682 s; later throughput 4.657 -> 6.560 tokens/s; sampled total GPU peaks
7249.75 -> 7250.30 MiB. FP16 host/device first-token times are 102.498/102.045 s,
later throughput 9.675/16.673 tokens/s, peaks 7252.52/7221.90 MiB.
Short matched 120-token decode is unchanged at 13.932 -> 13.933 tokens/s.
All FP32 comparisons and the sampled FP16 predictions agree.

The independent four-token 40-layer FP32 check remains 3.59125275e-6; the
full-layer/image state gate remains 8.85501504e-6. The isolated legacy/fused
attention benchmark is bit-identical and about 2x faster, but excludes model,
weight streaming and host transfers. Both FP32/FP16 pass 11 real API checks.
FP16 host/device/reset/image results agree exactly. Against FP32, the 64-token
sample's hidden drift is max 0.081905365, relative L2 0.0137232212. FP16 is
experimental and lossy; its sequential/batched rounding can exceed the FP32
1e-5 model gate. Do not relax that gate or call FP16 generation equivalent.
The CLI rejects FP16 on CPU before loading optional tokenizer dependencies.
A 4-layer/4097-varied-token/2048-chunk transfer test also matches host/device
exactly in both FP32 and FP16, including decode, exercising both slots and
slot-zero reuse. Twenty-one Python tests pass in the sibling venv.

The updated full 128K FP16 device profile completed: 131040 prompt + 16
response tokens, explicit 2048-token chunks, first token 1736.531 s, later
7.212 tokens/s, peak total GPU 7304.03 MiB, peak process RAM 12591.10 MiB.
All KV history is retained. This differs in precision and chunk size from the
earlier FP32 host run; do not claim an equal-precision speedup from that pair.
Prefill still takes 28 min 57 s. Exact command/hash/hardware are in
`fused-f16-device-128k.json` in the attention-update measurement directory.

Broader comparisons against official Transformers on capable hardware,
Linux GPU runtime remains unverified.
Build success alone does not establish correctness, quality or speed.
Model equations follow the pinned
[Qwen config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/config.json),
Transformers `qwen3_5_moe` and llama.cpp `conversion/qwen.py` orientation.
The NumPy reference uses independent gguf-py dequantization.


### Strata execution work (current)

The default compute mode is checked FP32. Enable optional reduced-precision
computation with `--compute-mode fast`; KV precision is a separate option.
For the streaming policy on a 64 GiB RAM machine, set `LAMINA_HOST_REGISTER=1`
to bypass staging memcpy and DMA directly from registered model pages. With the
fast-mode CPU policy it measured no faster than leaving it unset on the RTX 3050
machine; there it only enables admission. Registration failure
falls back to the persistent worker staging pool. Registration adds about
several seconds to cold startup and reserves roughly 21 GiB of pinned RAM.
It is not appropriate to set blindly on a low-RAM machine.

Reproducible new checks:

```powershell
python -m tools.quality_check
$env:LAMINA_HOST_REGISTER='1'
python -m tools.performance_check --long
python -m tools.runtime_check --compute-mode fast --report ../Lamina-data/runtime-fast.json
```

The quality fixture is `tests/lamina/quality_fixture.txt`: 256-token warmup
and 1025 teacher-forced targets, including prose, code and arithmetic. Non-ASCII fixture portions are
corrupted; multilingual quality has not been validated.
FP32 baseline logits are written under sibling Lamina-data, then the fast
process reads them. Gates: mean KL <=0.1 nats, perplexity ratio <=1.05.
This is a small deterministic regression fixture, not a broad quality eval.
`tools.performance_check` measures three serial 256-token runs for each of
prose/code/math and records the nine-run median against the 40 tokens/s target.
Its long profiles contain an early semantic needle; the 128K profile explicitly
uses lossy FP16 device KV. Do not transfer fast/F16 quality claims to FP32.

`LAMINA_DEV_PROFILE` records async DMA/expert event timings and host staging/
router waits; `LAMINA_PROFILE` prints model statistics. The GPU chain span
includes idle gaps and host delays and must not be described as GPU busy time.
Instrumentation has overhead and bounds its timing pool with a diagnostic
  fence at 1024 event pairs: never use profiled runs as headline benchmarks.

The source, commands and measured results for this phase are recorded in
`bench/results/2026-10-06-strata-plan/VALIDATION.md`. The 40 tokens/s target
must be assessed from that record, not inferred from kernel throughput.


Long CUDA text prompts use `PROMPT` through the CLI/API and benchmark tooling.
This changes the scheduling order across chunks, not the causal layer equations.
Only the residual stream has full prompt size; query/softmax/projection/MoE
workspaces remain bounded. `--prefill-chunk` sets the work tile size. This avoids
reloading all model layers for each work tile. Model weights stay RAM-backed;
one layer's experts fit the bounded GPU cache across its prompt tiles.
`LAMINA_PREFILL_PROGRESS=1` prints completed layers to stderr. Benchmark
`--prefill-schedule chunk` retains the earlier schedule for matched comparisons.
Fast MMVQ defaults to Strata's original launch arrangement. Experimental
`LAMINA_Q8_PERSISTENT=1` enables a bounded persistent grid and compact active
expert tables; it did not improve this machine's real-prompt measurements.

Long prefill releases previous layers' dense/shared weight pins, synchronizes
readers and invalidates graphs containing their addresses. Decode re-pins
weights on lookup. The regression suite includes long prefill after graph
capture, followed by reset. Retrieval benchmarks stop at both model stop IDs;
continuing generation past EOS does not count as a semantic success.

The <=8 GiB GPU cache target is capped at 5000 MiB after real-prompt cache
comparisons; a larger target reduced throughput. User overrides remain explicit
experiments. The CUDA copy-batch prototype faulted and is not included.
