# Work required to approach Strata performance

Status: 7 October 2026, based on committed implementation `e014949`.

Lamina has a working Qwen3.6 port, but the performance objective is unmet.
The next phase must reduce measured end-to-end inference cost. Reusing
Strata kernels alone has not reproduced its execution performance.

**Update, 7 October 2026 (RTX 3050 8 GB / Ryzen 7 5700X machine).** Step 1 has
a tool: `LAMINA_TIMELINE=1` partitions each token's stream time into stages.
Acceptance milestone 2 is reached on that machine: with CPU experts as the
fast-mode default and a coalesced DeltaNet step, the matched nine-run median is
35.84 tokens/s against 28.56 for llama.cpp b11474 (see also
[the host-path update](../bench/results/2026-10-07-rtx3050-host-path/README.md) and
[the greedy and cache update](../bench/results/2026-10-07-rtx3050-greedy-keep/README.md)), ahead in every prompt class
and with identical output across repeats. Milestone 3 (40 tokens/s) is not
reached by ordinary decode, which now measures 37.49 tokens/s. Greedy MTP
speculation measures 45.87 tokens/s and is reported separately in
[the speculation record](../bench/results/2026-10-08-rtx3050-mtp-speculation/README.md)
and [the hybrid prefill record](../bench/results/2026-10-08-rtx3050-hybrid-prefill/README.md),
which also cut the short-prompt first token from 2.18 to 0.69 s.
The largest remaining cost is the RAM-bandwidth-bound CPU expert batch
(it saturates near 16 GB/s of expert reads), then the DeltaNet projections,
GPU experts, attention layers and the DeltaNet output path; see
[the DeltaNet step update](../bench/results/2026-10-07-rtx3050-gdn-step/README.md).
Milestone 4 holds: 128K retrieval passes, with prefill unchanged and later
decode 6.02 -> 9.15 tokens/s. The RTX 4060 numbers below predate this work.
See [the CPU-expert record](../bench/results/2026-10-07-rtx3050-cpu-experts/README.md).

## Target and current evidence

The local target is at least **40 generated tokens/s** on the RTX 4060 8 GB,
with weights backed by the machine's 64 GiB RAM. Preserve usable **131072-token
context** with complete history. The short-context throughput target does
not imply that 128K decode will also reach 40 tokens/s; report those separately.

Hardware: RTX 4060, 8188 MiB VRAM, driver 610.88, PCIe Gen3 x8;
Ryzen 7 1700X, 8 cores/16 threads; 64 GiB RAM; Windows 11.
Model: pinned Qwen3.6-35B-A3B UD-Q4_K_M, architecture `qwen35moe`.

| Measurement | Lamina | CUDA llama.cpp b11474 |
|---|---:|---:|
| Prose, median later tokens/s | 20.24 | 25.28 |
| Code, median later tokens/s | 16.40 | 25.50 |
| Math, median later tokens/s | 16.27 | 25.47 |
| Nine-run median later tokens/s | 16.40 | 25.47 |
| Steady short-prompt first token | 2.45 s | 1.41 s |
| Peak total GPU memory, short runs | 6969 MiB | 6937 MiB |
| 128K cold first response, including startup | 306.52 s | 283.32 s |
| 128K retrieval | Pass | Pass |

Short tests use three repeats of each prompt, 256 generated tokens per run,
32768 context, identical input token IDs, greedy generation, resident processes,
and FP16 device KV. Lamina uses optional fast computation and registered model
RAM. Normal computation and KV defaults remain FP32. The 128K fixture contains
130928 prompt tokens and produces a nine-token answer including EOS; its answer
throughput is **not** a sustained long-context decode benchmark.

See [full comparison and commands](../bench/results/2026-10-07-llama-cuda/README.md)
and [implementation validation](../bench/results/2026-10-06-strata-plan/VALIDATION.md).
The earlier 28m57s capacity test is historical, not the current prefill result.

At 16.40 tokens/s the average decode budget is approximately 61 ms/token.
Matching the measured llama.cpp baseline requires about 39 ms/token; reaching
40 tokens/s requires **25 ms/token**, approximately a 2.44x throughput increase.
These are end-to-end budgets, not kernel-only targets.

There is no matched Strata benchmark on this machine. Do not label 40 tokens/s
as proven Strata parity. Record Strata's hardware, model, quantization, context,
prompt, precision and speculation settings before making that comparison.
If Strata cannot run a comparable model/configuration here, retain 40 tokens/s
as the local acceptance target and use llama.cpp as the measured baseline.

## Already implemented: preserve these foundations

- Checked Qwen3.6 layer math: 40 layers, 30 recurrent DeltaNet and ten gated
  attention layers, 256 experts with top-eight routing and a shared expert.
- GPU residual state, optional Q8 activation MMVQ decode, BF16 batched projection
  with FP32 accumulation, GPU norms/routing and CUDA mixer/MoE subgraphs.
- A mapped host-visible router doorbell. The host still waits for each layer's
  routing result; replacing a stream synchronization with polling has not
  removed that dependency.
- Separate weight-copy stream, persistent pinned staging workers, optional
  direct DMA from registered model RAM, resident/miss overlap, retirement
  fences, readiness events, leases and reusable allocations.
- Bounded LRU expert cache, capped at 5000 MiB on GPUs with at most 8 GiB.
- Layer-major long prefill, bounded work tiles, batched RoPE, tiled matrix
  attention, full-history host/device KV and optional FP16 KV storage.

Whole-token graph replay and a fully asynchronous routing scheduler are not
implemented. Existing subgraphs and the router doorbell must not be described
as either of those features.

## 1. Establish where the 61 ms/token goes

This is the first implementation task. The current measurements identify a
gap but do not establish one dominant cause.

Instrument a representative steady section of prose, code and math decode:

1. CPU submission time, router polling time and GPU idle intervals per layer.
2. Expert-only cache hits/misses, useful bytes uploaded per token, copy count,
   copy sizes, staging memcpy time and achieved host-to-device bandwidth.
   Separate dense/shared accesses so they do not inflate expert hit rates.
3. Resident expert work, missing expert work, dense projections, recurrent
   kernels, attention, logits/sampling, and exposed time waiting for copies.
4. Graph capture/replay counts, allocations, eviction activity and cache state
   immediately after prefill versus steady decode.
5. At 128K, KV traffic and attention cost separately from expert traffic.

Use `LAMINA_PROFILE=1` and `LAMINA_DEV_PROFILE=1` as starting points. Extend
their counters and add timeline ranges where attribution is missing. Capture
a CUDA/CPU timeline with Nsight Systems if available. Do not add overlapping
event durations together or call `gpu_chain_span_ms` GPU busy time. Existing
profiling can introduce diagnostic fences and overhead; final speed tests must
run with profiling disabled.

Measure sustained DMA bandwidth using the actual pinned/mapped source and
representative copy sizes. Compare required miss bytes/token with that measured
bandwidth. If the transfer budget alone exceeds 25 ms, reduce misses or move
some computation to CPU before expecting graph changes to reach 40 tokens/s.
PCIe Gen3 x8 is a hardware constraint; advertised GPU compute speed does not
remove it. Do not infer a hard speed ceiling without these measurements.

Deliverable: one reproducible profile, a per-token critical-path breakdown,
and a ranked list of costs that can realistically be removed.

## 2. Port scheduling around the kernels

Read the inherited Strata scheduling implementation, especially
`src/core/session.cpp`, `src/core/graph.cpp`, `src/core/expert_source.cpp`,
`src/core/expert_cache.cpp` and their headers. The source baseline is
`6f32ec070f23ced9f50e704d854d775da52591ab`. Local inherited code includes
per-layer pre/post graph replay and expert-source lookahead hooks; do not
assume Strata is simply one monolithic graph.

Adapt applicable scheduling techniques into the checked Lamina graph in
`src/model/inference.cpp` and `src/model/cuda_projection.cpp`:

- Preallocate routing arrays, expert descriptors and scratch buffers. Remove
  repeated host allocation/string construction from the hot path where profiling
  shows it matters.
- Keep graph addresses stable through device pointer tables and explicit
  ownership. Expand capture around remaining small operations where compatible
  with changing expert IDs, KV positions and recurrent state.
- Replace serialized host handling of routing/copy submission with a measured
  asynchronous producer/consumer design. Version every mailbox publication;
  preserve memory visibility, completion ordering and bounded backpressure.
- Start resident expert work while missing weights arrive. Minimize exposed
  copy waits without removing retirement or readiness dependencies.
- Evaluate larger graph replay units only after these dependencies are explicit.
  Whole-token capture is an experiment, not a guaranteed speedup or a prerequisite
  for matching Strata's scheduling.

For every change, test eviction while graphs exist, reuse of pooled memory,
long prefill after decode capture, reset, cancellation and subsequent recovery.
Never let a captured node read an evicted weight or overwrite a slot with active
readers. Preserve original expert-slot accumulation order where FP32 parity
depends on it.

**Do not run the inherited Qwen3.8 graph with Qwen3.6 weights.** Reuse scheduling
ideas and generic kernels while preserving Lamina's equations, routing,
positions, cache semantics, licenses and attribution.

## 3. Reduce expert transfer and cache costs

Use expert traces from step 1 to distinguish insufficient cache capacity,
poor replacement, partial expert residency and mistimed transfers.

- Evaluate cache allocation between dense/shared weights and routed experts
  using measured reuse. Tune by bytes and critical-path savings, not aggregate
  lookup hit rate alone.
- Compare complete gate/up/down expert residency against separate tensor entries.
  Whole-expert allocation is already an optional experiment; it has not earned
  promotion to the default.
- Examine cache state after layer-major prefill. Avoid unnecessary dense
  eviction/graph invalidation at short contexts if the memory budget permits it;
  preserve the bounded strategy that makes 128K fit.
- Compare transfer coalescing and persistent staging with representative misses.
  Keep direct registered-RAM DMA available, including its allocation failure
  fallback and startup cost.
- Treat lookahead as a bounded, speculative prefetch hint. Actual next-layer
  routing depends on next-layer activations. Measure useful prefetches, wasted
  bandwidth and cache pollution; always preserve a correct miss path.

Do not blindly raise the 5000 MiB cap. Larger caches, bounded LFU and atomic
expert caching have already regressed measured cases. Retest them only with
a specific explanation for why a new design changes the outcome.

The CUDA 13 batched-copy prototype caused illegal GPU memory access and was
removed. Any replacement requires isolated lifetime/order validation followed
by real inference correctness checks before speed claims.

## 4. Make CPU/GPU placement an evidence-based option

The faster llama.cpp configuration stores most model weights in host memory.
That is evidence to investigate hybrid execution, not proof that copying its
placement alone will deliver its throughput. Prefill and decode may choose
different execution paths.

Lamina's optional `cpu-miss` path uses AVX2 quantized dot products and workers,
but earlier trials were slower than GPU streaming. Its admission behavior after
eviction and its cache state after long prefill need inspection.

- Benchmark stable per-layer CPU expert placement as well as cache-miss fallback.
- Compare row/task partitioning, shared activation quantization, worker wakeup
  overhead and allocations with the ggml CPU implementation.
- Keep dense projections, normalization, routing and state handling on the
  appropriate device; measure each CPU/GPU boundary and synchronization cost.
- Tune threads on this 8-core Ryzen, and measure overlap with GPU work.
- Run an independent quality gate for quantized CPU execution. The existing
  fast GPU fixture does not establish CPU-quantized quality.

Retain a CPU policy only when full prompt-class benchmarks improve, not when
an isolated expert dot product gets faster.

## 5. Close the prefill and long-context gap

- Profile recurrence, expert grouping, projection GEMMs, attention and transfers
  separately during 4K, 16K, 32K and 128K prefill.
- Preserve layer-major weight reuse while tuning tile size and scratch capacity.
  Verify whether dequantization/reloading or GEMM/attention dominates before
  replacing kernels.
- Compare Lamina's attention kernels against an established Flash Attention
  implementation for the actual Qwen geometry and full-history decode. Port
  only after checking causal masking, grouped queries, RoPE and FP32/F16 behavior.
- Measure the prefill-to-decode transition: graph rebuilding, dense reuploads,
  expert cache refill and scratch release can dominate a nine-token answer.
- Add sustained long-context generation, separate from retrieval. Use a bounded
  output length that fits the remaining context and report prompt/response sizes.
- Measure host and device KV separately. Never improve reported speed by silently
  dropping history, changing precision or reducing context.

Cold startup and steady request latency are different metrics. Lamina initializes
CUDA lazily on its first prompt after RESET. A RESET acknowledgement does not
prove that GPU initialization and RAM registration have finished. Record cold
process-to-first-token latency and subsequent request latency separately.

## 6. Correctness, quality and acceptance gates

Keep the default FP32 numerical tolerance at **1e-5**. For layer-math or scheduling
changes, run the independent 40-layer reference and relevant CUDA attention,
projection, prefill and KV checks. Cover existing prefixes, one-column tails,
images/text positions, graph eviction, host/device KV, reset and cancellation.

Repair the corrupted non-ASCII paragraphs in `tests/lamina/quality_fixture.txt`,
then generate a fresh FP32 baseline and rerun fast-mode quality. Preserve
fixture/token/executable hashes. Existing thresholds are mean KL <=0.1 nats
and perplexity ratio <=1.05 over 1025 scored targets after 256 warmup tokens.
Expand quality coverage beyond this small fixture; do not reuse its results as
evidence for multilingual, CPU-quantized or 128K quality.

Acceptance milestones:

1. Reproduce the committed baseline and explain the main critical-path costs.
2. Match or beat the measured llama.cpp short-prompt baseline, without correctness
   regressions or a material prompt-class regression hidden by the aggregate.
3. Reach >=40 tokens/s on the established nine-run short-prompt test. Report all
   three prompt medians, individual runs and variability, not just the best run.
4. Preserve 128K capacity and EOS-correct retrieval; beat the current 306.52 s
   cold first response in a matched setup and report sustained long decode.
5. Claim Strata parity only after a defensible matched Strata comparison.

Do not combine results from different binaries or label an accuracy-changing
mode as FP32-equivalent. Run GPU workloads serially. Record exact commands,
engine/model hashes, hardware, precision, context, first-token time, later
tokens/s, peak VRAM including idle usage, RAM use and failure cases. Reject
changes that improve a microbenchmark but regress real inference.

## Reproduction and files to update

Start with [DEVELOPER_HANDOFF.md](DEVELOPER_HANDOFF.md) for full build/check
instructions. Use the sibling Python environment and keep data in
`../Lamina-data`.

```powershell
python -m tools.build_windows
cmake -S . -B build
cmake --build build --config Release --target lamina-gguf
python -m unittest discover -s tests/lamina
$env:OPENBLAS_NUM_THREADS='1'
python -m tools.reference_prefix --layers 40 --engine build-cuda/lamina-infer.exe --cuda --batched --tokens 42 43 44 45 --max-context 131072 --kv-cache host
$env:LAMINA_HOST_REGISTER='1'
python -m tools.quality_check
python -m tools.performance_check --long --report-dir ../Lamina-data/performance-candidate
python -m tools.compare_lamina --report-dir ../Lamina-data/comparison-lamina/candidate
python -m tools.compare_llama --server ../Lamina-data/llama-prebuilt-b11474/llama-server.exe --revision b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea --load-mode none --report-dir ../Lamina-data/comparison-llama/candidate
```

Run the first diagnostic with profiling enabled, then unset both profiling
variables before acceptance measurements. Keep optional cache/CPU experiments
out of the default environment unless that experiment is being measured.

Primary implementation files: `src/model/cuda_projection.cpp`,
`src/model/inference.cpp`, `src/model/cpu_experts.cpp`,
`src/model/cuda_kernels.cu`, `src/kernels/cuda/native_mmvq.cu`, and
`include/lamina/model/weight_stager.hpp`. Record each retained change and its
evidence in `README.md`, `docs/DEVELOPER_HANDOFF.md`, `docs/LAMINA_PORT.md`
and a dated directory under `bench/results/`.

Speculative decoding is now a separately measured greedy-only path using the
official checkpoint's MTP head, with acceptance, VRAM, determinism and
all-GPU identity measurements in
[its record](../bench/results/2026-10-08-rtx3050-mtp-speculation/README.md).
It must not conceal the ordinary decode gap: report single-token and
speculative throughput side by side.
