# Strata execution work, 6-7 October 2026

This phase ports scheduling and kernel techniques into Lamina's checked
Qwen3.6 graph. It never executes the inherited Qwen3.8 graph on these weights.
The 40 tokens/s acceptance target is **not met**. Final measurements are
recorded below; microbenchmarks are separate from real generated prompts.

## Hardware and precision

RTX 4060, 8188 MiB VRAM, driver 610.88, PCIe Gen3 x8; Ryzen 7 1700X,
8 cores/16 threads; 64 GiB RAM; Windows 11 build 26100; Release CUDA 13.3,
SM 89. ComfyUI was stopped. NVML peaks include desktop/driver allocations;
idle usage varies and is recorded per run. Model SHA256 is
`ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.

Default computation and KV remain FP32. Optional `--compute-mode fast` uses
Q8 activations for quantized decode projections and BF16 tensor-core prefill,
with FP32 norms, routing, recurrence, residuals, softmax and accumulation.
FP16 KV is an independently lossy storage option. Registration
`LAMINA_HOST_REGISTER=1` pins approximately 21.1 GiB RAM and adds a substantial
cold startup cost; it is opt-in. The worker-backed staging fallback remains.

## Implementation

- Separate weight-copy stream, persistent pinned workers, retirement/readiness
  fences, leases, pooled allocations and resident/miss expert overlap.
- GPU residual throughout prefill; layer-major traversal of all prompt tiles
  reuses one layer's weights instead of reloading every layer for every tile.
  At 128K the residual is at most 1 GiB. Work tiles are at most 2048 tokens.
- Previous layers' dense/shared entries become evictable during long prefill.
  Their readers complete and captured graphs are invalidated before eviction;
  decode re-pins weights on lookup. This fixes the 128K late-layer VRAM failure.
- Bounded causal QK/PV matrices, FP32 online softmax, full KV history, batched
  mRoPE; grouped-query decode shares KV reads at histories >=2048 tokens.
- Whole-expert cache, bounded LFU and AVX2 ggml CPU miss execution are optional
  experiments. Streaming/LRU remain default after slower machine-specific runs.
  Increasing the cache also regressed. The <=8 GiB default cache is now
  capped at 5000 MiB after a controlled real-prompt comparison. Compact persistent Q8 grids are opt-in
  (`LAMINA_Q8_PERSISTENT=1`), rather than an assumed improvement.
- Benchmark retrieval stops at both model stop IDs. Layer progress is available
  with `LAMINA_PREFILL_PROGRESS=1` and `--progress-file`.

## Correctness and quality

Default CMake configure and Release build of lamina-gguf, lamina-infer and
sampling passed, as did the pinned GGUF contract and 21 Python unit tests.
CUDA elementwise, sampling, projection, attention, prefill and KV tests passed.
See `final-*.txt` for logs. The prefill suite includes existing prefixes,
one-column tails, host/device caches, later decode, prefill after captured
decode graphs, image/text positions and reset. Long scheduling differs from
chunk scheduling by at most 1.90734863e-6 in the 40-layer residual comparison.

The independent NumPy/gguf-py 40-layer check retained the default 1e-5 gate:
max absolute difference **3.59125275e-6**, next token **369** in both engines.
Matrix attention matches the scalar/fused control at 2057 keys; the attention
suite checks all 131072 keys with signal only at the oldest key. Compilation
alone is not runtime validation.

The varied held-out fixture uses 256 warmup tokens and **1025 scored targets**.
Separate FP32 and fast processes from the same build produce/compare logits.
The fresh gate passed: mean KL **0.00256116943 nats**, perplexity ratio
**0.99936955**, argmax agreement **98.05%**. Gates remain <=0.1 nats and
<=1.05. This small fixture covers prose, code and mathematics. Non-ASCII portions
were corrupted to question marks during creation; multilingual coverage is
not validated and the fixture needs repair followed by a fresh gate. It is
not a broad quality evaluation or a 128K/F16 quality gate.
It also does not validate optional CPU-quantized computation's quality.

## Commands

Use the sibling virtual environment. All GPU runs are serial. Full commands,
arguments, environment, executable hashes, hardware, timings and memory peaks
are embedded in each benchmark JSON. Data and large baseline logits stay in
sibling Lamina-data.

```powershell
python -m tools.build_windows
cmake -S . -B build
cmake --build build --config Release --target lamina-gguf lamina-infer lamina-sampling-check
python -m unittest discover -s tests/lamina
$env:OPENBLAS_NUM_THREADS='1'
python -m tools.reference_prefix --layers 40 --engine build-cuda/lamina-infer.exe --cuda --batched --tokens 42 43 44 45 --max-context 131072 --kv-cache host
$env:LAMINA_HOST_REGISTER='1'
python -m tools.quality_check
$env:LAMINA_PREFILL_PROGRESS='1'
python -m tools.performance_check --long --report-dir ../Lamina-data/performance-final-capped
python -m tools.runtime_check --compute-mode f32 --report ../Lamina-data/runtime-f32.json
python -m tools.runtime_check --compute-mode fast --report ../Lamina-data/runtime-fast.json
python -m tools.runtime_check --compute-mode fast --kv-type f16 --report ../Lamina-data/runtime-fast-f16.json
```

## Final capped-cache measurements

The default 5000 MiB cache/full-grid build (`shipping-*.json`) measured
three runs of 256 generated tokens per prompt: prose **20.580**, code
**16.388**, mathematics **16.325 tokens/s**; nine-run median **16.388**.
Cold first-token time was approximately 11 seconds. EOS-correct 128K
retrieval used 130928 prompt tokens: **306.515 seconds** to first token,
**6.366 tokens/s** across eight later tokens and **6791 MiB** total GPU peak.
The short answer is not a sustained long-context decode benchmark.

A separate resident-process/F16-KV comparison found CUDA llama.cpp ahead:
**25.468** versus Lamina **16.397 tokens/s**. See the
[matched comparison](../2026-10-07-llama-cuda/README.md).

## Earlier exploratory measurements

`fast-*.json` are separately hashed development snapshots, mostly one matched
120-token teacher-forced run. They are not the nine-run acceptance result.
Examples: worker staging 16.957 tokens/s, fast dense projections 18.436,
registered RAM 23.172, resident/miss overlap 23.690. Corrected quantized CPU
miss execution reached 14.925; LFU 21.275; an enlarged atomic cache 14.521.
Do not combine these as if they measured one final binary or real chat prompts.
The initial fixture in `quality-initial.json` is superseded by the fresh check.

The earlier FP32-compute/F16-device capacity stress took 1736.531 seconds to
first token (28m57s), at 7.212 later tokens/s and 7304 MiB total GPU peak.
`capacity-repeat-128k.json` took 278.141 seconds, 8.344 tokens/s over 63 later
tokens and 7099 MiB peak, with 130917 prompt tokens. These differ in prompts and
compute precision; this is not an equal-precision matched speedup claim.
That repeated-text capacity run emitted EOS first. A code appearing only
while forcing generation past EOS **does not count as retrieval success**.
The failure is retained alongside the EOS-correct varied-record tests.

## Remaining performance work

The 40 tokens/s target and Strata parity remain unachieved. Per-layer host/GPU
routing rendezvous and expert misses remain significant. Profiling event spans
include queue/host gaps; cumulative DMA, expert and router times overlap and
must not be added together. Whole-token graph replay with asynchronous routing
mailboxes is a possible next step, not implemented or a promised result.
Official Transformers generation comparison and Linux CUDA runtime are still
unverified. Full-context retrieval is a narrow single-needle test, not broad
long-context quality evidence.

The CUDA 13 batched-copy prototype faulted during real decode and was removed.
Its individual-copy control, with a fixed 5000 MiB cache, measured 20.689
prose tokens/s versus approximately 17 with the uncapped cache. The shipping
cap was then checked across all three prompt classes, not inferred from this
single prompt class. No batched-copy API is present in the delivered engine.

Cancellation also monitors disconnects while large PROMPT writes are blocked.
The hardware-free regression fills the stdin pipe of a non-reading child,
then verifies cancellation terminates it. Real API checks explicitly abort the
connection and verify a fresh request recovers.
