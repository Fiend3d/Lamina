# CPU experts by default in fast mode: RTX 3050 8 GB

This directory records the change that makes Lamina faster than CUDA llama.cpp
on this machine, together with the diagnosis behind it, the bugs found on the
way, and the validation of the final binary. It is written for developers who
need exact commands, hardware and limits.

## Summary

On this machine Lamina's fast mode now decodes at a nine-run median of
**29.76 tokens/s against 28.56 tokens/s for CUDA llama.cpp b11474**, ahead in
all three prompt classes, with identical output across repeated requests. Like
for like with the session's starting point, the nine-run `performance_check`
median rose from **15.29 to 29.62 tokens/s (1.94x)**. The 40 tokens/s target
is **not met**, and this is **not** a matched Strata comparison (see Limits).

## Hardware and build

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3050 8 GB, compute capability 8.6, driver 616.92 |
| PCIe | Gen3 maximum, x8 negotiated width |
| CPU / RAM | AMD Ryzen 7 5700X (8 cores / 16 threads), 64 GiB DDR4-2667 dual channel |
| Build | `python -m tools.build_windows --cuda-arch 86`, Release, CUDA 13.3 |
| Engine | SHA-256 recorded in every JSON (`38511851af579556...`) |
| Model | Qwen3.6-35B-A3B-UD-Q4_K_M, checksum verified |

## Results

### Matched comparison with llama.cpp

Both engines use the same prompts and token IDs, 256 greedy tokens, three
repeats per prompt, resident processes, FP16 device KV, 32K context, no
speculative decoding. llama.cpp is the official Windows CUDA 12.4 release
b11474 (`b9acf138...`, both ZIPs checked against the published SHA-256
digests), with automatic placement, eight threads and `--load-mode none`, as in
the earlier 4060 comparison. Lamina runs in fast mode with its new default
expert policy. Note that `tools.compare_lamina` always sets
`LAMINA_HOST_REGISTER=1`.

```powershell
python -m tools.compare_lamina --report-dir <dir>
python -m tools.compare_llama --load-mode none --revision b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea --threads 8 --report-dir <dir>
```

| Prompt (median of 3) | llama.cpp | Lamina default | Lamina, `stream` forced |
| --- | ---: | ---: | ---: |
| Prose | 28.54 | **31.21** | 24.18 |
| Code | 28.56 | **29.76** | 19.20 |
| Math | 28.65 | **29.50** | 18.51 |
| **Nine-run median** | **28.56** | **29.76** | **19.20** |

Steady first-token time is 2.1-2.2 s for Lamina against about 1.4 s for
llama.cpp, so prefill remains slower. Sampled total-GPU peaks, desktop
included, are about 6.5 GiB for Lamina and 6.8 GiB for llama.cpp. All three
repeats of each prompt produce identical Lamina token streams.

### Like-for-like with the session baseline

The same command as the first measurement of the session (registered RAM, fast
mode, default FP32 KV and context):

```powershell
$env:LAMINA_HOST_REGISTER='1'
python -m tools.performance_check --report-dir <dir>
```

| Binary | Prose | Code | Math | Nine-run median |
| --- | ---: | ---: | ---: | ---: |
| Session start (commit `1da685f`) | 18.64 | 15.29 | 15.19 | 15.29 |
| After next-layer prefetch (`ec37e16`) | 20.37 | 17.34 | 16.89 | 17.34 |
| This change | 30.99 | 29.62 | 28.97 | **29.62** |

The first two rows are from `../2026-10-07-rtx3050-prefetch/`.

### 128K retrieval

`python -m tools.performance_check --only-long --contexts 131072` with 130,928
prompt tokens. Both runs find the needle with EOS-correct stopping.

| Policy | First token | Later tokens/s | Peak total GPU |
| --- | ---: | ---: | ---: |
| `stream` (forced) | 516.1 s | 6.02 | 6807 MiB |
| New fast default | 517.9 s | **9.15** | 7012 MiB |

The long layer-major prefill does not use the decode expert path, so its time
is unchanged. The answer throughput covers only eight tokens; it is not a
sustained long-context decode benchmark.

## Diagnosis

There is no Nsight on this machine, so a stage timeline was added
(`LAMINA_TIMELINE=1`). It records stream-ordered CUDA events at every stage
boundary of every layer, which partition the stream's wall time, including
idle gaps, into stages, and prints per-token averages. Its markers cost about
2 ms per token, so it is never used for headline numbers.

The first streaming profile (59.5 ms per token on a code prompt) showed four
costs: expert kernels 17.7 ms, dense mixer 13.6 ms, copy waits 12.0 ms and a
9.4 ms router stage. The host round trip itself cost only 0.7 ms. Following
these numbers led to the changes below.

The decisive measurement came from the CPU path. Expert reads from RAM are
memory-bandwidth bound (about 24 GB/s achieved; two worker threads already
come close). Under the old CPU-miss policy 158-224 of the 320 routed experts
per token ran on the CPU, because the policy never admitted anything into VRAM
after the first eviction, so the cache stayed frozen with prefill experts.

## Changes

Per-change numbers below are development measurements from intermediate
binaries. Only the tables above come from the final binary.

1. **Parallel router top-k.** `router_topk_kernel` ran on one GPU thread
   (`<<<1, 1>>>`), scanning 256 logits eight times through local memory. It
   is now a block-wide argmax with the same tie-breaking, and the softmax over
   the chosen logits is unchanged, so ids and weights are bit-identical. The
   router stage fell from 9.4 to 4.2 ms per token in streaming mode.
2. **Row-split CPU expert pool.** Each CPU expert used to be one job on one
   thread. Rows are now split across all workers, and the calling thread
   helps. Every output row is computed exactly as before, so results are
   bit-identical. The batch starts before any GPU setup. The router input
   reaches the host through mapped memory written before the doorbell, and
   CPU results return through mapped memory read by a kernel. Neither
   transfer can queue behind copy-engine DMA.
3. **CPU worker default.** Workers default to a quarter of the hardware
   threads, at most eight (4 here). Additional spinning workers slow the host
   thread that drives the GPU: 4 threads gave 27.8 tok/s and 15 threads 23.3.
4. **Request-local admission** (registered RAM only). After a layer's
   combine is queued, up to `LAMINA_ADMIT_PER_LAYER` (default 1) of its CPU
   experts are copied into VRAM in the background. Admitted experts form a
   separate pool (`LAMINA_ADMIT_MB`, default 2048) with its own LRU, are
   promoted only at token boundaries after their copy completes, and are
   dropped at RESET. Placement, and therefore fast-mode output, then does not
   depend on copy timing or on earlier requests. The GPU share of routed
   experts rose from 51% to 85%. Caps of 2 and 3 saturated PCIe and lost
   speed.
5. **Warp-per-row MMVQ for short rows.** The quantized table kernel used a
   128-thread block per output row. For 512-wide Q4_K/Q5_K rows (the expert
   down projections) three warps had no work. Such rows now use one warp each,
   four rows per block, with bitwise-equal results: about +4%.
6. **Expert graph replay with CPU slots**, and removal of two unused memsets
   per CPU expert. This was neutral on its own, because the host setup it
   shortened was already hidden behind the CPU batch.
7. **Default policy.** Fast mode now defaults to CPU experts for cache misses.
   FP32 keeps GPU streaming, the checked reference path. `lamina.py` gains
   `--expert-policy auto` (the default) and only exports `LAMINA_CPU_THREADS`
   when `--cpu-threads` is given. `LAMINA_EXPERT_POLICY` still overrides.

### Registered RAM is no longer needed in fast mode

| Configuration (single repeat) | Prose | Code | Math |
| --- | ---: | ---: | ---: |
| Registered, admission off | 27.41 | 27.70 | 27.37 |
| Registered, admission on (default) | 31.06 | 29.82 | 29.41 |
| Unregistered (admission unavailable) | 30.95 | 29.85 | 29.56 |

Registration costs several seconds at startup and pins about 21 GiB, and
admission only recovers what registration loses. Fast mode is therefore
recommended without `LAMINA_HOST_REGISTER`.

## Bug fixed: prefill attention race

`lamina-prefill-check` failed its 40-layer cases on this GPU with run-to-run
varying errors (0.0073, 0.0636, 0.0077) against the 1e-5 gate. Disabling the
cuBLAS projections did not help; disabling matrix attention did. The cause is
a missing barrier in `attn_matrix_softmax_kernel`: every thread reads the row
maximum from `reduce[0]`, and lane 0 then overwrites `reduce[0]` with its
partial denominator. A late warp could read the denominator as the maximum.
With one `__syncthreads()` added, three consecutive runs pass with a largest
difference of `8.85501504e-6`, the documented value. The race depends on warp
scheduling, which explains why the 4060 did not show it.

## Validation of the final binary

| Check | Result |
| --- | --- |
| 21 Python tests, sampling check | pass |
| CUDA elementwise, attention, projection, KV precision | pass |
| `lamina-prefill-check` | pass, largest `8.85501504e-6` |
| Independent 40-layer reference, FP32 default | `3.59125275e-6`, next token 369 |
| Reference, FP32 CPU-miss, forced 2200 MiB cache | `2.16125275e-6`, next token 369 |
| `tools.runtime_check --compute-mode fast` | all 12 checks pass, including seeded repeat and reset |
| `tools.quality_check` (fast default) | mean KL 0.00253 nats, perplexity ratio 0.998, argmax agreement 98.1% |
| Token streams across matched repeats | identical for every prompt |

The quality gates are mean KL <= 0.1 nats and perplexity ratio <= 1.05 on the
existing 1025-target fixture. This is a small regression fixture, not a broad
quality evaluation.

## Limits

- Everything here was measured on one RTX 3050 machine. The RTX 4060 machine
  of the earlier records is unmeasured with these changes.
- Strata's published decode numbers (65-95 tokens/s) come from an RTX 5070
  12 GB on PCIe 5.0 x16 with 2-3-bit quantizations of a different model and
  MTP speculative decoding. They are not comparable, and no Strata parity is
  claimed.
- The 40 tokens/s target is not met. At about 34 ms per token the remaining
  costs are the dense mixer (about 13 ms), the CPU expert batch, which is RAM
  bandwidth bound, and GPU expert kernels that reach only a fraction of the
  card's bandwidth.
- Prefill is slower than llama.cpp: 2.1 s against 1.4 s steady first-token
  time for short prompts.
- Fast-mode output depends on which experts are resident, because CPU and GPU
  arithmetic differ in their last bits. It is reproducible for a given request
  sequence from the same starting cache, not bitwise equal to the streaming
  policy.
