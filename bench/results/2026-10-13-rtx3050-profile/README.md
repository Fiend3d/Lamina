# Decode profile on RTX 3050 8 GB: where CPU and GPU wait

This record profiles greedy decode of the pinned Qwen3.6 on the RTX 3050 /
Ryzen 7 5700X machine to find idle CPU or GPU time that could be removed. It is
written for developers choosing the next optimization. The only code change is
a diagnostics fix; no decode path changed, so no speed claim is made.

Hardware: NVIDIA GeForce RTX 3050 (8192 MiB, PCIe Gen3 running at x8), driver
616.92, AMD Ryzen 7 5700X (8 cores/16 threads, 32 MiB L3), 64 GiB DDR4-2667 in
two channels, Windows 11. Engine built from `d3b40c4` with
`tools.build_windows --cuda-arch 86`. Workload: `tools.compare_lamina` with the
three standard prompts, 256 greedy tokens, MTP off, 32768 context, FP16 device
KV; about 39 tokens/s (25.5 ms per token) without profiling.

## Result

Lamina keeps each side near its hardware limit during its own phase. The idle
time that remains comes from the layer dependency chain, not from scheduling:

- During each layer's MoE step, the CPU computes the experts that are not in
  VRAM while the GPU computes the resident ones. The GPU finishes first and
  waits about 3.4 ms per token (`miss_wait`), about 13% of the token.
- The CPU expert batch reads its weights at about 28 GB/s. Plain reads with the
  same access pattern reach 30-36 GB/s on this machine, so at most 15-35 us of
  a 210 us batch, about 1 ms per token, is CPU overhead rather than RAM time.
- Outside the MoE steps the CPU has no work: the next layer cannot start before
  the MoE output exists, and the next layer's routing cannot be predicted well
  enough to start its CPU experts early (see
  [the decode-tuning record](../2026-10-08-rtx3050-decode-tuning/README.md)).
- Large GPU projections run at 78-89% of the card's 224 GB/s.
- The native protocol and the Python server cost less than 1%.

## Measurements

### Stage timeline

`LAMINA_TIMELINE=1` (which runs DeltaNet layers eagerly and therefore reports
31.6 ms per token instead of 25.5) gave these GPU-stream times per token:

| Stage | ms | Notes |
| --- | ---: | --- |
| DeltaNet, 30 layers | 9.7 | projections 4.6, output 2.6, small ops 1.6, recurrence 1.0 |
| GPU experts (`resident`) | 4.0-4.2 | |
| Router and early shared expert | 3.6 | the shared expert runs behind the doorbell |
| Waiting for CPU experts (`miss_wait`) | 3.4 | GPU idle |
| Attention, 10 layers | 3.0 | |
| LM head kernel | 2.0 | near peak bandwidth |
| After the head (`head_copy_and_host`) | 2.4 | mostly timeline overhead; see protocol below |
| Combine, setup, other | 2.4 | |

The CPU ran 35.4 decode batches per token with 3.10 experts (1.90 MB each) and
210 us per batch. The host starts each CPU batch before it builds GPU tables or
launches the expert graph, so host setup is off the critical path.

### Utilization

Sampling every 100 ms during a normal run gave a median NVML GPU utilization of
91% (mean 77%) and 4.3 busy cores for the engine. Most of those cores are the
four expert workers spinning between batches, which keeps their wake-up time at
about 1 us (`first_claim_us`).

### CPU expert bandwidth

| Measurement | Batch time | Bandwidth |
| --- | ---: | ---: |
| Lamina decode batch, 4 workers (5.9 MB) | 210 us | 28 GB/s |
| Same with 1 / 2 / 3 workers (6.35 MB) | 330 / 290 / 254 us | 19 / 22 / 25 GB/s |
| Same with 6 / 8 workers | 211 / 223 us | 28 / 27 GB/s |
| Plain reads, 9 scattered 650 KB matrices, 36 KB chunks, 4 threads | 164-197 us | 30-37 GB/s |
| Plain reads from the mapped GGUF instead of private memory | 174-197 us | 30-34 GB/s |

More than four workers do not shorten the batch, and extra spinning threads slow
the host thread that drives the GPU (`resident` 4.2 -> 5.1 ms per token at eight
workers; 38.9 -> 38.1 tokens/s). Chunk size and the memory-mapped model file do
not explain the remaining gap; it is the cost of the ggml `vec_dot` calls,
especially for the short 512-wide down-projection rows, and batch start/finish.

### GPU kernels

`lamina-cuda-ncols-check` timings of single launches on real weights:

| Projection | Weights | Time | Bandwidth |
| --- | ---: | ---: | ---: |
| DeltaNet `attn_qkv`, Q8_0, 2048 -> 8192 | 17.8 MB | 89.7 us | 199 GB/s |
| DeltaNet `ssm_out`, Q8_0, 4096 -> 2048 | 8.9 MB | 51.2 us | 174 GB/s |
| Shared-expert down, Q8_0, 512 -> 2048 | 1.1 MB | 9.7 us | 115 GB/s |

Bringing `ssm_out` to the `attn_qkv` level would save about 0.3 ms per token.

### Protocol and server

| Path | Without MTP | With MTP |
| --- | ---: | ---: |
| Native protocol, one command per token | 38.89 | |
| Native `GENERATE`, no pipe round trip per token | 39.02 | 46.42 |
| Server `Engine.events`, streaming | 38.58 | 46.72 |

## Diagnostics fix

The timeline's `cpu_batches` line averaged prefill batches together with decode
batches. Prefill batches are much larger, so it reported 251 us and 3.59 experts
per decode batch instead of 210 us and 3.10. `CpuExperts::stats()` now counts
only decode (non-wide) batches. Results and speed are unaffected.

## What could still help

- CPU expert kernels that process several rows per call, to close the last
  15-35 us per batch (at most about 1 ms per token, about 4%).
- Fewer bytes on the CPU path: a lower-bit model or more VRAM. This is the only
  lever for the 3.4 ms per token the GPU spends waiting.
- Reading part of each missed expert over PCIe while the CPU reads the rest
  could add a few GB/s, but PCIe Gen3 x8 also carries expert admissions.

## Exact commands

```powershell
$env:LAMINA_TIMELINE = "1"
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --repeats 2 --tokens 256 --report-dir <out>/timeline
$env:LAMINA_CPU_THREADS = "6"   # also 1, 2, 3, 4 and 8
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --repeats 1 --tokens 256 --report-dir <out>/threads-6
build-cuda/lamina-cuda-ncols-check.exe ../Lamina-data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

The utilization sampler, the protocol/server overhead script and the
bandwidth tests were scratch programs and are not part of the repository; their
method is described above.
