# Next-layer expert prefetch on RTX 3050 8 GB

This directory records the effect of speculative next-layer expert prefetch on
decode throughput. It is written for developers who need the exact commands,
hardware and limits of the measurement.

## Hardware and build

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3050 8 GB, compute capability 8.6, driver 616.92 |
| PCIe | Gen3 maximum, x8 negotiated width (`nvidia-smi` `pcie.link.gen.max` / `pcie.link.width.current`) |
| CPU / RAM | AMD Ryzen 7 5700X (8 cores), 64 GiB |
| Build | `python -m tools.build_windows --cuda-arch 86`, Release, CUDA 13.3 |
| Model | Qwen3.6-35B-A3B-UD-Q4_K_M, checksum verified by `lamina-gguf --check` |

These are **not** the RTX 4060 numbers in the other result directories. Do not
compare absolute values across machines.

## What was measured

The nine-run real-prompt benchmark (three prompts, three runs each, 256 tokens,
fast compute mode, registered model RAM, default FP32 KV, engine-default
context because `--max-context` was not passed, 2048-token prefill chunk):

```powershell
$env:LAMINA_HOST_REGISTER='1'
$env:LAMINA_PREFETCH='0'   # baseline; '1' for the prefetch run
python -m tools.performance_check --report-dir <directory>
```

| Prompt | Baseline tok/s | Prefetch tok/s | Change |
| --- | ---: | ---: | ---: |
| Prose (median of 3) | 18.64 | 20.37 | +9% |
| Code (median of 3) | 15.29 | 17.34 | +13% |
| Math (median of 3) | 15.19 | 16.89 | +11% |
| **Nine-run median** | **15.29** | **17.34** | **+13.5%** |

First-token time (median over nine runs, includes registration and prefill of
the short prompt) is unchanged within noise: 6.93 s versus 6.97 s. Sampled
total-GPU peak (includes the desktop) was 7779 MiB versus 7686 MiB. The 40
tokens/s target is **not met**.

The generated token streams of all nine prefetch runs are identical to the
baseline runs. Prefetch changes only when weights are copied, never the math.

## Mechanism and diagnosis

While layer L is routed, layer L+1's router is applied to layer L's normalized
input. The predicted top-eight experts that are not resident are copied on the
weight-copy stream behind layer L's own misses. In the profile run 79% of the
predicted experts were selected by the real router, and the real-lookup hit
rate rose from 85.0% to 94.1% (`LAMINA_PROFILE=1`, which prints a
`prefetch predicted=... useful=... accuracy=...` line).

Misses fell by about 60% but throughput rose by only 13%. Expert uploads are
therefore no longer the dominant cost. The remaining time is expert kernel
execution (about 18.8 ms/token of event time, roughly 15% of the card's memory
bandwidth) and the per-layer host routing rendezvous. A timeline capture is
needed before choosing the next change.

## Limits

- Measured on one RTX 3050 only. The 4060 is unmeasured with this change.
- It defaults on only when model RAM is registered (`LAMINA_HOST_REGISTER=1`).
  On the worker-staging path it measured 20% slower (16.09 to 12.83 tok/s on the
  120-token teacher-forced run), so there it defaults off.
  `LAMINA_PREFETCH=0` or `1` overrides the default.
- A 5500 MiB expert cache looked better on the 120-token fixture (20.35 tok/s)
  but was worse on real prompts (nine-run median 16.25 against 17.34), so the
  5000 MiB cap is unchanged. A 6000 MiB cache dropped to about 16.7 tok/s.
- Only short-prompt decode with the default context was measured. Long-context
  (32K and 128K) behavior with prefetch is unmeasured.
- The report JSON records only `LAMINA_HOST_REGISTER` in its environment field,
  so the baseline's `LAMINA_PREFETCH=0` setting is documented here, not in the
  JSON.

## Validation

Independent 40-layer reference check with prefetch enabled: maximum difference
`3.59125275e-6`, next token 369, unchanged. `python -m tools.runtime_check
--compute-mode fast` and the 21 Python tests pass.

`lamina-prefill-check` fails its 40-layer case on this GPU
(`prefill max_abs_diff` 0.0073 and 0.0636 on two runs, gate 1e-5) with and
without prefetch. The 4-layer cases pass. This was not investigated further.

The JSON reports in `baseline/` and `prefetch/` carry the engine SHA-256,
command line and environment of each run.
