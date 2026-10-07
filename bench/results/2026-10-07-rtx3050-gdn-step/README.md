# Coalesced DeltaNet recurrent step: RTX 3050 8 GB

This directory records one decode optimization that follows
`../2026-10-07-rtx3050-cpu-experts/`: a coalesced DeltaNet recurrent-step
kernel. It is written for developers; hardware, model and comparison method
are the same as in that record.

## Result

The matched nine-run median (`python -m tools.compare_lamina`, which sets
`LAMINA_HOST_REGISTER=1`; fast mode, FP16 device KV, 32K context) rose from
**29.76 to 31.24 tokens/s**. CUDA llama.cpp b11474 measured 28.56 on the same
machine, so Lamina is now 9.4% ahead.

| Prompt (median of 3) | Before | After | llama.cpp |
| --- | ---: | ---: | ---: |
| Prose | 31.21 | 32.71 | 28.54 |
| Code | 29.76 | 31.24 | 28.56 |
| Math | 29.50 | 30.84 | 28.65 |

All nine token streams are identical to those of the previous binary (in
`../2026-10-07-rtx3050-cpu-experts/lamina-default/`), and repeats are identical.
The engine SHA-256 begins with `cc1414784d0d3f35`. The 40 tokens/s target is
not met.

## Diagnosis

`LAMINA_TIMELINE=1` now also splits DeltaNet layers into `gdn_project`,
`gdn_small`, `gdn_step` and `gdn_out`, and the LM head into `head_kernel` and
`head_copy_and_host`. With the timeline on, DeltaNet layers run eagerly
instead of as captured graphs, because markers cannot be recorded inside a
capture. Per token over 30 DeltaNet layers the split was:

| Sub-stage | Before | After | Bytes per token |
| --- | ---: | ---: | --- |
| qkv + gate projection | 4.35 ms | 4.34 ms | about 800 MB |
| Recurrent step | **3.35 ms** | **0.84 ms** | about 120 MB (2 MiB state read and written per layer) |
| Output norm + ssm_out | 2.09 ms | 2.1-2.5 ms | about 267 MB |
| Small operations | 0.83 ms | 0.84 ms | negligible |

The old step kernel let the 32 lanes of a warp read rows of one state column
16 KiB apart, so every 4-byte load touched its own 32-byte sector (about 36 GB/s
effective). `step_tiled` in `src/kernels/cuda/native_gdn.cu` stages a 128 x 32
tile of one head through padded shared memory with 128-byte row loads and
stores. Each warp then runs the unchanged per-column computation with the same
lane-to-row mapping, sums and warp reductions, which is why the results are
bitwise equal. The prefill (multi-column) step is unchanged.

## Measured but not kept

- Issuing CPU-expert admissions during the next layer's doorbell wait,
  instead of right after the combine, measured slower on code (31.0 -> 29.0
  tokens/s, single runs) and was reverted.
- Raising the CPU worker count does not help: the CPU expert batch saturates
  near 16 GB/s of expert reads (19.8 ms per token at 4 workers, 19.1 ms at 7),
  a memory-bandwidth ceiling for this access pattern.

## Validation

Python tests, CUDA elementwise/attention/projection/KV-precision checks and
`lamina-prefill-check` (largest `8.85501504e-6`) pass. The independent 40-layer
FP32 reference reproduces `3.59125275e-6`, next token 369.

## Remaining per-token costs

The timeline of the current code shows the CPU expert wait as the largest
stage when registered RAM and admission are off (about 13 ms), followed by the
DeltaNet projections (4.3 ms), GPU experts (4.2 ms), attention layers (2.6 ms),
the DeltaNet output path (2.5 ms) and the LM head kernel (2.0 ms, already near
peak bandwidth). Logits transfer, sampling and the per-token pipe round trip
cost about 1.4 ms.
