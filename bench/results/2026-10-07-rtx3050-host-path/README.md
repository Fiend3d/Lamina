# Decode host path and expert kernels: RTX 3050 8 GB

This directory records a batch of decode optimizations that follow
`../2026-10-07-rtx3050-gdn-step/`. It is written for developers; hardware,
model and the comparison method are the same as in
`../2026-10-07-rtx3050-cpu-experts/`.

## Result

The matched nine-run median (`python -m tools.compare_lamina`, which sets
`LAMINA_HOST_REGISTER=1`; fast mode, FP16 device KV, 32K context) rose from
**31.24 to 33.98 tokens/s**. CUDA llama.cpp b11474 measured 28.56 on the same
machine, so Lamina is now 19% ahead. The 40 tokens/s target is not met.

| Prompt (median of 3) | Before | After | llama.cpp |
| --- | ---: | ---: | ---: |
| Prose | 32.71 | 37.07 | 28.54 |
| Code | 31.24 | 33.98 | 28.56 |
| Math | 30.84 | 33.39 | 28.65 |

Every change in this batch is bitwise neutral. All nine token streams are
identical to those of the previous binary, repeats are identical, the FP32
reference reproduces `3.59125275e-6`, and the fast quality gate reproduces the
previous values exactly (mean KL 0.00252725509, perplexity ratio 0.997558516).
The engine SHA-256 begins with `77c74a8fbebaa849`. Steady first-token time is
2.1-2.2 s.

## Changes kept

Single-repeat matched measurements on intermediate binaries, in order:

| Change | Prose / code / math |
| --- | --- |
| Starting point (previous binary) | 32.44 / 31.22 / 30.81 |
| Warp-per-row MMVQ for all single-pass table rows | 33.91 / 32.10 / 31.45 |
| Fused router tail | 34.41 / 32.31 / 31.67 |
| Shared expert queued behind the doorbell | 34.85 / 32.79 / 32.26 |
| Hot-path settings read once | 36.29 / 33.72 / 33.05 |
| Expert tables and scales as kernel arguments | 36.68 / 33.88 / 33.34 |
| Admission without per-copy events | about the same, simpler |

1. **Warp-per-row MMVQ.** Rows that the four-warp table kernel covers in one
   pass (`n_in / DIV * T <= 128`: Q4_K expert gate/up and all 512-wide down
   rows) run one warp per row. Each lane emulates the four warps in turn and
   adds their partials in the original order, so results are bitwise equal.
   Extending it to wider dense rows was slower (`LAMINA_WARP_ROWS` selects the
   threshold for experiments).
2. **Fused router tail.** The shared-gate dot, top-k, publication of the CPU
   expert input and the doorbell run in one launch (`router_finish`) instead
   of four; each part keeps its arithmetic.
3. **Early shared expert.** The shared expert does not depend on routing. In
   fast CPU-miss decode it is queued directly behind the doorbell, so the GPU
   computes it while the host reads the routing; the expert graph then skips
   it. The GPU-idle `host_gap` stage fell from about 1.4 to 0.15 ms per token.
   `LAMINA_SHARED_EARLY=0` disables it.
4. **Settings read once.** Several `getenv` calls ran per layer, two inside
   every `projection_weight`; on Windows each takes a CRT lock and scans the
   environment. They are now read once.
5. **Publish kernel.** Expert pointer tables and combine scales are passed as
   kernel arguments to one small launch instead of two `cudaMemcpyAsync`
   calls, which cost more host time under WDDM.
6. **Admission without events.** Promotion at a token boundary synchronizes
   the copy stream once instead of waiting on one event per admitted tensor.

## Measured and reverted

- Running the shared expert concurrently with the routed experts on a second
  stream was about 1% slower: the routed launches already fill every SM.

## Diagnosis tools

`LAMINA_TIMELINE=1` now also reports attention sub-stages (`attn_project`,
`attn_rope_kv`, `attn_core`, `attn_out`), per-kernel expert stages on the
ungraphed path (`LAMINA_MOE_GRAPHS=0`), and host-side times of the CPU-miss
path (`host_cpu_start`, `host_setup` with `host_scales`, `host_slots` and
`host_tables`, `host_cpu_wait`, `host_admit`).

## Validation

Python tests, sampling, CUDA elementwise/attention/projection/KV-precision
checks and `lamina-prefill-check` (largest `8.85501504e-6`) pass. The FP32
reference gives `3.59125275e-6` (default) and `2.16125275e-6` (CPU-miss, forced
2200 MiB cache), next token 369. `tools.runtime_check --compute-mode fast`
passes. Reports are in this directory.
