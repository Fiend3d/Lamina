# Retuning after hybrid prefill, and a failed miss prefetch

This record covers the settings re-measured after hybrid prefill changed how
much data moves through the GPU cache, one retained change (more CPU workers
for prefill only), and one rejected experiment (prefetching predicted expert
misses during decode). It is written for developers choosing the next step.

## Result

Prefill batches now use 12 CPU expert workers instead of 4, while decode keeps
4. The warm first token of the short benchmark prompts fell from **0.87 s to
0.57 s** in interleaved runs, with unchanged decode speed and bit-identical
output. Everything else that was tried measured no better or worse.

The machine ran slower overall than in the hybrid-prefill record of the same
day: the old configuration measured 44.0 instead of 45.9 tokens/s
(speculative) and 0.87 instead of 0.69 s first token. Compare only
interleaved runs within this record.

Nine-run medians with the retained change (engine run directories `single`
and `speculative`):

| Mode | Prose | Code | Math | Nine-run median | Warm first token |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single-token | 39.51 | 36.69 | 35.84 | 36.69 | 0.572 s |
| Speculative (`--mtp`) | 43.20 | 44.17 | 44.14 | 44.14 | 0.579 s |

Generated tokens equal those of the hybrid-prefill record in all 18 runs. Each
output row is computed by one worker, so the worker count cannot change
results. Hardware: RTX 3050 8 GB, Ryzen 7 5700X (8 cores, 16 threads), 64 GiB
DDR4, Windows 11.

## Interleaved measurements

Each line is one engine process, all three prompts run twice, warm second
runs reported (`compare_lamina --repeats 2`).

| Setting | Decode median | First token |
| --- | ---: | ---: |
| Speculative, 4 prefill workers (old) | 44.11, 43.99 | 0.865, 0.867 s |
| Speculative, 12 prefill workers (new) | 44.13, 44.11 | 0.567, 0.567 s |
| Single-token, 4 prefill workers (old) | 36.56, 36.10 | 0.877, 0.869 s |
| Single-token, 12 prefill workers (new) | 36.35, 36.59 | 0.567, 0.565 s |
| Single-token, 8 or 16 prefill workers | 36.6-36.8, 36.5-36.6 | 0.64, 0.63-0.65 s |

Rejected settings (single-token, against a 36.5-37.3 default in the same runs):

| Setting | Decode median | Note |
| --- | ---: | --- |
| `LAMINA_CUDA_CACHE_MB` 5400 / 5800 / 6200 alone | 37.30-37.41 | No effect: after prefill the base holds about 2.95 GB (dense weights and kept experts); decode grows only through the 2048 MB admission pool. |
| Cache 5600 + `LAMINA_ADMIT_MB` 2600 | 38.62 | +3.5% single-token, but speculative fell 44.17 -> 42.21 and the first token rose to 1.07 s; peak 7258 MiB. |
| Cache 5600-6200 with admission 2800-3200 or `LAMINA_BASE_KEEP` 20-24 | crash | VRAM headroom exceeded with the 32K FP16 device KV cache. |
| `LAMINA_CPU_THREADS` 6 / 8 (decode workers) | 37.20 / 36.66 | Decode is RAM-bandwidth bound; extra spinning workers cost. |
| `LAMINA_ADMIT_PER_LAYER=2` | 34.36 | PCIe and copy fences slow decode. |
| CPU work-item rows: gate 16/64, down 64/256 | 36.46-36.69 | Within noise of 32/128. |

## Rejected: prefetching predicted misses

The experiment applied the next layer's router to the current layer's input
(the streaming mode's prediction), uploaded the top predicted experts that
were not resident while the current layer ran, and promoted them at the next
layer behind one stream fence. Decode fell to 32.3-33.6 tokens/s (one or two
experts per layer, with or without the regular admission). Only 128 of 6,416
prefetched experts (2%) were routed by the next layer: the prediction is good
for frequently used experts, which are already resident, and almost always
wrong for exactly the experts that miss. The code was removed.

## What remains

Decode is bound by reading cache-missed experts from DDR4 (about 29 GB/s,
insensitive to work-item size). Remaining options are deeper speculation
(two drafts), which helps more on GPUs with larger caches, or opt-in lossy
reductions of CPU expert traffic, which need the quality gate.
