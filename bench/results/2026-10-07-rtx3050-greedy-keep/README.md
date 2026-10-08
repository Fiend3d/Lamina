# GPU greedy selection and balanced expert cache: RTX 3050 8 GB

This directory records the decode changes that follow
`../2026-10-07-rtx3050-host-path/`. It is written for developers; hardware,
model and the comparison method are the same as in
`../2026-10-07-rtx3050-cpu-experts/`.

## Result

The matched nine-run median (`python -m tools.compare_lamina`, which sets
`LAMINA_HOST_REGISTER=1`; fast mode, FP16 device KV, 32K context) rose from
**33.98 to 35.84 tokens/s**. CUDA llama.cpp b11474 measured 28.56 on the same
machine, so Lamina is now 25% ahead. The 40 tokens/s target is not met.

| Prompt (median of 3) | Before | After | llama.cpp |
| --- | ---: | ---: | ---: |
| Prose | 37.07 | 38.83 | 28.54 |
| Code | 33.98 | 35.84 | 28.56 |
| Math | 33.39 | 35.18 | 28.65 |

All three repeats of every prompt, including the cold first request of the
process, produce identical token streams. Steady first-token time is
2.1-2.2 s. The engine SHA-256 begins with `e05e8b354d48afc8`.

## Changes

1. **GPU greedy selection.** With temperature 0 the engine selects the token
   with an argmax kernel on the device (`Inference::greedy`,
   `CudaProjection::matvec_argmax`) instead of downloading 248,320 logits
   (1 MB, pageable) and scanning them on the CPU. The kernel keeps the first
   maximum, as `std::max_element` does; a non-finite logit falls back to the
   full path so the sampler raises the same error. Bitwise neutral; about +4%.
2. **Admission without per-tensor fences.** Blocks evicted from the admission
   pool are reused only by later admissions. One epoch event on the compute
   stream fences their readers, and the copy stream waits on it once, instead
   of a retirement event and a wait per tensor. Bitwise neutral; about +1%.
3. **Balanced base cache.** During layer-major prefill in fast CPU-miss mode,
   each layer's most-routed prompt experts (`LAMINA_BASE_KEEP`, default 12)
   are kept out of the base LRU. When prefill finishes, every other routed
   expert is dropped from the base, so the base depends on the current prompt
   alone. Before this change the base held whatever the last few layers of
   prefill had left behind:

   | CPU experts per layer (of 8) | Before | After |
   | --- | ---: | ---: |
   | Layers 0-9 | 4.49 | about 4.0 |
   | Layers 10-29 | 3.24 | about 2.6 |
   | Layers 30-39 | 2.25 | about 2.7 |

   This change moves experts between GPU and CPU, so fast-mode output differs
   from the previous binary in the last bits (about +2-5%). Two determinism
   problems found while measuring it are fixed: leftover base entries that
   depended on earlier requests, and kept experts that could already have been
   evicted under cache pressure (they are now uploaded).

## Measured and not kept

- Admitting more than one CPU expert per layer, or only experts already used
  two or three times on the CPU: slower or no better. At two or more, the
  copies overlap the next layer's zero-copy reads of CPU results on
  PCIe Gen3 x8.
- A larger expert cache (5500 or 6000 MiB), with or without a larger admission
  pool: no better.
- Pinning CPU workers to separate physical cores: about 1%, within noise.

## Diagnosis

`LAMINA_TIMELINE=1` additionally prints `cpu_experts_by_layer` and the CPU
pool's batch statistics (`cpu_batches`: batches per token, batch duration,
first-claim latency, experts per batch). A CPU batch of about 3.7 experts
(7 MB) takes about 240 us, about 29 GB/s, which is the practical bandwidth of
this DDR4-2667 system; seven workers are no faster than four. The client's
per-token pipe round trip (`client_wait`) measured about 0.1 ms.

## Validation

Python tests, CUDA projection and KV-precision checks and
`lamina-prefill-check` (largest `8.85501504e-6`) pass. The independent FP32
reference reproduces `3.59125275e-6`, next token 369.
`tools.runtime_check --compute-mode fast` passes. The fast quality gate gives
mean KL 0.00241239854 nats, perplexity ratio 1.00131595 and argmax agreement
98.3% (gates: 0.1 and 1.05). 128K retrieval (130,928 prompt tokens) still
finds the needle with EOS-correct stopping: first token 513.8 s (previously
517.9), later answer throughput 9.49 tokens/s (previously 9.15), peak total
GPU 6820 MiB (`retrieval-131072.json`).
