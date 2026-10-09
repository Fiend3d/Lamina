# Ornith benchmark on RTX 3050 8 GB

This record measures Ornith-1.5-35B-A3B on the RTX 3050 / Ryzen 7 5700X machine
and lists the correctness checks and tuning experiments that followed. It is
written for developers who want to know what this machine reaches and which
settings have already been tried on it. No source code was changed.

Hardware: RTX 3050 (8192 MiB, PCIe Gen3 running at x8), NVIDIA driver 616.92,
Ryzen 7 5700X (8 cores/16 threads), 64 GiB DDR4, Windows 11 Enterprise LTSC
2024. CUDA toolkit 13.3.33, build `--cuda-arch 86`. Source commit
`042edd7ac65d3a689f37c71b33732e2e9b8eafb8`; the engine SHA-256 begins with
`609225878bd6506b` and is recorded in every run JSON.

Model: Ornith-1.5-35B-A3B Q4_K_M, prepared by `tools.lamina_ornith setup`
(60 SSM tensors re-encoded to F32, in-file MTP head packed with `--norms raw`).
The setup log does not report a checksum line, so this record does not claim a
verified download hash.

## Results

The method matches [the RTX 4060 Ornith record](../2026-10-09-ornith-rtx4060/README.md):
one resident engine per mode, three serial repetitions of each prose/code/math
prompt from `tools.performance_check.PROMPTS`, 256 greedy tokens per request,
RESET before every request, fast compute, FP16 device KV, 32768 context and
registered host model pages. No profiling was enabled.

| Prompt | MTP off, tokens/s | MTP on, tokens/s | MTP draft acceptance |
| --- | ---: | ---: | ---: |
| Prose | 45.38 | 42.44 | 49.1% |
| Code | 42.03 | 46.15 | 78.9% |
| Math | 41.21 | 48.58 | 92.4% |
| **Median of all nine runs** | **42.03** | **46.15** | 73-75% cumulative |

Table entries are medians of three runs. Later throughput excludes the first
generated token. MTP helps code (+10%) and math (+18%) but slows prose by about
6%, because the head drafts prose badly and the acceptance guard still spends
pair passes probing it.

The warm first token takes 0.54 s in both modes. The first request of each fresh
engine takes 4.83 s (MTP off) and 4.66 s (MTP on) because it initializes CUDA and
the model; startup before it takes 0.25 s. The file cache was warm.

Peak sampled **total** GPU memory was 6499.9 MiB without MTP and 6567.9 MiB with
MTP, against 1347 MiB idle (desktop). These are NVML totals, not process-only VRAM.

All three repetitions are token-identical within each mode. MTP-on and MTP-off
outputs differ, first at generated token 19 (prose), 41 (code) and 65 (math).
The cause is the known CPU/GPU expert split described under Correctness, not
an error in the speculative path. No run reached EOS within 256 tokens.

This machine was faster than the RTX 4060 8 GB / Ryzen 7 1700X machine
(36.35 / 40.79 tokens/s). The Ryzen 7 5700X CPU expert path is the likely
reason, but that comparison was not isolated.

## Correctness checks

The checks below ran on this machine with the same engine build. Everything
passed except two Ornith-specific results, which are explained after the table.

| Check | Qwen3.6 | Ornith |
| --- | --- | --- |
| `python -m unittest discover -s tests/lamina` | pass | - |
| `lamina-sampling-check`, `lamina-cuda-elementwise-check`, `lamina-cuda-attention-check` | pass | - |
| `lamina-cuda-projection-check` | pass | pass |
| `lamina-cuda-ncols-check` | - | pass |
| `lamina-kv-precision-check` | pass | pass |
| `lamina-prefill-check` | pass (largest `8.855e-6`) | fails one case at `1.359e-5` |
| `lamina-gguf --check` | pass | 30 tensors "out of range" |
| `tools.reference_prefix --layers 40 --cuda --batched` | next 369, logit 12.448531 vs 12.448534 | next 369, logit 10.40105 vs 10.401051 |
| All-GPU MTP identity, 96 tokens per prompt | - | identical for prose, code and math |

**`lamina-gguf --check` on Ornith.** The checker requires each tensor's data to
fill the gap to the next tensor. Setup appends the F32 `ssm_alpha`/`ssm_beta`
copies at the end of the file and leaves their old Q4_K slots unused. Exactly
the 30 `blk.N.ssm_a` tensors that precede those vacated slots fail the gap test;
no tensors overlap. This is a checker assumption, not file damage.

**`lamina-prefill-check` on Ornith.** Only the 40-layer synthetic image-embedding
case exceeds the absolute `1e-5` limit. A temporary diagnostic build (reverted)
showed values up to 16.78 and a relative L2 error of `2.05e-6`, about seven
FP32 ulps at that magnitude. Every other case passed, including device KV and
layer-major prefill. This looks like FP32 accumulation-order noise, not a defect;
the tolerance was left unchanged.

**All-GPU MTP identity.** With `LAMINA_EXPERT_POLICY=stream`,
`LAMINA_EXPERT_PIPELINE=0` and `LAMINA_PREFETCH=0`, speculative and single-token
greedy output are identical for 96 tokens of each prompt. Ornith's pair path,
including its split dispatch for mixed Q4_K/Q6_K encodings, therefore reproduces
single-token decode. In the default CPU-miss mode, the outputs diverge because
some experts run on the CPU in one mode and on the GPU in the other, as recorded
in [the speculation record](../2026-10-08-rtx3050-mtp-speculation/README.md).

## Experiments that were not kept

These experiments are recorded so they are not repeated on this machine. Two
runs of the unchanged default ranged 41.50-42.24 tokens/s, so differences below
about 0.7 tokens/s are noise. The raw run directories stayed in `../Lamina-data`
and are not part of this record.

### Engine settings, MTP off

Each line is one engine process with one repetition of each prompt.

| Setting | Median tokens/s | Note |
| --- | ---: | --- |
| Default (two runs) | 41.50 / 42.05 | |
| `LAMINA_CPU_THREADS` 3 / 5 / 6 | 41.42 / 42.13 / 41.54 | noise |
| `LAMINA_CUDA_CACHE_MB` 5500 / 6000 | 42.03 / 42.06 | No effect; decode grows only through the admission pool. |
| `LAMINA_MOE_GRAPH_CACHE=1024` | 42.04 | noise |
| `LAMINA_ADMIT_PER_LAYER` 0 / 2 | 36.75 / 37.99 | Admission matters; two per layer costs PCIe time. |
| `LAMINA_BASE_KEEP` 6 / 16 / 20 | 40.72 / 41.82 / 42.67 | 20 adds 0.1 s to the first token and 640 MiB of VRAM. |
| `LAMINA_ADMIT_MB=2560` (cache 5120) | 42.49 | Peak 7025 MiB. |
| `LAMINA_ADMIT_MB=3072` (cache 6144) | 41.08 | First token 0.82 s, peak 7451 MiB. |
| `LAMINA_ADMIT_MB=3584` (cache 6400) | 35.00 | VRAM pressure. |
| Keep 20, admission 3072, cache 6144 | 32.42 | VRAM pressure. |

These results agree with [the decode-tuning record](../2026-10-08-rtx3050-decode-tuning/README.md),
which tried the same settings with Qwen3.6 on this machine.

### Faster MTP acceptance guard

`Inference::generate_greedy` backs off from speculation when an exponential
average of acceptance (weight 0.97) falls below 0.55. With Ornith prose at 49%
acceptance, it spends about 60 losing pair passes before the first backoff.

A variant with weight 0.92 and doubling backoffs (32 up to 256 single steps,
reset after a probe survives 32 pairs) raised Ornith prose with MTP from 42.2 to
43.1 tokens/s in interleaved runs, with code and math token-identical. It also
caused two needless backoffs in Qwen3.6 prose (44.49 -> 44.22). A weight of 0.95
removed the Qwen cost but also the Ornith gain (prose 42.04 -> 42.04). The change
was reverted.

### Q8_0 MTP head

The packed head is mostly Q4_K (`eh_proj`, attention and gate/up experts) with
three Q6_K tensors; the Qwen3.6 head is Q8_0. The official
`ornith-ai/Ornith-1.5-35B-A3B-GGUF` repository (revision
`12393612fd4f730ff5aadc23e9b8f9648aa49ceb`) publishes a Q8_0 GGUF. Its 20
`blk.40.*` tensors were fetched with HTTP range requests and packed like
`tools.lamina_inline_mtp` does (857 MiB). The F32 norms were bit-identical to
the current head, and the matrices correlated at 0.997.

| Head | Prose | Code | Math | Median | Acceptance (prose / code / math) |
| --- | ---: | ---: | ---: | ---: | --- |
| Q4 (current), run 1 | 41.43 | 44.89 | 47.26 | 44.89 | 49.1% / 78.9% / 92.4% |
| Q8_0, run 1 | 40.10 | 43.31 | 43.89 | 43.07 | 52.0% / 78.3% / 92.4% |
| Q4 (current), run 2 | 40.79 | 43.72 | 45.13 | 43.56 | same |
| Q8_0, run 2 | 38.57 | 41.92 | 45.17 | 41.92 | same |

Higher precision raised prose acceptance by only three points, and the larger
head made drafting slower. Head precision does not explain the weak prose
drafts. The head was not kept. The machine ran slower later in the session, so
compare only interleaved rows.

## What remains

Decode on this machine is limited by the CPU expert path, which reads cache-missed
experts from DDR4, and by about 1.7 GiB of free VRAM at peak. A lower-bit Ornith
build (for example bartowski's IQ3_XXS at 14.3 GiB) would reduce both RAM traffic
and cache misses, but Lamina's projection adapter does not yet dispatch IQ
formats on the GPU, and the result would need the quality gate. Deeper
speculation is not promising here: a second verified token already costs about
11.5 ms against 23.8 ms for a single step, mostly in additional CPU experts.

## Exact commands

Run from the repository root with the sibling virtual environment. GPU
workloads ran serially.

```powershell
../Lamina-data/venv/Scripts/python.exe -m tools.lamina_ornith setup
../Lamina-data/venv/Scripts/python.exe -m tools.build_windows --cuda-arch 86 --jobs 6
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --model ../Lamina-data/models/Ornith-1.5-35B-Q4_K_M.gguf --tokenizer ../Lamina-data/ornith/tokenizer.json --context 32768 --repeats 3 --tokens 256 --host-register 1 --report-dir bench/results/2026-10-09-ornith-rtx3050/mtp-off
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --model ../Lamina-data/models/Ornith-1.5-35B-Q4_K_M.gguf --tokenizer ../Lamina-data/ornith/tokenizer.json --context 32768 --repeats 3 --tokens 256 --host-register 1 --mtp ../Lamina-data/mtp/ornith-mtp.gguf --report-dir bench/results/2026-10-09-ornith-rtx3050/mtp-on
```

The reference check used `OPENBLAS_NUM_THREADS=1` and, for each model:

```powershell
../Lamina-data/venv/Scripts/python.exe -m tools.reference_prefix --model <model.gguf> --layers 40 --engine build-cuda/lamina-infer.exe --cuda --batched --tokens 42 43 44 45 --max-context 131072 --kv-cache host
```

Raw reports, per-mode summaries and native stderr are in `mtp-off/` and
`mtp-on/`.
