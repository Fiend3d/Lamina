# Ornith benchmark on RTX 4060 8 GB

9 October 2026. RTX 4060 (8188 MiB), Ryzen 7 1700X (8 cores/16 threads),
64 GiB installed RAM, Windows 11, NVIDIA driver 610.88, CUDA toolkit 13.3.
Source commit: `5cbe58ae5e3a056dc36a4cf646f5626ea45c813e`; the benchmark
helper additionally accepts explicit model/tokenizer paths and records tokenizer
hashes and EOS positions. The native executable hash is in every run JSON.

Model: Ornith-1.5-35B-A3B Q4_K_M. Original download SHA-256 was verified as
`42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`
before `tools.lamina_ornith` re-encoded 60 SSM tensors to F32 and packed the
521 MiB in-file MTP head with `--norms raw`. Assets remain in sibling Lamina-data.

## Results

Each mode uses one resident engine and three serial repetitions of each
prose/code/math prompt in `tools.performance_check.PROMPTS`, generating exactly
256 greedy tokens per request. RESET precedes every request; no prefix reuse.
Fast compute, FP16 device KV, 32768 context, default fast CPU-miss policy,
registered host model pages. No timeline/profiling instrumentation. Ornith's
own tokenizer is used, and the text prompts were checked against its official
non-thinking chat template. No run reached EOS within 256 tokens.

| Prompt | MTP off, tokens/s | MTP on, tokens/s | Off first token, s | On first token, s |
| --- | ---: | ---: | ---: | ---: |
| Prose | 40.10 | 39.85 | 0.735 | 0.716 |
| Code | 36.44 | 40.79 | 0.715 | 0.727 |
| Math | 35.50 | 42.79 | 0.735 | 0.731 |
| Median of all nine runs | 36.35 | 40.79 | 0.735 | 0.728 |

Table entries are medians. Later throughput excludes the first generated token
and includes native protocol overhead. MTP has 12.2% higher median throughput
in this workload, with differing output trajectories (see below).

Peak sampled **total** GPU memory: 5759.56 MiB (5.62 GiB) without MTP,
5855.56 MiB (5.72 GiB) with MTP. NVML was sampled every 50 ms; these figures
include desktop use, rather than process-only VRAM. Idle total was recorded in
each JSON. The registered model pages occupy 20722.6 MiB of host RAM.

The first request in each fresh engine initializes CUDA/model resources:
8.084 s without MTP, 7.777 s with MTP. Initial RESET/startup additionally takes
0.337/0.267 s, making process-start-to-first-token 8.422/8.045 s. File cache
was already warm after download/checksum verification; these are not cold-disk
loading measurements. The warm first-token medians above exclude no runs, but
are dominated by the eight subsequent requests.

## Output and validation limits

All three repetitions are token-identical within each mode for each prompt.
**MTP-on and MTP-off outputs differ**, first at generated token 11 for prose
and code, and token 65 for math (one-based). This is not an output-equivalent
speedup or a demonstrated quality improvement. The cause of this divergence
in the fast/FP16, 8 GB configuration has not been isolated. The MTP log reports
885 accepted drafts across 1152 speculative steps (76.8%, cumulative).
Generated text and token IDs are preserved in the run JSONs for review.

The source build, default CMake configure/lamina-gguf build, all 49 Python
unit checks, CUDA elementwise checks and CUDA attention checks passed. No
model layer math was changed for this benchmark. These checks do not establish
Ornith reference-logit parity or broad model quality. This record measures short
prompts/256-token generation on one PC; it does not establish long-context
throughput or performance on other GPUs.

## Exact commands

Run from the repository root, with the sibling virtual environment:

```powershell
../Lamina-data/venv/Scripts/python.exe -m tools.lamina_ornith setup
../Lamina-data/venv/Scripts/python.exe -m tools.build_windows --cuda-arch 89 --jobs 6
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --model ../Lamina-data/models/Ornith-1.5-35B-Q4_K_M.gguf --tokenizer ../Lamina-data/ornith/tokenizer.json --context 32768 --repeats 3 --tokens 256 --host-register 1 --report-dir bench/results/2026-10-09-ornith-rtx4060/mtp-off
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --model ../Lamina-data/models/Ornith-1.5-35B-Q4_K_M.gguf --tokenizer ../Lamina-data/ornith/tokenizer.json --context 32768 --repeats 3 --tokens 256 --host-register 1 --mtp ../Lamina-data/mtp/ornith-mtp.gguf --report-dir bench/results/2026-10-09-ornith-rtx4060/mtp-on
```

The initial download used parallel validated HTTP ranges after the standard
single-stream downloader proved slow, followed by the full published SHA-256
check and the normal setup conversion/packing. No launcher preferences were
changed. Raw reports, per-mode summaries, and native stderr are in `mtp-off/`
and `mtp-on/`; hardware/source metadata is in `hardware.json`.
