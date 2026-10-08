# Hybrid prefill: rarely used experts on the CPU

This record covers a prefill change in fast CPU-miss mode. During prompt
processing, an expert routed by only a few prompt tokens now runs on the CPU
expert pool instead of being uploaded over PCIe. It is written for developers
continuing the performance work.

## Result

Warm first-token time for the short benchmark prompts fell from **2.18 s to
0.69 s**. Speculative decoding no longer pays a first-token penalty (0.70 s
warm), and its throughput rose to a **45.87 tokens/s** nine-run median.
Single-token decode is unchanged within noise at 37.49 tokens/s.

| Mode | Prose | Code | Math | Nine-run median | Warm first token | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-token, before | 39.99 | 37.15 | 35.37 | 37.15 | 2.18 s | 6800 MiB |
| Single-token, hybrid prefill | 40.06 | 37.49 | 36.70 | **37.49** | **0.69 s** | 6615 MiB |
| Speculative, before | 42.58 | 42.91 | 43.97 | 42.94 | 2.61 s | 6900 MiB |
| Speculative, hybrid prefill | 44.70 | 45.97 | 45.92 | **45.87** | **0.70 s** | 6719 MiB |

"Before" is [the speculation record](../2026-10-08-rtx3050-mtp-speculation/README.md).
Prompt columns are three-run medians of later tokens/s; the first-token column
is the median of the eight warm requests. Hardware: RTX 3050 8 GB (sm_86,
PCIe Gen3 x8, WDDM), Ryzen 7 5700X, 64 GiB DDR4. Engine SHA-256 prefix
`4d977ead0e0ebb4c`.

```powershell
python -m tools.build_windows --cuda-arch 86
python -m tools.compare_lamina --repeats 3 --report-dir <out>/single
python -m tools.compare_lamina --repeats 3 --mtp ..\Lamina-data\mtp\qwen36-mtp-q8_0.gguf --report-dir <out>/speculative
python -m tools.quality_check --report <out>/quality.json
$env:LAMINA_HOST_REGISTER='1'
python -m tools.performance_check --only-long --contexts 4096 16384 32768 --report-dir <out>/long-on
$env:LAMINA_PREFILL_CPU_TOKENS='0'   # the previous all-GPU prefill
python -m tools.performance_check --only-long --contexts 4096 16384 32768 --report-dir <out>/long-off
```

## Why it helps

Before this change a 95-token prompt uploaded about 8.6 GB of experts. Most
experts serve only a few prompt tokens, and after prefill the engine keeps
only each layer's top prompt experts (`LAMINA_BASE_KEEP`), so most of those
uploads were discarded. Over PCIe Gen3 x8 an upload ran at about 4.4 GB/s,
while the CPU pool reads expert rows from RAM at about 29 GB/s and applies
each row to all of the expert's tokens. With the split, the same prompt
uploads about 1.3 GB (`LAMINA_PREFILL_STATS=1` prints these numbers).

## How it works

`CudaProjection::moe_columns` counts the prompt tokens routed to each expert.
Every expert with at most `LAMINA_PREFILL_CPU_TOKENS` tokens (default 12, at
most 16) runs on the CPU while the GPU processes the remaining experts and the
shared expert. The layer's normalized inputs reach the CPU through one
mapped-memory copy that rides along with the routing download; the CPU results
are scattered into the expert slots before the combine.

The split depends on token counts only, never on cache residency, so output
does not depend on earlier requests (all repeats are identical). `CpuExperts`
now takes jobs of one expert with up to 16 token inputs of its own; each
distinct input row is quantized once per batch, and the decode entry points are
wrappers over the same path.

Thresholds measured on short prompts (warm first token): 2 tokens 1.30-1.47 s,
4 tokens 0.99-1.10 s, 8 tokens 0.71-0.76 s, 12 tokens 0.67-0.71 s, 16 tokens
0.74-0.77 s. A 1,270-token prompt went from 5.04 s to 4.03 s.

## Checks

- Quality gate (`tools.quality_check`, 1,024 teacher-forced tokens against
  FP32): mean KL 0.00277 nats, perplexity ratio 0.993, argmax agreement
  97.4%. With hybrid prefill disabled on the same binary: 0.00241, 1.001 and
  98.3%. Both are far inside the gate (KL at most 0.1, ratio at most 1.05).
- Long-context retrieval passes at 4K, 16K and 32K with and without the split.
  Prefill rose from 260 to 291, 388 to 421 and 383 to 409 tokens/s.
- Unit tests, `lamina-prefill-check` and `lamina-kv-precision-check` pass. The
  FP32 path and its reference check are unaffected (the split requires fast
  CPU-miss mode).
- Generated tokens differ from the previous binary, because CPU experts use
  ggml's activation quantization rather than the GPU prefill arithmetic. The
  prose prompt changes from its first token (an open-ended story opening,
  both versions coherent); code and math match for 184 and 93 tokens.

## Notes

- The speculative path's earlier first-token regression (about 0.45 s after
  the first MTP draft) is no longer visible, because prefill now moves little
  data over PCIe. Its cause, slower device-side expert uploads, was not found.
- The 128K profile was not rerun.
