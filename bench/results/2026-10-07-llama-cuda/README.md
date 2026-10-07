# CUDA llama.cpp comparison, 7 October 2026

Lamina trails the official CUDA llama.cpp build on this machine. Nine-run
median later throughput is **16.40 tokens/s versus 25.47**: llama.cpp is about
55% faster. Neither reaches the requested 40 tokens/s. This is not a matched
Strata measurement and does not establish Strata parity.

## Hardware and setup

RTX 4060 8188 MiB, driver 610.88, PCIe Gen3 x8; Ryzen 7 1700X, 8 cores/16
threads; 64 GiB RAM; Windows 11. Same pinned Qwen3.6-35B-A3B UD-Q4_K_M GGUF,
SHA256 `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
All GPU runs were serial. NVML peaks include the desktop; idle values are
recorded, so differing desktop usage should not be attributed to an engine.

llama.cpp is the official Windows CUDA 12.4 x64 release **b11474**, commit
`b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea`, built with Clang 20.1.8. Both
binary and runtime ZIPs matched the published GitHub SHA256 digests in
`llama-release.json`. The attempted source build was stopped before testing.
The release differs from Lamina's pinned ggml dependency revision; this
comparison uses the downloadable product, not a same-revision kernel test.

Lamina is Release CUDA 13.3/SM89, native executable SHA256
`3ed0fba0f9c9c1d749f44b52adf015845ea69224cb1d5e74b7a129de257c067d`.
Optional fast mode and direct mapped-model RAM registration are enabled;
normal computation/KV defaults remain FP32. Both engines use FP16 device KV
and 32768 context for short tests. No speculative decoding is enabled.
llama.cpp uses eight CPU threads, automatic GPU placement, Flash Attention,
2048 batch/microbatch and `--load-mode none`. It placed 4528 MiB of model
weights on CUDA and 16570 MiB in host memory, with all dense layers offloaded.
Lamina uses its bounded streaming expert cache, capped at 5000 MiB.

## Short prompts

Three repeats each, 256 greedy generated tokens including forced generation
past EOS if encountered. All nine input token hashes and output lengths
match between engines. Same Qwen non-thinking chat template and token IDs.
Generated sequences may diverge because computation differs. Both engines
remain resident; KV is reset between requests and llama.cpp prompt reuse is
disabled. Native pipe versus HTTP/SSE transport differs. Wall-time throughput
excludes the first token; llama.cpp's internal timings agree with it.

| Prompt | Lamina tokens/s | llama.cpp tokens/s |
|---|---:|---:|
| prose | 20.24 | 25.28 |
| code | 16.40 | 25.50 |
| math | 16.27 | 25.47 |
| Nine-run median | 16.40 | 25.47 |

Steady request first-token medians: Lamina **2.45 s**, llama.cpp **1.41 s**.
Lamina's first request initializes CUDA lazily after RESET, so it is excluded
from this steady-latency number. First cold request including startup:
Lamina **11.26 s**, llama.cpp **20.16 s**. Model startup and request prefill
must not be conflated. Total GPU peaks: **6969 MiB** Lamina, **6937 MiB**
llama.cpp. Per-run timings, token IDs, commands and executable hashes are in
the accompanying JSON files.

64-token prose pilots (not acceptance results) measured auto/mmap/registered
22.43, CPU-MoE-34/mmap/registered 22.86, CPU-MoE-34/four-threads/load-none
22.79, CPU-MoE-34/eight-threads/load-none 26.14, and auto/eight-threads/load-none
25.91 tokens/s. The last two are close; automatic placement was selected for
its similar speed and lower VRAM use. Only that configuration received the
full nine-run benchmark.

## Long retrieval

The exact varied station-record prompt places code `LAMINA847263` near the
beginning. Both engines return it before EOS at 4K and 128K. No truncated
history or sliding window. Each answer is nine tokens including EOS;
answer throughput is not a sustained long-context decode measurement.

At 128K both consume **130928 prompt tokens**. Lamina's prior final capped
cache run uses the identical executable, prompt file and precision options;
it is reused rather than repeating a five-minute measurement. Cold first
response: Lamina **306.52 s**, llama.cpp **283.32 s** including its **19.99 s**
model startup. llama.cpp's request-only prefill/first token is **263.33 s**.
Later short-answer throughput is **6.37** versus **15.46 tokens/s**, with total
GPU peaks **6791** versus **7049 MiB**. Idle GPU usage was 701 versus 878 MiB.

At 4K the first Lamina request took 19.95 s after RESET, including deferred
CUDA initialization. llama.cpp's request-only time was 6.03 s after loading.
Those two request timings are not a steady-prefill comparison; the raw
records retain startup time separately.

## Reproduce

Use the sibling virtual environment. Download binary/runtime assets from
https://github.com/ggml-org/llama.cpp/releases/tag/b11474 and extract both into
`../Lamina-data/llama-prebuilt-b11474`. Keep model data outside the source tree.

```powershell
python -m tools.compare_llama --server ../Lamina-data/llama-prebuilt-b11474/llama-server.exe --revision b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea --load-mode none --report-dir ../Lamina-data/comparison-llama/final
python -m tools.compare_lamina --report-dir ../Lamina-data/comparison-lamina/final
python -m tools.compare_llama --load-mode none --context 131072 --prompt-file ../Lamina-data/retrieval-varied-131072.txt --tokens 64 --repeats 1 --timeout 600
```

The retrieval files are produced by `tools.performance_check --only-long`.
Full native/server commands are saved per run. `--no-warmup` deliberately
keeps first-request graph setup visible. Lamina's independent FP32 gate and
fast quality evidence are recorded in the adjacent Strata-plan validation.
The observed gap makes CPU expert execution, expert transfers and routing
synchronization candidates for profiling; this comparison alone does not
prove which dominates. Whole-token asynchronous graph replay is not implemented.
