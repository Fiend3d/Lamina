# Expert-pipeline warm-up hybrid (RTX 4060 Ti 16 GB)

Date: 9 October 2026. Source base: `524c92d` (plus the change below).
Owner: Vlad Tatintsev.

## Problem

Fast-mode decode runs the serial `moe_core` path under the `cpu-miss` policy
(`pipeline_mode()` false), which does not overlap missing-expert uploads with
resident-expert compute (`src/model/cuda_projection.cpp` `moe_pipeline` vs
`moe_core`). CPU experts only run after an eviction, so during the initial cache
fill the serial path exposes the upload latency of every miss. The result is a
long warm-up ramp: the first ~32 generated tokens run far below the steady rate,
and short/agent turns and time-to-first-token suffer.

Baseline on this machine with the published v0.1.1 engine
(`--compute-mode fast --kv-type f16 --kv-cache device --max-context 32768`,
256 generated tokens, prose prompt):

| Run | First token | First 32 median | Last 128 median | Overall |
| --- | ---: | ---: | ---: | ---: |
| v0.1.1 `cpu-miss` (default) | 4.068 s | 51.5 ms (19.4 tok/s) | 21.3 ms (47.0 tok/s) | 32.4 tok/s |
| v0.1.1 `stream` (`LAMINA_EXPERT_POLICY=stream`) | 2.781 s | 36.1 ms (27.7 tok/s) | 26.8 ms (37.3 tok/s) | 33.2 tok/s |

The per-token crossover is near token 96: `stream`/pipeline wins while the cache
fills; the serial `cpu-miss` path wins once it is warm.

## Change

`Impl::pipeline_mode()` in `src/model/cuda_projection.cpp` now lets the
`cpu-miss` policy use the overlapping `moe_pipeline` path while the expert cache
is still filling, and fall back to the serial path once it is warm:

```c++
bool pipeline_mode() const {
    if (!cpu_misses) return true;
    static const bool disabled = [] { const char* v = std::getenv("LAMINA_HYBRID"); return v && v[0] == '0'; }();
    if (disabled) return false;
    return stat_evicted == 0 && cached_bytes < cache_limit - cache_limit / 10;
}
```

The same predicate gates next-layer prefetch (`predict`) and the early shared
expert (`shared_early`), so each decode token uses exactly one consistent path.
`LAMINA_HYBRID=0` restores the previous always-serial behavior for A/B runs.
The default `f32` path is unaffected (`cpu_misses` is false there).

## Results (source build, same prompt/config, 256 tokens)

| Run | First token | First 32 median | Last 128 median | Overall |
| --- | ---: | ---: | ---: | ---: |
| Serial (`LAMINA_HYBRID=0`) | 5.274 s | 58.4 ms | 22.0 ms | 31.2 tok/s |
| Hybrid | 2.832 s | 34.9 ms | 22.1 ms | 36.4 tok/s |
| Hybrid + `LAMINA_HOST_REGISTER=1` | 2.771 s | 35.3 ms | 22.8 ms | 35.8 tok/s |

Warm-up throughput roughly doubles (17 → 29 tok/s on the first 32 tokens),
first-token latency drops ~2.4 s, and steady-state decode is unchanged.

## Correctness

- Greedy `next_tokens` for the 256-token, 32768-context run are **identical** to
  the previous serial `cpu-miss` output (no eviction during the session).
- Forced eviction (`LAMINA_CUDA_CACHE_MB=2500`) exercises the CPU-expert path and
  the pipeline→serial transition; the run completes at the same rate (28.57 vs
  28.72 tok/s) and produces a valid, deterministic trajectory. Under eviction the
  tokens differ from the serial run, which is already true of the existing
  `stream` vs `cpu-miss` policies because CPU and GPU expert arithmetic differ in
  their last bits (release `stream` vs `cpu-miss` first diverge at token 30 with
  the same 2500 MiB cache).
- Independent 40-layer FP32 reference unchanged: max abs diff `3.59125275e-06`,
  next token `369` (matches the documented gate).
- `python -m unittest discover -s tests/lamina` passes.
- MTP smoke test (`lamina.py serve ... --mtp`, one request) returns correctly.

## Limitations

- Measured on one machine (RTX 4060 Ti 16 GB / i5-12400F); the 8 GB path is
  exercised only by forcing a 2500 MiB cache, not on 8 GB hardware.
- The warm-switch trigger (`cached_bytes` ≥ 90% of `cache_limit`) is a heuristic;
  a workload that reaches steady decode with a much smaller resident set could
  stay on the pipeline longer than ideal. Real prose/code/math generations reach
  the threshold, and the `stat_evicted` guard bounds it at saturation.
- Fast-mode quality under sustained eviction was not re-run through the KL /
  perplexity fixture (the fixture's non-ASCII paragraphs are still corrupted per
  the roadmap); output identity holds on the no-eviction path.

## Commands

```powershell
python -m tools.benchmark --engine build-cuda/Release/lamina-infer.exe --cuda `
  --prompt "Explain the concept of recursion in computer science, with a short example." `
  --tokens 256 --compute-mode fast --kv-type f16 --kv-cache device --max-context 32768 `
  --quiet --json <out>.json
$env:LAMINA_HYBRID='0'    # serial baseline
$env:LAMINA_HOST_REGISTER='1'  # registered-RAM / prefetch path
```
