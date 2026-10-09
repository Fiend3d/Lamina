# Decode admission overlap with 128K capacity preserved

9 October 2026. Source baseline: `5cbe58ae5e3a056dc36a4cf646f5626ea45c813e`.
Hardware: RTX 4060, 8188 MiB, driver 610.88, CUDA 13.3, SM89;
Ryzen 7 1700X (8 cores / 16 threads), 64 GiB RAM, PCIe Gen3 x8,
Windows 11 Enterprise LTSC. See [hardware.json](hardware.json).

## Changes and result

The CPU-miss path previously finished its CPU expert batch, submitted the
combine, and then submitted cache admissions. Single-token and MTP pair decode
now submit admissions after GPU expert launches and before the CPU wait. Host
submission and DMA can overlap the remaining CPU rows. Retirement epochs still
protect reused GPU blocks. Admissions still become visible only at the next
token, after copy-stream completion, so placement does not depend on DMA timing.
`LAMINA_ADMIT_EARLY=0` restores the previous submission order.

Ordinary decode retains 512 MoE graph variants instead of 128, reducing
eviction/recapture. MTP keeps its original 128 budget: an unconditional increase
regressed its measurements and was rejected. `LAMINA_MOE_GRAPH_CACHE` overrides
the budget (1..4096). Expert-cache/admission budgets, precision, model math,
worker count and 131072-token capacity are unchanged.

Nine runs per row: prose/code/math, three repetitions each, 256 output tokens.
Greedy generation, fast computation, FP16 device KV, registered model RAM.
GPU workloads and accepted builds were serial. The before and after runs are
successive sets rather than randomized trials; small differences can be timing
variation. All nine output-token lists match before/after in each mode.

| Model / mode | Original tokens/s | Final tokens/s | Gain | Warm first token, original / final | Peak total GPU MiB, original / final |
|---|---:|---:|---:|---:|---:|
| Ornith ordinary | 36.07 | 39.17 | 8.6% | 0.727 / 0.728 s | 5821 / 5567 |
| Ornith MTP | 40.58 | 41.27 | 1.7% | 0.724 / 0.724 s | 5917 / 5629 |
| Pinned Qwen ordinary | 35.26 | 38.53 | 9.3% | 0.783 / 0.791 s | 6417 / 6158 |
| Pinned Qwen MTP | 41.51 | 43.88 | 5.7% | 0.787 / 0.789 s | 6529 / 6240 |

These are **short-prompt rates with 131072 capacity configured**, not rates at
a full 128K history. Warm first-token medians exclude the first request, which
initializes CUDA/model registration lazily. Peak memory is NVML total-device
usage sampled every 50 ms, including desktop applications. Desktop usage changed
between sets; the lower final peaks do not establish engine memory savings.
Cold initialization and file-cache differences are not claimed as improvements.

Isolating admission order from the previously accepted graph-cache change:

| Model / mode | Graph-cache change only | With early admission | Additional gain |
|---|---:|---:|---:|
| Ornith ordinary | 36.91 | 39.17 | 6.1% |
| Ornith MTP | 40.71 | 41.27 | 1.4% |
| Pinned Qwen ordinary | 35.88 | 38.53 | 7.4% |
| Pinned Qwen MTP | 42.26 | 43.88 | 3.8% |

The short preliminary screen toggled admission order on the same new binary;
its gains were 6.9% ordinary and 6.1% MTP. The longer repeated MTP result above
supersedes that screen. Raw screen records are retained, not mixed into medians.
Ornith was checked before starting base-model measurements.

## Commands and records

Build:

```powershell
../Lamina-data/venv/Scripts/python.exe -m tools.build_windows --cuda-arch 89 --jobs 6
```

Final Ornith ordinary command (add `--mtp ../Lamina-data/mtp/ornith-mtp.gguf`
for speculation):

```powershell
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --engine build-cuda/lamina-infer.exe --model ../Lamina-data/models/Ornith-1.5-35B-Q4_K_M.gguf --tokenizer ../Lamina-data/ornith/tokenizer.json --context 131072 --repeats 3 --tokens 256 --host-register 1 --report-dir ../Lamina-data/performance/ornith-optimization/acceptance/ornith-after-off
```

Pinned Qwen ordinary command (add
`--mtp ../Lamina-data/mtp/qwen36-mtp-q8_0.gguf` for speculation):

```powershell
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --engine build-cuda/lamina-infer.exe --model ../Lamina-data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --tokenizer ../Lamina-data/tokenizer/tokenizer.json --context 131072 --repeats 3 --tokens 256 --host-register 1 --report-dir ../Lamina-data/performance/ornith-optimization/acceptance/qwen-after-off
```

Profilers and CPU/cache/admission overrides were unset for headline runs.
`LAMINA_HOST_REGISTER=1`; compare_lamina also sets `LAMINA_SPEC_STATS=1`.
Each raw JSON records its exact native command, environment, engine/tokenizer
hashes, prompt-token hash, output tokens, timings and memory. Invocation JSONs
record the benchmark command. Original native executable was saved locally as
`build-cuda/lamina-infer-ornith-baseline.exe` before edits; graph-cache-only engine
is saved as `build-cuda/lamina-infer-graph512-baseline.exe`. With the final binary,
`LAMINA_MOE_GRAPH_CACHE=128` and `LAMINA_ADMIT_EARLY=0` reproduce original policy
settings; executable hashes will differ.

[metrics.json](metrics.json) contains exact medians, warm first-token times,
peaks and engine hashes. `*-before-*`, `*-graph512-*` and `*-after-*` are the
original, graph-cache-only and final short-prompt records respectively.
The scripts here are snapshots of the runners used from `../Lamina-data/tmp`;
run from the source root. Models, text fixtures and reference logits stay
outside the source tree. Prior compilation-overlap and exploratory tuning
screens are excluded from headline results.

## Context measurements before the admission change

These records use the graph-cache-only engine, hash beginning `a15c3d17`,
not the final admission-overlap engine. **No final full-context speed gain is
claimed.** Subsequent experiments use short decode tests to avoid repeatedly
paying minutes of prefill.

Stress fixture: exactly 130816 chat-prompt tokens, varied field records,
an early unique code, then retrieval plus a long narrative request; 256 generated
tokens reach the 131072 budget. Full history is retained, with no sliding window.
Both models/modes retrieved the early code and generated 256 tokens without EOS.
Each process first generated 64 warmup tokens, RESET, then ingested the full
fixture, exercising long prefill after existing decode captures.

| Model / mode | Fresh full-prompt first token | Later tokens/s | Peak total GPU MiB |
|---|---:|---:|---:|
| Ornith ordinary | 303.78 s | 15.98 | 7388 |
| Ornith MTP | 305.10 s | 15.21 | 7042 |
| Pinned Qwen ordinary | 295.49 s | 14.67 | 7042 |
| Pinned Qwen MTP | 297.20 s | 14.59 | 7100 |

Original Ornith ordinary engine on the identical full fixture measured
304.64 s / 16.19 tokens/s / 7079 MiB. Prompt token hashes match, but generated
tokens first differ at index 71; both retrieve the early code. The cause of that
long-context divergence is not isolated. Those different decode trajectories
and one run per build prevent attributing the rate difference to graph-cache
size. There is no demonstrated graph-cache gain or token-equivalence claim at
full context.

Exact Ornith full-context command; replace executable for original:

```powershell
$env:LAMINA_PREFILL_PROGRESS='1'
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --engine build-cuda/lamina-infer-graph512-baseline.exe --model ../Lamina-data/models/Ornith-1.5-35B-Q4_K_M.gguf --tokenizer ../Lamina-data/ornith/tokenizer.json --context 131072 --repeats 1 --tokens 256 --host-register 1 --report-dir ../Lamina-data/performance/ornith-optimization/full-context --prompt-file ../Lamina-data/performance/ornith-optimization/ornith-full-128k.txt --fixed-tokens --warmup-tokens 64
```

### An 80K cached server conversation

The graph-cache-only engine also ran through `tools.lamina_chat.Engine`, the
server request path, using Ornith MTP, greedy generation and official template.
The configured capacity remained 131072. Three 256-token coding responses:

| Turn | Prompt tokens | Reused tokens | First token | Later tokens/s | Peak total GPU MiB |
|---|---:|---:|---:|---:|---:|
| Initial | 80000 | 0 | 159.71 s | 19.50 | 7065 |
| Follow-up | 80290 | 79993 | 2.52 s | 20.15 | 7081 |
| Follow-up | 80580 | 80283 | 2.54 s | 20.62 | 7081 |

The follow-ups ingest only 297 new tokens, rather than 80K from scratch.
This distinguishes fresh-history ingestion from normal cached agent turns.
This is a synthetic coding conversation without Pi's actual tools; it does not
establish another PC's performance. Exact runner is [ornith_pi_80k.py](ornith_pi_80k.py).
Run `../Lamina-data/venv/Scripts/python.exe ../Lamina-data/tmp/ornith_pi_80k.py` from
the source root. Raw turns, content, backend settings and memory are in
`ornith-pi-80k-after/`; its metadata identifies the earlier engine hash.

## Checks and limitations

The independent 40-layer FP32 reference passed at default `1e-5`:
maximum hidden difference `3.59125275e-6`, next token 369 in both, native logit
12.448531 versus reference 12.448533724. The default build and Python suite,
CUDA elementwise/projection/attention checks (including 131072-cell history),
prefix restore under default and one-entry graph caches, and prefill/reset/state
checks are recorded under `validation/`. See `validation/results.json` for exact
commands and completion status.

An additional registered-RAM prefix stress check passed with
`LAMINA_HOST_REGISTER=1`, `LAMINA_MOE_GRAPH_CACHE=1`, `LAMINA_ADMIT_MB=4`:
`build-cuda/lamina-prefix-check.exe ../Lamina-data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
Restored suffix error was zero in both FP32 and fast modes; process exit was 0.
The PowerShell-captured log formats native stderr as `NativeCommandError`; this
is its stderr rendering, not a nonzero test exit. Main Python suite: 49 passed.

Compilation alone is not runtime validation. The before/after greedy equality
checks verify this scheduling change on the measured prompts; they do not prove
general model quality. MTP and ordinary outputs differ in the original engine
and are compared only within their own mode.

There is no matched Strata run on this PC. User-reported Strata performance on
another PC, another model (Qwen3.8-Flash-Next) and 3-bit quantization is a target,
not this benchmark's baseline. Its inherited QSA implementation selects up to
2051 history cells for final attention after scoring pooled history; Ornith
attends the complete history. Scheduling can be improved without substituting
that other model's trained selection mechanism or truncating Ornith's context.
