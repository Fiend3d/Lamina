# System/tools prefix reuse

Windows 10, RTX 4060 8 GiB, driver 610.88, Ryzen 7 1700X, 64 GiB RAM,
PCIe Gen3 x8. Pinned Qwen3.6 UD-Q4_K_M and Q8 MTP head. CUDA fast mode,
FP16 device KV, 32K context, greedy generation, thinking disabled. Engine
remained resident; first run populated the prefix, median uses three later
requests. No other model server was running. Total GPU memory includes desktop
usage; a 20 ms NVML sampler measured peak. No timeline instrumentation in these
matched measurements. `LAMINA_HOST_REGISTER=1`, `LAMINA_PREFILL_STATS=1`.

| Request | Reuse off TTFT | Reuse on TTFT | Later tok/s off / on | Peak MiB off / on |
|---|---:|---:|---:|---:|
| Pi system + tools, 1602 tokens, answer `Ready` | 5.030 s | 0.328 s | N/A (one token) | 5658 / 5416 |
| Same system/tools, 1613 tokens, 96-token sky explanation | 5.311 s | 0.466 s | 26.80 / 32.45 | 6908 / 7006 |

First uncached `Ready` request: 4.221 s with reuse off, 4.534 s with reuse on.
First visible delta median: 5.042 vs 0.344 s. This improves later agent turns;
it does not remove the initial model load or first prefix prefill. The longer
cached run overlapped a portable compiler build, so its timings are conservative
and are not an isolated decode-speed claim. Warm cached answers were identical.
Splitting prefill can change fast-mode rounding/routing compared with unsplit
prefill; exact equivalence to the old fast path is not claimed.

Reproduce from the repo with runtime/reference dependencies plus nvidia-ml-py:

```powershell
$env:LAMINA_PREFIX_CACHE='0'
..\Lamina-data\venv\Scripts\python.exe -m tools.benchmark_prefix build-cuda\lamina-infer.exe ..\Lamina-data\releases\ttft\off bench\results\2026-10-08-rtx4060-prefix\pi-request.json
$env:LAMINA_PREFIX_CACHE='1'
..\Lamina-data\venv\Scripts\python.exe -m tools.benchmark_prefix build-cuda\lamina-infer.exe ..\Lamina-data\releases\ttft\on bench\results\2026-10-08-rtx4060-prefix\pi-request.json
```

Original runs used the equivalent local `measure_ttft.py` script outside the
repo; raw results are committed here. The exact original commands were:

```powershell
$env:LAMINA_PREFIX_CACHE='0'
..\Lamina-data\venv\Scripts\python.exe ..\Lamina-data\releases\measure_ttft.py build-cuda\lamina-infer.exe ..\Lamina-data\releases\ttft\uncached
$env:LAMINA_PREFIX_CACHE='1'
..\Lamina-data\venv\Scripts\python.exe ..\Lamina-data\releases\measure_ttft.py build-cuda\lamina-infer.exe ..\Lamina-data\releases\ttft\cached
```

For the long case replace the final user
message with `Explain why the sky is blue in about 100 words.` and set max_tokens
to 96. `--profile` enables timeline instrumentation for diagnosis only.

Validation: 47 Python tests, default CMake/lamina-gguf, native 40-layer prefix
checkpoint replay in f32/fast (max_abs_diff=0), replay after a different long
suffix, and RESET invalidation. Independent reference check:

```powershell
build-cuda\lamina-prefix-check.exe ..\Lamina-data\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
..\Lamina-data\venv\Scripts\python.exe -m tools.reference_prefix --layers 40 --cuda --engine build-cuda\lamina-infer.exe --kv-cache device
```

Reference maximum absolute difference 5.682e-7; next token matched (20).
The immutable system/tool block is keyed by full BPE IDs (minimum 256 tokens),
with about 63 MiB of budgeted recurrent snapshots. Existing attention KV stores
the prefix. Changed system/tools, images, unsupported/older binaries, host KV,
process failure or cancellation fall back to RESET. No conversation suffix is
cached. Set `LAMINA_PREFIX_CACHE=0` to disable reuse.

Final portable v0.1.1 ZIP (`23f5a84`) passed startup/preload, greedy MTP,
streamed text/usage/tools/reasoning and cached requests with changed user text
using only Windows directories in PATH. Both asset checksums were verified.
A separate 3,088-token prefix at temperature 0.7, seed 123 produced identical
eight-token responses on the cache miss and two hits, covering the ignored
long-prefix PROMPT token before request RNG configuration.
