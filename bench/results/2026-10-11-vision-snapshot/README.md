# GPU vision handoff snapshots

Hardware: NVIDIA RTX 4060 8 GB (8188 MiB), AMD Ryzen 7 1700X, 64 GB RAM,
Windows. CUDA fast compute, device F16 KV, configured context 131072, MTP on.
Source: b1552e4 plus this change; layer formulas are unchanged.

Reproduction from the repository root with the sibling venv and installed model:

```powershell
..\Lamina-data\venv\Scripts\python.exe -m tools.build_windows --jobs 8
..\Lamina-data\venv\Scripts\python.exe -m tools.check_vision_snapshot --model qwen3.6 --audit-tokens 48 --output ..\Lamina-data\validation\vision-snapshot-qwen
.\build-cuda\lamina-prefix-snapshot-check.exe ..\Lamina-data\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf "..\Lamina-data\tmp\native snapshot test.bin"
```

The Qwen fixture has 1800 filler lines followed by ALPHA, then appends BETA
and GAMMA as genuinely uncached GPU images. All cached, reloaded and full-prefill
OCR answers match. Real prompt counts include image tokens.

| Request | Prompt tokens | Full-handoff first token | Snapshot first token | Reused tokens |
| --- | ---: | ---: | ---: | ---: |
| First image | 24500 | 45.71 s | 46.92 s | 0 |
| New second image | 25601 | 60.27 s | 25.75 s | 24493 |
| New third image | 26702 | 68.12 s | 24.78 s | 25594 |
| Identical second-image repeat | 25601 | 0.53 s | 0.44 s | 25594 |

Second-image answer: BETA; third-image answer: ALPHA BETA GAMMA. Later-token
rates on the third-image answer were 13.03 versus 7.43 tokens/s; these are
only six-token outputs, affected by speculative chunking, and not steady decode
benchmarks. Peak total GPU memory sampled with nvidia-smi: 6986 MiB. Full
prefill after the snapshot case also returned exactly ALPHA BETA GAMMA.

The first snapshot was 604.0 MiB: export 0.45 s, import/startup 0.91 s. The
next was 625.5 MiB: export 0.45 s, import/startup 0.88 s. Snapshot transfer
uses bounded host staging and a temporary file in Lamina-data/tmp, normally
benefiting from the OS file cache. It is not a durable session backup or a
promise that all state stays exclusively in RAM. Old model teardown, new
encoder startup, new image encoding (~6.2 s), model reload and processing
~1100 appended image/text tokens still take time.

The native reload check returned max_abs_diff=0 for continuation hidden states
in all four combinations of F16/F32 KV and F32/fast compute, including synthetic
image mRoPE positions. It rejects truncated snapshots and context mismatches.
Server unit checks cover export/import failure fallback, cleanup and signatures.
Python checks, default CMake/lamina-gguf, CPU native build and CUDA numerical
attention/elementwise checks passed.

Ornith's short image fixture also passed all nine image requests, including
cached/reloaded/full-prefill equality. Its subsequent independent greedy replay
audit failed exact cold/warm equality on a 28-token text prompt. That is retained
in the raw report, not presented as a passed full audit. The backend's existing
CPU-miss expert policy can change last-bit rounding as residency changes; these
records do not establish the cause of every observed agent retry loop.

Scope: Qwen runtime snapshot measurement is at 24K–27K actual tokens, not 128K.
Capacity remains 131072; this does not constitute a new full-context speed or
memory-stress claim. Changes to history/image identity, unsupported older engines,
host KV and failed snapshots still take full prefill. Server restarts clear state.

The raw Qwen report is [qwen.json](qwen.json). The independent short Ornith
report, including the failed greedy replay audit, is [ornith.json](ornith.json).
