# Current state on RTX 3050 8 GB (v0.2.0 plus sampled MTP)

This record re-checks and re-measures Lamina on the RTX 3050 / Ryzen 7 5700X
machine after v0.2.0 and the sampled-MTP change (`a201290`). It is written for
developers who need current numbers for this machine and want to know whether
the new commits changed correctness or speed. No layer math changed in these
commits, and none was changed for this record.

Hardware: NVIDIA GeForce RTX 3050 (8192 MiB, PCIe Gen3 running at x8), driver
616.92, AMD Ryzen 7 5700X (8 cores/16 threads), 64 GiB DDR4, Windows 11
Enterprise LTSC 2024. Source commit `a201290`, built with
`tools.build_windows --cuda-arch 86`; the engine SHA-256 begins with
`6fb7de6b338f4037` and is recorded in every greedy run JSON. Models: pinned
Qwen3.6-35B-A3B UD-Q4_K_M with its Q8_0 MTP head, and Ornith-1.5-35B-A3B
Q4_K_M with its packed in-file head. The new abliterated Qwen3.6 build was not
downloaded or measured.

## Greedy decode

This section uses the method of [the 9 October Ornith record](../2026-10-09-ornith-rtx3050/README.md):
`tools.compare_lamina`, one resident engine per mode, three repetitions of each
prose/code/math prompt, 256 greedy tokens, 32768 context, FP16 device KV, fast
compute and registered host model pages.

| Model | Mode | Prose | Code | Math | Nine-run median |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6 | MTP off | 41.30 | 39.40 | 38.70 | **39.40** |
| Qwen3.6 | MTP on | 45.73 | 47.61 | 47.55 | **47.05** |
| Ornith | MTP off | 43.63 | 42.02 | 40.63 | **42.02** |
| Ornith | MTP on | 41.85 | 45.35 | 48.03 | **45.35** |

Values are tokens/s and exclude the first generated token. Ornith matches the
9 October measurements (42.03 without MTP, 45.5-46.2 with MTP in that session's
runs), so the new commits did not change greedy speed. Qwen3.6 without MTP
measures well above the 36.7-37.5 tokens/s of the 7-8 October records on this
machine; that gain predates this change and comes from the 9 October admission
and graph-cache work.

The warm first token takes 0.55 s for both models. The first request after
startup took 4.7-4.9 s, and 7.1 s for Qwen3.6 without MTP, whose model file had
probably left the file cache. Peak sampled total GPU memory was 7083 / 7157 MiB
for Qwen3.6 and 6619 / 6657 MiB for Ornith (MTP off / on), with 1.3-1.4 GiB idle
desktop use. All repetitions are token-identical within each mode.

## Sampled decoding

This section uses `tools.check_sampled_mtp` with its defaults: temperature 0.6,
top_p 0.95, top_k 20, 128 output tokens, one warm-up round and three measured
rounds of three prompts (seeds 43-45). It runs through the Python server with a
131072-token context, so its rates are not comparable with the greedy table.

| Model | Ordinary sampling | Sampled MTP | Prose | Code | Math |
| --- | ---: | ---: | --- | --- | --- |
| Qwen3.6 | 34.15 | **38.63** (+13%) | 34.29 -> 35.33 | 34.35 -> 39.33 | 33.51 -> 38.63 |
| Ornith | 35.87 | **38.52** (+7%) | 35.87 -> 34.55 | 35.93 -> 39.38 | 34.72 -> 38.59 |

Medians are over nine requests per variant. In both models all nine paired
answers are identical with and without sampled MTP. The warm first visible
token took 0.44-0.46 s in every variant. Cumulative speculation statistics were
83.2% acceptance for Qwen3.6 and 78.3% for Ornith (two backoffs), with about
46 ms per verification pass against about 35 ms for the greedy pair pass.

The gain is larger than on the RTX 4060 (27.26 -> 28.81 tokens/s in
[the sampled-MTP record](../2026-10-11-sampled-mtp/README.md)). As with greedy
decoding, Ornith prose becomes slightly slower because its drafts are often
rejected.

Peak sampled total GPU memory reached 7241 MiB (Qwen3.6) and 7351 MiB (Ornith)
of 8192 MiB. At the 128K context this leaves about 0.8 GiB, less than in the 32K
greedy runs.

The tool wrote "RTX 4060 8 GB / Ryzen 7 1700X / 64 GB RAM" into both reports
because the hardware string was hard-coded. The copies here carry the detected
hardware in `hardware` and the original text in `hardware_as_written`. The
tool now detects the hardware with `tools.lamina_hardware`, as does
`tools.check_vision_snapshot`.

## Correctness checks

All checks below passed with the same build.

| Check | Result |
| --- | --- |
| `python -m unittest discover -s tests/lamina` | 73 tests pass after installing `imageio-ffmpeg==0.6.0` from `requirements.txt` into the sibling venv |
| `lamina-sampling-check` | pass |
| `lamina-sampled-mtp-check`, Qwen3.6 and Ornith | top-k=1 greedy parity, seeded replay and positions pass |
| `lamina-prefix-snapshot-check`, Qwen3.6 and Ornith | reload hidden `max_abs_diff=0`; no temporary snapshot files left |
| `lamina-cuda-elementwise-check`, `lamina-cuda-attention-check` | pass |
| `lamina-cuda-projection-check`, `lamina-kv-precision-check`, Qwen3.6 | pass |
| `lamina-prefill-check`, Qwen3.6 | pass, largest difference `8.855e-6` as before |
| `tools.reference_prefix --layers 40 --cuda --batched`, Qwen3.6 | next token 369, logit 12.448531 against reference 12.448534 |

The only initial failure was `test_video_sampling_order_eof_and_cleanup`, which
could not import `imageio_ffmpeg` because the existing venv predated v0.2.0.

## Exact commands

Run from the repository root with the sibling virtual environment. GPU
workloads ran serially.

```powershell
../Lamina-data/venv/Scripts/python.exe -m pip install -r requirements.txt
../Lamina-data/venv/Scripts/python.exe -m tools.build_windows --cuda-arch 86 --jobs 6
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --context 32768 --repeats 3 --tokens 256 --host-register 1 --report-dir <out>/qwen-off
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --context 32768 --repeats 3 --tokens 256 --host-register 1 --mtp ../Lamina-data/mtp/qwen36-mtp-q8_0.gguf --report-dir <out>/qwen-on
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --model ../Lamina-data/models/Ornith-1.5-35B-Q4_K_M.gguf --tokenizer ../Lamina-data/ornith/tokenizer.json --context 32768 --repeats 3 --tokens 256 --host-register 1 --report-dir <out>/ornith-off
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --model ../Lamina-data/models/Ornith-1.5-35B-Q4_K_M.gguf --tokenizer ../Lamina-data/ornith/tokenizer.json --context 32768 --repeats 3 --tokens 256 --host-register 1 --mtp ../Lamina-data/mtp/ornith-mtp.gguf --report-dir <out>/ornith-on
../Lamina-data/venv/Scripts/python.exe -m tools.check_sampled_mtp --model qwen3.6 --runs 3 --output <out>/sampled-qwen3.6
../Lamina-data/venv/Scripts/python.exe -m tools.check_sampled_mtp --model ornith --runs 3 --output <out>/sampled-ornith
```

Raw greedy reports, summaries and native stderr are in `qwen-off/`, `qwen-on/`,
`ornith-off/` and `ornith-on/`. The sampled reports are `sampled-qwen3.6.json`
and `sampled-ornith.json`.
