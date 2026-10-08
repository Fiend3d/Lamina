# Lamina


Fast mode now runs cache-missed experts on the CPU by default. On an RTX 3050
8 GB with a Ryzen 7 5700X, the matched comparison measured **35.84 tokens/s for
Lamina versus 28.56 tokens/s for CUDA llama.cpp b11474** (nine resident-process
runs, 32K context, FP16 KV), ahead in prose, code and math. The **40 tokens/s
target is not met**, prefill is still slower than llama.cpp, and this is not a
matched Strata comparison. See
[commands, results and limits](bench/results/2026-10-07-rtx3050-cpu-experts/README.md)
[the DeltaNet step update](bench/results/2026-10-07-rtx3050-gdn-step/README.md)
[the host-path update](bench/results/2026-10-07-rtx3050-host-path/README.md)
and [the greedy and cache update](bench/results/2026-10-07-rtx3050-greedy-keep/README.md).

Greedy speculative decoding with the model's own MTP head (`--mtp`, see below)
now measures a **42.94 tokens/s nine-run median against 37.15 tokens/s** for
single-token decode on the same machine and binary, and 45.21 tokens/s when
drafts come from the first 98,304 token ids. It costs about 0.45 s of
first-token time, for a reason not yet found. See
[the speculation record](bench/results/2026-10-08-rtx3050-mtp-speculation/README.md).

Earlier records from an RTX 4060 machine (where llama.cpp measured 25.47 and
Lamina 16.40 tokens/s before these changes) are in
[the previous comparison](bench/results/2026-10-07-llama-cuda/README.md) and
[validation](bench/results/2026-10-06-strata-plan/VALIDATION.md). That machine has
not been remeasured.

Lamina is a native [Strata](https://github.com/Niko1221/Strata) port for
Qwen3.6-35B-A3B, pinned to Strata commit `6f32ec070f23ced9f50e704d854d775da52591ab`.
Upstream MIT licenses and attribution are preserved. The inherited Qwen3.8
execution graph is not used for Lamina inference.

The native engine supports text and images, persistent CLI/API sessions, causal
batched prefill, sampling, streaming, reasoning and tool calls. The default
context is 32,768 tokens; 131,072 tokens uses a complete FP32 KV cache in system
RAM with bounded GPU staging. Model weights are memory-mapped and experts are
streamed through a bounded GPU cache. An 8 GB GPU does not hold the 22.1 GB model.

## Windows quick start

Requires Python 3.11+, Visual Studio 2022 C++ tools, CMake and an NVIDIA driver.
Use the sibling data directory for the environment, model and toolchain:

```powershell
python -m venv ../Lamina-data/venv
../Lamina-data/venv/Scripts/python.exe -m pip install -r requirements.txt -r requirements-build.txt -r requirements-reference.txt -r requirements-benchmark.txt
../Lamina-data/venv/Scripts/python.exe setup.py --download
../Lamina-data/venv/Scripts/python.exe setup.py --tokenizer
../Lamina-data/venv/Scripts/python.exe setup.py --index
../Lamina-data/venv/Scripts/python.exe -m tools.lamina_assets
../Lamina-data/venv/Scripts/python.exe -m tools.bootstrap_cuda
../Lamina-data/venv/Scripts/python.exe -m tools.build_windows --vision
```

The helper builds Release CUDA for SM 89 (RTX 4060) by default. For another GPU
pass its compute capability, for example `--cuda-arch 86` for an RTX 3050/3060.
A mismatched architecture builds without error but produces wrong kernels.
The helper installs runtime DLLs beside the binary and builds the pinned CPU
image encoder. CUDA
redistribution checksums and licenses are verified/preserved. The image encoder
uses CPU RAM so it does not compete for the text engine's VRAM.

```powershell
../Lamina-data/venv/Scripts/python.exe lamina.py chat --prompt "Hello" --max-tokens 64
../Lamina-data/venv/Scripts/python.exe lamina.py chat --image ../Lamina-data/validation/ocr-0.png --prompt "Read this image"
../Lamina-data/venv/Scripts/python.exe lamina.py serve --vision --max-context 131072 --kv-cache host
```

For faster, slightly lossy computation on an 8 GB GPU / 64 GB RAM machine:

```powershell
../Lamina-data/venv/Scripts/python.exe lamina.py serve --compute-mode fast --prefill-chunk 2048
```

`fast` uses Strata Q8 activation kernels and BF16 prefill with FP32 accumulation.
It is lossy; checked FP32 remains the default. In fast mode, experts missing from
the GPU cache run on CPU worker threads (a quarter of the hardware threads by
default) instead of being copied over PCIe. Optional `LAMINA_HOST_REGISTER=1`
pins about 21 GiB of model RAM and enables background admission of CPU experts
into VRAM. In fast mode it measured no faster than leaving it unset, while it
adds several seconds to startup, so it is not recommended there. For a 128K fast configuration add
`--max-context 131072 --kv-type f16 --kv-cache device`. FP16 KV is separately
lossy. Capacity and retrieval results are in
[the current measurement record](bench/results/2026-10-06-strata-plan/VALIDATION.md).

Greedy requests (temperature 0) can decode speculatively with the checkpoint's
multi-token-prediction head. The head is not in the GGUF; fetch it once from
the official BF16 checkpoint (range requests for 19 tensors) and pack it:

```powershell
../Lamina-data/venv/Scripts/python.exe -m tools.lamina_mtp fetch
../Lamina-data/venv/Scripts/python.exe -m tools.lamina_mtp pack
../Lamina-data/venv/Scripts/python.exe lamina.py serve --compute-mode fast --mtp ../Lamina-data/mtp/qwen36-mtp-q8_0.gguf
```

Each step drafts one token and verifies it together with the current token in
one two-token pass, so greedy output stays the model's own. In fast CPU-miss
mode the result can still differ in late tokens from single-token decode,
because different experts land on the CPU, whose results differ in their last
bits. Text the head drafts poorly falls back to single steps automatically.
`LAMINA_MTP_VOCAB=98304` drafts from the first 98,304 token ids only, which is
faster for English text but can never draft tokens above that bound.

The server binds to `127.0.0.1:8000`. Endpoints are `/health`, `/v1/models` and
`POST /v1/chat/completions`. Requests support `stream`, `stream_options.include_usage`,
`temperature`, `top_p`, `top_k`, `seed`, `stop`, `max_tokens`, thinking controls,
function tools/tool choice, and text/image content. JSON object/schema requests
are prompted and validated before delivery; this is not grammar-constrained
decoding. Tool/schema responses are buffered for validation. Other completions
stream UTF-8 content and separate `reasoning_content`. Requests are serialized
through one resident engine. Disconnecting during inference stops that request's
native process; the next request starts a fresh process.

128K is a capacity option, with slower prefill and decode than short contexts.
The context limit includes prompt, expanded image tokens and generated tokens.
There is no KV history truncation. This machine has 64 GB RAM; RAM-backed caches
and mapped model pages need ample system memory. `--vram-limit-mb` bounds engine
CUDA allocations; automatic sizing leaves GPU headroom and includes scratch/KV.
Long text prefill processes all prompt tiles one layer at a time, reusing that
layer's expert weights. Its full residual needs at most 1 GiB of GPU memory;
attention and projection workspaces remain tiled. `--prefill-chunk` defaults
to 2048 (maximum 2048). `--cpu` selects scalar inference.
`--expert-policy auto` (default) runs cache-missed experts on the CPU in fast
mode and streams them to the GPU in f32 mode; `stream` and `cpu-miss` force a
policy. `--cpu-threads` sets the image encoder threads (default 8) and, when
given, the CPU expert workers.

`--kv-type f16` enables experimental, lossy FP16 KV storage with FP32 attention
accumulation. It halves KV memory and transfer bytes. For 128K the cache is
about 2.5 GiB instead of 5 GiB; `--kv-cache device` can keep that cache on this
GPU while reducing expert residency. `auto` retains its conservative host-KV
choice above 32K. FP32 remains the default. FP16 can change routing and output;
a 64-token sample showed 1.37% relative hidden-state drift against FP32.
That sample does not establish generation quality at long contexts.

## Earlier measurements on this machine

RTX 4060 8 GB, Ryzen 7 1700X, 64 GB RAM, Release CUDA 13.3.
The following FP32 capacity runs precede the attention/KV update:

| Run | First token | Later tokens/s | Peak total GPU memory |
| --- | ---: | ---: | ---: |
| Short matched decode (three-run median) | 1.54-1.57 s | 14.13 | 6642 MiB |
| 32K capacity: 32,736 prompt + 16 response tokens | 351.47 s | 10.87 | 7662 MiB |
| 128K capacity: 131,040 prompt + 16 response tokens | 3115.67 s | 0.689 | 6980 MiB |

Short decode is 6.55% faster than the original Lamina Release build on the
same machine. These results do not establish Strata-class performance.
The 128K FP32 RAM-backed mode meets the capacity target but has substantial
latency on this hardware. Peaks include the desktop and are sampled with NVML.
Context runs use repeated token 42 as a capacity stress, not a retrieval-quality
benchmark. Exact commands, hardware, executable hashes and validation are in
[the measurement record](bench/results/2026-10-06-rtx4060-8gb/VALIDATION.md).

The attention/KV update was measured against the saved, completed port binary
on the same machine, with 2048-token chunks:

| 16K prompt mode | First token | Later tokens/s | Peak total GPU memory |
| --- | ---: | ---: | ---: |
| Previous FP32 host KV | 125.81 s | 4.66 | 7250 MiB |
| Updated FP32 host KV | 104.68 s | 6.56 | 7250 MiB |
| Updated FP16 host KV (lossy) | 102.50 s | 9.68 | 7253 MiB |
| Updated FP16 device KV (lossy) | 102.04 s | 16.67 | 7222 MiB |

These are single matched runs. FP32 16K first-token time fell 16.8%, and later
throughput rose 40.9%, with identical continuations. Matched short decode
remained 13.93 tokens/s before/after; all 120 predictions agreed in FP32 and
FP16. Both precision modes passed 11 real API checks. Full benchmark commands,
hashes, precision differences and boundaries are in the
[attention update record](bench/results/2026-10-06-attention/VALIDATION.md).

The full 128K FP16 GPU-cache run (131040 prompt + 16 generated tokens) completed
at **28 min 57 s to first token, 7.21 later tokens/s**, with **7304 MiB peak total
GPU memory** and 12591 MiB process RAM. This changes precision and uses 2048-token
chunks; it is a capacity stress, not a retrieval-quality benchmark. Prefill still
has substantial latency. To select this experimental profile:

```powershell
../Lamina-data/venv/Scripts/python.exe lamina.py serve --vision --max-context 131072 --kv-cache device --kv-type f16
```

## Build and validate

The default CMake build needs no CUDA toolkit:

```sh
cmake -S . -B build
cmake --build build --config Release --target lamina-gguf lamina-infer lamina-sampling-check
python -m unittest discover -s tests/lamina
```

On Windows, inspector output is `build/Release/lamina-gguf.exe`. Pass the model
path and `--check`. CUDA checks, the independent 40-layer reference, and measured
performance commands are in [the developer handoff](docs/DEVELOPER_HANDOFF.md).
See [port status](docs/LAMINA_PORT.md) for validation boundaries and
[RTX 4060 measurements](bench/results/2026-10-06-rtx4060-8gb/) for exact hardware,
commands, first-token time, later throughput and sampled peak total GPU memory.
The earlier RTX 4060 Ti result is a different machine and is not a performance
promise for this RTX 4060/Ryzen 1700X system.

Model assets are pinned in `tools/lamina_model.py` and `tools/lamina_assets.py`;
all downloads live under `../Lamina-data`. Original Strata documentation in
`ref/` is retained as porting source material.
