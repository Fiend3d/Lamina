# Lamina

Lamina is a native port of [Strata](https://github.com/Niko1221/Strata) for
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The source baseline is Strata commit
`6f32ec070f23ced9f50e704d854d775da52591ab`. Strata is MIT-licensed; its license and third-party notices remain
in this repository.

## Current status

The Qwen3.6 execution graph now has a scalar C++ path. It reads the published GGUF directly and has DeltaNet, full
attention, routed experts, a text CLI, and a limited OpenAI chat endpoint. A two-token, full-model check matched an
independent NumPy implementation of the Qwen layer equations and selected the same next token and logit. Broader
generation quality has not been checked. The scalar path dequantizes matrix rows on the CPU for each token and will
be slow. Large scalar projections use up to eight CPU threads (`LAMINA_CPU_THREADS` sets 1..64). On a Ryzen 5 7520U,
a matched warm-cache 40-layer pass took 4.46 seconds with one worker and 2.98 with eight. A four-token decode sample
gave 0.395 later tokens/s with eight workers. An opt-in hybrid path now sends Q8_0, Q4_K, Q5_K and Q6_K projections through Strata's native CUDA
MMVQ kernels; its state updates and routing still run on the CPU. The hybrid path [compiles in Linux CUDA 12.6 CI](https://github.com/Fiend3d/Lamina/actions/runs/37366756905),
but has not been run on NVIDIA hardware or compiled with Windows CUDA. The inherited Strata GPU execution graph
still implements Qwen3.8 and is not used by Lamina.

## Build and inspect

The default build needs CMake 3.24+, a C++20 compiler, and no GPU toolkit:

```sh
cmake -S . -B build
cmake --build build --target lamina-gguf lamina-infer
```

On a machine with the CUDA toolkit and an Ampere or newer NVIDIA GPU, configure with
`-DLAMINA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80` and run `python lamina.py chat --cuda ...`.
The flag is explicit: a default build stays CPU-only, and a CPU-only binary reports an error if `--cuda` is used.
The hybrid path keeps an LRU device weight cache sized to 75% of free VRAM at startup and transfers activations for
each projection. On a 16 GB card, set `LAMINA_CUDA_CACHE_MB=6144` to test a roughly 8 GB-card-sized cache budget;
an actual 8 GB card chooses its limit automatically. This setting caps the cache, not all CUDA allocations. It is an
experimental step toward a full GPU engine, with no verified speedup or model parity yet.

Measure decode speed with one persistent process, first at the automatic cache limit, then with a 6 GiB cap to
approximate an 8 GB card's weight budget:

```sh
python -m tools.benchmark --engine build-cuda/lamina-infer --cuda --tokens 4
python -m tools.benchmark --engine build-cuda/lamina-infer --cuda --cache-mb 6144 --tokens 4
```

On Windows, use `build-cuda/Release/lamina-infer.exe`. The benchmark reports the first token separately from later
tokens. A 6 GiB cache cap on a 16 GB card does not reproduce an 8 GB card's bandwidth or exact free memory.

The model is kept in `../Lamina-data/models`, outside the source tree. Downloads are explicit:

```sh
python setup.py --download
python setup.py --tokenizer
python setup.py --index
build/lamina-gguf ../Lamina-data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --check
```

On Windows, use `build/lamina-gguf.exe`. `setup.py --data-dir PATH` overrides the sibling data directory.
The pinned input is Unsloth's `UD-Q4_K_M` GGUF at revision
`a483e9e6cbd595906af30beda3187c2663a1118c`; setup checks its size, SHA-256, metadata, tensors, and offsets.
The tensor index goes under `../Lamina-data/packs`.

For text prompts, install `requirements.txt` and point the client at the build directory:

```sh
python -m pip install -r requirements.txt
python lamina.py chat --prompt "Hello" --max-tokens 32
python lamina.py serve --host 127.0.0.1 --port 8000
```

On Windows, pass `--engine build/Release/lamina-infer.exe` if your build is in a different location. The API
implements text-only, non-streaming `POST /v1/chat/completions`, greedy decoding with
`temperature: 0`, and `GET /v1/models`. Large prompts and generated responses will be very slow on CPU.

The original Strata setup and README are preserved under `ref/` solely as porting references. They still describe a
different model.

Developers continuing the port should start with [the developer handoff](docs/DEVELOPER_HANDOFF.md). It maps active
code, exact validation, the unported GPU work, and performance gates.
