# Lamina

Lamina is a native port of [Strata](https://github.com/Niko1221/Strata) for
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The source baseline is Strata commit
`6f32ec070f23ced9f50e704d854d775da52591ab`. Strata is MIT-licensed; its license and third-party notices remain
in this repository.

## Current status

The Qwen3.6 execution graph now has an experimental scalar C++ path. It reads the published GGUF directly and has
DeltaNet, full attention, routed experts, a text CLI, and a limited OpenAI chat endpoint. Its output has **not yet
been checked against reference logits**, so it should not be treated as a validated inference engine. It dequantizes
matrix rows on the CPU for each token and will be very slow. The CUDA path remains outstanding. The inherited Strata
GPU code still implements Qwen3.8 and is not used by Lamina.

## Build and inspect

The default build needs CMake 3.24+, a C++20 compiler, and no GPU toolkit:

```sh
cmake -S . -B build
cmake --build build --target lamina-gguf lamina-infer
```

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
python lamina.py chat --engine build/lamina-infer --prompt "Hello" --max-tokens 32
python lamina.py serve --engine build/lamina-infer --host 127.0.0.1 --port 8000
```

On Windows the executable is `build/lamina-infer.exe` (or `build-lamina/lamina-infer.exe` for this workspace's
configured build). The API implements text-only, non-streaming `POST /v1/chat/completions`, greedy decoding with
`temperature: 0`, and `GET /v1/models`. Large prompts and generated responses will be very slow on CPU.

The original Strata setup and README are preserved under `ref/` solely as porting references. They still describe a
different model.
