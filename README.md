# Lamina

Lamina is an in-progress native port of [Strata](https://github.com/Niko1221/Strata) for
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The source baseline is Strata commit
`6f32ec070f23ced9f50e704d854d775da52591ab`. Strata is MIT-licensed; its license and third-party notices remain
in this repository.

## Current status

The pinned Qwen3.6 GGUF can be inspected and validated. The CUDA inference engine, terminal chat, and local API are
**not yet ported**. The inherited Strata GPU code still implements Qwen3.8 and must not be used with Qwen3.6 weights.
See [port status](docs/LAMINA_PORT.md) for the implementation work still required.

## Build and inspect

The default build needs CMake 3.24+, a C++20 compiler, and no GPU toolkit:

```sh
cmake -S . -B build
cmake --build build --target lamina-gguf
```

The model is kept in `../Lamina-data/models`, outside the source tree. Downloads are explicit:

```sh
python setup.py --download
python setup.py --index
build/lamina-gguf ../Lamina-data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --check
```

On Windows, use `build/lamina-gguf.exe`. `setup.py --data-dir PATH` overrides the sibling data directory.
The pinned input is Unsloth's `UD-Q4_K_M` GGUF at revision
`a483e9e6cbd595906af30beda3187c2663a1118c`; setup checks its size, SHA-256, metadata, tensors, and offsets.
The tensor index goes under `../Lamina-data/packs`.

The original Strata setup and README are preserved under `ref/` solely as porting references. They still describe a
different model.
