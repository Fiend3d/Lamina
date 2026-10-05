# Lamina development

Lamina targets Qwen3.6-35B-A3B. The pinned GGUF contract is in `tools/lamina_model.py`; its engine architecture name
is `qwen35moe`. The source baseline is Strata commit `6f32ec070f23ced9f50e704d854d775da52591ab`.

The inherited CUDA engine in `src/core/`, `src/kernels/`, and `src/prefill/` still assumes Strata's Qwen3.8 model.
Do not advertise it as Lamina inference or feed it Qwen3.6 weights until its layer math, weights, cache, and routing
are ported and checked against a reference implementation. Preserve upstream license notices and attribution.

Model data belongs in `../Lamina-data`, never in this source tree. Hardware-free checks: configure CMake with its
defaults, build `lamina-gguf`, and run `python -m unittest discover -s tests/lamina`.
