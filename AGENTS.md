# Lamina development

Start with `docs/DEVELOPER_HANDOFF.md`. It maps the active Lamina files, exact
build and numerical checks, performance evidence, and the order of remaining
GPU work. `README.md` is the user quick start; `docs/LAMINA_PORT.md` records
port status. Keep these aligned with implementation changes.

Lamina targets Qwen3.6-35B-A3B. The pinned GGUF contract is in `tools/lamina_model.py`; its engine architecture name
is `qwen35moe`. The source baseline is Strata commit `6f32ec070f23ced9f50e704d854d775da52591ab`.

The inherited CUDA engine in `src/core/`, `src/kernels/`, and `src/prefill/` still assumes Strata's Qwen3.8 model.
Do not advertise it as Lamina inference or feed it Qwen3.6 weights until its layer math, weights, cache, and routing
are ported and checked against a reference implementation. Preserve upstream license notices and attribution.

Model data belongs in `../Lamina-data`, never in this source tree. Hardware-free checks: configure CMake with its
defaults, build `lamina-gguf`, and run `python -m unittest discover -s tests/lamina`.
For layer-math changes, also run `python -m tools.reference_prefix --layers 40`
against the pinned model. For performance claims, report the hardware, exact
command, first-token time, later tokens/s, and peak VRAM where applicable.
Successful compilation alone does not establish CUDA runtime correctness or speed.
