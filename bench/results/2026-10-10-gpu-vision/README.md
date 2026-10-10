# CUDA vision enablement

**Superseded memory lifecycle:** the resident-encoder setup below later failed
at a 109022-token pi text request (VRAM working set exceeded available headroom).
The initial short-image smoke checks did not establish long-context correctness.
GPU vision now stops the text model before encoding, releases the encoder after
all frames, and reloads the text model. Text-only requests never start vision.
The historical timings below describe the resident setup, not the corrected
lifecycle. GPU encoding remains enabled; context capacity remains 131072.
See the [109K/129K regression record](../2026-10-10-gpu-vision-memory/README.md)
for validation of the corrected lifecycle.


Windows, RTX 4060 8 GB, Ryzen 7 1700X, 64 GB RAM. Ornith Q4_K_M with its
pinned BF16 projector, fast computation, FP16 device KV, MTP and 131072 context
capacity. The encoder is built for SM89 from the pinned llama.cpp checkout;
flash attention is disabled, matching the CPU vision configuration.

Commands from the repository root:

```powershell
../Lamina-data/venv/Scripts/python.exe -m tools.build_windows --vision-gpu --vision-only --cuda-arch 89 --jobs 8
.\START-HERE.bat --vision-device gpu
../Lamina-data/venv/Scripts/python.exe ../Lamina-data/validation/launcher-media/validate_gpu.py
```

The local validation script, raw `gpu-server.log`, `gpu-validation.log` and
`gpu-runtime.json` are in sibling `Lamina-data/validation/launcher-media`.
Fixtures are the earlier red ALPHA / blue BETA images and two-second H.264 MP4.
The additional user MOV is kept outside the source tree. Its request used a
16-frame cap and one-second interval, yielding six actual frames before EOF;
it is not a test of sixteen actual image frames. The answer was capped at 128
tokens with thinking disabled and temperature zero. This is a local smoke
measurement, not a general video-quality benchmark or matched CPU/GPU comparison.

The launcher preload warmed the GPU encoder before the language model allocated
its expert cache. `/health` reports `vision_device: gpu`. Both processes remain
resident and the server is available to pi on port 8000.

| Request | First content, seconds | Later tokens/s | Total, seconds | Result |
| --- | ---: | ---: | ---: | --- |
| Single image OCR | 10.26 | 2.81 | 10.62 | ALPHA |
| Two-frame word order | 16.47 | 9.78 | 16.78 | ALPHA BETA |
| User MOV, six frames | 42.90 | 19.97 | 46.82 | Nonempty chronological description |
| Text after video | 0.67 | 13.42 | 0.97 | GPU vision is ready. |

First content includes encoding and prefill. Later rates exclude the first
content token; two/four/five-token outputs are too short for a steady decode
claim. The MOV description used 79 output tokens. Its first frame encoded in
2.23 seconds, subsequent frames in 1.01–1.03 seconds. Peak **total** GPU memory
observed by nvidia-smi at approximately 0.5-second intervals was **7869 MiB**,
including desktop use. This leaves little spare VRAM on this 8 GB card.

Default CMake configuration and the Release `lamina-gguf` target passed. The
CPU and CUDA encoders both compiled, and all 58 Python tests passed, including
GPU encoder startup before the native model and launcher device forwarding.
GPU requests reject an encoder without a GPU backend and fail startup if GPU
warm-up fails; they do not silently select CPU execution. No model layer math
was changed and no CPU/GPU embedding equivalence is claimed.

CPU mode remains selectable with `START-HERE.bat --vision-device cpu`. The
portable release continues to contain the CPU encoder; this CUDA enablement is
for the source checkout and does not publish or replace release assets.
