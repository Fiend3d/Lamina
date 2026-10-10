# Image embedding and conversation cache validation

Hardware: RTX 4060 8 GB, Ryzen 7 1700X, 64 GB RAM, Windows.
Model: Ornith-1.5-35B-A3B Q4_K_M with its packed MTP head. Native CUDA
engine uses fast compute, FP16 device KV, context 131072; CUDA vision uses
the model-specific BF16 projector, 2048 maximum image tokens, 8 CPU threads.
No native inference math or protocol changes were made.

Image embeddings use a per-engine 256 MiB RAM LRU keyed by SHA-256 of source
bytes. Repeated images do not start the encoder or unload the native model.
Conversation checkpoints include ordered image identities, text token IDs and
the existing native image mRoPE state. New GPU images still require the VRAM-safe
model/encoder handoff and full prefill. Eviction, edits and restarts can also
require full prefill. HTTP image URLs are fetched again to detect changed bytes.

Exact local commands from the repository root:

```powershell
& ../Lamina-data/venv/Scripts/python.exe ../Lamina-data/validation/launcher-media/validate_image_cache.py
& ../Lamina-data/venv/Scripts/python.exe -m unittest discover -s tests/lamina
cmake -S . -B ../Lamina-data/launcher-hardware-free
cmake --build ../Lamina-data/launcher-hardware-free --config Release --target lamina-gguf
```

The validation script initializes `Engine` with the saved model/encoder paths,
`cuda=True, max_context=131072, kv_cache='device', kv_type='f16',
compute_mode='fast', vision_gpu=True`, the saved MTP head and projector.
Local scripts, logs and raw JSON are under sibling
`Lamina-data/validation/launcher-media`, outside the source tree.

| Request | Prompt tokens | First content, s | Later tokens/s | Result |
| --- | ---: | ---: | ---: | --- |
| First image | 3185 | 23.05 | 2.51 | ALPHA |
| Identical image/history | 3185 | 0.18 | 3.13 | ALPHA |
| Appended follow-up | 3213 | 0.64 | 2.37 | ALPHA |
| Same follow-up, prefix cache disabled | 3213 | 14.07 | 2.25 | ALPHA |
| Changed image bytes | 1085 | 20.94 | 1.45 | BETA |
| Two cached images | 2146 | 12.32 | 9.35 | ALPHA BETA |
| Reversed images | 2146 | 12.74 | 5.25 | BETA ALPHA |
| Large image-containing conversation | 102485 | 261.10 | 1.82 | ALPHA |
| Identical large conversation | 102485 | 1.47 | 1.85 | ALPHA |

The repeated large request reused 102478 tokens and fed only seven new tokens.
The script asserts that native process identity is preserved on cache hits and
that the cached follow-up answer equals a full-prefill answer. Sampled total GPU
memory peaked at 7027 MiB, including desktop usage, sampled every 0.5 seconds.
These short OCR replies do not establish sustained generation speed or general
output equivalence. The large synthetic conversation checks allocation and reuse,
not the user's private pi conversation.

All 63 hardware-free Python tests passed, including embedding survival after
temporary-file cleanup, GPU handoff on misses/errors, changed URL bytes, RAM
eviction, changed image identity, prefix extension and appended-image indexing.
Default CMake configuration and the `lamina-gguf` build passed.

The server was then restarted on port 8000 with the current source and checked
through the same streaming API used by pi:

```powershell
.\START-HERE.bat --vision-device gpu
& ../Lamina-data/venv/Scripts/python.exe ../Lamina-data/validation/launcher-media/validate_image_cache_http.py
```

| API request | Prompt tokens | First content, s | Later tokens/s |
| --- | ---: | ---: | ---: |
| First short image | 1082 | 26.02 | 2.39 |
| Same image/history | 1082 | 0.22 | 3.03 |
| Image follow-up | 1103 | 0.58 | 2.30 |
| Two images, one newly encoded | 2146 | 30.60 | 7.62 |
| Same two images/history | 2146 | 0.19 | 9.14 |
| Reversed images, prefix invalidated | 2146 | 10.63 | 5.51 |

All API assertions passed. This also checks prefixes with fewer than 256 text
tokens: image-containing histories can reuse checkpoints even with short text.
The 7027 MiB sampled peak above applies to the direct-engine validation run.
