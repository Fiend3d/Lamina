# Long text prefill around images

Hardware: RTX 4060 8 GB, Ryzen 7 1700X, 64 GB RAM, Windows.
Model: Ornith-1.5-35B-A3B Q4_K_M and its packed MTP head, fast CUDA compute,
FP16 device KV, context 131072, prefill chunk 2048. GPU vision uses the pinned
Ornith BF16 projector and 2048 maximum image tokens. No native inference math
or protocol changes were made.

Image-containing cached prefixes previously sent every text chunk as BATCH,
running all 40 layers before moving to the next chunk. They now accumulate
text up to each IMAGE boundary and use the existing layer-major PROMPT path
for spans longer than the configured chunk. Short spans still use BATCH.
This reduces repeated weight uploads while preserving image order and mRoPE.
Sampling remains configured after the prefix, so ignored PROMPT results do
not consume the request's seeded sampling state. Requests that cannot use a
prefix checkpoint retain their previous mixed-image path.

Exact commands from the repository root:

```powershell
& ../Lamina-data/venv/Scripts/python.exe ../Lamina-data/validation/launcher-media/validate_image_prefill.py
& ../Lamina-data/venv/Scripts/python.exe -m unittest discover -s tests/lamina
cmake -S . -B ../Lamina-data/launcher-hardware-free
cmake --build ../Lamina-data/launcher-hardware-free --config Release --target lamina-gguf
```

The local benchmark initializes Engine with `cuda=True, max_context=131072,
kv_cache='device', kv_type='f16', compute_mode='fast', vision_gpu=True`,
the saved executable/model/projector paths and MTP head. It reconstructs the
previous Python feed loop in memory for the baseline. Both variants receive
the same previously encoded ALPHA/BETA images, avoiding encoder time in the
comparison; each starts its own native process. Timing includes native startup.
The 32047-token request contains text before, between and after two images.

| Request | First content, s | Later tokens/s |
| --- | ---: | ---: |
| Previous mixed prefix | 90.32 | 9.27 |
| Optimized mixed prefix | 63.25 | 8.74 |
| Previous cached repeat | 0.49 | 10.73 |
| Optimized cached repeat | 0.48 | 11.17 |
| Previous appended follow-up | 0.94 | 7.87 |
| Optimized appended follow-up | 0.97 | 7.63 |
| Optimized 102485-token image request | 182.19 | 1.81 |
| Optimized 102485-token cached repeat | 1.35 | 2.07 |

First-content time fell about 30% in this single paired comparison. Both initial
answers were exactly `ALPHA\nBETA`; cached repeats matched their initial answers,
and both follow-ups returned `ALPHA BETA`. These short replies do not establish
sustained decode speed or general numerical/output equivalence. The benchmark
uses synthetic text, rather than the user's private pi conversation.

The large request used the identical 102485-token fixture from the earlier
[image-cache record](../2026-10-10-image-cache/README.md), which reached first
content in 261.10 s with the chunk-major image path. The optimized path took
182.19 s (about 30% less waiting), and repeated it in 1.35 s while reusing
102478 tokens. This large comparison is against the preceding recorded run,
not a same-run paired baseline; initial encoder/model warm-up differs. Both
returned ALPHA. GPU memory was sampled every 0.5 seconds, including desktop
usage; sampled peak was 6712 MiB for this validation run.

All 64 Python tests passed, including text spans on both sides of an image,
command ordering before seeded sampling, checkpoint extension, image cache
eviction and changed image identity. Default CMake and lamina-gguf build passed.
Raw scripts, logs and JSON live in sibling
`Lamina-data/validation/launcher-media`, outside the source tree.
