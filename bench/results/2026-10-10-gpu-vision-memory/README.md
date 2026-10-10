# Release GPU vision before text inference

The initial GPU-vision implementation kept the encoder resident alongside the
native model. It passed short image/video smoke tests but later failed in pi at
109022 prompt tokens, while extending a text prefix: allocation requested 10 MiB
with zero free headroom. The encoder was idle. The original failure is recorded
in sibling `Lamina-data/validation/launcher-media/gpu-server.log`.

GPU image/video requests now stop the native model, encode all images with one
GPU encoder process, close that process, and reload the native model to infer
from the resulting embeddings. Cleanup also runs when encoding fails. Text-only
startup never loads vision; append-only text history still reuses its prefix.
The tradeoff is model reload time for image requests. CPU mode is unchanged.

Validation: Windows, RTX 4060 8 GB / Ryzen 7 1700X / 64 GB RAM, Ornith Q4_K_M,
BF16 projector, fast compute, FP16 device KV, MTP, 131072 context capacity.
Context was not reduced and KV remained on the GPU.

Commands from the repository root:

```powershell
.\START-HERE.bat --vision-device gpu --port 18047
../Lamina-data/venv/Scripts/python.exe ../Lamina-data/validation/launcher-media/validate_gpu_shared.py
```

The local validation script, `gpu-shared-runtime.json`, `gpu-shared-validation.log`
and `gpu-shared-test-server.log` are under sibling `Lamina-data/validation/launcher-media`.
The synthetic long prompt puts `LAMINA847263` near its beginning and repeats a
rainfall-observation sentence; the next user message extends that exact history.
Both retrieval requests return the code. This tests the allocation boundary and
history growth, not the user's private pi conversation or all possible workloads.

| Request | Prompt tokens | First content, seconds | Later tokens/s | Result |
| --- | ---: | ---: | ---: | --- |
| Two-frame video after text preload | 2183 | 30.99 | 7.48 | ALPHA BETA |
| Large fresh text prompt | 109014 | 196.30 | 13.77 | LAMINA847263 |
| Append-only history | 129883 | 62.08 | 12.46 | LAMINA847263; 109007 prefix tokens reused |
| GPU image after large text | 1082 | 24.88 | 2.52 | ALPHA |

The short outputs are not steady decode benchmarks. First-content times include
encoding, model reload where applicable, and prefill. The full sequence passed,
including a final text request after the image. Peak total GPU use sampled by
nvidia-smi approximately every 0.5 seconds was **7137 MiB**, including desktop
usage. There was no vision process during either long text request.

All 59 Python tests passed, including cleanup after failed GPU encoding and
ensuring text startup does not construct an encoder. Default CMake configuration
and the Release `lamina-gguf` build passed. Native layer math was unchanged.
The corrected server was restarted on the normal port 8000 with GPU vision
selected and the original 128K context setting.
