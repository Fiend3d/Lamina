# Launcher image and sampled video validation

Windows, RTX 4060 8 GB, Ryzen 7 1700X, 64 GB RAM; pinned Qwen3.6 model,
local CUDA engine and CPU strata-vision encoder. Correctness smoke checks only;
no new throughput, latency or model-quality claim. Ornith runtime was not repeated.

Default CMake configuration and the Release `lamina-gguf` target passed:

```powershell
cmake -S . -B ../Lamina-data/launcher-hardware-free
cmake --build ../Lamina-data/launcher-hardware-free --config Release --target lamina-gguf
../Lamina-data/venv/Scripts/python.exe -m unittest discover -s tests/lamina
```

All 55 Python tests passed. These include saved media choices, image conversion,
frame ordering, EOF handling, frame limits and temporary-file cleanup.

Fixtures and raw logs are in sibling `Lamina-data/validation/launcher-media`.
The image fixture is a 560x280 red RGB image with white Arial 84 text `ALPHA`.
The two-second H.264/yuv420p MP4 contains this image for one second, followed by
a blue image with `BETA` for one second. The video filename contains a space.

```powershell
.\START-HERE.bat chat --vision on --image "..\Lamina-data\validation\launcher-media\ALPHA.png" --max-context 8192 --prompt "Read the word in this image. Reply with the word only." --max-tokens 32
.\START-HERE.bat chat --video "..\Lamina-data\validation\launcher-media\ordered clip.mp4" --video-frames 2 --video-interval 1 --max-context 8192 --prompt "Read the word in each frame and list the two words in time order. Reply with only those words." --max-tokens 32
.\START-HERE.bat --vision on --video-input on --no-preload --port 18047
```

Image output: `ALPHA`. Video output: `ALPHA BETA`, exit code 0.
The server's `/health` returned status `ok`, images `true`, compute mode `fast`,
and FP16 KV. The test stopped its server process tree afterwards.

The bundled imageio-ffmpeg 0.6.0 decoder extracted two frames and stopped at EOF
with a four-frame cap from MP4/MOV/MKV (H.264), AVI (MPEG-4) and WebM (VP9).
These tests verify those container/codec combinations, not every possible codec
or arbitrary damaged files. No audio, native video temporal positions, remote
video URLs or direct API video payloads are supported. Video uses the ordinary
multi-image inference path with timestamp text between frames.

Pi integration follow-up: the local Lamina provider now advertises text/image
input; `pi --list-models lamina` reports `images: yes`. Pi 1.1.0's installed
skill loader discovers the registered `lamina-video` skill without diagnostics.
The existing-server client's HTTP contract test brings the Python total to 56
passing tests. The skill wrapper also extracted both real MP4 fixture frames and
submitted their image data and timestamps to a local stub HTTP server; this
checks transport, not an additional inference run. The native recognition
evidence remains the image/video smoke checks above. Local pi JSON files were
backed up before modification.
