---
name: lamina-video
description: Inspect, describe, or answer questions about a local video using timestamped sampled frames and the running Lamina image API. Use when the user asks to look at an MP4, MOV, MKV, AVI, or WebM video.
---

Use `scripts/ask_video.py` relative to this skill directory. It sends frames to
the existing Lamina server; it does not start another model instance.

The Lamina installation is three directories above this skill directory. Its
Python runtime is `<installation>/python/python.exe` for a portable installation
or `<installation>/../Lamina-data/venv/Scripts/python.exe` for a source checkout.
Quote all paths. For example, from Git Bash on this machine:

```bash
"D:/Projects/lamina/Lamina-data/venv/Scripts/python.exe" "D:/Projects/lamina/Lamina/.pi/skills/lamina-video/scripts/ask_video.py" --video "C:/media/clip.mp4" --prompt "Describe what happens in time order" --frames 4 --interval 1
```

The server must already be running with image input enabled. The default URL is
http://127.0.0.1:8000/v1; override with `--base-url` when the user's provider uses
another address. Never launch `START-HERE.bat chat` while the server is running:
that would load a second model. If the server is unavailable, report it and ask
the user to start `START-HERE.bat --vision on`.

Sampling starts at 0 seconds, uses the chosen interval and stops at EOF or the
frame cap. Defaults inspect only the first four seconds. Choose an interval and
1..16 frames appropriate to the requested portion; do not describe unseen parts
of a clip. The helper prints the model's answer; explain that it is based on
sampled frames. Audio, direct video attachments and native video temporal
positions are unsupported. MP4/H.264 is recommended; common MOV/MKV H.264,
AVI MPEG-4 and WebM VP9 also work. The running server selects CPU or GPU encoding (`START-HERE.bat --vision-device gpu` enables a CUDA-built encoder). CPU encoding can take minutes; let the command
finish rather than repeatedly launching it.

For still images, use pi's image attachment or image-reading tool directly;
the Lamina provider must advertise `input: ["text", "image"]`.
