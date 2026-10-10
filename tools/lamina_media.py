"""Local images and timestamped sampled video frames for Lamina's image API.

Video is a sequence of independent images, without audio or native video mRoPE.
"""
import base64
import math
import subprocess
import tempfile
from pathlib import Path


def image_content(path):
    from PIL import Image
    import io
    with Image.open(path) as picture:
        output = io.BytesIO()
        picture.convert("RGB").save(output, format="PNG")
    payload = output.getvalue()
    if len(payload) > 20 * 1024 * 1024:
        raise ValueError("image exceeds 20 MiB after PNG conversion")
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(payload).decode()}}


def video_content(path, data, frames=4, interval=1.0):
    """Sample at 0, interval, ... seconds, stopping at EOF or the frame cap."""
    if not 1 <= frames <= 16 or not math.isfinite(interval) or interval <= 0:
        raise ValueError("video frames must be 1..16 and interval must be positive")
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"video missing: {path}")
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise ValueError("video decoder missing; run START-HERE.bat setup --video-input on") from error
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    data = Path(data) / "tmp"
    data.mkdir(parents=True, exist_ok=True)
    content = [{"type": "text", "text":
                "These are sampled video frames in time order; audio is not included."}]
    with tempfile.TemporaryDirectory(prefix="video-", dir=data) as directory:
        for index in range(frames):
            timestamp = index * interval
            target = Path(directory) / "frame.png"
            if target.exists():
                target.unlink()
            result = subprocess.run([executable, "-nostdin", "-hide_banner", "-loglevel", "error",
                                     "-ss", str(timestamp), "-i", str(path), "-frames:v", "1",
                                     "-vf", "scale=1024:1024:force_original_aspect_ratio=decrease",
                                     "-y", str(target)], capture_output=True, text=True, timeout=120,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode:
                raise ValueError("video decode failed: " + result.stderr.strip()[-2000:])
            if not target.is_file():
                break
            content += [{"type": "text", "text": f"Frame {index + 1} at {timestamp:.3f} seconds:"},
                        image_content(target)]
    if len(content) == 1:
        raise ValueError("video contains no decodable frames")
    return content


def main(argv=None):
    """Send sampled frames to an existing server, without loading another engine."""
    import argparse
    import json
    import urllib.error
    import urllib.request
    parser = argparse.ArgumentParser(description="Ask a running Lamina server about a local video (sampled frames, no audio).")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    if not 1 <= args.max_tokens <= 131072:
        parser.error("max-tokens must be 1..131072")
    data = Path(__file__).resolve().parents[2] / "Lamina-data"
    try:
        content = [{"type": "text", "text": args.prompt}]
        content += video_content(args.video, data, args.frames, args.interval)
        body = json.dumps({"messages": [{"role": "user", "content": content}],
                           "max_tokens": args.max_tokens, "temperature": 0,
                           "enable_thinking": False}).encode()
        request = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions", body,
                                         {"Content-Type": "application/json"})
        print(f"Sending {sum(i['type'] == 'image_url' for i in content)} sampled frames to Lamina; see server logs for encoder device and progress.", flush=True)
        with urllib.request.urlopen(request, timeout=1800) as response:
            result = json.load(response)
        print(result["choices"][0]["message"]["content"])
        return 0
    except urllib.error.HTTPError as error:
        parser.exit(1, "Lamina request failed: " + error.read(4096).decode(errors="replace") + "\n")
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Video request failed: {error}. Start Lamina with image input enabled.\n")


if __name__ == "__main__":
    raise SystemExit(main())
