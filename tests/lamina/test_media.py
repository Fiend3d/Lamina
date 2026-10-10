import base64
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools import quickstart
from tools.lamina_media import image_content, video_content, main


class MediaTest(unittest.TestCase):
    def test_saved_choices_and_vision_disabled_launch(self):
        with TemporaryDirectory() as folder, patch.object(quickstart, "CONFIG", Path(folder) / "settings.json"), \
                patch.object(quickstart.sys.stdin, "isatty", return_value=False):
            config = quickstart.ask({"vision": False, "video_input": False}, False)
            self.assertFalse(config["vision"])
            with patch.object(quickstart, "vision_executable", return_value=Path("encoder.exe")):
                self.assertNotIn("--vision", quickstart.engine_options(config, 8192))
            config["video_input"] = True
            self.assertTrue(quickstart.ask(config, False)["vision"])

    def test_gpu_launcher_selects_cuda_encoder_and_passes_flag(self):
        with patch.object(quickstart, "selected_vision", return_value=Path("cuda-vision.exe")):
            options = quickstart.engine_options({"vision": True, "vision_device": "gpu"}, 8192)
        self.assertIn("--vision-gpu", options)
        self.assertEqual(options[options.index("--vision-engine") + 1], "cuda-vision.exe")

    def test_image_is_rgb_png(self):
        from PIL import Image
        with TemporaryDirectory() as folder:
            path = Path(folder) / "picture.jpg"
            Image.new("RGB", (20, 10), "red").save(path)
            item = image_content(path)
            payload = base64.b64decode(item["image_url"]["url"].split(",", 1)[1])
            with Image.open(io.BytesIO(payload)) as image:
                self.assertEqual((image.format, image.mode, image.size), ("PNG", "RGB", (20, 10)))

    def test_video_sampling_order_eof_and_cleanup(self):
        from PIL import Image
        from subprocess import CompletedProcess
        commands = []
        def decode(command, **kwargs):
            commands.append(command)
            if float(command[command.index("-ss") + 1]) < 2:
                Image.new("RGB", (20, 10), "red" if len(commands) == 1 else "blue").save(command[-1])
            return CompletedProcess(command, 0, "", "")
        with TemporaryDirectory() as folder:
            path = Path(folder) / "clip with spaces.mp4"; path.touch()
            with patch("imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg"), \
                    patch("tools.lamina_media.subprocess.run", side_effect=decode):
                items = video_content(path, Path(folder), frames=4)
            self.assertEqual([i["text"] for i in items if i.get("text", "").startswith("Frame")],
                             ["Frame 1 at 0.000 seconds:", "Frame 2 at 1.000 seconds:"])
            self.assertEqual(len(commands), 3)
            self.assertEqual(list((Path(folder) / "tmp").iterdir()), [])

    def test_video_rejects_invalid_limits(self):
        for frames, interval in ((0, 1), (17, 1), (4, 0), (4, float("nan"))):
            with self.assertRaises(ValueError):
                video_content(Path("missing.mp4"), Path("unused"), frames, interval)

    def test_client_submits_image_api_request_and_prints_answer(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "ALPHA BETA"}}]}).encode())
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        frames = [{"type": "text", "text": "Frame 1 at 0.000 seconds:"},
                  {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}]
        try:
            output = io.StringIO()
            with patch("tools.lamina_media.video_content", return_value=frames), \
                    patch("sys.stdout", output):
                result = main(["--video", "clip.mp4", "--prompt", "Read the words", "--max-tokens", "32",
                               "--base-url", f"http://127.0.0.1:{server.server_port}/v1/"])
            self.assertEqual(result, 0)
            self.assertIn("ALPHA BETA", output.getvalue())
            path, body = received[0]
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(body["messages"][0]["content"][1:], frames)
            self.assertEqual(body["max_tokens"], 32)
            self.assertFalse(body["enable_thinking"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
