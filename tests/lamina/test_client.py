import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch

from lamina import Engine, chat_prompt, make_handler


class FakeEngine:
    def completion(self, messages, max_tokens):
        return ("Hello", 12, 1, "stop")


class ClientTest(unittest.TestCase):
    def test_prompt_tokens_skip_unused_logits(self):
        class Tokenizer:
            def encode(self, prompt, add_special_tokens):
                return type("Encoding", (), {"ids": [1, 2]})()

            def decode(self, ids, skip_special_tokens):
                return "Hello" if ids == [5] else ""

        class Input:
            def __init__(self):
                self.writes = []

            def write(self, value):
                self.writes.append(value)

            def flush(self):
                pass

            def close(self):
                pass

        class Process:
            def __init__(self):
                self.stdin = Input()
                self.stdout = self
                self.stderr = self

            def readline(self):
                value = self.stdin.writes[-1]
                return (
                    ".\n"
                    if value.startswith("+")
                    else "5\n"
                    if value == "2\n"
                    else "248046\n"
                )

            def poll(self):
                return 0

            def wait(self, timeout):
                return 0

        engine = object.__new__(Engine)
        engine.tokenizer = Tokenizer()
        engine.model = Path("model.gguf")
        engine.executable = Path("lamina-infer")
        engine.cuda = False
        process = Process()
        with patch("lamina.subprocess.Popen", return_value=process):
            result = engine.completion([{"role": "user", "content": "Hi"}], 2)
        self.assertEqual(result, ("Hello", 2, 1, "stop"))
        self.assertEqual(process.stdin.writes, ["+1\n", "2\n", "5\n"])

    def test_cuda_option_reaches_native_engine(self):
        engine = object.__new__(Engine)
        engine.tokenizer = type(
            "Tokenizer",
            (),
            {
                "encode": lambda self, *args, **kwargs: type(
                    "Encoding", (), {"ids": [1]}
                )(),
                "decode": lambda self, *args, **kwargs: "",
            },
        )()
        engine.model = Path("model.gguf")
        engine.executable = Path("lamina-infer")
        engine.cuda = True
        with patch("lamina.subprocess.Popen") as start:
            process = start.return_value
            process.stdin = type(
                "Input",
                (),
                {
                    "write": lambda self, value: None,
                    "flush": lambda self: None,
                    "close": lambda self: None,
                },
            )()
            process.stdout.readline.return_value = "248046\n"
            process.poll.return_value = 0
            engine.completion([{"role": "user", "content": "Hi"}], 1)
        self.assertEqual(
            start.call_args.args[0],
            ["lamina-infer", "model.gguf", "--cuda", "--interactive"],
        )

    def test_qwen_text_template(self):
        self.assertEqual(
            chat_prompt([{"role": "user", "content": "Hi"}]),
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        with self.assertRaises(ValueError):
            chat_prompt([{"role": "user", "content": [{"type": "image"}]}])

    def test_chat_completions_endpoint(self):
        server = HTTPServer(("127.0.0.1", 0), make_handler(FakeEngine()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
            body = json.dumps(
                {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2}
            ).encode()
            with urllib.request.urlopen(
                urllib.request.Request(url, body, {"Content-Type": "application/json"})
            ) as response:
                result = json.load(response)
            self.assertEqual(result["choices"][0]["message"]["content"], "Hello")
            self.assertEqual(result["usage"]["total_tokens"], 13)
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(
                    urllib.request.Request(
                        url,
                        json.dumps({"messages": [], "stream": True}).encode(),
                        {"Content-Type": "application/json"},
                    )
                )
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
            for invalid in ([], {"messages": [], "top_p": 0.9}):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(
                        urllib.request.Request(
                            url,
                            json.dumps(invalid).encode(),
                            {"Content-Type": "application/json"},
                        )
                    )
                self.assertEqual(error.exception.code, 400)
                error.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
