import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer

from lamina import chat_prompt, make_handler


class FakeEngine:
    def completion(self, messages, max_tokens):
        return ("Hello", 12, 1, "stop")


class ClientTest(unittest.TestCase):
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
            body = json.dumps({"messages": [{"role": "user", "content": "Hi"}],
                               "max_tokens": 2}).encode()
            with urllib.request.urlopen(urllib.request.Request(url, body,
                                                                {"Content-Type": "application/json"})) as response:
                result = json.load(response)
            self.assertEqual(result["choices"][0]["message"]["content"], "Hello")
            self.assertEqual(result["usage"]["total_tokens"], 13)
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(
                    url, json.dumps({"messages": [], "stream": True}).encode(),
                    {"Content-Type": "application/json"}))
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
