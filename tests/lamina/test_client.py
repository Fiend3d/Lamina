import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch
from lamina import Engine, chat_prompt, make_handler
from tools.lamina_chat import validate_options, split_reasoning, hold_suffix, tool_message


class FakeEngine:
    max_context = 32768
    vision_engine = None
    def validate(self, messages, options):
        validate_options(options, self.max_context)
        if not isinstance(messages, list) or not messages: raise ValueError("messages required")
    def events(self, messages, max_tokens, **options):
        yield {"delta": {"content": "Hello"}}
        yield {"finish_reason": "stop", "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13}}


class ClientTest(unittest.TestCase):
    def engine(self, directory, tokens=(1, 2)):
        engine = object.__new__(Engine)
        engine.tokenizer = type("Tokenizer", (), {
            "encode": lambda self, *a, **kw: type("Encoding", (), {"ids": list(tokens)})(),
            "decode": lambda self, ids, **kw: "Hello" if ids else ""})()
        engine.template = type("Template", (), {"render": lambda self, **kw: "prompt"})()
        engine.model = Path(directory) / "model"
        engine.executable = Path(directory) / "engine"
        engine.data = Path(directory)
        engine.max_context = 32768; engine.prefill_chunk = 32
        engine.kv_type = "f32"; engine.kv_cache = "auto"; engine.vram_limit_mb = 0; engine.cuda = True
        engine.native = engine.vision = engine.vision_engine = None
        engine.lock = threading.Lock(); engine.max_image_tokens = 1024; engine.cpu_threads = 8
        return engine

    def test_resident_process_reset_and_unused_logits(self):
        class Native:
            def __init__(self): self.commands = []
            def command(self, value, acknowledgement=False):
                self.commands.append(value)
                if acknowledgement: return None
                return 5 if value.startswith("PREFILL") else 248046
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory); native = Native()
            with patch("tools.lamina_chat.NativeProcess", return_value=native) as start:
                for _ in range(2):
                    self.assertEqual(engine.completion([{"role": "user", "content": "Hi"}], 2), ("Hello", 2, 1, "stop"))
                self.assertEqual(start.call_count, 1)
                self.assertIn("--cuda", start.call_args.args[0])
            self.assertEqual(native.commands, ["RESET", "SAMPLE 0 1 20 0", "PREFILL 1 2", "5"] * 2)

    def test_text_startup_does_not_start_gpu_encoder(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            engine.vision_gpu = True
            engine.vision_engine = Path(directory) / "vision.exe"
            engine.vision_projector = Path(directory) / "projector.gguf"
            engine._cache_supported = False
            calls = []
            def vision(*args, **kwargs):
                calls.append("vision")
                self.assertTrue(kwargs["gpu"])
                return Mock()
            def native(*args, **kwargs):
                calls.append("native")
                return Mock()
            with patch("tools.lamina_chat.Vision", side_effect=vision), \
                    patch("tools.lamina_chat.NativeProcess", side_effect=native):
                engine._start_native()
            self.assertEqual(calls, ["native"])

    def test_gpu_image_encoding_releases_model_then_encoder_even_on_error(self):
        from unittest.mock import Mock
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                engine = self.engine(directory)
                engine.vision_gpu = True
                engine.vision_engine = Path(directory) / "vision.exe"
                engine.vision_projector = Path(directory) / "projector.gguf"
                calls = []
                native = Mock()
                native.close.side_effect = lambda: calls.append("model-close")
                engine.native = native
                encoder = Mock()
                def encode(*args):
                    calls.append("encode")
                    if fail:
                        raise RuntimeError("encoding failed")
                    destination = Path(directory) / "frame.sve"
                    destination.write_bytes(b"embedding")
                    return destination, 1
                encoder.encode_payload.side_effect = encode
                encoder.close.side_effect = lambda: calls.append("encoder-close")
                def start(*args, **kwargs):
                    calls.append("encoder-start")
                    return encoder
                messages = [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}]}]
                with patch("tools.lamina_chat.Vision", side_effect=start):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "encoding failed"):
                            engine._encode_images(messages, Path(directory))
                    else:
                        self.assertEqual(len(engine._encode_images(messages, Path(directory))), 1)
                self.assertEqual(calls, ["model-close", "encoder-start", "encode", "encoder-close"])
                self.assertIsNone(engine.native)
                self.assertIsNone(engine.vision)

    def test_repeated_images_keep_native_and_restore_embedding_after_temp_cleanup(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            engine.vision_gpu = True
            engine.vision_engine = Path(directory) / "vision"
            engine.vision_projector = Path(directory) / "projector"
            encoder = Mock()
            def encode(payload, folder, index):
                destination = folder / f"{index}.sve"
                destination.write_bytes(payload + b"embedding")
                return destination, 1024
            encoder.encode_payload.side_effect = encode
            def message(data):
                return [{"role": "user", "content": [{"type": "image_url",
                    "image_url": "data:image/png;base64," + data}]}]
            with patch("tools.lamina_chat.Vision", return_value=encoder) as start:
                with tempfile.TemporaryDirectory() as first:
                    engine._encode_images(message("eA=="), Path(first))
                native = Mock(); engine.native = native; engine._cache_prefix = [1, 2]
                with tempfile.TemporaryDirectory() as second:
                    image, count = engine._encode_images(message("eA=="), Path(second))[0]
                    self.assertEqual(image.read_bytes(), b"xembedding")
                    self.assertEqual(count, 1024)
                    self.assertEqual(engine._cache_prefix, [1, 2])
                    native.close.assert_not_called()
                    self.assertEqual(start.call_count, 1)
                    engine._encode_images(message("eQ=="), Path(second))
                    native.close.assert_called_once()
                    self.assertEqual(start.call_count, 2)

    def test_image_prefix_restore_extension_and_changed_image_reset(self):
        from tools.lamina_chat import IMAGE_ID
        class Native:
            def __init__(self): self.commands = []
            def command(self, value, acknowledgement=False):
                self.commands.append(value)
                return None if acknowledgement else 5
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory); native = Native(); engine.native = native
            engine._cache_supported = True; engine._cache_prefix = []
            prefix = [1] * 256 + [IMAGE_ID, 2]
            key = ["image-a"]
            def prepare(*args):
                engine._prepared_prefix = prefix.copy()
                engine._prepared_signature = [key[0] if t == IMAGE_ID else t for t in prefix]
                return prefix + [300], [(Path(directory) / "a.sve", 1024)], len(prefix) + 1024
            engine._prepare = prepare
            for _ in range(2): list(engine.events([{"role": "user", "content": "Hi"}], 1))
            prefix.extend([3, 4])
            list(engine.events([{"role": "user", "content": "Hi"}], 1))
            self.assertEqual(sum(c.startswith("IMAGE ") for c in native.commands), 1)
            self.assertEqual(native.commands.count("RESET"), 1)
            self.assertEqual(native.commands.count("RESTORE_PREFIX"), 2)
            key[0] = "image-b"
            list(engine.events([{"role": "user", "content": "Hi"}], 1))
            self.assertEqual(native.commands.count("RESET"), 2)
            self.assertEqual(sum(c.startswith("IMAGE ") for c in native.commands), 2)

    def test_image_cache_eviction_and_changed_http_content(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            encoder = Mock(); engine.vision = encoder
            def encode(payload, folder, index):
                destination = folder / f"{index}.sve"
                destination.write_bytes(payload * 8)
                return destination, 1
            encoder.encode_payload.side_effect = encode
            messages = [{"role": "user", "content": [{"type": "image_url",
                         "image_url": "https://example.test/image.png"}]}]
            with patch("tools.lamina_chat.IMAGE_CACHE_BYTES", 8), \
                    patch("tools.lamina_chat.image_payload", side_effect=[b"a", b"a", b"b", b"a"]):
                for _ in range(4): engine._encode_images(messages, Path(directory))
            # Same URL/same bytes hits; changed content misses; evicted content
            # must be encoded again, and total cached bytes stay bounded.
            self.assertEqual(encoder.encode_payload.call_count, 3)
            self.assertEqual(engine._image_cache_bytes, 8)
            self.assertEqual(len(engine._image_cache), 1)

    def test_image_prefix_extension_feeds_only_appended_image(self):
        from tools.lamina_chat import IMAGE_ID
        class Native:
            def __init__(self): self.commands = []
            def command(self, value, acknowledgement=False):
                self.commands.append(value)
                return None if acknowledgement else 5
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory); native = Native(); engine.native = native
            engine._cache_supported = True; engine._cache_prefix = []
            prefix = [1] * 256 + [IMAGE_ID, 2]
            images = [(Path(directory) / "first.sve", 1024)]
            def prepare(*args):
                engine._prepared_prefix = prefix.copy()
                keys = iter(["first", "second"])
                engine._prepared_signature = [next(keys) if t == IMAGE_ID else t for t in prefix]
                return prefix + [300], images.copy(), len(prefix) + sum(n - 1 for _, n in images) + 1
            engine._prepare = prepare
            list(engine.events([{"role": "user", "content": "Hi"}], 1))
            prefix.extend([3, IMAGE_ID, 4])
            images.append((Path(directory) / "second.sve", 1024))
            list(engine.events([{"role": "user", "content": "Hi"}], 1))
            self.assertEqual(native.commands.count("RESET"), 1)
            self.assertEqual([c for c in native.commands if c.startswith("IMAGE ")],
                             ["IMAGE " + str(path) for path, _ in images])

    def test_long_image_prefix_uses_prompt_on_each_side_of_image_before_sampling(self):
        from tools.lamina_chat import IMAGE_ID
        class Native:
            def __init__(self): self.commands = []
            def command(self, value, acknowledgement=False):
                self.commands.append(value)
                return None if acknowledgement else 5
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory); native = Native(); engine.native = native
            engine._cache_supported = True; engine._cache_prefix = []
            prefix = [1] * 256 + [IMAGE_ID] + [2] * 256
            def prepare(*args):
                engine._prepared_prefix = prefix.copy()
                engine._prepared_signature = ["image" if t == IMAGE_ID else t for t in prefix]
                return prefix + [300], [(Path(directory) / "a.sve", 1024)], len(prefix) + 1024
            engine._prepare = prepare
            list(engine.events([{"role": "user", "content": "Hi"}], 1, temperature=0.7, seed=42))
            prompts = [c for c in native.commands if c.startswith("PROMPT ")]
            self.assertEqual(prompts, ["PROMPT 32 " + " ".join([str(t)] * 256) for t in (1, 2)])
            self.assertLess(native.commands.index(prompts[0]), native.commands.index("IMAGE " + str(Path(directory) / "a.sve")))
            self.assertLess(native.commands.index(prompts[1]), native.commands.index("CACHE_PREFIX"))
            self.assertLess(native.commands.index("CACHE_PREFIX"), native.commands.index("SAMPLE 0.7 1 20 42"))

    def test_fp16_mode_reaches_native_process(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            engine.kv_type = "f16"
            with patch("tools.lamina_chat.NativeProcess") as start:
                engine._start_native()
                command = start.call_args.args[0]
                self.assertEqual(command[command.index("--kv-type") + 1], "f16")

    def test_fast_mode_reaches_native_process(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            engine.compute_mode = "fast"
            with patch("tools.lamina_chat.NativeProcess") as start:
                engine._start_native()
                command = start.call_args.args[0]
                self.assertEqual(command[command.index("--compute-mode") + 1], "fast")

    def test_long_text_uses_layer_major_prompt(self):
        class Native:
            def __init__(self): self.commands=[]
            def command(self,value,acknowledgement=False):
                self.commands.append(value)
                return None if acknowledgement else 5
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            engine=self.engine(directory,tokens=tuple(range(64)))
            native=Native();engine.native=native
            self.assertEqual(engine.completion([{"role":"user","content":"Hi"}],1)[0],"Hello")
            self.assertEqual(native.commands,["RESET","SAMPLE 0 1 20 0","PROMPT 32 "+" ".join(map(str,range(64)))])

    def test_prefix_restored_and_changed_prefix_invalidated(self):
        class Native:
            def __init__(self): self.commands = []
            def command(self, value, acknowledgement=False):
                self.commands.append(value)
                return None if acknowledgement else 5
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            native = Native(); engine.native = native
            engine._cache_supported = True; engine._cache_prefix = []
            prefix = list(range(256))
            def prepare(*args):
                engine._prepared_prefix = prefix.copy()
                return prefix + [300], [], 257
            engine._prepare = prepare
            for _ in range(2):
                list(engine.events([{"role": "user", "content": "Hi"}], 1))
            self.assertEqual(native.commands.count("RESET"), 1)
            self.assertEqual(native.commands.count("CACHE_PREFIX"), 1)
            self.assertEqual(native.commands.count("RESTORE_PREFIX"), 1)
            self.assertEqual(native.commands.count("PREFILL 300"), 2)
            self.assertLess(native.commands.index("CACHE_PREFIX"),
                            native.commands.index("SAMPLE 0 1 20 0"),
                            "ignored long-prefix token must precede request RNG configuration")
            prefix[0] = 999
            list(engine.events([{"role": "user", "content": "Changed"}], 1))
            self.assertEqual(native.commands.count("RESET"), 2)
            self.assertEqual(native.commands.count("CACHE_PREFIX"), 2)
            engine._stop_native()
            self.assertEqual(engine._cache_prefix, [])

    def test_prefix_checkpoint_advances_when_conversation_grows(self):
        class Native:
            def __init__(self): self.commands = []
            def command(self, value, acknowledgement=False):
                self.commands.append(value)
                return None if acknowledgement else 5
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory); native = Native(); engine.native = native
            engine._cache_supported = True; engine._cache_prefix = []
            state = {"ids": list(range(66)), "prefix": list(range(64))}
            def prepare(*args):
                engine._prepared_prefix = list(state["prefix"])
                return list(state["ids"]), [], len(state["ids"])
            engine._prepare = prepare
            list(engine.events([{"role": "user", "content": "a"}], 1))
            # the next turn appends to the conversation: the checkpoint extends
            state["prefix"] = list(range(70)); state["ids"] = list(range(70)) + [7, 8, 9]
            list(engine.events([{"role": "user", "content": "b"}], 1))
            self.assertEqual(native.commands.count("RESET"), 1)
            self.assertEqual(native.commands.count("RESTORE_PREFIX"), 1)
            self.assertEqual(native.commands.count("CACHE_PREFIX"), 2)
            restore = native.commands.index("RESTORE_PREFIX")
            self.assertIn("CACHE_PREFIX", native.commands[restore:])  # restore, then re-cache the longer prefix

    def test_fast_rejected_for_cpu_before_loading_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("model", "tokenizer", "engine")]
            for path in paths: path.touch()
            with self.assertRaisesRegex(ValueError, "fast.*CUDA"):
                Engine(*paths, cuda=False, compute_mode="fast")

    def test_fp16_rejected_for_cpu_before_loading_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("model", "tokenizer", "engine")]
            for path in paths: path.touch()
            with self.assertRaisesRegex(ValueError, "f16 with CUDA"):
                Engine(*paths, cuda=False, kv_type="f16")

    def test_generation_budget_does_not_feed_an_extra_token(self):
        class Native:
            def command(self, value, acknowledgement=False):
                if value == "5": raise AssertionError("extra decode")
                return None if acknowledgement else 5
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory); engine.native = Native()
            self.assertEqual(engine.completion([{"role": "user", "content": "Hi"}], 1)[-1], "length")

    def test_prompt_overflow_before_native_start(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory); engine.max_context = 2
            with patch("tools.lamina_chat.NativeProcess") as start:
                with self.assertRaises(ValueError): engine.completion([{"role": "user", "content": "Hi"}], 1)
                start.assert_not_called()

    def test_qwen_text_template(self):
        self.assertEqual(chat_prompt([{"role": "user", "content": "Hi"}]),
                         "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        with self.assertRaises(ValueError): chat_prompt([{"role": "user", "content": [{"type": "image"}]}])

    def test_sampling_validation(self):
        for options in ({"top_p": 0}, {"temperature": float("nan")}, {"seed": -1}, {"max_tokens": True}, {"stop": [""]}, {"stream": "yes"}):
            with self.assertRaises(ValueError): validate_options(options, 131072)
        self.assertEqual(validate_options({"max_tokens": 100000}, 131072)["max_tokens"], 100000)

    def test_reasoning_and_stop_prefixes(self):
        self.assertEqual(split_reasoning("analysis</think>\n\nanswer", True), ("analysis", "answer"))
        self.assertEqual(hold_suffix("hello</thi", ["</think>"]), "hello")
        self.assertEqual(hold_suffix("answer ST", ["STOP"]), "answer ")

    def test_cancel_discards_native_and_recovers(self):
        class Native:
            def __init__(self): self.closed = False
            def command(self, value, acknowledgement=False):
                if value.startswith("PREFILL"): raise ConnectionResetError("disconnected")
                return None
            def close(self): self.closed = True
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory); native = Native(); engine.native = native
            with self.assertRaises(ConnectionResetError): engine.completion([{"role": "user", "content": "Hi"}], 1)
            self.assertTrue(native.closed); self.assertIsNone(engine.native)

    def test_developer_and_late_system_roles_normalize(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            messages, _, _ = engine.validate([
                {"role": "developer", "content": "D"},
                {"role": "user", "content": "u"},
                {"role": "system", "content": "late"},
                {"role": "developer", "content": "later"}], {})
            self.assertEqual([m["role"] for m in messages], ["system", "user", "user", "user"])
            with self.assertRaises(ValueError):
                engine.validate([{"role": "robot", "content": "x"}], {})

    def test_invalid_tool_schema_and_selection(self):
        try: import jsonschema
        except ImportError: self.skipTest("jsonschema not installed")
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine(directory)
            tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
            with self.assertRaises(ValueError):
                engine.validate([{"role": "user", "content": "Hi"}], {"tools": tools, "tool_choice": {"type": "function", "function": None}})
            tools[0]["function"]["parameters"] = {"type": "unknown"}
            with self.assertRaises(ValueError): engine.validate([{"role": "user", "content": "Hi"}], {"tools": tools})

    def test_tool_output(self):
        tools = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        try: import jsonschema
        except ImportError: self.skipTest("jsonschema not installed")
        text, calls = tool_message("<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>", tools, "required")
        self.assertEqual(text, ""); self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Paris"})
        with self.assertRaises(RuntimeError): tool_message("No call", tools, "required")

    def test_chat_and_streaming_endpoint(self):
        server = HTTPServer(("127.0.0.1", 0), make_handler(FakeEngine()))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        def request(body):
            return urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}))
        try:
            with request({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2}) as response:
                result = json.load(response)
            self.assertEqual(result["choices"][0]["message"]["content"], "Hello")
            self.assertEqual(result["usage"]["total_tokens"], 13)
            with request({"messages": [{"role": "user", "content": "Hi"}], "stream": True}) as response:
                data = response.read().decode()
                self.assertIn("text/event-stream", response.headers["Content-Type"])
                self.assertIn("Hello", data); self.assertTrue(data.endswith("data: [DONE]\n\n"))
            for body in ([], {"messages": [], "top_p": 0}):
                with self.assertRaises(urllib.error.HTTPError) as error: request(body)
                self.assertEqual(error.exception.code, 400); error.exception.close()
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)


if __name__ == "__main__": unittest.main()
