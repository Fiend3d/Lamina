#!/usr/bin/env python3
"""Text-only Lamina client for the native inference executable."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from tools.lamina_model import FILENAME

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "Lamina-data"
MODEL_NAME = "Qwen3.6-35B-A3B-UD-Q4_K_M"
STOP_IDS = {248044, 248046}


def default_engine() -> Path:
    name = "lamina-infer.exe" if sys.platform == "win32" else "lamina-infer"
    candidates = [
        ROOT / "build" / "Release" / name,
        ROOT / "build" / name,
        ROOT / "build-lamina" / name,
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def chat_prompt(messages: list[dict]) -> str:
    if (
        not isinstance(messages, list)
        or not messages
        or not isinstance(messages[-1], dict)
        or messages[-1].get("role") != "user"
    ):
        raise ValueError("messages must end with a user message")
    result = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role: {role}")
        if role == "system" and index != 0:
            raise ValueError("system message must be first")
        if not isinstance(content, str):
            raise ValueError("only text message content is supported")
        result.append(f"<|im_start|>{role}\n{content.strip()}<|im_end|>\n")
    # This is the official Qwen text template's enable_thinking=false suffix.
    result.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(result)


class Engine:
    def __init__(
        self, model: Path, tokenizer: Path, executable: Path, cuda: bool = False
    ):
        from tokenizers import Tokenizer

        if not model.is_file():
            raise FileNotFoundError(
                f"model missing: {model}; run python setup.py --download"
            )
        if not tokenizer.is_file():
            raise FileNotFoundError(
                f"tokenizer missing: {tokenizer}; run python setup.py --tokenizer"
            )
        if not executable.is_file():
            raise FileNotFoundError(f"engine missing: {executable}; build lamina-infer")
        self.model = model
        self.executable = executable
        self.cuda = cuda
        self.tokenizer = Tokenizer.from_file(str(tokenizer))

    def completion(
        self, messages: list[dict], max_tokens: int = 128
    ) -> tuple[str, int, int, str]:
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or not 1 <= max_tokens <= 32768
        ):
            raise ValueError("max_tokens must be 1..32768")
        prompt_ids = self.tokenizer.encode(
            chat_prompt(messages), add_special_tokens=False
        ).ids
        if not prompt_ids or len(prompt_ids) + max_tokens > 32768:
            raise ValueError("prompt and response exceed the 32768-token context")
        process = subprocess.Popen(
            [
                str(self.executable),
                str(self.model),
                *(["--cuda"] if self.cuda else []),
                "--interactive",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        generated: list[int] = []
        finish = "length"
        try:

            def feed(token: int, need_logits: bool = True) -> int:
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(f"{'' if need_logits else '+'}{token}\n")
                process.stdin.flush()
                line = process.stdout.readline()
                if not line:
                    error = process.stderr.read() if process.stderr else ""
                    raise RuntimeError(f"native engine exited: {error.strip()}")
                if not need_logits:
                    if line.strip() != ".":
                        raise RuntimeError(
                            f"native engine rejected prompt token: {line.strip()}"
                        )
                    return -1
                return int(line)

            next_id = -1
            for index, token in enumerate(prompt_ids):
                next_id = feed(token, need_logits=index == len(prompt_ids) - 1)
            for _ in range(max_tokens):
                if next_id in STOP_IDS:
                    finish = "stop"
                    break
                generated.append(next_id)
                next_id = feed(next_id)
            return (
                self.tokenizer.decode(generated, skip_special_tokens=True),
                len(prompt_ids),
                len(generated),
                finish,
            )
        finally:
            if process.stdin:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self.reply(
                    200,
                    {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]},
                )
            else:
                self.reply(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.reply(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 1024 * 1024:
                    raise ValueError("request body must be 1..1048576 bytes")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("request body must be an object")
                if body.get("stream", False):
                    raise ValueError("streaming responses are not supported")
                if body.get("temperature", 0) != 0:
                    raise ValueError("only greedy temperature=0 is supported")
                if body.get("top_p", 1) != 1 or body.get("n", 1) != 1:
                    raise ValueError("only top_p=1 and n=1 are supported")
                if "tools" in body or "tool_choice" in body:
                    raise ValueError("tool calling is not supported")
                if body.get("model", MODEL_NAME) != MODEL_NAME:
                    raise ValueError("unknown model")
                text, prompt_count, completion_count, finish = engine.completion(
                    body["messages"],
                    body.get("max_tokens", body.get("max_completion_tokens", 128)),
                )
                self.reply(
                    200,
                    {
                        "id": "chatcmpl-" + uuid.uuid4().hex,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": MODEL_NAME,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": finish,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_count,
                            "completion_tokens": completion_count,
                            "total_tokens": prompt_count + completion_count,
                        },
                    },
                )
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self.reply(400, {"error": {"message": str(error)}})
            except Exception as error:
                self.reply(500, {"error": {"message": str(error)}})

        def reply(self, status: int, payload: dict):
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["chat", "serve"])
    parser.add_argument("--prompt", help="user prompt for chat mode")
    parser.add_argument("--model", type=Path, default=DATA / "models" / FILENAME)
    parser.add_argument(
        "--tokenizer", type=Path, default=DATA / "tokenizer" / "tokenizer.json"
    )
    parser.add_argument("--engine", type=Path, default=default_engine())
    parser.add_argument(
        "--cuda", action="store_true", help="use the hybrid CUDA projection path"
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    engine = Engine(args.model, args.tokenizer, args.engine, cuda=args.cuda)
    if args.command == "serve":
        HTTPServer((args.host, args.port), make_handler(engine)).serve_forever()
        return 0
    prompt = args.prompt if args.prompt is not None else input("You: ")
    answer, _, _, _ = engine.completion(
        [{"role": "user", "content": prompt}], args.max_tokens
    )
    print(answer)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Lamina: {error}", file=sys.stderr)
        sys.exit(1)
