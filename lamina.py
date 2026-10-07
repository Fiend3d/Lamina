#!/usr/bin/env python3
"""Native Qwen3.6 chat and local chat-completions API for Lamina."""
import argparse
import atexit
import os
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from tools.lamina_model import FILENAME
from tools.lamina_chat import Engine, make_handler, MODEL_NAME, STOP_IDS

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "Lamina-data"


def default_engine():
    name = "lamina-infer.exe" if sys.platform == "win32" else "lamina-infer"
    candidates = [ROOT / "build-cuda" / "Release" / name, ROOT / "build-cuda" / name,
                  ROOT / "build" / "Release" / name, ROOT / "build" / name]
    return next((p for p in candidates if p.is_file()), candidates[-1])


def chat_prompt(messages):
    """Compatibility helper for the original text-only, non-thinking template."""
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "user":
        raise ValueError("messages must end with a user message")
    result = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("invalid message role")
        if message["role"] == "system" and index != 0: raise ValueError("system message must be first")
        if not isinstance(message.get("content"), str): raise ValueError("only text content supported by compatibility helper")
        result.append(f"<|im_start|>{message['role']}\n{message['content'].strip()}<|im_end|>\n")
    return "".join(result) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main():
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"): sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("chat", "serve"))
    parser.add_argument("--prompt")
    parser.add_argument("--model", type=Path, default=DATA / "models" / FILENAME)
    parser.add_argument("--tokenizer", type=Path, default=DATA / "tokenizer/tokenizer.json")
    parser.add_argument("--engine", type=Path, default=default_engine())
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--cuda", dest="cuda", action="store_true", default=None)
    backend.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--compute-mode", choices=("f32", "fast"), default="f32", help="fast: experimental Strata-style reduced-precision computation")
    parser.add_argument("--kv-type", choices=("f32", "f16"), default="f32", help="KV storage precision; experimental lossy f16 requires CUDA")
    parser.add_argument("--kv-cache", choices=("auto", "host", "device"), default="auto")
    parser.add_argument("--vram-limit-mb", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="CPU threads for the image encoder (default 8) and CPU experts (engine default: a quarter of hardware threads)")
    parser.add_argument("--expert-policy", choices=("auto", "stream", "cpu-miss"), default="auto",
                        help="auto: CPU experts for cache misses in fast mode, GPU streaming in f32")
    parser.add_argument("--prefill-chunk", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--vision-engine", type=Path, default=ROOT / "build-vision/bin/Release/strata-vision.exe" if sys.platform == "win32" else ROOT / "build-vision/bin/strata-vision")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--max-image-tokens", type=int, default=1024)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.cpu_threads is not None:
        if not 1 <= args.cpu_threads <= 64: parser.error("cpu-threads must be 1..64")
        os.environ["LAMINA_CPU_THREADS"] = str(args.cpu_threads)
    if args.expert_policy != "auto":
        os.environ["LAMINA_EXPERT_POLICY"] = args.expert_policy
    cuda = args.cuda if args.cuda is not None else "build-cuda" in str(args.engine)
    engine = Engine(args.model, args.tokenizer, args.engine, cuda, args.max_context, args.kv_cache,
                    args.vram_limit_mb, args.prefill_chunk, args.vision_engine if args.vision or args.image else None,
                    args.max_image_tokens, args.cpu_threads or 8, args.kv_type, args.compute_mode)
    atexit.register(engine.close)
    try:
        if args.command == "serve":
            print(f"Lamina {MODEL_NAME}: {'CUDA' if cuda else 'CPU'}, compute={args.compute_mode}, context={args.max_context}, KV={args.kv_type}/{args.kv_cache}, http://{args.host}:{args.port}", flush=True)
            server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
            try: server.serve_forever()
            finally: server.server_close()
            return 0
        prompt = args.prompt if args.prompt is not None else input("You: ")
        content = prompt
        if args.image:
            import base64
            content = [{"type": "text", "text": prompt}]
            for image in args.image:
                content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()}})
        for event in engine.events([{"role": "user", "content": content}], args.max_tokens,
                                  temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                  seed=args.seed, enable_thinking=args.thinking, stop=args.stop):
            delta = event.get("delta", {})
            if delta.get("reasoning_content"): print(delta["reasoning_content"], end="", file=sys.stderr, flush=True)
            if delta.get("content"): print(delta["content"], end="", flush=True)
        print()
        return 0
    finally: engine.close()


if __name__ == "__main__":
    try: sys.exit(main())
    except KeyboardInterrupt: sys.exit(130)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Lamina: {error}", file=sys.stderr); sys.exit(1)
