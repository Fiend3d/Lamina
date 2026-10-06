"""Exercise real native chat, SSE, sampling, tools, JSON, OCR and cancellation.

Uses the installed pinned model and CPU vision encoder. Generated image fixtures
live in sibling Lamina-data; request reports live under bench/results.
"""
import argparse
import base64
import hashlib
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from tools.lamina_chat import Engine, make_handler
from tools.lamina_model import FILENAME


def main():
    root = Path(__file__).resolve().parents[1]; data = root.parent / "Lamina-data"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, default=root / "build-cuda/lamina-infer.exe")
    p.add_argument("--vision-engine", type=Path, default=root / "build-vision/bin/Release/strata-vision.exe")
    p.add_argument("--kv-type", choices=("f32", "f16"), default="f32")
    p.add_argument("--report", type=Path, default=root / "bench/results/2026-10-06-rtx4060-8gb/runtime.json")
    args = p.parse_args()
    engine = Engine(data / "models" / FILENAME, data / "tokenizer/tokenizer.json", args.engine,
                    True, prefill_chunk=2048, vision_engine=args.vision_engine, kv_type=args.kv_type)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine))
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    report, passed = [], False
    def request(label, content, **options):
        body = {"messages": [{"role": "user", "content": content}], "max_tokens": 64, **options}
        start = time.perf_counter()
        req = urllib.request.Request(endpoint, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            if options.get("stream"):
                raw = response.read().decode(); assert raw.endswith("data: [DONE]\n\n")
                chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
                text = "".join(c["choices"][0].get("delta", {}).get("content", "") for c in chunks if c["choices"])
                result = {"text": text, "chunks": chunks}
            else: result = json.load(response)
        report.append({"label": label, "seconds": time.perf_counter()-start, "request": body, "response": result})
        print(label, f"{report[-1]['seconds']:.3f}s", json.dumps(result)[:400], flush=True)
        return result
    def text(result): return result["choices"][0]["message"]["content"]
    try:
        first = text(request("greedy", "Say hello in one short sentence.", max_tokens=32))
        assert "hello" in first.lower()
        seeded = text(request("sample-seed", "Give one short creative greeting.", temperature=0.6, top_p=0.9, seed=42, max_tokens=32))
        assert seeded == text(request("sample-seed-repeat", "Give one short creative greeting.", temperature=0.6, top_p=0.9, seed=42, max_tokens=32))
        assert first == text(request("reset", "Say hello in one short sentence.", max_tokens=32))
        streamed = request("sse-unicode", "Reply exactly: Hello 🌍", stream=True, stream_options={"include_usage": True}, max_tokens=32)
        assert "🌍" in streamed["text"]
        reasoning = request("thinking", "What is 2 + 3? Answer briefly.", enable_thinking=True, max_tokens=128)
        assert reasoning["choices"][0]["message"].get("reasoning_content")
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
        structured = request("json-schema", 'Return {"ok": true}.', response_format={"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}}, max_tokens=32)
        assert json.loads(text(structured)) == {"ok": True}
        tools = [{"type": "function", "function": {"name": "weather", "description": "Get weather in a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}}}]
        called = request("forced-tool", "What is the weather in Paris?", tools=tools, tool_choice={"type": "function", "function": {"name": "weather"}}, max_tokens=64)
        call = called["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == "weather" and json.loads(call["function"]["arguments"])["city"] == "Paris"
        from PIL import Image, ImageDraw, ImageFont
        fixtures = data / "validation"; fixtures.mkdir(exist_ok=True)
        def picture(word, index):
            im = Image.new("RGB", (560, 140), "white")
            ImageDraw.Draw(im).text((22, 25), word, fill="black", font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 72))
            path = fixtures / f"ocr-{index}.png"; im.save(path)
            return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()}}
        ocr = request("ocr", [{"type": "text", "text": "Read the text in this image. Reply only with that text."}, picture("LAMINA 42", 0)], max_tokens=32)
        assert "LAMINA" in text(ocr).upper() and "42" in text(ocr)
        both = request("multiple-images", [{"type": "text", "text": "Read both images. Reply with the two words."}, picture("ALPHA", 1), picture("BETA", 2)], max_tokens=32)
        assert "ALPHA" in text(both).upper() and "BETA" in text(both).upper()
        old = engine.native.process
        connection = socket.create_connection(server.server_address)
        body = json.dumps({"messages": [{"role": "user", "content": "Hello " * 4096}], "max_tokens": 1, "stream": True}).encode()
        connection.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        time.sleep(0.5); connection.close()
        deadline = time.monotonic() + 5
        while old.poll() is None and time.monotonic() < deadline: time.sleep(0.1)
        assert old.poll() is not None, "disconnect did not stop native prefill"
        assert first == text(request("recover-after-cancel", "Say hello in one short sentence.", max_tokens=32))
        passed = True
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=2); engine.close()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"passed": passed, "engine": str(args.engine), "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(), "kv_type": args.kv_type, "vision_engine": str(args.vision_engine), "requests": report}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
