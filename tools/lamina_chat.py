"""Lamina's persistent CLI/chat API backend; text inference is always native."""
import copy
import json
import math
import os
import re
import select
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler
from serve.structured import prepare_format, validated_json, StructuredOutputError
from tools.lamina_protocol import NativeProcess
from tools.lamina_toolcalls import ToolStream, hold_suffix
from tools.lamina_vision import Vision

MODEL_NAME = "Qwen3.6-35B-A3B-UD-Q4_K_M"
STOP_IDS = {248044, 248046}
IMAGE_ID = 248056


def validate_options(options, context):
    o = dict(options)
    count = o.get("max_tokens", o.get("max_completion_tokens", 128))
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= context:
        raise ValueError(f"max_tokens must be 1..{context}")
    o["max_tokens"] = count
    for key, default, low, high in (("temperature", 0, 0, 10), ("top_p", 1, 0, 1)):
        v = o.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low <= v <= high or (key == "top_p" and v == 0):
            raise ValueError(f"invalid {key}")
        o[key] = v
    for key, default, high in (("top_k", 20, 248320), ("seed", 0, 2**64-1)):
        v = o.get(key, default)
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= high:
            raise ValueError(f"invalid {key}")
        o[key] = v
    if o.get("n", 1) != 1:
        raise ValueError("only n=1 is supported")
    if not isinstance(o.get("stream", False), bool):
        raise ValueError("stream must be boolean")
    stream_options = o.get("stream_options", {})
    if not isinstance(stream_options, dict) or not isinstance(stream_options.get("include_usage", False), bool):
        raise ValueError("stream_options.include_usage must be boolean")
    thinking = o.get("enable_thinking", False)
    if "reasoning_effort" in o:
        # The model only thinks or does not; the finer levels that clients such as pi send all mean "think".
        if o["reasoning_effort"] not in ("none", "off", "minimal", "low", "medium", "high", "xhigh", "max"):
            raise ValueError("reasoning_effort must be none, minimal, low, medium, high, xhigh or max")
        thinking = o["reasoning_effort"] not in ("none", "off")
    if not isinstance(thinking, bool):
        raise ValueError("enable_thinking must be boolean")
    o["enable_thinking"] = thinking
    stop = o.get("stop", [])
    stop = [stop] if isinstance(stop, str) else stop
    if not isinstance(stop, list) or len(stop) > 4 or any(not isinstance(v, str) or not v for v in stop):
        raise ValueError("stop must be a string or up to four nonempty strings")
    o["stop"] = stop
    return o


def split_reasoning(text, thinking):
    if thinking or text.startswith("<think>"):
        text = text.removeprefix("<think>")
        if "</think>" in text:
            reason, content = text.split("</think>", 1)
            return reason, content.lstrip("\n")
        return text, ""
    return "", text


def tool_message(content, tools, choice):
    available = {t["function"]["name"]: t["function"] for t in tools}
    calls = []
    for match in re.finditer(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", content, re.S):
        name, body = match.groups()
        if name not in available:
            raise RuntimeError(f"model requested an unknown tool: {name}")
        args = {}
        properties = available[name].get("parameters", {}).get("properties", {})
        for parameter in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.S):
            key, value = parameter.groups(); value = value.strip("\n")
            if key in args:
                raise RuntimeError("duplicate tool parameter")
            if properties.get(key, {}).get("type") == "string":
                args[key] = value
            else:
                try: args[key] = json.loads(value)
                except ValueError: args[key] = value
        import jsonschema
        try: jsonschema.validate(args, available[name].get("parameters", {"type": "object"}))
        except jsonschema.ValidationError as error: raise RuntimeError("invalid generated tool arguments: " + error.message) from error
        calls.append({"id": "call_" + uuid.uuid4().hex, "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    if "<tool_call>" in content and not calls:
        raise RuntimeError("incomplete generated tool call")
    forced = choice.get("function", {}).get("name") if isinstance(choice, dict) else None
    if (choice == "required" or forced) and not calls:
        raise RuntimeError("model did not produce the required tool call")
    if forced and any(call["function"]["name"] != forced for call in calls):
        raise RuntimeError("model did not call the selected tool")
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.S).strip() if calls else content
    return cleaned, calls


class Engine:
    def __init__(self, model, tokenizer, executable, cuda=False, max_context=32768,
                 kv_cache="auto", vram_limit_mb=0, prefill_chunk=2048, vision_engine=None,
                 max_image_tokens=1024, cpu_threads=8, kv_type="f32", compute_mode="f32", mtp=None):
        if kv_type not in ("f32", "f16") or (kv_type == "f16" and not cuda):
            raise ValueError("kv-type must be f32, or f16 with CUDA")
        if compute_mode not in ("f32", "fast") or (compute_mode == "fast" and not cuda):
            raise ValueError("compute-mode must be f32, or fast with CUDA")
        self.compute_mode = compute_mode
        if mtp is not None and (compute_mode != "fast" or not Path(mtp).is_file()):
            raise ValueError("an MTP head needs --compute-mode fast and an existing packed file")
        self.mtp = mtp
        self.kv_type = kv_type
        from tokenizers import Tokenizer
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        for path, label in ((model, "model"), (tokenizer, "tokenizer"), (executable, "engine")):
            if not path.is_file(): raise FileNotFoundError(f"{label} missing: {path}")
        if not 1 <= max_context <= 131072 or kv_cache not in ("auto", "device", "host"):
            raise ValueError("max-context must be 1..131072 and kv-cache auto, device or host")
        if not 1 <= prefill_chunk <= 2048 or vram_limit_mb < 0 or not 1 <= cpu_threads <= 64:
            raise ValueError("invalid prefill chunk, VRAM budget or CPU worker count")
        if not 1 <= max_image_tokens <= 4096: raise ValueError("max-image-tokens must be 1..4096")
        self.model, self.executable, self.cuda = model, executable, cuda
        self.max_context, self.kv_cache, self.vram_limit_mb = max_context, kv_cache, vram_limit_mb
        self.prefill_chunk = prefill_chunk
        self.data = tokenizer.parent.parent
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        template = tokenizer.parent / "chat_template.jinja"
        if not template.is_file(): raise FileNotFoundError("official chat template missing; run python -m tools.lamina_assets")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        def raise_exception(message): raise ValueError(message)
        env.globals["raise_exception"] = raise_exception
        self.template = env.from_string(template.read_text(encoding="utf-8"))
        self.lock = threading.Lock()
        self.native = self.vision = None
        self._cache_supported = None
        self._cache_prefix = []
        self._prepared_prefix = []
        self.vision_engine, self.max_image_tokens, self.cpu_threads = vision_engine, max_image_tokens, cpu_threads

    log_requests = False  # the server prints one line per request and progress lines during long answers

    def _log(self, text):
        if self.log_requests:
            print(f"{time.strftime('%H:%M:%S')}  {text}", flush=True)

    def close(self):
        # A running request holds the lock for its whole answer, which can take minutes. Do not wait for it:
        # stopping the engine process below makes that request fail and release the lock.
        locked = self.lock.acquire(timeout=1)
        try:
            self._stop_native()
            if self.vision: self.vision.close(); self.vision = None
        finally:
            if locked: self.lock.release()

    def _stop_native(self):
        self._cache_prefix = []
        if self.native: self.native.close(); self.native = None

    def _start_native(self):
        if self.native is None:
            if getattr(self, "_cache_supported", None) is None:
                self._cache_supported = False
                if self.cuda and self.kv_cache == "device" and os.environ.get("LAMINA_PREFIX_CACHE", "1") != "0":
                    try:
                        caps = subprocess.run([str(self.executable.resolve()), "--capabilities"], capture_output=True,
                                              text=True, timeout=10,
                                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        self._cache_supported = caps.returncode == 0 and json.loads(caps.stdout).get("prefix_cache") is True
                    except (OSError, ValueError, subprocess.TimeoutExpired):
                        pass  # older release binaries retain their RESET path
            command = [str(self.executable.resolve()), str(self.model.resolve()), *(["--cuda"] if self.cuda else []),
                       "--max-context", str(self.max_context), "--kv-cache", self.kv_cache, "--kv-type", self.kv_type, "--compute-mode", getattr(self, "compute_mode", "f32"),
                       "--vram-limit-mb", str(self.vram_limit_mb)]
            if getattr(self, "mtp", None): command += ["--mtp", str(Path(self.mtp).resolve())]
            command.append("--interactive")
            self.native = NativeProcess(command)

    def validate(self, messages, options):
        o = validate_options(options, self.max_context)
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages) or messages[-1].get("role") not in ("user", "tool"):
            raise ValueError("messages must end with a user or tool message")
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant", "tool"):
                raise ValueError("invalid message role")
            if message["role"] == "system" and index != 0: raise ValueError("system message must be first")
            content = message.get("content")
            if content is not None and not isinstance(content, (str, list)): raise ValueError("invalid message content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict) or item.get("type") not in ("text", "image_url", "image"):
                        raise ValueError("content supports text and images")
                    if item["type"] == "text" and not isinstance(item.get("text"), str): raise ValueError("invalid text content")
        tools = o.get("tools", [])
        if not isinstance(tools, list): raise ValueError("tools must be an array")
        names = set()
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
                raise ValueError("only function tools are supported")
            name = tool["function"].get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) or name in names:
                raise ValueError("invalid or duplicate function name")
            names.add(name)
            parameters = tool["function"].get("parameters", {"type": "object"})
            if not isinstance(parameters, dict): raise ValueError("tool parameters must be a JSON Schema object")
            import jsonschema
            try: jsonschema.validators.validator_for(parameters).check_schema(parameters)
            except jsonschema.SchemaError as error: raise ValueError("invalid tool parameter schema: " + error.message) from error
        choice = o.get("tool_choice", "auto")
        if isinstance(choice, dict):
            if choice.get("type") != "function" or not isinstance(choice.get("function"), dict) or choice["function"].get("name") not in names:
                raise ValueError("unknown tool_choice function")
        elif choice not in ("auto", "none", "required"): raise ValueError("invalid tool_choice")
        if choice == "required" and not tools: raise ValueError("required tool_choice needs tools")
        response_format = o.get("response_format")
        if response_format is not None and not isinstance(response_format, dict): raise ValueError("response_format must be an object")
        if tools and (response_format or {}).get("type", "text") != "text":
            raise ValueError("structured output with tools is not supported")
        messages, validator = prepare_format(o.get("response_format"), copy.deepcopy(messages))
        if choice == "required" or isinstance(choice, dict):
            directive = "You must respond with a tool call."
            if isinstance(choice, dict): directive += " Call the function " + choice["function"]["name"] + "."
            if messages[0]["role"] == "system":
                content = messages[0].get("content") or ""
                if isinstance(content, list): messages[0]["content"] = content + [{"type": "text", "text": directive}]
                else: messages[0]["content"] = content + "\n" + directive
            else: messages.insert(0, {"role": "system", "content": directive})
        if choice == "none": o["tools"] = []
        return messages, o, validator

    def _prepare(self, messages, o, directory):
        images = []
        for message in messages:
            for item in message.get("content", []) if isinstance(message.get("content"), list) else []:
                if item["type"] in ("image", "image_url"):
                    if self.vision is None:
                        if self.vision_engine is None: raise ValueError("image input requires --vision")
                        self.vision = Vision(self.vision_engine, self.model, self.data / "vision/mmproj-F16.gguf",
                                             self.data / "tmp", self.max_image_tokens, self.cpu_threads)
                    images.append(self.vision.encode(item, directory, len(images)))
        for message in messages:
            for call in message.get("tool_calls", []) or []:
                args = call.get("function", {}).get("arguments")
                if isinstance(args, str): call["function"]["arguments"] = json.loads(args)
        prompt = self.template.render(messages=messages, tools=o.get("tools", []), add_generation_prompt=True,
                                      enable_thinking=o["enable_thinking"], add_vision_id=False)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        self._prepared_prefix = []
        if not images and self.cuda and self.kv_cache == "device" and os.environ.get("LAMINA_PREFIX_CACHE", "1") != "0":
            # Cache only the stable system/tools block, before the first user.
            # Verify against full tokenization: splitting a string at an arbitrary
            # byte boundary can change the BPE token at the join.
            boundary = prompt.find("<|im_start|>user\n")
            if boundary > 0:
                prefix = self.tokenizer.encode(prompt[:boundary], add_special_tokens=False).ids
                if 256 <= len(prefix) < len(ids) and ids[:len(prefix)] == prefix:
                    self._prepared_prefix = prefix
        if ids.count(IMAGE_ID) != len(images): raise ValueError("image marker count differs from image input")
        count = len(ids) + sum(n - 1 for _, n in images)
        if not ids or count + o["max_tokens"] > self.max_context:
            raise ValueError(f"prompt and response exceed the {self.max_context}-token context")
        return ids, images, count

    def events(self, messages, max_tokens=128, cancelled=None, **options):
        options["max_tokens"] = max_tokens
        messages, o, validator = self.validate(messages, options)
        # A streaming client gets reasoning, text and the arguments of tool calls as they are generated.
        # Without streaming, or with a JSON validator, the answer is checked as a whole before it is delivered.
        incremental = bool(o.get("tools")) and validator is None and bool(o.get("stream"))
        buffered = (bool(o.get("tools")) and not incremental) or validator is not None
        tool_stream = ToolStream(o["tools"], o.get("tool_choice", "auto")) if incremental else None
        with self.lock:
            tmp = self.data / "tmp"; tmp.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=tmp, prefix="chat-") as path:
                ids, images, prompt_count = self._prepare(messages, o, Path(path))
                start = time.perf_counter()
                self._log(f"request: {prompt_count} prompt tokens, up to {o['max_tokens']} answer tokens"
                          + (f", {len(o['tools'])} tools" if o.get("tools") else "")
                          + (", thinking" if o["enable_thinking"] else "") + (", streaming" if o.get("stream") else ""))
                self._start_native()
                self.native.cancelled = cancelled
                try:
                    prefix = self._prepared_prefix if getattr(self, "_cache_supported", False) else []
                    cached = bool(prefix) and prefix == self._cache_prefix
                    if cached:
                        self.native.command("RESTORE_PREFIX", True)
                        self._log(f"  reused {len(prefix)} system/tool prefix tokens")
                    else:
                        self._cache_prefix = []
                        self.native.command("RESET", True)
                    if prefix:
                        if not cached:
                            if len(prefix) > self.prefill_chunk:
                                self.native.command(f"PROMPT {self.prefill_chunk} " + " ".join(map(str, prefix)))
                            else:
                                self.native.command("BATCH " + " ".join(map(str, prefix)), True)
                            self.native.command("CACHE_PREFIX", True)
                            self._cache_prefix = prefix
                        ids = ids[len(prefix):]
                    # Long prefix PROMPT returns an ignored token. Configure
                    # sampling afterwards so cache misses cannot consume this
                    # request's seeded RNG before its first answer token.
                    self.native.command(f"SAMPLE {o['temperature']} {o['top_p']} {o['top_k']} {o['seed']}", True)
                    if self.cuda and not images and len(ids)>self.prefill_chunk:
                        next_id=self.native.command(f"PROMPT {self.prefill_chunk} " + " ".join(map(str,ids)))
                    else:
                        pending, image_index = [], 0
                        for i, token in enumerate(ids):
                            if token == IMAGE_ID:
                                if pending: self.native.command("BATCH " + " ".join(map(str, pending)), True); pending = []
                                self.native.command("IMAGE " + str(images[image_index][0]), True); image_index += 1
                            else:
                                pending.append(token)
                                if len(pending) >= self.prefill_chunk or i == len(ids) - 1:
                                    last = i == len(ids) - 1
                                    next_id = self.native.command(("PREFILL " if last else "BATCH ") + " ".join(map(str, pending)), not last)
                                    pending = []
                    first_token = time.perf_counter() - start
                    self._log(f"  prompt read in {first_token:.1f} s ({prompt_count / max(first_token, 1e-9):.0f} tokens/s)")
                    generation_start = last_progress = time.perf_counter()
                    generated, raw, prior_reason, prior_content = [], "", "", ""
                    finish = "length"
                    # Greedy requests with an MTP head take tokens in chunks from
                    # GENERATE (speculative); a chunk may run past a stop token,
                    # which costs only time: the next request resets or restores
                    # its immutable prefix before processing a new suffix.
                    speculative = bool(getattr(self, "mtp", None)) and o["temperature"] == 0
                    queued = []
                    for index in range(o["max_tokens"]):
                        if next_id in STOP_IDS: finish = "stop"; break
                        generated.append(next_id)
                        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
                        stops = [raw.find(s) for s in o["stop"] if s in raw]
                        stopped = bool(stops)
                        if stopped: raw = raw[:min(stops)]; finish = "stop"
                        visible = raw.rstrip("\ufffd") if not stopped else raw
                        if not stopped: visible = hold_suffix(visible, o["stop"] + ["</think>", "<think>"])
                        reason, content = split_reasoning(visible, o["enable_thinking"])
                        if incremental:
                            if len(reason) > len(prior_reason): yield {"delta": {"reasoning_content": reason[len(prior_reason):]}}
                            for delta in tool_stream.feed(content): yield {"delta": delta}
                        elif not buffered:
                            delta = {}
                            if len(reason) > len(prior_reason): delta["reasoning_content"] = reason[len(prior_reason):]
                            if len(content) > len(prior_content): delta["content"] = content[len(prior_content):]
                            if delta: yield {"delta": delta}
                        prior_reason, prior_content = reason, content
                        if self.log_requests and time.perf_counter() - last_progress >= 10:
                            last_progress = time.perf_counter()
                            self._log(f"  writing: {len(generated)} tokens so far, "
                                      f"{(len(generated) - 1) / (last_progress - generation_start):.1f} tokens/s")
                        if stopped or index + 1 == o["max_tokens"]: break
                        if speculative:
                            if not queued: queued = self.native.generate(min(8, o["max_tokens"] - index - 1), next_id)
                            next_id = queued.pop(0)
                        else: next_id = self.native.command(str(next_id))
                    reason, content = split_reasoning(raw, o["enable_thinking"])
                    calls = []
                    if incremental:
                        if len(reason) > len(prior_reason): yield {"delta": {"reasoning_content": reason[len(prior_reason):]}}
                        for delta in tool_stream.finish(content, truncated=finish == "length"): yield {"delta": delta}
                        if tool_stream.calls: finish = "tool_calls"
                    elif o.get("tools"):
                        content, calls = tool_message(content, o["tools"], o.get("tool_choice", "auto"))
                        if calls: finish = "tool_calls"
                    if validator: content = validated_json(content, validator, finish)
                    if incremental:
                        pass
                    elif buffered:
                        delta = {"content": content}
                        if reason: delta["reasoning_content"] = reason
                        if calls: delta["tool_calls"] = [dict(call, index=i) for i, call in enumerate(calls)]
                        yield {"delta": delta}
                    else:
                        tail = {}
                        if len(reason) > len(prior_reason): tail["reasoning_content"] = reason[len(prior_reason):]
                        if len(content) > len(prior_content): tail["content"] = content[len(prior_content):]
                        if tail: yield {"delta": tail}
                    seconds = time.perf_counter() - generation_start
                    self._log(f"  done: {len(generated)} answer tokens in {seconds:.1f} s"
                              + (f" = {(len(generated) - 1) / seconds:.1f} tokens/s" if len(generated) > 1 and seconds > 0 else "")
                              + f", finish: {finish}")
                    yield {"finish_reason": finish, "usage": {"prompt_tokens": prompt_count,
                           "completion_tokens": len(generated), "total_tokens": prompt_count + len(generated)},
                           "timings": {"first_token_seconds": first_token, "total_seconds": time.perf_counter()-start,
                                       "cached_prompt_tokens": len(prefix) if cached else 0}}
                except BaseException:
                    self._stop_native()
                    raise
                finally:
                    if self.native: self.native.cancelled = None

    def completion(self, messages, max_tokens=128, **options):
        content, final = "", None
        for event in self.events(messages, max_tokens, **options):
            content += event.get("delta", {}).get("content", "")
            if "finish_reason" in event: final = event
        return content, final["usage"]["prompt_tokens"], final["usage"]["completion_tokens"], final["finish_reason"]


def make_handler(engine):
    log = getattr(engine, "_log", lambda text: None)  # test doubles have no request log
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded))); self.end_headers(); self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/health": self.reply(200, {"status": "ok", "compute_mode": getattr(engine, "compute_mode", "f32"), "max_context": engine.max_context, "kv_type": getattr(engine, "kv_type", "f32"), "images": engine.vision_engine is not None})
            elif self.path == "/v1/models": self.reply(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})
            else: self.reply(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions": self.reply(404, {"error": {"message": "not found"}}); return
            started = False
            events = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 32 * 1024 * 1024: raise ValueError("request body must be 1..33554432 bytes")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict): raise ValueError("request must be an object")
                if body.get("model", MODEL_NAME) != MODEL_NAME: raise ValueError("unknown model")
                messages = body.pop("messages")
                engine.validate(messages, body)
                count = body.pop("max_tokens", body.pop("max_completion_tokens", 128))
                def disconnected():
                    try:
                        ready, _, _ = select.select([self.connection], [], [], 0)
                        return bool(ready) and self.connection.recv(1, socket.MSG_PEEK) == b""
                    except (OSError, ValueError): return True
                events = engine.events(messages, count, cancelled=disconnected, **body)
                identifier, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
                base = {"id": identifier, "created": created, "model": MODEL_NAME}
                stream = body.get("stream", False)
                message = {"role": "assistant", "content": ""}
                final = None
                def send(payload):
                    self.wfile.write(("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8")); self.wfile.flush()
                for event in events:
                    if stream and not started:
                        self.send_response(200); self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache"); self.end_headers(); started = True
                        send(dict(base, object="chat.completion.chunk", choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]))
                    if "delta" in event:
                        if stream: send(dict(base, object="chat.completion.chunk", choices=[{"index": 0, "delta": event["delta"], "finish_reason": None}]))
                        else:
                            for key, value in event["delta"].items():
                                if key == "tool_calls": message[key] = [{k:v for k,v in call.items() if k != "index"} for call in value]
                                else: message[key] = message.get(key, "") + value
                    else: final = event
                if stream:
                    send(dict(base, object="chat.completion.chunk", choices=[{"index": 0, "delta": {}, "finish_reason": final["finish_reason"]}]))
                    if body.get("stream_options", {}).get("include_usage"):
                        send(dict(base, object="chat.completion.chunk", choices=[], usage=final["usage"]))
                    self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                else:
                    self.reply(200, dict(base, object="chat.completion", choices=[{"index": 0, "message": message,
                               "finish_reason": final["finish_reason"]}], usage=final["usage"]))
            except (BrokenPipeError, ConnectionResetError): pass
            except (ValueError, KeyError, TypeError) as error:
                log(f"  request rejected: {error}")
                if not started: self.reply(400, {"error": {"message": str(error)}})
                else: self.wfile.write(("data: " + json.dumps({"error": {"message": str(error)}}) + "\n\n").encode())
            except Exception as error:
                log(f"  request failed: {error}")
                if not started: self.reply(502, {"error": {"message": str(error)}})
                else: self.wfile.write(("data: " + json.dumps({"error": {"message": str(error)}}) + "\n\n").encode())
            finally:
                if events is not None: events.close()
    return Handler
