"""Measure native first-token latency, prompt throughput, decode and peak memory.

Use --input-tokens for matched teacher-forced comparisons between engines.
NVML reports total GPU memory on WDDM, including the desktop, not process VRAM.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import statistics
import threading
import time
from pathlib import Path
from tools.lamina_model import FILENAME
from tools.lamina_protocol import NativeProcess


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=root.parent / "Lamina-data/models" / FILENAME)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--cache-mb", type=int)
    p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--input-tokens", type=Path)
    p.add_argument("--prompt", help="text chat prompt, using the non-thinking Qwen template")
    p.add_argument("--prompt-file", type=Path, help="UTF-8 text chat prompt (also suitable for long retrieval tests)")
    p.add_argument("--prompt-tokens", type=int, default=0, help="stress prefill with repeated token 42")
    p.add_argument("--max-context", type=int)
    p.add_argument("--compute-mode", choices=("f32", "fast"), default="f32")
    p.add_argument("--kv-type", choices=("f32", "f16"))
    p.add_argument("--kv-cache", choices=("auto", "host", "device"))
    p.add_argument("--vram-limit-mb", type=int)
    p.add_argument("--prefill-schedule", choices=("layer", "chunk"), default="layer")
    p.add_argument("--chunk", type=int, default=1)
    p.add_argument("--json", type=Path)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--warm-start", action="store_true", help="send RESET before measuring; CUDA may still initialize lazily on the first prompt")
    p.add_argument("--stop-at-eos", action="store_true", help="stop semantic retrieval runs at the pinned EOS token")
    p.add_argument("--progress-file", type=Path, help="write completed prompt tokens and elapsed time after each chunk")
    a = p.parse_args()
    if a.prompt and a.prompt_file: p.error("choose prompt or prompt-file")
    if a.prompt_file: a.prompt = a.prompt_file.read_text(encoding="utf-8")
    if a.tokens < 1 or not 1 <= a.chunk <= 2048 or a.prompt_tokens < 0: p.error("invalid token/chunk count")
    if a.cache_mb is not None:
        if not a.cuda or a.cache_mb < 1: p.error("cache-mb requires CUDA and a positive value")
        os.environ["LAMINA_CUDA_CACHE_MB"] = str(a.cache_mb)
    command = [str(a.engine.resolve()), str(a.model.resolve())]
    if a.cuda: command.append("--cuda")
    for name, value in (("--max-context", a.max_context), ("--kv-cache", a.kv_cache), ("--kv-type", a.kv_type), ("--compute-mode", a.compute_mode), ("--vram-limit-mb", a.vram_limit_mb)):
        if value is not None: command += [name, str(value)]
    command.append("--interactive")
    inputs = list(map(int, a.input_tokens.read_text().split())) if a.input_tokens else None
    if inputs and len(inputs) < a.tokens: p.error("input token file is too short")
    prompt = []
    if a.prompt:
        from tokenizers import Tokenizer
        from lamina import chat_prompt
        tokenizer = Tokenizer.from_file(str(root.parent / "Lamina-data/tokenizer/tokenizer.json"))
        prompt = tokenizer.encode(chat_prompt([{"role": "user", "content": a.prompt}]), add_special_tokens=False).ids
    elif a.prompt_tokens: prompt = [42] * a.prompt_tokens
    if len(prompt) + a.tokens > (a.max_context or 32768): p.error("prompt and response exceed context")
    peak_gpu = idle_gpu = peak_ram = None
    gpu = process_info = None
    try:
        import psutil
    except ImportError: psutil = None
    try:
        import pynvml
        pynvml.nvmlInit(); gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        idle_gpu = peak_gpu = pynvml.nvmlDeviceGetMemoryInfo(gpu).used
    except (ImportError, Exception): gpu = None
    native = NativeProcess(command)
    if psutil: process_info = psutil.Process(native.process.pid)
    done = threading.Event()
    benchmark_start=time.perf_counter()
    prefill_complete=threading.Event()
    progress_lock=threading.Lock()
    def write_progress(payload):
        with progress_lock:
            if prefill_complete.is_set() and "prompt_tokens_completed" not in payload: return
            a.progress_file.parent.mkdir(parents=True,exist_ok=True)
            a.progress_file.write_text(json.dumps(payload),encoding="utf-8")
    def monitor():
        last_layer=0
        nonlocal peak_gpu, peak_ram
        while not done.is_set():
            try:
                if gpu is not None: peak_gpu = max(peak_gpu or 0, pynvml.nvmlDeviceGetMemoryInfo(gpu).used)
                if process_info: peak_ram = max(peak_ram or 0, process_info.memory_info().rss)
                if a.progress_file:
                    for line in reversed(native.errors.copy()):
                        match=re.fullmatch(r"long prefill layer=(\d+)/(\d+) tokens=(\d+)",line)
                        if match:
                            layer,layers,total=map(int,match.groups())
                            if layer>last_layer:
                                write_progress({"layers_completed":layer,"layers":layers,"prompt_tokens":total,"elapsed_seconds":time.perf_counter()-benchmark_start,"peak_total_gpu_mib":peak_gpu/2**20 if peak_gpu else None})
                                last_layer=layer
                            break
            except Exception: pass
            done.wait(0.05)
    watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
    times, outputs = [], []
    try:
        startup_seconds = None
        if a.warm_start:
            native.command("RESET", True)
            startup_seconds = time.perf_counter() - benchmark_start
        start = time.perf_counter()
        if prompt:
            if a.chunk == 1:
                for token in prompt[:-1]: native.command("+" + str(token), True)
                token = native.command(str(prompt[-1]))
            elif a.cuda and a.prefill_schedule == "layer" and len(prompt)>a.chunk:
                token=native.command(f"PROMPT {a.chunk} " + " ".join(map(str,prompt)))
                if a.progress_file:
                    prefill_complete.set()
                    write_progress({"prompt_tokens_completed":len(prompt),"prompt_tokens":len(prompt),"elapsed_seconds":time.perf_counter()-start})
            else:
                for offset in range(0, len(prompt), a.chunk):
                    chunk = prompt[offset:offset+a.chunk]
                    last = offset + len(chunk) == len(prompt)
                    token = native.command(("PREFILL " if last else "BATCH ") + " ".join(map(str, chunk)), not last)
                    if a.progress_file:
                        a.progress_file.parent.mkdir(parents=True, exist_ok=True)
                        a.progress_file.write_text(json.dumps({"prompt_tokens_completed": offset+len(chunk), "prompt_tokens": len(prompt),
                            "elapsed_seconds": time.perf_counter()-start, "peak_total_gpu_mib": peak_gpu/2**20 if peak_gpu else None,
                            "peak_process_ram_mib": peak_ram/2**20 if peak_ram else None}), encoding="utf-8")
            times.append(time.perf_counter()-start); outputs.append(token)
        else:
            token = inputs[0] if inputs else 42
            token = native.command(str(token)); times.append(time.perf_counter()-start); outputs.append(token)
        if not a.quiet: print(f"token 1: {times[0]:.3f} s, next={token}", flush=True)
        for index in range(1, a.tokens):
            if a.stop_at_eos and token in (248044,248046): break
            start = time.perf_counter()
            token = native.command(str(inputs[index] if inputs else token))
            times.append(time.perf_counter()-start); outputs.append(token)
            if not a.quiet: print(f"token {index+1}: {times[-1]:.3f} s, next={token}", flush=True)
    finally:
        done.set(); watcher.join(timeout=2); native.close()
    report = {"command": command,
              "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "engine_sha256": hashlib.sha256(a.engine.read_bytes()).hexdigest(),
              "benchmark_arguments": vars(a) | {"engine": str(a.engine), "model": str(a.model), "prompt": None if a.prompt_file else a.prompt},
              "prompt_file_sha256": hashlib.sha256(a.prompt_file.read_bytes()).hexdigest() if a.prompt_file else None,
              "environment": {key: os.environ[key] for key in ("LAMINA_CUDA_CACHE_MB", "LAMINA_CPU_THREADS", "LAMINA_EXPERT_POLICY", "LAMINA_PROFILE", "LAMINA_PINNED_UPLOADS", "LAMINA_MOE_GRAPHS", "LAMINA_PREFILL_BLAS", "LAMINA_HOST_REGISTER", "LAMINA_EXPERT_PIPELINE", "LAMINA_CACHE_POLICY", "LAMINA_ATOMIC_EXPERT_CACHE", "LAMINA_MATRIX_ATTN", "LAMINA_DEV_PROFILE", "LAMINA_PREFILL_PROGRESS", "LAMINA_Q8_PERSISTENT", "LAMINA_PREFETCH", "LAMINA_ADMIT_PER_LAYER", "LAMINA_ADMIT_MB", "LAMINA_PREFILL_CPU_TOKENS", "LAMINA_MTP_VOCAB", "LAMINA_TIMELINE") if key in os.environ},
              "first_token_seconds": times[0], "prompt_tokens": len(prompt),
              "startup_seconds": startup_seconds,
              "generated_tokens":len(outputs), "eos_positions":[i for i,t in enumerate(outputs) if t in (248044,248046)],
              "prefill_tokens_per_second": len(prompt)/times[0] if prompt else None,
              "later_tokens_per_second": (len(times)-1)/sum(times[1:]) if len(times)>1 else None,
              "later_median_seconds": statistics.median(times[1:]) if len(times)>1 else None,
              "peak_total_gpu_mib": peak_gpu/2**20 if peak_gpu is not None else None,
              "idle_total_gpu_mib": idle_gpu/2**20 if idle_gpu is not None else None,
              "peak_process_ram_mib": peak_ram/2**20 if peak_ram is not None else None,
              "seconds_per_token": times, "next_tokens": outputs, "native_stderr_tail": list(native.errors)}
    report["platform"] = platform.platform()
    report["python"] = platform.python_version()
    if psutil: report["system_ram_mib"] = psutil.virtual_memory().total / 2**20
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as processor:
            report["cpu"] = winreg.QueryValueEx(processor, "ProcessorNameString")[0].strip()
    if gpu is not None:
        report["gpu"] = pynvml.nvmlDeviceGetName(gpu)
        report["driver"] = pynvml.nvmlSystemGetDriverVersion()
        report["gpu_total_mib"] = pynvml.nvmlDeviceGetMemoryInfo(gpu).total / 2**20
        pynvml.nvmlShutdown()
    print(f"first token: {times[0]:.3f} s")
    if len(times)>1: print(f"later throughput: {report['later_tokens_per_second']:.3f} tokens/s")
    print(f"peak total GPU MiB: {report['peak_total_gpu_mib']}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(report, default=str, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
