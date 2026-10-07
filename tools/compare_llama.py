"""Compare CUDA llama-server with Lamina's identical prompt tokens.

Model/build/report assets remain in ../Lamina-data. Run one GPU engine at a time.
"""
import argparse
import datetime
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer
from lamina import chat_prompt
from tools.performance_check import PROMPTS
from tools.lamina_model import FILENAME


def main():
    root = Path(__file__).resolve().parents[1]
    data = root.parent / "Lamina-data"
    p = argparse.ArgumentParser(description=__doc__)
    prebuilt = data / "llama-prebuilt-b11474/llama-server.exe"
    p.add_argument("--server", type=Path, default=prebuilt if prebuilt.exists() else data / "llama-cuda/bin/llama-server.exe")
    p.add_argument("--report-dir", type=Path, default=data / "comparison-llama")
    p.add_argument("--cpu-moe", type=int, help="first N expert layers on CPU; otherwise auto placement")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--prompts", nargs="+", choices=list(PROMPTS), default=list(PROMPTS))
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--prompt-file", type=Path)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--host-register", action="store_true", help="pin mmap weights via GGML_CUDA_REGISTER_HOST=1")
    p.add_argument("--revision", help="actual source revision of a downloaded release")
    p.add_argument("--load-mode", choices=("auto", "none"), help="model load policy on newer llama.cpp releases")
    a = p.parse_args()
    a.report_dir.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output([str(a.server.resolve()), "--version"], stderr=subprocess.STDOUT, text=True).strip()
    command = [str(a.server.resolve()), "-m", str(data / "models" / FILENAME),
               "-c", str(a.context), "-np", "1", "-t", str(a.threads), "-tb", str(a.threads),
               "-b", "2048", "-ub", "2048", "-fa", "on", "-ctk", "f16", "-ctv", "f16",
               "--host", "127.0.0.1", "--port", "8097", "--fit-target", "1024", "--no-warmup", "-lv", "4"]
    if a.load_mode:
        command += ["--load-mode", a.load_mode]
    environment = os.environ.copy()
    if a.host_register:
        environment["GGML_CUDA_REGISTER_HOST"] = "1"
    if a.cpu_moe is not None:
        command += ["-ngl", "99", "--n-cpu-moe", str(a.cpu_moe)]
    tokenizer = Tokenizer.from_file(str(data / "tokenizer/tokenizer.json"))
    url = "http://127.0.0.1:8097"
    import pynvml
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    peak = [pynvml.nvmlDeviceGetMemoryInfo(gpu).used]
    idle = peak[0]
    done = threading.Event()
    def monitor():
        while not done.wait(.05):
            peak[0] = max(peak[0], pynvml.nvmlDeviceGetMemoryInfo(gpu).used)
    watcher = threading.Thread(target=monitor, daemon=True)
    log = (a.report_dir / "server.log").open("w", encoding="utf-8")
    started = time.perf_counter()
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    watcher.start()
    reports = []
    try:
        while True:
            if server.poll() is not None:
                raise RuntimeError("llama-server exited; inspect server.log")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.HTTPError):
                pass
            if time.perf_counter() - started > 180:
                raise TimeoutError("server startup exceeded 180 seconds")
            time.sleep(.2)
        startup = time.perf_counter() - started
        sources = {"retrieval": a.prompt_file.read_text(encoding="utf-8")} if a.prompt_file else {k: PROMPTS[k] for k in a.prompts}
        for name, text in sources.items():
            ids = tokenizer.encode(chat_prompt([{"role": "user", "content": text}]), add_special_tokens=False).ids
            for repeat in range(a.repeats):
                peak[0] = pynvml.nvmlDeviceGetMemoryInfo(gpu).used
                body = {"prompt": ids, "n_predict": a.tokens, "temperature": 0, "seed": 0,
                        "stream": True, "cache_prompt": False, "ignore_eos": not bool(a.prompt_file),
                        "return_tokens": True}
                request = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
                start = time.perf_counter()
                first = None
                finish = None
                tokens = []
                content = []
                final = {}
                # A deadline timer bounds the whole streamed request, including silent prefill.
                timer = threading.Timer(a.timeout, server.kill)
                timer.start()
                try:
                    with urllib.request.urlopen(request, timeout=a.timeout) as response:
                        for line in response:
                            if not line.startswith(b"data: "):
                                continue
                            event = json.loads(line[6:])
                            if event.get("tokens"):
                                if first is None:
                                    first = time.perf_counter()
                                tokens.extend(event["tokens"])
                                finish = time.perf_counter()
                            content.append(event.get("content", ""))
                            if event.get("stop"):
                                final = event
                except Exception as error:
                    (a.report_dir / f"{name}-{repeat+1}-failed.json").write_text(json.dumps({
                        "command": command, "prompt_tokens": len(ids), "timeout_seconds": a.timeout,
                        "elapsed_seconds": time.perf_counter()-start, "error": str(error),
                        "peak_total_gpu_mib": peak[0]/2**20}, indent=2)+"\n", encoding="utf-8")
                    raise
                finally:
                    timer.cancel()
                if first is None:
                    raise RuntimeError("no generated tokens returned")
                report = {"command": command, "engine_sha256": hashlib.sha256(a.server.read_bytes()).hexdigest(),
                          "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          "revision": a.revision, "version": version,
                          "model": str(data / "models" / FILENAME), "prompt": name,
                          "host_register": a.host_register, "gpu": pynvml.nvmlDeviceGetName(gpu),
                          "driver": pynvml.nvmlSystemGetDriverVersion(),
                          "prompt_tokens": len(ids), "prompt_token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                          "generated_tokens": len(tokens), "next_tokens": tokens, "content": "".join(content),
                          "startup_seconds": startup, "loaded_first_token_seconds": first - start,
                          "startup_plus_first_token_seconds": startup + first - start if not reports else None,
                          "later_tokens_per_second": (len(tokens)-1)/(finish-first) if len(tokens)>1 else None,
                          "peak_total_gpu_mib": peak[0]/2**20, "idle_total_gpu_mib": idle/2**20,
                          "server_timings": final.get("timings"), "request": body}
                if a.prompt_file:
                    report["retrieval_passed"] = "LAMINA847263" in report["content"]
                path = a.report_dir / f"{name}-{repeat+1}.json"
                path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
                reports.append(report)
                print(name, repeat+1, "ttft", round(first-start,3), "tps", report["later_tokens_per_second"], "VRAM", report["peak_total_gpu_mib"], flush=True)
        def median_speed(rows):
            values = [r["later_tokens_per_second"] for r in rows if r["later_tokens_per_second"] is not None]
            return statistics.median(values) if values else None
        summary = {"command": command, "startup_seconds": startup,
                   "prompt_medians": {name: median_speed([r for r in reports if r["prompt"]==name]) for name in sources},
                   "median_later_tokens_per_second": median_speed(reports)}
        (a.report_dir / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    finally:
        done.set()
        watcher.join(timeout=2)
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        log.close()


if __name__ == "__main__":
    main()
