"""Resident Lamina counterpart to tools.compare_llama (no prompt-cache reuse)."""
import argparse
import datetime
import hashlib
import json
import os
import re
import statistics
import threading
import time
from pathlib import Path

from tokenizers import Tokenizer
from lamina import chat_prompt
from tools.lamina_model import FILENAME
from tools.lamina_protocol import NativeProcess
from tools.performance_check import PROMPTS


def main():
    root = Path(__file__).resolve().parents[1]
    data = root.parent / "Lamina-data"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, default=root / "build-cuda/lamina-infer.exe")
    p.add_argument("--model", type=Path, default=data / "models" / FILENAME)
    p.add_argument("--tokenizer", type=Path, default=data / "tokenizer/tokenizer.json")
    p.add_argument("--report-dir", type=Path, default=data / "comparison-lamina")
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--prompts", nargs="+", choices=list(PROMPTS), default=list(PROMPTS))
    p.add_argument("--prompt-file", type=Path)
    p.add_argument("--fixed-tokens", action="store_true", help="also generate the full token count for prompt-file runs (capacity/throughput stress)")
    p.add_argument("--warmup-tokens", type=int, default=0, help="capture decode graphs before RESET and the measured requests")
    p.add_argument("--progress-file", type=Path, help="record long-prefill layer progress when LAMINA_PREFILL_PROGRESS=1")
    p.add_argument("--host-register", choices=("0", "1"), default="1", help="LAMINA_HOST_REGISTER for the engine")
    p.add_argument("--mtp", type=Path, help="packed MTP head: generate with greedy speculation (GENERATE)")
    a = p.parse_args()
    if a.tokens < 1 or a.repeats < 1 or not 0 <= a.warmup_tokens < a.context:
        p.error("invalid token, repetition or warmup count")
    a.report_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LAMINA_HOST_REGISTER"] = a.host_register
    os.environ.setdefault("LAMINA_SPEC_STATS", "1")  # the engine's per-request speculation line, kept in native-stderr.txt
    command = [str(a.engine.resolve()), str(a.model.resolve()), "--cuda", "--compute-mode", "fast",
               "--kv-type", "f16", "--kv-cache", "device", "--max-context", str(a.context)]
    if a.mtp:
        command += ["--mtp", str(a.mtp.resolve())]
    command.append("--interactive")
    tokenizer = Tokenizer.from_file(str(a.tokenizer.resolve()))
    import pynvml
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    peak = [pynvml.nvmlDeviceGetMemoryInfo(gpu).used]
    idle = peak[0]
    done = threading.Event()
    def monitor():
        last_layer = 0
        while not done.wait(.05):
            peak[0] = max(peak[0], pynvml.nvmlDeviceGetMemoryInfo(gpu).used)
            if a.progress_file:
                for line in reversed(native.errors.copy()):
                    match = re.fullmatch(r"long prefill layer=(\d+)/(\d+) tokens=(\d+)", line)
                    if match:
                        layer, layers, tokens = map(int, match.groups())
                        if layer != last_layer:
                            a.progress_file.parent.mkdir(parents=True, exist_ok=True)
                            a.progress_file.write_text(json.dumps({"layers_completed": layer, "layers": layers, "prompt_tokens": tokens, "process_elapsed_seconds": time.perf_counter()-started, "peak_total_gpu_mib": peak[0]/2**20}), encoding="utf-8")
                            last_layer = layer
                        break
    watcher = threading.Thread(target=monitor, daemon=True)
    started = time.perf_counter()
    native = NativeProcess(command)
    watcher.start()
    reports = []
    try:
        native.command("RESET", True)
        startup = time.perf_counter()-started
        warmup_seconds = 0.0
        if a.warmup_tokens:
            warmup_start = time.perf_counter()
            token = native.command("42")
            if a.mtp and a.warmup_tokens > 1:
                native.generate(a.warmup_tokens-1, token)
            else:
                for _ in range(a.warmup_tokens-1):
                    token = native.command(str(token))
            native.command("RESET", True)
            warmup_seconds = time.perf_counter()-warmup_start
        sources = {"retrieval": a.prompt_file.read_text(encoding="utf-8")} if a.prompt_file else {k:PROMPTS[k] for k in a.prompts}
        for name, text in sources.items():
            ids = tokenizer.encode(chat_prompt([{"role": "user", "content": text}]), add_special_tokens=False).ids
            for repeat in range(a.repeats):
                native.command("RESET", True)
                peak[0] = pynvml.nvmlDeviceGetMemoryInfo(gpu).used
                start = time.perf_counter()
                operation = "PROMPT 2048 " if len(ids)>2048 else "PREFILL "
                token = native.command(operation+" ".join(map(str,ids)))
                first = time.perf_counter()
                tokens = [token]
                finish = first
                if a.mtp and (not a.prompt_file or a.fixed_tokens) and a.tokens > 1:
                    tokens += native.generate(a.tokens-1, token)
                    finish = time.perf_counter()
                for _ in range(a.tokens-len(tokens)):
                    if a.prompt_file and not a.fixed_tokens and token in (248044,248046):
                        break
                    token = native.command(str(token))
                    tokens.append(token)
                    finish = time.perf_counter()
                report = {"command": command, "engine_sha256": hashlib.sha256(a.engine.read_bytes()).hexdigest(),
                          "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          "prompt": name, "prompt_tokens": len(ids),
                          "tokenizer_sha256": hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(),
                          "prompt_token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                          "generated_tokens": len(tokens), "next_tokens": tokens,
                          "eos_positions": [i for i, t in enumerate(tokens) if t in (248044, 248046)],
                          "content": tokenizer.decode(tokens,skip_special_tokens=True),
                          "startup_seconds": startup, "loaded_first_token_seconds": first-start,
                          "warmup_tokens": a.warmup_tokens, "warmup_seconds": warmup_seconds,
                          "startup_plus_first_token_seconds": startup+first-start if not reports and not a.warmup_tokens else None,
                          "later_tokens_per_second": (len(tokens)-1)/(finish-first) if len(tokens)>1 else None,
                          "peak_total_gpu_mib": peak[0]/2**20, "idle_total_gpu_mib": idle/2**20,
                          "gpu": pynvml.nvmlDeviceGetName(gpu), "driver": pynvml.nvmlSystemGetDriverVersion(),
                          "environment": {k:v for k,v in os.environ.items() if k.startswith("LAMINA_")}}
                if a.prompt_file:
                    report["retrieval_passed"] = "LAMINA847263" in report["content"]
                (a.report_dir/f"{name}-{repeat+1}.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
                reports.append(report)
                print(name,repeat+1,"ttft",round(first-start,3),"tps",report["later_tokens_per_second"],"VRAM",report["peak_total_gpu_mib"],flush=True)
        def median_speed(rows):
            values = [r["later_tokens_per_second"] for r in rows if r["later_tokens_per_second"] is not None]
            return statistics.median(values) if values else None
        summary = {"command":command,"startup_seconds":startup,
                   "prompt_medians":{name:median_speed([r for r in reports if r["prompt"]==name]) for name in sources},
                   "median_later_tokens_per_second":median_speed(reports)}
        (a.report_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    finally:
        done.set()
        watcher.join(timeout=2)
        native.close()
        (a.report_dir/"native-stderr.txt").write_text("\n".join(native.errors)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
