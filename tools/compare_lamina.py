"""Resident Lamina counterpart to tools.compare_llama (no prompt-cache reuse)."""
import argparse
import datetime
import hashlib
import json
import os
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
    p.add_argument("--report-dir", type=Path, default=data / "comparison-lamina")
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--prompts", nargs="+", choices=list(PROMPTS), default=list(PROMPTS))
    p.add_argument("--prompt-file", type=Path)
    p.add_argument("--host-register", choices=("0", "1"), default="1", help="LAMINA_HOST_REGISTER for the engine")
    p.add_argument("--mtp", type=Path, help="packed MTP head: generate with greedy speculation (GENERATE)")
    a = p.parse_args()
    a.report_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LAMINA_HOST_REGISTER"] = a.host_register
    command = [str(a.engine.resolve()), str(data / "models" / FILENAME), "--cuda", "--compute-mode", "fast",
               "--kv-type", "f16", "--kv-cache", "device", "--max-context", str(a.context)]
    if a.mtp:
        command += ["--mtp", str(a.mtp.resolve())]
    command.append("--interactive")
    tokenizer = Tokenizer.from_file(str(data / "tokenizer/tokenizer.json"))
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
    started = time.perf_counter()
    native = NativeProcess(command)
    watcher.start()
    reports = []
    try:
        native.command("RESET", True)
        startup = time.perf_counter()-started
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
                if a.mtp and not a.prompt_file and a.tokens > 1:
                    tokens += native.generate(a.tokens-1, token)
                    finish = time.perf_counter()
                for _ in range(a.tokens-len(tokens)):
                    if a.prompt_file and token in (248044,248046):
                        break
                    token = native.command(str(token))
                    tokens.append(token)
                    finish = time.perf_counter()
                report = {"command": command, "engine_sha256": hashlib.sha256(a.engine.read_bytes()).hexdigest(),
                          "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          "prompt": name, "prompt_tokens": len(ids),
                          "prompt_token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                          "generated_tokens": len(tokens), "next_tokens": tokens,
                          "content": tokenizer.decode(tokens,skip_special_tokens=True),
                          "startup_seconds": startup, "loaded_first_token_seconds": first-start,
                          "startup_plus_first_token_seconds": startup+first-start if not reports else None,
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
