"""Compare ordinary sampling and sampled MTP on the installed CUDA engine."""
import argparse
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from tools.lamina_chat import Engine
from tools.quickstart import load_config, model_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen3.6", "ornith"], default="qwen3.6")
    parser.add_argument("--temperature", type=float, default=.6)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--proposal", choices=["greedy", "sampled"], default="greedy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(); config["model"] = args.model; paths = model_paths(config)
    prompts = [
        "Explain how to frame three characters in a medium-wide shot. Give concrete camera and composition advice.",
        "Write a Python function that computes the bounding box of a list of 3D points. Explain empty input handling.",
        "Find the sum of the integers from 1 through 100 using two different methods and explain why they agree."]
    report = {"hardware": "RTX 4060 8 GB / Ryzen 7 1700X / 64 GB RAM", "model": args.model,
              "temperature": args.temperature, "top_p": .95, "top_k": 20,
              "context": 131072, "kv_type": "f16", "compute_mode": "fast", "proposal": args.proposal, "runs": []}
    os.environ["LAMINA_SPEC_STATS"] = "1"
    os.environ["LAMINA_MTP_PROPOSAL"] = args.proposal
    peak, stop = [0], threading.Event()
    def monitor():
        while not stop.is_set():
            reply = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if reply.returncode == 0: peak[0] = max(peak[0], int(reply.stdout.strip().splitlines()[0]))
            stop.wait(.5)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    engine = None
    try:
        for speculative in (False, True):
            os.environ["LAMINA_SAMPLED_MTP"] = "1" if speculative else "0"
            engine = Engine(paths["model"], paths["tokenizer"], Path(config["engine"]), cuda=True,
                max_context=131072, kv_cache="device", kv_type="f16", compute_mode="fast", mtp=paths["mtp"])
            engine.log_requests = True
            for iteration in range(args.runs + 1):
                for index, prompt in enumerate(prompts):
                    answer = ""; first = last = None; started = time.perf_counter(); final = None
                    for event in engine.events([{"role": "user", "content": prompt}], max_tokens=args.max_tokens,
                            temperature=args.temperature, top_p=.95, top_k=20, seed=42 + iteration):
                        content = event.get("delta", {}).get("content", "")
                        if content:
                            last = time.perf_counter()
                            if first is None: first = last
                            answer += content
                        if "finish_reason" in event: final = event
                    row = {"speculative": speculative, "warmup": iteration == 0, "iteration": iteration, "prompt": index,
                           "answer": answer, **final, "first_content_seconds": first - started if first else None}
                    if first and last > first:
                        row["later_tokens_per_second"] = (final["usage"]["completion_tokens"] - 1) / (last - first)
                    if engine.native:
                        stats = [line for line in engine.native.errors if line.startswith("speculation steps=")]
                        if stats: row["native_stats"] = stats[-1]
                    report["runs"].append(row); print(json.dumps(row), flush=True)
                    assert answer and final["usage"]["completion_tokens"] <= args.max_tokens
                    assert final["finish_reason"] in ("stop", "length")
            engine.close(); engine = None
        report["passed"] = True
    finally:
        if engine is not None: engine.close()
        stop.set(); thread.join(timeout=5)
        report["peak_total_gpu_mib_sampled"] = peak[0]
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.environ.pop("LAMINA_SAMPLED_MTP", None); os.environ.pop("LAMINA_SPEC_STATS", None)
        os.environ.pop("LAMINA_MTP_PROPOSAL", None)


if __name__ == "__main__": main()
