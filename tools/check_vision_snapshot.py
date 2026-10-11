"""GPU vision handoff/full-prefill parity and latency check; assets stay in Lamina-data."""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tools.lamina_chat import Engine
from tools.lamina_hardware import describe as describe_hardware
from tools.lamina_media import image_content
from tools.quickstart import load_config, model_paths, selected_vision, vision_projector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen3.6", "ornith"], default="qwen3.6")
    parser.add_argument("--filler-lines", type=int, default=1800)
    parser.add_argument("--audit-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(); config.update(model=args.model, vision=True, vision_device="gpu")
    paths = model_paths(config)
    for word in ["ALPHA", "BETA", "GAMMA"]:
        image = Image.new("RGB", (640, 360), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 90)
        draw.text((60, 120), word, fill="black", font=font)
        image.save(args.output / (word + ".png"))
    report = {"hardware": describe_hardware(), "model": args.model,
              "settings": {"max_context": 131072, "kv_type": "f16", "kv_cache": "device", "compute_mode": "fast"},
              "requests": []}
    stop, peak = threading.Event(), [0]
    def monitor():
        while not stop.is_set():
            result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if result.returncode == 0: peak[0] = max(peak[0], int(result.stdout.strip().splitlines()[0]))
            stop.wait(.5)
    monitor_thread = threading.Thread(target=monitor, daemon=True); monitor_thread.start()
    def create(mtp=True):
        engine = Engine(paths["model"], paths["tokenizer"], Path(config["engine"]), cuda=True,
            max_context=131072, kv_cache="device", kv_type="f16", compute_mode="fast",
            mtp=paths["mtp"] if mtp else None, vision_engine=selected_vision(config), vision_gpu=True,
            vision_projector=vision_projector(config))
        engine.log_requests = True
        return engine
    def request(engine, label, messages, **options):
        started = time.perf_counter(); first = last = None; answer = ""; final = None
        for event in engine.events(messages, max_tokens=options.pop("max_tokens", 48), **options):
            content = event.get("delta", {}).get("content", "")
            if content:
                last = time.perf_counter()
                if first is None: first = last
                answer += content
            if "finish_reason" in event: final = event
        row = dict(label=label, answer=answer, **final)
        row["first_content_seconds"] = first - started if first else None
        if first and last > first:
            row["later_tokens_per_second"] = (final["usage"]["completion_tokens"] - 1) / (last - first)
        report["requests"].append(row); print(json.dumps(row), flush=True)
        return answer, final
    filler = "The observers recorded rainfall and checked all instruments before each measurement.\n" * args.filler_lines
    base = [{"role": "user", "content": [{"type": "text", "text": filler + "Read this image word. Reply with its word only."}, image_content(args.output / "ALPHA.png")]}]
    engine = None
    try:
        # Matched old/new handoffs include a genuinely uncached new image.
        answers = []
        for enabled in (False, True):
            os.environ["LAMINA_PREFIX_SNAPSHOT"] = "1" if enabled else "0"
            label = "snapshot" if enabled else "baseline"
            engine = create()
            first_answer, _ = request(engine, label + "-initial", base)
            follow = base + [{"role": "assistant", "content": first_answer}, {"role": "user", "content": [
                {"type": "text", "text": "Read the new image word. Reply with its word only."}, image_content(args.output / "BETA.png")]}]
            answer, final = request(engine, label + "-new-image", follow)
            assert "BETA" in answer.upper(), answer
            assert (final["timings"]["cached_prompt_tokens"] > 0) == enabled, final
            answers.append(answer)
            repeat, _ = request(engine, label + "-repeat", follow)
            assert repeat == answer, (repeat, answer)
            # Another handoff must preserve the earlier images' mRoPE history too.
            third = follow + [{"role": "assistant", "content": answer}, {"role": "user", "content": [
                {"type": "text", "text": "Give all three image words in chronological order, nothing else."}, image_content(args.output / "GAMMA.png")]}]
            ordered, _ = request(engine, label + "-third-image", third)
            assert all(word in ordered.upper() for word in ("ALPHA", "BETA", "GAMMA")), ordered
            if enabled:
                # Same request with no checkpoint must match greedy cached output.
                engine._stop_native()
                full, _ = request(engine, "snapshot-third-full-prefill", third)
                assert full == ordered, (full, ordered)
            engine.close(); engine = None
        assert answers[0] == answers[1], answers
        report["snapshot_checks_passed"] = True
        # Repetition audit: cached/uncached and MTP/single-token generation.
        prompt = [{"role": "user", "content": "Return a numbered list of exactly five different camera-framing tips, then stop. " + filler[:2500]}]
        outputs = []
        for mtp in (True, False):
            engine = create(mtp)
            original, _ = request(engine, "loop-audit-" + ("mtp" if mtp else "single"), prompt, max_tokens=args.audit_tokens)
            repeated, _ = request(engine, "loop-audit-repeat-" + ("mtp" if mtp else "single"), prompt, max_tokens=args.audit_tokens)
            report.setdefault("greedy_replay_matches", {})["mtp" if mtp else "single"] = original == repeated
            outputs.append(original)
            engine.close(); engine = None
        report["greedy_mtp_single_match"] = outputs[0] == outputs[1]
        report["passed"] = True
    finally:
        if engine is not None: engine.close()
        os.environ.pop("LAMINA_PREFIX_SNAPSHOT", None)
        stop.set(); monitor_thread.join(timeout=5)
        report["peak_total_gpu_mib_sampled"] = peak[0]
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
