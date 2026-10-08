"""Windows resident-engine TTFT benchmark: ENGINE OUTPUT REQUEST.json [--profile].
Install requirements-reference.txt plus nvidia-ml-py. Toggle LAMINA_PREFIX_CACHE=0
for the matched baseline. First run populates cache; median uses last three.
"""
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time
import winreg
import hashlib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "Lamina-data"
sys.path.insert(0, str(ROOT))
from tools.lamina_chat import Engine
from tools.lamina_model import FILENAME
import pynvml

os.environ["LAMINA_HOST_REGISTER"] = "1"
os.environ["LAMINA_PREFILL_STATS"] = "1"
if "--profile" in sys.argv:
    os.environ["LAMINA_TIMELINE"] = "1"
engine_path = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
fixture = json.loads(Path(sys.argv[3]).read_text())
fixture.pop("model", None)
fixture.pop("stream_options", None)
fixture.pop("stream", None)
fixture["enable_thinking"] = False
fixture["max_tokens"] = min(fixture.get("max_tokens", 32), 256)
fixture["temperature"] = 0
messages = fixture.pop("messages")
e = Engine(DATA / "models" / FILENAME, DATA / "tokenizer/tokenizer.json", engine_path,
           cuda=True, max_context=32768, kv_type="f16", kv_cache="device", compute_mode="fast",
           mtp=DATA / "mtp/qwen36-mtp-q8_0.gguf")
pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
done = threading.Event()
peak = [0]


def monitor():
    while not done.wait(.02):
        peak[0] = max(peak[0], pynvml.nvmlDeviceGetMemoryInfo(gpu).used)


threading.Thread(target=monitor, daemon=True).start()
reports = []
try:
    e.completion([{"role": "user", "content": "Hi"}], max_tokens=1)
    for repeat in range(4):
        start = time.perf_counter()
        visible = None
        deltas = []
        for event in e.events(messages, **fixture, stream=True):
            if event.get("delta") and visible is None:
                visible = time.perf_counter() - start
            deltas.append(event)
        final = deltas[-1]
        report = {"repeat": repeat, **final, "visible_first_seconds": visible,
                  "content": "".join(event.get("delta", {}).get("content", "") for event in deltas)}
        reports.append(report)
        print("repeat", repeat, "prompt", final["usage"]["prompt_tokens"], "first", final["timings"]["first_token_seconds"], "visible", visible, flush=True)
    stderr = list(e.native.errors)
finally:
    e.close()
    done.set()
    gpu_name = pynvml.nvmlDeviceGetName(gpu)
    driver = pynvml.nvmlSystemGetDriverVersion()
    pynvml.nvmlShutdown()
with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
    cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
result = {"reports": reports, "median_first_seconds": statistics.median(r["timings"]["first_token_seconds"] for r in reports[1:]),
          "peak_total_gpu_mib": peak[0] / 2**20, "cpu": cpu, "platform": platform.platform(),
          "engine": str(engine_path), "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
          "gpu": gpu_name, "driver": driver, "environment": {k:v for k,v in os.environ.items() if k.startswith("LAMINA_")},
          "native_stderr": stderr}
(output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("warm median", result["median_first_seconds"], "peak", result["peak_total_gpu_mib"], flush=True)
