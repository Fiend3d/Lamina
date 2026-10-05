"""Measure Lamina decode latency with a persistent model process.

The first token includes model paging and (on CUDA) weight uploads. Later
tokens reuse whatever fits in the device cache. Token 42 is only a repeatable
starting ID; this measures decode mechanics, not answer quality.
"""

import argparse
import os
import statistics
import subprocess
import time
from pathlib import Path

from tools.lamina_model import FILENAME

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model", type=Path, default=root.parent / "Lamina-data" / "models" / FILENAME
)
parser.add_argument("--engine", type=Path, required=True)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--cache-mb", type=int, help="lower CUDA weight-cache limit in MiB")
parser.add_argument("--tokens", type=int, default=4)
args = parser.parse_args()
if args.tokens < 1:
    parser.error("--tokens must be positive")
if args.cache_mb is not None and (not args.cuda or args.cache_mb < 1):
    parser.error("--cache-mb requires --cuda and a positive value")
environment = os.environ.copy()
if args.cache_mb is not None:
    environment["LAMINA_CUDA_CACHE_MB"] = str(args.cache_mb)
command = [str(args.engine.resolve()), str(args.model.resolve())]
if args.cuda:
    command.append("--cuda")
command.append("--interactive")

times = []
process = subprocess.Popen(
    command,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
    env=environment,
)
try:
    token = 42
    for index in range(args.tokens):
        start = time.perf_counter()
        process.stdin.write(f"{token}\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(process.stderr.read().strip() or "native engine exited")
        token = int(line.strip())
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"token {index + 1}: {elapsed:.3f} s, next={token}", flush=True)
finally:
    process.stdin.close()
    if process.poll() is None:
        process.terminate()
    process.wait(timeout=10)

if len(times) > 1:
    warm = times[1:]
    print(f"later tokens: median {statistics.median(warm):.3f} s")
    print(f"later throughput: {len(warm) / sum(warm):.3f} tokens/s")
else:
    print("Run at least two tokens to measure latency after the first token.")
