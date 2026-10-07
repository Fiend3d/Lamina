"""Teacher-force held-out text against serial FP32 and fast CUDA engines.

The >=1024-token gate is mean KL <=0.1 nats and perplexity ratio <=1.05.
Large temporary FP32 logits are stored only in sibling Lamina-data.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from tools.lamina_model import FILENAME, SHA256


def main():
    root = Path(__file__).resolve().parents[1]
    data = root.parent / "Lamina-data"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, default=root / "build-cuda/lamina-quality-check.exe")
    p.add_argument("--model", type=Path, default=data / "models" / FILENAME)
    p.add_argument("--fixture", type=Path, default=root / "tests/lamina/quality_fixture.txt")
    p.add_argument("--report", type=Path, default=data / "quality-report.json")
    a = p.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(data / "tokenizer/tokenizer.json"))
    tokens = tokenizer.encode(a.fixture.read_text(encoding="utf-8"), add_special_tokens=False).ids[:1281]
    if len(tokens) < 1281: p.error("fixture needs >=1281 tokens")
    token_file = data / "quality-heldout-tokens.txt"
    token_file.write_text(" ".join(map(str, tokens)), encoding="utf-8")
    logits = data / "quality-baseline-logits.bin"
    evidence = {"engine_sha256": hashlib.sha256(a.engine.read_bytes()).hexdigest(),
                "fixture_sha256": hashlib.sha256(a.fixture.read_bytes()).hexdigest(),
                "token_ids_sha256": hashlib.sha256(token_file.read_bytes()).hexdigest(),
                "pinned_model_sha256": SHA256, "scored_tokens": len(tokens)-256,
                "environment": {k: v for k,v in os.environ.items() if k.startswith("LAMINA_")}, "runs": []}
    for mode in ("write", "compare"):
        command = [str(a.engine.resolve()), str(a.model.resolve()), str(token_file.resolve()), mode, str(logits.resolve())]
        result = subprocess.run(command, capture_output=True, text=True)
        evidence["runs"].append({"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        print(result.stdout, end="", flush=True)
        if result.returncode: break
    last = evidence["runs"][-1]
    found = re.search(r"mean_kl_nats=(\S+) perplexity_ratio=(\S+) argmax_agreement=(\S+)",last["stdout"])
    if found: evidence.update(zip(("mean_kl_nats","perplexity_ratio","argmax_agreement"), map(float,found.groups())))
    evidence["passed"] = last["returncode"] == 0 and found is not None
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    raise SystemExit(0 if evidence["passed"] else 1)


if __name__ == "__main__": main()
