#!/usr/bin/env python3
"""Download and inspect Lamina's pinned model without loading its weights."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from tools.lamina_model import FILENAME, SIZE, URL, validate_file
from tools.gguf_reader import GGUFFile

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "Lamina-data"


def download(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        validate_file(target)
        print(f"Already verified: {target}")
        return
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > SIZE:
        raise ValueError(f"Partial download is larger than the pinned model: {partial}")
    request = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        if offset and response.status != 206:
            raise RuntimeError("Model server ignored the resume range; existing partial file was preserved")
        if not offset and response.status != 200:
            raise RuntimeError(f"Unexpected download status {response.status}")
        with partial.open("ab" if offset else "wb") as output:
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
    validate_file(partial)
    partial.replace(target)
    print(f"Verified model: {target}")


def index(target: Path, packs: Path) -> Path:
    model = GGUFFile(target)
    packs.mkdir(parents=True, exist_ok=True)
    destination = packs / "Qwen3.6-35B-A3B-UD-Q4_K_M.index.json"
    payload = {
        "source": str(target.resolve()),
        "data_start": model.data_start,
        "tensors": {
            tensor.name: {"shape": tensor.shape, "type": tensor.type_name,
                          "offset": model.data_start + tensor.offset,
                          "bytes": tensor.expected_bytes()}
            for tensor in model.tensors
        },
    }
    temporary = destination.with_suffix(".json.part")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(destination)
    print(f"Wrote tensor index: {destination}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA,
                        help="model and pack directory (default: ../Lamina-data)")
    parser.add_argument("--download", action="store_true", help="download the pinned 22 GB GGUF")
    parser.add_argument("--verify", action="store_true", help="verify the complete model and SHA-256")
    parser.add_argument("--index", action="store_true", help="write a validated tensor offset index")
    args = parser.parse_args()
    target = args.data_dir / "models" / FILENAME
    if args.download:
        download(target)
    if args.verify or args.index:
        if not target.exists():
            parser.error(f"model missing: {target}; run --download first")
        validate_file(target)
        print(f"Model verified: {target}")
    if args.index:
        index(target, args.data_dir / "packs")
    if not (args.download or args.verify or args.index):
        print(f"Lamina data: {args.data_dir.resolve()}")
        print(f"Model: {target} ({'present' if target.exists() else 'absent'})")
        print("Run --download to fetch the model, then --index to inspect its tensors.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Lamina setup: {error}", file=sys.stderr)
        sys.exit(1)
