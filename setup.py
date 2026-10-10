#!/usr/bin/env python3
"""Download and inspect Lamina's pinned model without loading its weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

from tools.lamina_model import FILENAME, SIZE, URL, validate_file
from tools.gguf_reader import GGUFFile

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "Lamina-data"
TOKENIZER_URL = "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/995ad96eacd98c81ed38be0c5b274b04031597b0/tokenizer.json"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
# Bahushruth's abliterated (uncensored) build of the pinned base model. Same
# qwen35moe graph and tokenizer; the revision and SHA-256 pin the exact weights.
ABLITERATED_REPO = "Bahushruth/Qwen3.6-35B-A3B-abliterated-v4-GGUF"
ABLITERATED_REVISION = "b8a9ab20c8bde880621a7bab65073c0078ef33f7"
ABLITERATED_FILENAME = "Qwen3.6-35B-A3B-abliterated-v4-Q4_K_M.gguf"
ABLITERATED_SIZE = 21_166_757_792
ABLITERATED_SHA256 = "0a9c6bcf5bf63877aa1eb686841e3d5538fcdc825500f1a715ce0f7e74da5f9f"
ABLITERATED_URL = f"https://huggingface.co/{ABLITERATED_REPO}/resolve/{ABLITERATED_REVISION}/{ABLITERATED_FILENAME}"


def download_tokenizer(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == TOKENIZER_SHA256:
        print(f"Already verified: {target}")
        return
    temporary = target.with_suffix(".json.part")
    urllib.request.urlretrieve(TOKENIZER_URL, temporary)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError("Downloaded tokenizer checksum differs from the pinned file")
    temporary.replace(target)
    print(f"Verified tokenizer: {target}")


def download_stream(url: str, size: int, target: Path, verify) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verify(target)
        print(f"Already verified: {target}")
        return
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError(f"Partial download is larger than the expected file: {partial}")
    failures = 0
    next_report = offset + 512 * 1024 * 1024
    while offset < size:
        start = offset
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset:
                    expected_range = f"bytes {offset}-"
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected_range):
                        raise RuntimeError("Model server ignored the resume range; existing partial file was preserved")
                elif response.status != 200:
                    raise RuntimeError(f"Unexpected download status {response.status}")
                with partial.open("ab" if offset else "wb") as output:
                    while block := response.read(8 * 1024 * 1024):
                        output.write(block)
                        offset += len(block)
                        if offset > size:
                            raise RuntimeError("Model server sent more data than the expected file size")
                        if offset >= next_report:
                            print(f"Downloaded {offset / 1e9:.1f}/{size / 1e9:.1f} GB", flush=True)
                            next_report = offset + 512 * 1024 * 1024
            if offset == start:
                raise OSError("Model server returned no bytes")
            if offset < size:
                print(f"Model stream ended at {offset}/{size}; resuming", file=sys.stderr)
            failures = 0
        except (OSError, TimeoutError) as error:
            failures += 1
            if failures > 10:
                raise RuntimeError(f"Model download failed repeatedly at {offset}/{size}: {error}") from error
            print(f"Model download interrupted at {offset}/{size}: {error}; retrying", file=sys.stderr)
            time.sleep(min(failures * 2, 20))
    verify(partial)
    partial.replace(target)
    print(f"Verified model: {target}")


def download(target: Path) -> None:
    download_stream(URL, SIZE, target, validate_file)


def verify_abliterated(path: Path) -> None:
    size = path.stat().st_size
    if size < ABLITERATED_SIZE:
        raise ValueError(f"{path.name} is {size} bytes, expected at least {ABLITERATED_SIZE}")
    if size > ABLITERATED_SIZE:
        # The setup step re-encodes ssm_alpha/beta to F32 in place, appending bytes;
        # a larger file is that patched copy, not a foreign model.
        return
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != ABLITERATED_SHA256:
        raise ValueError(f"{path.name} SHA-256 differs from the pinned abliterated model")


def download_abliterated(target: Path) -> None:
    download_stream(ABLITERATED_URL, ABLITERATED_SIZE, target, verify_abliterated)


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
    parser.add_argument("--abliterated", action="store_true", help="download the abliterated (uncensored) 20 GB GGUF")
    parser.add_argument("--tokenizer", action="store_true", help="download the pinned tokenizer.json")
    parser.add_argument("--verify", action="store_true", help="verify the complete model and SHA-256")
    parser.add_argument("--index", action="store_true", help="write a validated tensor offset index")
    args = parser.parse_args()
    target = args.data_dir / "models" / FILENAME
    abliterated_target = args.data_dir / "models" / ABLITERATED_FILENAME
    if args.download:
        download(target)
    if args.abliterated:
        download_abliterated(abliterated_target)
    if args.tokenizer:
        download_tokenizer(args.data_dir / "tokenizer" / "tokenizer.json")
    if args.verify or args.index:
        if not target.exists():
            parser.error(f"model missing: {target}; run --download first")
        validate_file(target)
        print(f"Model verified: {target}")
    if args.index:
        index(target, args.data_dir / "packs")
    if not (args.download or args.abliterated or args.verify or args.index or args.tokenizer):
        print(f"Lamina data: {args.data_dir.resolve()}")
        print(f"Model: {target} ({'present' if target.exists() else 'absent'})")
        print(f"Abliterated: {abliterated_target} ({'present' if abliterated_target.exists() else 'absent'})")
        print("Run --download to fetch the model, then --index to inspect its tensors.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Lamina setup: {error}", file=sys.stderr)
        sys.exit(1)
