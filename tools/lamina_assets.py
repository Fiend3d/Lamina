"""Fetch pinned chat/vision assets. All data stays outside the source tree."""
import argparse
import hashlib
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "Lamina-data"
QWEN_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
VISION_REVISION = "a483e9e6cbd595906af30beda3187c2663a1118c"
ASSETS = {
    "vision/ornith-mmproj-BF16.gguf": ("ornith-ai/Ornith-1.5-35B-A3B-GGUF", "12393612fd4f730ff5aadc23e9b8f9648aa49ceb", "mmproj-Ornith-1.5-35B-BF16.gguf", 902822240, "sha256", "1921a36a85aee56cd2abd27f46701802c9d85a33474792e600df6c3b282a135d"),
    "tokenizer/chat_template.jinja": ("Qwen/Qwen3.6-35B-A3B", QWEN_REVISION, "chat_template.jinja", 7764, "git", "a8755d827c0a7b614c246c4060dfd58ab352a8ff"),
    "tokenizer/tokenizer_config.json": ("Qwen/Qwen3.6-35B-A3B", QWEN_REVISION, "tokenizer_config.json", 16718, "git", "28d96ff303d1d20350185caf4bf037045916ed35"),
    "vision/preprocessor_config.json": ("Qwen/Qwen3.6-35B-A3B", QWEN_REVISION, "preprocessor_config.json", 390, "git", "2ea84a437d448ff71b08df68fdd949d5cc4ebb64"),
    "vision/mmproj-F16.gguf": ("unsloth/Qwen3.6-35B-A3B-GGUF", VISION_REVISION, "mmproj-F16.gguf", 899283680, "sha256", "8971ee4f331ff0a4c609374f32984b3d4e6dc086c0aa35f1d637fad1829e887f"),
}


def verified(path, size, kind, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    h = hashlib.sha1(f"blob {size}\0".encode()) if kind == "git" else hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest() == digest


def download_assets(data=DATA, text_only=False, model="qwen3.6"):
    for name, (repo, revision, remote, size, kind, digest) in ASSETS.items():
        if model == "ornith" and name != "vision/ornith-mmproj-BF16.gguf":
            continue
        if model != "ornith" and name == "vision/ornith-mmproj-BF16.gguf":
            continue
        if text_only and name.startswith("vision/"):
            continue
        target = data / name
        if verified(target, size, kind, digest):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError(f"Oversized partial asset: {partial}")
        if offset < size:
            req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/{revision}/{remote}", headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(req, timeout=120) as response:
                if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise RuntimeError("Server ignored resume range; partial asset preserved")
                with partial.open("ab" if offset else "wb") as out:
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        offset += len(block)
                        if offset > size:
                            raise ValueError("Asset exceeds pinned size")
                        out.write(block)
        if not verified(partial, size, kind, digest):
            raise ValueError(f"Asset checksum mismatch: {partial}")
        partial.replace(target)
        print(f"Verified {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-only", action="store_true", help="skip optional vision assets")
    parser.add_argument("--model", choices=("qwen3.6", "ornith"), default="qwen3.6")
    args = parser.parse_args()
    download_assets(text_only=args.text_only, model=args.model)
