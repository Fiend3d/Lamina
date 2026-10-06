"""Install the minimal NVIDIA Windows CUDA redistributables beside Lamina data."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="13.3.0")
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[2] / "Lamina-data/toolchains/cuda")
    parser.add_argument("--packages", nargs="+", default=["cuda_nvcc", "cuda_crt", "cuda_cudart", "cccl", "libnvvm", "visual_studio_integration", "libcublas"])
    args = parser.parse_args()
    base = "https://developer.download.nvidia.com/compute/cuda/redist/"
    with urllib.request.urlopen(base + f"redistrib_{args.version}.json", timeout=60) as response:
        manifest = json.load(response)
    args.destination.mkdir(parents=True, exist_ok=True)
    packages = args.packages
    for name in packages:
        package = manifest[name]["windows-x86_64"]
        archive = args.destination.parent / Path(package["relative_path"]).name
        if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != package["sha256"]:
            print(f"Downloading {name}", flush=True)
            urllib.request.urlretrieve(base + package["relative_path"], archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != package["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {archive}")
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                parts = Path(entry.filename).parts[1:]
                if not parts or entry.is_dir():
                    continue
                target = args.destination.joinpath(*parts).resolve()
                if not target.is_relative_to(args.destination.resolve()):
                    raise RuntimeError("Archive path escapes destination")
                # Keep each package's license instead of overwriting it.
                if len(parts) == 1 and "license" in parts[0].lower():
                    target = args.destination / "licenses" / f"{name}-{parts[0]}"
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as inp, target.open("wb") as out:
                    shutil.copyfileobj(inp, out)
        print(f"Verified {name}", flush=True)
    (args.destination / "version.json").write_text(json.dumps({"cuda": {"version": args.version}}))
    print(args.destination.resolve())


if __name__ == "__main__":
    main()
