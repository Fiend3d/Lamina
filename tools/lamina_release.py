"""Download and verify the pinned Windows engine release (no compiler needed)."""
import hashlib
import json
import re
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

REPOSITORY = "Fiend3d/Lamina"
VERSION = "v0.2.0"
ENGINE_ASSET = "lamina-windows-x64-engine.zip"
ARCHITECTURES = ("86", "89", "120")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_extract(archive, destination):
    """Validate all paths before writing, including Windows drive/ADS names."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            parts = PurePosixPath(info.filename.replace("\\", "/")).parts
            if (not parts or parts[0] == "/" or any(p in (".", "..") or ":" in p for p in parts)
                    or stat.S_ISLNK(info.external_attr >> 16)
                    or not destination.joinpath(*parts).resolve().is_relative_to(destination)):
                raise ValueError(f"Unsafe release archive path: {info.filename}")
        source.extractall(destination)


def validate_runtime(directory, arch):
    directory = Path(directory)
    manifest = json.loads((directory / "release.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION or arch not in manifest.get("cuda_architectures", []):
        raise ValueError(f"Release {VERSION} does not support compute capability {arch}")
    files = manifest.get("files", {})
    if ("lamina-infer.exe" not in files or any(not any(n.startswith(prefix) for n in files)
            for prefix in ("cudart64_", "cublas64_", "cublasLt64_"))):
        raise ValueError("Release manifest lacks the inference engine or CUDA/cuBLAS runtimes")
    for name, digest in files.items():
        if Path(name).name != name or ":" in name or "\\" in name or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid release manifest entry")
        if sha256(directory / name) != digest:
            raise ValueError(f"Release file checksum mismatch: {name}")
    return manifest


def install_engine(data, arch):
    if arch not in ARCHITECTURES:
        raise ValueError(f"No prebuilt engine for compute capability {arch}; use --build-source --cuda-arch {arch}")
    destination = Path(data) / "engine" / VERSION
    if (destination / "release.json").is_file():
        validate_runtime(destination, arch)
        return destination / "lamina-infer.exe"
    cache = Path(data) / "releases"
    cache.mkdir(parents=True, exist_ok=True)
    base = f"https://github.com/{REPOSITORY}/releases/download/{VERSION}/"
    request = urllib.request.Request(base + ENGINE_ASSET + ".sha256", headers={"User-Agent": "Lamina"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            expected = response.read(4096).decode("ascii").split()[0]
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid release archive checksum")
        archive = cache / ENGINE_ASSET
        if not archive.is_file() or sha256(archive) != expected:
            partial = archive.with_suffix(".zip.part")
            print(f"Downloading Lamina {VERSION} prebuilt engine", flush=True)
            with urllib.request.urlopen(base + ENGINE_ASSET, timeout=120) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            if sha256(partial) != expected:
                raise ValueError("Release archive checksum mismatch; download was not installed")
            partial.replace(archive)
    except OSError as error:
        raise RuntimeError(f"Cannot download Lamina {VERSION}: {error}. "
                           "Download the portable ZIP from GitHub Releases, or use --build-source.") from error
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        stage = Path(temporary) / "engine"
        safe_extract(archive, stage)
        validate_runtime(stage, arch)
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(destination)
    return destination / "lamina-infer.exe"
