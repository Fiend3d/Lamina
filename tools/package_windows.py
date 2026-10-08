"""Package a portable Windows release and a downloadable engine-only asset.

Build with tools.build_windows --portable --cuda-arch "86;89;120" first.
Generated staging files, dependency wheels and ZIPs stay in ../Lamina-data.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tools.lamina_release import VERSION, ENGINE_ASSET, ARCHITECTURES, safe_extract, sha256

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "Lamina-data"
PYTHON_VERSION = "3.13.12"
PYTHON_SHA256 = "76f238f606250c87c6beac75dccd35ee99070a13490555936abb6cb64ecce3d0"
PORTABLE_ASSET = "lamina-windows-x64-portable.zip"


def cache_values(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.split(":", 1)[0]] = value
    return values


def validate_build(build):
    values = cache_values(build / "CMakeCache.txt")
    required = ("STRATA_PORTABLE", "LAMINA_ENABLE_CUDA", "LAMINA_PREFILL_BLAS", "LAMINA_CPU_QUANT")
    if any(values.get(key) != "ON" for key in required):
        raise ValueError("Release build must enable portable, CUDA, prefill BLAS and CPU quant options")
    if values.get("CMAKE_BUILD_TYPE") != "Release":
        raise ValueError("Release package needs a Release build")
    architectures = {part.split("-", 1)[0] for part in values.get("CMAKE_CUDA_ARCHITECTURES", "").split(";")}
    if not set(ARCHITECTURES) <= architectures:
        raise ValueError(f"Release build must contain all GPU architectures: {ARCHITECTURES}")
    return values


def write_zip(directory, output):
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                archive.write(path, path.relative_to(directory).as_posix())
    output.with_suffix(output.suffix + ".sha256").write_text(f"{sha256(output)}  {output.name}\n", encoding="ascii")


def stage_engine(build, stage, revision):
    values = validate_build(build)
    stage.mkdir(parents=True, exist_ok=True)
    for name in ("lamina-infer.exe", "lamina-gguf.exe", "lamina-sampling-check.exe",
                 "lamina-cuda-elementwise-check.exe", "lamina-cuda-attention-check.exe"):
        shutil.copy2(build / name, stage / name)
    dlls = sorted(build.glob("*.dll"))
    for prefix in ("cudart64_", "cublas64_", "cublasLt64_"):
        if not any(d.name.startswith(prefix) for d in dlls):
            raise ValueError(f"Missing release runtime DLL: {prefix}")
    for dll in dlls:
        shutil.copy2(dll, stage / dll.name)
    licenses = stage / "licenses"
    licenses.mkdir()
    shutil.copy2(ROOT / "LICENSE", licenses / "Lamina-LICENSE")
    shutil.copy2(ROOT / "third_party/llama.cpp/LICENSE", licenses / "llama.cpp-LICENSE")
    cuda = Path(values["CUDAToolkit_ROOT"]) / "licenses"
    cuda_version = json.loads((cuda.parent / "version.json").read_text())["cuda"]["version"]
    for name in ("cuda_cudart-LICENSE", "libcublas-LICENSE"):
        shutil.copy2(cuda / name, licenses / name)
    manifest = {"version": VERSION, "source_revision": revision, "cuda_architectures": list(ARCHITECTURES),
                "cpu_baseline": "AVX2/FMA/F16C/BMI2", "cuda_toolkit": cuda_version,
                "files": {p.name: sha256(p) for p in stage.iterdir() if p.is_file()}}
    (stage / "release.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def stage_python(stage, cache):
    archive = cache / f"python-{PYTHON_VERSION}-embed-amd64.zip"
    if not archive.exists() or sha256(archive) != PYTHON_SHA256:
        urllib.request.urlretrieve(f"https://www.python.org/ftp/python/{PYTHON_VERSION}/{archive.name}", archive)
    if sha256(archive) != PYTHON_SHA256:
        raise ValueError("Embedded Python checksum mismatch")
    python = stage / "python"
    safe_extract(archive, python)
    (python / "python313._pth").write_text("python313.zip\n.\nLib/site-packages\n..\nimport site\n", encoding="ascii")
    packages = python / "Lib/site-packages"
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--only-binary=:all:", "--platform", "win_amd64", "--python-version", "3.13",
                    "--implementation", "cp", "--abi", "cp313", "--target", str(packages),
                    "--report", str(stage / "python-packages.json"),
                    "-r", str(ROOT / "requirements.txt"), "-r", str(ROOT / "requirements-reference.txt")], check=True)
    # Import from the embedded runtime, not the developer's Python or venv.
    subprocess.run([str(python / "python.exe"), "-c",
                    "import tokenizers, jinja2, PIL, jsonschema, numpy, gguf; import tools.quickstart"],
                   cwd=stage, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DATA / "release-build")
    parser.add_argument("--output", type=Path, default=DATA / "releases")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Windows x64 packaging must be run on Windows")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise ValueError("Commit release source changes before packaging so the manifest names the exact source revision")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix="lamina-release-", dir=args.output) as temporary:
        stage = Path(temporary)
        engine = stage / "build-cuda"
        stage_engine(args.build_dir.resolve(), engine, revision)
        write_zip(engine, args.output / ENGINE_ASSET)
        for name in ("START-HERE.bat", "lamina.py", "setup.py", "LICENSE", "README.md",
                     "requirements.txt", "requirements-reference.txt"):
            shutil.copy2(ROOT / name, stage / name)
        (stage / "tools").mkdir()
        for name in ("__init__.py", "quickstart.py", "lamina_release.py", "lamina_model.py", "gguf_reader.py",
                     "lamina_assets.py", "lamina_mtp.py", "lamina_chat.py", "lamina_protocol.py",
                     "lamina_vision.py", "lamina_toolcalls.py"):
            source = ROOT / "tools" / name
            if source.is_file():
                shutil.copy2(source, stage / "tools" / name)
        (stage / "docs").mkdir()
        (stage / "serve").mkdir()
        shutil.copy2(ROOT / "serve/structured.py", stage / "serve/structured.py")
        for name in ("WINDOWS_RELEASE.md", "ADVANCED.md", "DEVELOPER_HANDOFF.md", "LAMINA_PORT.md"):
            shutil.copy2(ROOT / "docs" / name, stage / "docs" / name)
        (stage / "portable.json").write_text(json.dumps({"version": VERSION, "source_revision": revision}) + "\n")
        stage_python(stage, args.output)
        write_zip(stage, args.output / PORTABLE_ASSET)
    print(f"Release assets: {args.output}")


if __name__ == "__main__":
    main()
