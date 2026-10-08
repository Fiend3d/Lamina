"""Build Lamina Release with MSVC and the sibling portable CUDA toolkit."""
import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "Lamina-data"


def quoted(path):
    value = str(path)
    if any(c in value for c in '\r\n"%'): raise ValueError("unsupported build path")
    return '"' + value + '"'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-cuda")
    parser.add_argument("--cuda-root", type=Path, default=DATA / "toolchains/cuda")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--vision", action="store_true", help="also build the pinned CPU image encoder")
    parser.add_argument("--portable", action="store_true", help="AVX2 baseline and static MSVC runtime for release packages")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--cuda-arch", default="89", help="CMAKE_CUDA_ARCHITECTURES (89 = RTX 4060, 86 = RTX 3050/3060/3090)")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]+(?:-(?:real|virtual))?(?:;[0-9]+(?:-(?:real|virtual))?)*", args.cuda_arch):
        parser.error("--cuda-arch must be a semicolon-separated list of CUDA architectures")
    if os.name != "nt": parser.error("this helper is for Windows")
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    visual_studio = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
    if not visual_studio: raise RuntimeError("Visual Studio C++ tools are missing")
    if args.vision or not args.cpu:
        source = ROOT / "third_party/llama.cpp"
        revision = (ROOT / "third_party/ggml/VERSION.txt").read_text().split()[0]
        if not source.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "checkout", revision], check=True)
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision: raise RuntimeError(f"ggml/vision source must be at pinned commit {revision}; existing checkout preserved")
    import ninja
    ninja_path = Path(ninja.BIN_DIR) / "ninja.exe"
    args.build_dir.mkdir(parents=True, exist_ok=True)
    configure = f"cmake -S {quoted(ROOT)} -B {quoted(args.build_dir.resolve())} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM={quoted(ninja_path)}"
    configure += f" -DSTRATA_PORTABLE={'ON' if args.portable else 'OFF'}"
    targets = "lamina-gguf lamina-infer lamina-sampling-check lamina-prefix-check"
    if not args.cpu:
        configure += f" -DLAMINA_ENABLE_CUDA=ON -DLAMINA_PREFILL_BLAS=ON -DLAMINA_CPU_QUANT=ON -DCMAKE_CUDA_ARCHITECTURES={quoted(args.cuda_arch)} -DCMAKE_CUDA_COMPILER={quoted(args.cuda_root.resolve() / 'bin/nvcc.exe')} -DCUDAToolkit_ROOT={quoted(args.cuda_root.resolve())}"
        targets += " lamina-cuda-projection-check lamina-cuda-ncols-check lamina-cuda-elementwise-check lamina-cuda-attention-check lamina-prefill-check lamina-kv-precision-check lamina-quality-check"
    else: configure += " -DLAMINA_ENABLE_CUDA=OFF -DLAMINA_PREFILL_BLAS=OFF -DLAMINA_CPU_QUANT=OFF"
    commands = ["@echo off", "call " + quoted(Path(visual_studio) / "VC/Auxiliary/Build/vcvars64.bat"),
                "if errorlevel 1 exit /b 1", configure, "if errorlevel 1 exit /b 1",
                f"cmake --build {quoted(args.build_dir.resolve())} --target {targets} -j {args.jobs}", "if errorlevel 1 exit /b 1"]
    if args.vision:
        commands += [f"cmake -S {quoted(ROOT / 'tools/vision')} -B {quoted(ROOT / 'build-vision')} -DLLAMA_DIR={quoted(source)} -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE={'ON' if args.portable else 'OFF'} -DCMAKE_BUILD_TYPE=Release",
                     "if errorlevel 1 exit /b 1", f"cmake --build {quoted(ROOT / 'build-vision')} --config Release --target strata-vision -j {args.jobs}"]
    commands += ["exit /b %errorlevel%"]
    script = args.build_dir.resolve() / "build-lamina.cmd"
    script.write_text("\n".join(commands) + "\n", encoding="utf-8")
    subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(script)], check=True, cwd=ROOT)
    if not args.cpu:
        import shutil
        for directory in (args.cuda_root / "bin", args.cuda_root / "bin/x64"):
            for pattern in ("cudart64*.dll", "cublas*.dll"):
                for dll in directory.glob(pattern):
                    target = args.build_dir / dll.name
                    # Windows locks loaded DLLs. Leave an identical runtime in
                    # place so a build can finish while a saved engine is tested.
                    if target.is_file() and hashlib.sha256(dll.read_bytes()).digest() == hashlib.sha256(target.read_bytes()).digest():
                        continue
                    shutil.copy2(dll, target)


if __name__ == "__main__": main()
