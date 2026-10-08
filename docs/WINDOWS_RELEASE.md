# Windows releases

The portable ZIP includes the native Lamina engine, CUDA/cuBLAS runtime DLLs,
an isolated Python 3.13 runtime and its Python packages. Users need no Python
installation, Git, Visual Studio, CMake, Ninja or CUDA toolkit. They still need
Windows x64, an AVX2/FMA/F16C/BMI2 CPU, an NVIDIA RTX 30/40/50 GPU (8 GB or more),
a driver compatible with CUDA 13.3, sufficient RAM (64 GB recommended), internet
for the first model download, and about 40 GB free disk space.

Download `lamina-windows-x64-portable.zip` from
[GitHub Releases](https://github.com/Fiend3d/Lamina/releases), extract it into
a writable folder, then double-click `START-HERE.bat`. Do not run inside the ZIP.
The model, tokenizer, optional MTP head and settings go into the sibling
`Lamina-data` folder. The first launch asks for context length and MTP, downloads
the pinned model and assets, and starts the server. Later launches reuse them.
The GPU driver is not bundled. The initial portable package supports text,
tools and reasoning; the optional image encoder is not included.

The first release, [v0.1.0](https://github.com/Fiend3d/Lamina/releases/tag/v0.1.0),
is published as a prerelease. Both ZIPs and checksums are public; source setup
can download the pinned engine automatically.

Source checkouts also support downloading the pinned engine-only release on
first setup. Its ZIP checksum and the manifest's per-file checksums are verified
before installation into `../Lamina-data/engine/v0.1.3`. The version is pinned in
`tools/lamina_release.py`, rather than following whatever release is newest.
An existing local developer build remains usable. Use `START-HERE.bat setup
--build-source` to compile from source instead.

## Building and checking a release

Run from a clean source checkout with the developer requirements installed:

```powershell
python -m tools.bootstrap_cuda
python -m tools.build_windows --portable --cuda-arch '86;89;120' --build-dir ../Lamina-data/release-build
../Lamina-data/release-build/lamina-sampling-check.exe
python -m tools.package_windows
```

The packager refuses a build without all three GPU architectures or the
portable/static-MSVC-runtime configuration. It preserves Lamina, llama.cpp,
NVIDIA and Python/dependency license notices, and records the source commit,
GPU architectures, Python package versions and file hashes. It emits two ZIPs
and their SHA-256 files in `../Lamina-data/releases`. Model weights are excluded.
Update the pinned release version before preparing a new tag. The tag workflow
builds the same assets and creates a draft release; compilation on a hosted
runner does not establish GPU correctness.

Before publishing, extract the portable ZIP into a fresh directory, remove
developer tools/Python/CUDA from PATH, and exercise `START-HERE.bat`, a streaming
tool request and MTP on available hardware. Check DLL imports and run the native
elementwise/attention checks. Record which GPU generations actually ran; support
compiled into the binary is not a runtime validation claim. Use
`docs/DEVELOPER_HANDOFF.md` for the full numerical/runtime gates when changing
layer math. The embedded Python archive is checksum-pinned; dependency wheels
are installed at packaging time and their versions/hashes are recorded in
`python-packages.json`.

Replace the application folder with a new extracted release when upgrading;
keep its sibling `Lamina-data`. Do not mix binaries or Python packages between
releases.
