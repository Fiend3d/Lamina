Lamina's first portable Windows x64 release runs Qwen3.6-35B-A3B with a local
OpenAI-compatible API, streaming tool calls, thinking, and optional greedy MTP
speculative decoding.

Download **lamina-windows-x64-portable.zip**, extract into a writable folder,
and run **START-HERE.bat**. Python, the engine and runtime libraries are included;
no developer toolchain is needed. The model downloads separately on first use
into the sibling Lamina-data folder (22 GB plus optional MTP/assets).

Requirements: Windows x64, AVX2/FMA/F16C/BMI2 CPU, NVIDIA RTX 30/40/50 GPU with
at least 8 GB VRAM, CUDA 13.3-compatible driver, 64 GB RAM recommended, and about
40 GB free disk space. This package includes text inference; the optional image
encoder is excluded. SHA-256 files accompany both release ZIPs.

The engine-only ZIP is for the source launcher's automatic download after
publication. Ordinary users should choose the portable ZIP.

Draft validation on an RTX 4060 / Ryzen 7 1700X / 64 GB Windows machine:

- Hardware-free default CMake configure and lamina-gguf build passed.
- All 46 Python tests passed, including verified release download/install,
  corrupt checksums, archive traversal and compiler-free portable setup.
- Portable native sampling, CUDA elementwise and attention checks passed.
- Engine DLL imports contain cuBLAS and Windows system DLLs; no MSVC runtime
  dependency. Embedded Python imports and START-HERE.bat setup passed in a
  fresh extracted directory with PATH limited to Windows system directories.
- Full generation, streaming tools and MTP through this packaged engine are
  pending: the existing local server was kept running at the owner's request.
  Attempts with a restricted second-engine cache failed due to insufficient
  pinned working-set capacity. RTX 30/50 kernels compiled; their runtime is
  unverified. No new performance claim is made.

Finish the pending runtime checks before publishing this draft.
