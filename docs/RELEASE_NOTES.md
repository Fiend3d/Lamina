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
- After the owner stopped the existing server, the exact uploaded portable ZIP
  passed START-HERE.bat model preload, greedy generation with MTP enabled,
  streaming text and usage, streaming function name/arguments, and streamed
  reasoning on the RTX 4060, with only Windows system directories in PATH.
- RTX 30/50 kernels compiled; their runtime is unverified. This is a functional
  packaging check, not a performance benchmark or a clean-Windows-install test.
  No new performance claim is made.

Published as v0.1.0 (prerelease). RTX 30/50 runtime validation remains unverified.
# v0.1.1

Reuse unchanged system/tool prefixes for CUDA device-KV server requests.
On RTX 4060 / Ryzen 7 1700X / 64 GiB, repeated 1,602-token Pi requests reached
the first token in 0.33 s versus 5.03 s with reuse disabled. The first request
still builds the prefix and took 4.53 s. Host-KV and image requests retain the
existing path. Old engine binaries are detected and use RESET. Disable reuse
with `LAMINA_PREFIX_CACHE=0`. See the benchmark record for exact conditions.

Fix runtime dependency installation and a Windows cancellation-test race in CI.
# v0.1.2

Reuse the whole conversation prefix across agent turns. v0.1.1 could only reuse an
unchanged system/tools block, so coding agents that resend the conversation each
turn re-prefilled the entire prompt every request: 41-47 s to the first token at
38K prompt tokens. The server now caches the message history up to the generation
prompt and advances that checkpoint when the conversation grows by an append,
prefilling only the new messages. On the RTX 4060 Ti 16 GB the second turn of a
grown conversation reached the first token in 0.5 s versus 5.2 s cold, with
identical output; a checkpoint that is not an exact BPE prefix still falls back to
RESET. The server log now reports new versus reused prompt tokens.

Fast mode also uses the overlapping expert pipeline while the GPU expert cache is
still filling and switches to the serial path once it is warm, roughly doubling
the first ~32 decode tokens of a cold request without changing steady decode;
`LAMINA_HYBRID=0` restores the previous behavior. Under cache eviction the
selected trajectory can differ from the previous release, as it already does
between the stream and cpu-miss policies. `LAMINA_PREFIX_CACHE=0` disables prefix
reuse. See the conversation-prefix and warm-up benchmark records for conditions.
