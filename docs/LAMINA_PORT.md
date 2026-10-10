# Qwen3.6 port status

Source launchers support saved `--vision-device cpu|gpu`. GPU mode selects the
separate CUDA encoder in `build-vision-gpu/bin/strata-vision.exe`, built with
`python -m tools.build_windows --vision-gpu --vision-only --cuda-arch 89` on
RTX 4060. CPU/portable builds are preserved. `lamina.py --vision-gpu` passes
`--gpu` to mtmd. GPU image requests stop the native model before encoding and
close the encoder before model inference; text startup does not load vision.
This fixes the resident-encoder OOM reported at 109K pi tokens on RTX 4060 8 GB.
Source now caches image embeddings in a bounded 256 MiB per-engine RAM LRU,
keyed by image bytes. Repeated images keep the text engine resident. CUDA
device-KV conversation checkpoints include image identity and order, and retain
the native image mRoPE state; follow-ups prefill only appended content. New GPU
images or evicted embeddings still require a model reload and full prefill.
HTTP URLs are fetched again; server restarts clear both caches.
See [image cache validation](../bench/results/2026-10-10-image-cache/README.md)
for cached/full-prefill OCR comparison and 102K image-context reuse.
Image-containing CUDA device-KV prefixes now route long text spans on either
side of images through the existing layer-major PROMPT path. Native image
position handling and the VRAM-exclusive encoder lifecycle are preserved;
this is a Python request-routing change, without native math changes.
See [image prefill measurements](../bench/results/2026-10-10-image-prefill/README.md)
for the paired comparison and correctness checks.
See the [109K/129K regression](../bench/results/2026-10-10-gpu-vision-memory/README.md)
for checks of GPU release before long text inference.
`/health` reports the
selected device and requests log per-image encoding progress. Request first-token
timing now includes image encoding. v0.2.0 packages the API server, both CPU/GPU
encoders, video decoder and pi video skill. The GPU encoder uses portable CPU
flags and SM86/89/120; RTX 30/50 runtime coverage remains unverified.
See [GPU vision runtime validation](../bench/results/2026-10-10-gpu-vision/README.md)
for first-content timing, later decode and sampled peak VRAM on RTX 4060.


The launcher now saves CPU image input (`--vision on|off`) and sampled terminal
video input (`--video-input on|off`); enabling video enables vision. `chat --image`
and `chat --video` accept local files, `--prompt` makes a single request, and
interactive chat accepts `/image "path" question` and `/video "path" question`.
`tools/lamina_media.py` uses the bundled imageio-ffmpeg 0.6.0 decoder to extract
up to 4 frames at 1-second intervals by default (overrides: `--video-frames`
1..16 and positive `--video-interval`), with timestamps and a 1024-pixel bound.
This uses ordinary independent-image embeddings, without audio or native video
mRoPE. HTTP clients continue to submit image_url content; video_url is unsupported.
Packaging includes the helper and decoder wheel. Model data and temporary frames
stay in sibling Lamina-data; temporary frames are removed after conversion.
See [launcher media validation](../bench/results/2026-10-10-launcher-media/README.md).

Pi integration includes a project `lamina-video` skill and an existing-server
client in `tools.lamina_media`; the image provider advertises text/image input.
Video questions are answered from sampled image frames via HTTP, without a
second engine. Direct video attachments and audio remain unsupported.


The quickstart launcher saves model, context length and MTP. `--reconfigure`
asks the settings again with saved values as defaults; an explicit `--model`
skips its question. See the [user guide](../README.md) for setup and commands.

CUDA device-KV server requests can reuse a prefix checkpoint across turns. v0.1.1
reused an exact unchanged system/tools prefix; v0.1.2 caches the whole message
history up to the generation prompt and advances that checkpoint when the
conversation grows by an append (restore, prefill only the new messages, re-cache),
so an agent client reuses everything but the new content. Independent DeltaNet
state snapshots and unchanged prefix attention KV; a checkpoint that is not an
exact token/image prefix falls back to RESET, as does host KV. See the
[conversation-prefix validation](../bench/results/2026-10-09-conversation-prefix/README.md)
and the earlier [system/tools record](../bench/results/2026-10-08-rtx4060-prefix/README.md).

The same `qwen35moe` graph also runs `Ornith-1.5-35B-A3B` (Q4_K_M) after the
contract tolerates its in-file MTP layer and the grouped qkv/gate and q/k/v
dispatches (single and pair) split per encoding; its `ssm_alpha/beta` are
re-encoded to F32 and its in-file MTP head is packed to an `mtp.*` side file.
`START-HERE.bat --model ornith` selects it. No reference-quality claim is made.
Short-prompt RTX 4060 8 GB measurements are 36.35 tokens/s without MTP and
40.79 with MTP; mode outputs differ. See
[the RTX 4060 benchmark](../bench/results/2026-10-09-ornith-rtx4060/README.md) and
[the Ornith bring-up](../bench/results/2026-10-09-ornith-bringup/README.md).

The latest short-prompt scheduling measurement, with 131072 capacity configured,
is 39.17 tokens/s ordinary Ornith decode and 41.27 with MTP; ordinary pinned Qwen
measures 38.53 (43.88 with MTP). CPU-expert admission now overlaps the CPU wait; ordinary MoE
graphs retain 512 variants while MTP retains 128. Matched per-mode output tokens
are unchanged. Full-context and cached 80K server measurements are separate;
see the [decode admission record](../bench/results/2026-10-09-decode-admission-128k/README.md).

Windows distribution: portable ZIP packaging includes Python, dependencies,
SM86/89/120 native binaries and CUDA runtime DLLs; source setup can download a
checksum-verified pinned engine instead of compiling. `--build-source` keeps
the developer path. See [Windows releases](WINDOWS_RELEASE.md). The v0.1.4 ZIP
supports text/tools/reasoning and images for both Qwen3.6 and Ornith. It includes
the portable CPU encoder; setup downloads each model's own verified projector. Compiled
GPU coverage does not establish runtime correctness on all those GPUs.


Fast mode now runs cache-missed experts on the CPU by default. On an RTX 3050
8 GB machine the matched comparison measured **35.84 tokens/s for Lamina versus
28.56 for CUDA llama.cpp b11474**. The **40 tokens/s target is not met**, and no
Strata parity is claimed. See [the CPU-expert record](../bench/results/2026-10-07-rtx3050-cpu-experts/README.md);
earlier RTX 4060 records are in [the previous validation](../bench/results/2026-10-06-strata-plan/VALIDATION.md).

Lamina uses the pinned Unsloth Qwen3.6-35B-A3B UD-Q4_K_M GGUF at revision
`a483e9e6cbd595906af30beda3187c2663a1118c`, engine architecture `qwen35moe`.
Its 40-layer graph has 30 recurrent DeltaNet and ten gated attention layers,
256 routed experts, top-eight routing and a shared expert. Strata source
baseline: `6f32ec070f23ced9f50e704d854d775da52591ab`.

Implemented in the native Lamina path:

- GGUF architecture/tensor guards, checksum-pinned downloads and scalar reference path.
- CUDA layer math, FP32 activations, accurate quantized projections, persistent
  recurrent/KV state, partial interleaved mRoPE and original-slot MoE reduction.
- Bounded expert streaming, pinned uploads, LRU residency, dynamic pointer-table
  CUDA MoE graphs, DeltaNet graphs and an optional persistent CPU miss pool.
- Causal layer-major batched text/image prefill, GPU routing/gather/SwiGLU/scatter,
  strict FP32 GEMM for larger batches, native kernels for small groups, and
  prefill scratch reclamation before decode.
- Fused shared-KV FP32 attention without the batch partial matrix, all-column
  history reuse and event-protected double-buffered host transfers.
- 32K default context and 128K complete FP32 RAM-backed KV with bounded staging;
  optional lossy FP16 storage in host or device memory, selected with --kv-type.
- Resident CLI/chat API, SSE UTF-8 streaming/usage, seeded sampling and stops,
  thinking, tool/schema validation, CPU image encoder/multiple images,
  request reset, context preflight and disconnect cancellation/recovery.
- Reproducible Release Windows build and verified sibling-directory model,
  tokenizer/template, mmproj and portable CUDA/cuBLAS assets.

The inherited Qwen3.8 graph remains disabled for Lamina. FP32 activation
accuracy is retained by default. Optional `--compute-mode fast` enables
Strata Q8 activations and BF16 batched projections/matrix attention, checked
against a separate KL/perplexity gate. It is lossy and is not FP32 equivalence. Structured
JSON is prompted/validated, not guaranteed by a grammar decoder. Video, a web
UI and full OpenAI API feature parity are not included.

On this RTX 4060 8 GB/Ryzen 1700X/64 GB Windows machine, the independent
four-token 40-layer reference passes at maximum hidden difference
`3.59125275e-6`, next token369 and matching logit within `3e-6`.
The state suite passes full-layer causal prefill/decode/reset and synthetic
image mRoPE with both host/device KV, maximum difference `8.85501504e-6`,
below the unchanged `1e-5` gate. Isolated attention checks exercise all
131,072 positions including a distant-first-token signal. Twenty Python
checks pass. These numerical tests cover specific inputs, not all generation.

Measured on this machine: matched short decode reaches a three-run median
14.129 tokens/s versus 13.260 for the original Lamina Release build (6.55%
faster), with identical predictions on all 120 teacher-forced inputs.
A 4096-token host-KV prompt reaches its first token in 26.783 seconds with
2048-token chunks, now the client default, versus 35.923 seconds with 1024.

Before the attention/KV update, the real 32K capacity run (32736 prompt + 16 generated tokens) completed at
351.465 seconds to first token, 10.870 later tokens/s and 7661.67 MiB peak
total GPU memory. The real 128K run (131040 prompt + 16 generated tokens)
completed with full FP32 host KV: 3115.672 seconds to first token, 0.689 later
tokens/s, 6980.10 MiB peak total GPU memory and 17693.29 MiB process RAM.
Both used explicit 1024-token chunks. Peaks include desktop memory. Repeated
token capacity stress does not establish semantic retrieval quality.
The 128K latency is substantial; these results do not establish Strata-class
performance. Previous RTX 4060 Ti measurements describe different hardware.

The optional CPU expert-miss policy passes an independent four-token,
40-layer reference check at maximum difference 3.33945929e-6. It measured
6.459 tokens/s with the normal cache and 3.908 with a forced 2200 MiB cache;
GPU expert streaming remains the default because it was faster.
The real API suite covers 11 checks, including images and cancellation
recovery. MoE graph profiling records 4760 replays and 40 initial misses.

The attention/KV update retains the FP32 numerical gates. Its single matched
16K host-KV run improves first-token time from 125.814 to 104.682 seconds and
later throughput from 4.657 to 6.560 tokens/s at the same sampled VRAM peak.
Short decode is unchanged at 13.93 tokens/s. Optional FP16 host/device KV reaches
9.675/16.673 tokens/s at 16K. Both precision modes pass 11 real API checks.
FP16 is experimental and lossy: a 64-token comparison against FP32 measured
1.37% relative hidden-state drift; host/device/reset/image behavior matches.
[Current measurements](../bench/results/2026-10-06-attention/VALIDATION.md)
record exact commands, hashes, precision differences and validation boundaries.

The updated FP16 device capacity profile also completed the full 128K run:
131040 prompt + 16 generated tokens, 2048-token chunks, first token 1736.531 s,
later 7.212 tokens/s, peak total GPU 7304.03 MiB and process RAM 12591.10 MiB.
Prefill remains slow (28 min 57 s). This changes precision and chunk size from
the earlier FP32 host run; it is capacity/performance evidence, not an equal-
precision speed comparison or a semantic long-context quality test.

See [the developer handoff](DEVELOPER_HANDOFF.md) for active files, exact build
and validation commands, measurement records and remaining validation boundaries.
Official Transformers generation comparison and Linux GPU runtime remain
unverified; semantic retrieval results and failures are recorded separately; compilation does not establish
runtime performance.


The Strata execution phase adds worker-backed RAM staging, a separate weight
copy stream with reader-retirement fences, resident/miss overlap, optional
whole-model RAM registration, GPU-resident layer-major prefill, bounded QK/PV
matrix attention with full-history causal online softmax, and shared grouped-
query KV reads during tiled decode. Optional quantized AVX2 CPU experts use
pinned ggml; streaming remains the default after machine-specific comparisons.
Default cache sizing remains conservative: enlarging it caused a regression.
Whole-expert admission and frequency-aware eviction remain experimental options.

Current correctness, quality and performance evidence is in
[the Strata execution record](../bench/results/2026-10-06-strata-plan/VALIDATION.md).
These changes do not by themselves establish 40 tokens/s or Strata parity.

Long text prefill now has a complete GPU residual and traverses all work tiles
of one layer before advancing. The PROMPT protocol retains model weights in
RAM while reusing the current layer's expert slices across tiles. The whole
residual is bounded to 1 GiB at 128K; query/softmax/MoE workspaces remain tiled.
Existing-prefix/reset/later-decode comparisons pass against the earlier
chunk-major path with host and device KV (including a one-column tail).

Long prefill releases previous layers' dense/shared weight pins, synchronizes
readers and invalidates graphs containing their addresses. Decode re-pins
weights on lookup. The regression suite includes long prefill after graph
capture, followed by reset. Retrieval benchmarks stop at both model stop IDs;
continuing generation past EOS does not count as a semantic success.

The <=8 GiB GPU cache target is capped at 5000 MiB after real-prompt cache
comparisons; a larger target reduced throughput. User overrides remain explicit
experiments. The CUDA copy-batch prototype faulted and is not included.

Before the CPU-expert default, the matched CUDA llama.cpp comparison on the
RTX 4060 machine measured **25.47 tokens/s** for llama.cpp versus **16.40
tokens/s** for Lamina (nine resident-process runs, 32K context, FP16 KV, fast
Lamina computation). See [commands and full results](../bench/results/2026-10-07-llama-cuda/README.md).
That machine has not been remeasured with the current code.

The MTP head of the official checkpoint is ported as a draft model for greedy
speculative decoding (fast mode, device KV). Its forward pass was checked
against an independent NumPy implementation (99.2-100% draft agreement), and
with every expert on the GPU the speculative and single-token greedy outputs
are identical. On the RTX 3050 it measured 42.94 against 37.15 tokens/s; see
[the speculation record](../bench/results/2026-10-08-rtx3050-mtp-speculation/README.md).

In fast CPU-miss mode, prefill runs experts routed by at most 12 prompt tokens
on the CPU. The quality gate passes (mean KL 0.0028 nats, perplexity ratio
0.993), 4K-32K retrieval passes, and the short-prompt first token fell from
2.18 to 0.69 s; see [the hybrid prefill record](../bench/results/2026-10-08-rtx3050-hybrid-prefill/README.md).
