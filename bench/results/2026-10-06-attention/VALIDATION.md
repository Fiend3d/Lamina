# Attention/KV update validation, RTX 4060 8 GB

Machine: RTX 4060, 8188 MiB, driver 610.88, SM89, Ryzen 7 1700X, 64 GiB RAM,
PCIe Gen3 x8, Windows, Release CUDA 13.3/cuBLAS. All GPU tests run serially.
Model: pinned Qwen3.6-35B-A3B UD-Q4_K_M in sibling Lamina-data. No competing
ComfyUI process. GPU peaks are sampled total memory including desktop.

Before means the completed first port's saved binary, SHA256
`e2623cb10cd09d4e9580863c4e4c62ec93633518632bd0b1a24ec170864c613d`,
not original Lamina HEAD or upstream Strata. Updated binary SHA256
`152b597382051d3a3888228f6157fd9496ceb14718f41fd85c2ecd2a1a672b89`.
Every benchmark JSON records command, hardware, environment, UTC timestamp,
arguments, timings, sampled peaks and executable hash. These are single
matched runs, not a statistically established distribution of speedups.

## Implementation

Fused FP32 prefill shares a KV head across eight query warps, retains the
128-key reduction/merge order and writes the running accumulator directly.
It removes the per-key-tile partial matrix and lets all up to 2048 queries
reuse one host-history sweep. The running accumulator is at most 32.25 MiB.
Host staging has two pinned/device slots, a nonblocking transfer stream and
copy/consumer events. DMA completion protects CPU pinned-buffer reuse;
consumer completion protects device-buffer reuse. Staging stays bounded:
16 MiB pinned + 16 MiB device for FP32, half those sizes for FP16.

--kv-type f16 stores only KV as IEEE FP16. Queries, projections, softmax and
attention accumulation stay FP32. It is experimental and lossy; FP32 remains
the default. Both host/device caches support it, without history truncation.
128K KV storage is 2.5 GiB versus 5 GiB for FP32. Auto still selects host above
32K; choose device explicitly to keep the compressed cache on this GPU.

## Correctness and precision

- Default hardware-free CMake configure/Release build: passed.
- Python tests: 17 passed in the sibling venv; system Python passes with two
  optional-dependency skips. Sampling check passed.
- Independent NumPy four-token/40-layer batched host-KV reference retains its
  unchanged 1e-5 gate: max 3.59125275e-6, mean 1.38403221e-7, next token 369,
  reference/native logits 12.448533724/12.448531. See reference-f32.txt.
- FP32 causal prefill/decode/reset/image state suite: passed unchanged 1e-5
  gate, maximum 8.85501504e-6; 4 layers/1025 tokens/1024 chunks and 40 layers/
  64 tokens/32 chunks, both host/device KV. Captured terminal output.
- Independent scalar attention checks for FP32 and explicitly rounded FP16
  inputs: 1,127,128,129,513,2049,131072 positions. Causal columns include an
  incomplete query group and a signal only at the oldest history position.
  All errors below 1e-5; see attention.txt. This is not a full 128K numerical
  reference of the complete model.
- Isolated fused/legacy attention benchmark, 128 distinct queries, full
  4K/16K/128K history: bit-identical outputs; see kernel-benchmark.txt.
  GPU times (legacy/fused ms): 16.219/8.332, 61.814/30.667, 495.151/242.146.
  These kernel times exclude model, weight streaming and host transfers.
- FP16 host/device, reset, decode, image mRoPE and following text checks:
  identical outputs; see kv-precision.txt. Against FP32, the 64-token sample
  has max hidden drift 0.081905365 and relative L2 0.0137232212. This is a drift
  measurement, not a generation-quality score. An exploratory FP16 sequential/
  batched comparison exceeded the strict FP32 gate (0.000198364258); its raw
  output remains in prefill-f16.txt. The FP32 gate was not relaxed.
- Three-tile transfer integration: 4 layers, 4097 varied tokens, 2048-token
  chunks and following decode. Both FP32 and FP16 host caches match device
  caches exactly (max difference zero), exercising both staging slots and
  slot-zero reuse. See kv-precision.txt.
- Full 128K FP16 GPU-cache sample: all 16 predicted tokens match the earlier
  FP32 host capacity sample. This is not general generation equivalence.
- Real API: both FP32 and FP16 passed all 11 requests, covering sampling,
  repeat/reset, SSE UTF-8/usage, reasoning, JSON schema, forced tools, OCR,
  multiple images and cancellation recovery. See runtime-f32.json and
  runtime-f16.json, which include precision and native hash.
- Matched short decode: all 120 predictions identical before/FP32/FP16.
  The 16-token 4K and 16K FP32 continuations match before; both FP16 16K
  continuations also match the FP32 sample. This does not guarantee general
  output equivalence or long-context quality.

## End-to-end measurements

Prefill stress repeats token 42, uses 2048-token chunks, and generates 16
response tokens. 4K/16K host tests set max-context 131072; device FP16 is
explicit. Short decode uses the original teacher-forced-120.txt inputs.

| Run | First token s | Later tokens/s | Peak total GPU MiB |
| --- | ---: | ---: | ---: |
| before-short | 1.622 | 13.932 | 7155.34 |
| fused-short | 1.570 | 13.933 | 7156.90 |
| fused-f16-short | 1.616 | 13.992 | 7156.71 |
| before-4k | 28.018 | 10.759 | 7249.12 |
| fused-4k | 27.299 | 11.594 | 7247.94 |
| before-16k | 125.814 | 4.657 | 7249.75 |
| fused-16k | 104.682 | 6.560 | 7250.30 |
| fused-f16-host-16k | 102.498 | 9.675 | 7252.52 |
| fused-f16-device-16k | 102.045 | 16.673 | 7221.90 |
| Updated FP16 device KV, 131040 prompt + 16 response | 1736.531 | 7.212 | 7304.03 |

The full 128K FP16 device run completed with all history retained: 28 min 57 s
to first token, 7.212 later tokens/s, 7304.03 MiB peak total GPU memory and
12591.10 MiB peak process RAM. Compared with the earlier FP32 host/1024-chunk
capacity run (3115.672 s, 0.689 tokens/s), this changes both precision and chunk
size as well as implementation. It is a usable capacity profile, not an equal-
precision speedup or a semantic retrieval benchmark.

The matched 16K FP32 host result reduces first-token time by 16.80% and
increases later throughput by 40.86%, with identical predictions. Short
FP32 decode is unchanged within measurement noise. Full-context results
are separate from these smaller-prompt timings.

## Reproduce

```powershell
../Lamina-data/venv/Scripts/python.exe -m tools.build_windows
build-cuda/lamina-cuda-attention-check.exe
build-cuda/lamina-cuda-attention-check.exe --benchmark
build-cuda/lamina-prefill-check.exe ../Lamina-data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
build-cuda/lamina-kv-precision-check.exe ../Lamina-data/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
$env:OPENBLAS_NUM_THREADS='1'
../Lamina-data/venv/Scripts/python.exe -m tools.reference_prefix --layers 40 --engine build-cuda/lamina-infer.exe --cuda --batched --tokens 42 43 44 45 --max-context 131072 --kv-cache host
../Lamina-data/venv/Scripts/python.exe -m tools.runtime_check --kv-type f32 --report bench/results/2026-10-06-attention/runtime-f32.json
../Lamina-data/venv/Scripts/python.exe -m tools.runtime_check --kv-type f16 --report bench/results/2026-10-06-attention/runtime-f16.json
../Lamina-data/venv/Scripts/python.exe -m tools.benchmark --engine build-cuda/lamina-infer.exe --cuda --prompt-tokens 16384 --chunk 2048 --tokens 16 --max-context 131072 --kv-cache host --kv-type f32 --quiet --json bench/results/2026-10-06-attention/fused-16k.json
../Lamina-data/venv/Scripts/python.exe -m tools.benchmark --engine build-cuda/lamina-infer.exe --cuda --prompt-tokens 131040 --chunk 2048 --tokens 16 --max-context 131072 --kv-cache device --kv-type f16 --quiet --json bench/results/2026-10-06-attention/fused-f16-device-128k.json
```

Official Transformers comparisons, semantic long-context retrieval and Linux
GPU runtime remain unverified. No Strata-class performance claim. Full-context
stress establishes capacity and timing, not retrieval or generation quality.
