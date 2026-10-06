# RTX 4060 8 GB validation, 2026-10-06

Hardware: RTX 4060, 8188 MiB, driver 610.88; Ryzen 7 1700X, 8 cores/16 threads;
64 GiB RAM; PCIe Gen3 x8; Windows; Release CUDA 13.3. Model: pinned
Qwen3.6-35B-A3B UD-Q4_K_M. ComfyUI was stopped before measurement.
NVML peaks are sampled total GPU memory including desktop; process RAM is RSS.
Every benchmark JSON records its exact native command, arguments, hardware,
timings, sampled peaks and executable SHA256. Run GPU measurements serially.

## Executed correctness checks

- Default CMake configure and Release build: lamina-gguf, lamina-infer and sampling.
- GGUF architecture/tensor geometry inspection: passed.
- `python -m unittest discover -s tests/lamina`: 15 passed using the sibling venv.
- Sampling, CUDA elementwise and quantized 17-column projection checks: passed.
  Projection relative L2 below 1e-6; both cuBLAS and native fallback are checked.
- Scalar-independent CUDA attention: 1,127,128,129,513,2049,131072 positions;
  max error below 9e-8. Oldest-position-only signal is preserved at 128K;
  batched causal error at 128K is 6.985e-10.
- Largest-angle rotary first-pair analytic double sin/cos reference:
  positions 0,32767,65536,131071; max error 9.13092e-7. This is not an
  all-frequency/full-model 128K numerical reference.
- Prefill state suite: 4 layers/1025 tokens/1024 chunks and 40 layers/64 tokens/32
  chunks, host/device KV, decode after prefill, reset, eight-position image mRoPE
  and following text. Maximum difference 8.85501504e-6; unchanged 1e-5 gate.
  Raw results: `prefill-parity.txt`.
- Independent NumPy four-token, 40-layer batched host-KV reference: maximum
  difference 3.59125275e-6, mean 1.38403221e-7, next token 369 in both,
  reference/native logits 12.448533724/12.448531. Command:
  `OPENBLAS_NUM_THREADS=1 python -m tools.reference_prefix --layers 40 --engine build-cuda/lamina-infer.exe --cuda --batched --tokens 42 43 44 45 --max-context 131072 --kv-cache host`.
- Forced CPU expert misses: independent sequential four-token, 40-layer
  reference passed at max 3.33945929e-6, mean 2.76536596e-7, next token 369,
  logits 12.448533724/12.448532. Set OPENBLAS_NUM_THREADS=1,
  LAMINA_EXPERT_POLICY=cpu-miss, LAMINA_CPU_THREADS=8 and
  LAMINA_CUDA_CACHE_MB=2200; run the command above without --batched.
  Raw results: `reference-cpu-miss.txt`.
- Real CLI/chat API suite: `python -m tools.runtime_check`; 11 checks cover
  greedy generation, seeded repeat, reset, SSE UTF-8/usage, reasoning, JSON
  schema, forced tools, OCR LAMINA42, multiple images ALPHA BETA, cancellation
  and next-request recovery. Results: `runtime.json`, `runtime-check.txt`.
  Final run uses the 2048-token client chunk default.

## Performance

| Run | First token (s) | Later tokens/s | Peak total GPU MiB |
| --- | ---: | ---: | ---: |
| Original Lamina, matched three-run median | 1.437-1.477 | 13.260 | 6635.91 |
| New Lamina, matched three-run median | 1.542-1.574 | 14.129 | 6641.79 |
| Final binary, matched 120 | 3.665 | 14.078 | 6771.33 |
| 4096 prompt, host KV, chunk 1024 | 35.923 | 12.062 | 6982.00 |
| 4096 prompt, host KV, chunk 2048 | 26.783 | 11.080 | 6982.00 |
| 32736 prompt + 16 response, device KV, chunk 1024 | 351.465 | 10.870 | 7661.67 |
| 131040 prompt + 16 response, host KV, chunk 1024 | 3115.672 | 0.689 | 6980.10 |
| CPU misses, normal cache, 120 matched | 1.580 | 6.459 | 6892 |
| CPU misses, 2200 MiB cache, 120 matched | 1.558 | 3.908 | 3796 |

Matched inputs are `teacher-forced-120.txt`; all baseline/new predictions agree.
Median short decode improved 6.55%. MoE graph profiling records 4760 replays,
40 initial misses and zero CPU experts. GPU streaming remains the default
because the optional CPU policy was slower (its counters confirm actual CPU
expert work). The three-run comparison predates only host-KV query grouping
and the expert-count guard; final-current-matched.json records the final binary.
Final executable SHA256:
`e2623cb10cd09d4e9580863c4e4c62ec93633518632bd0b1a24ec170864c613d`.

128K retains complete FP32 host KV and peaks at 17693.29 MiB process RAM.
Its first-token latency is 51 min 56 s. Context stress repeats token 42; it
establishes capacity and timing, not semantic retrieval or generation quality.
Both full-context runs use explicit 1024-token chunks; the client now defaults
to 2048, selected from the 4K comparison. No Strata-class performance claim.

Baseline means original Lamina Release commit
`81c0c41808442ba8a31c787f42a3b5a448b2d1c8`, not the inherited Qwen3.8 engine.
The port source baseline remains Strata `6f32ec070f23ced9f50e704d854d775da52591ab`.
Exact reproduction commands are also in `docs/DEVELOPER_HANDOFF.md`.

## Audit and boundaries

Earlier experiments remain for audit, not accepted final configurations:
initial-updated used an undersized budget; expert-graphs keyed exact expert
addresses and had no hits; early prefill-4096-blas failed the image tolerance.
Native small groups plus strict FP32 GEMM large groups pass that gate.
Debug performance is excluded. partial-128k-query256.json stopped at 30720
prompt tokens and is not full-context validation; context-128k.json completed.
Official Transformers generation comparison, semantic long-context retrieval
and Linux GPU runtime remain unverified. Compilation alone is not runtime proof.
