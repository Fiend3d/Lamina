# Lamina performance report

This report covers the CUDA performance pass on Qwen3.6-35B-A3B-UD-Q4_K_M.
It states what changed, the measured result, how it was validated, and what
still has to be done.

## Measured result (i5-12400F, RTX 4060 Ti 16 GB, warm file cache)

| | Before | After |
| --- | --- | --- |
| Later tokens/s (`tools.benchmark.py --cuda --tokens 120`) | 4.22 (4-token run) | **19.6** (median 0.043 s) |
| 40-layer reference parity (max hidden diff) | 6.50e-7 | **1.02e-6** |
| Next token / logit | 20 / 9.544323 | 20 / 9.544324 |
| GPU-busy time per step (`LAMINA_PROFILE=1`) | not measured | ~26 ms |

The scalar CPU path is unchanged; the CUDA projection suite, the elementwise
suite, and all `tests/lamina` unit tests pass.

## What changed

The backend went from "each projection is a synchronized host/device round trip
plus scalar recurrence/attention/routing on the CPU" to a device-resident token
chain, and the quantized matvec kernels were rewritten.

- **Device-resident chain** (`CudaProjection`): `hidden_upload`/`hidden_download`,
  a separate normalized buffer, `hidden_rms`, `add_hidden_mix`, and mixer/MoE
  entry points that read `norm_dev` and write `mix_dev`. The hidden state never
  leaves VRAM during a token.
- **Device attention** (`attention_into_mix`): growing device KV cache,
  `attn_norm_rope` (per-head weighted RMS + partial RoPE over 64 of 256), and
  `attn_decode` (GQA 16 Q / 2 KV heads, online softmax, sigmoid gate).
- **Device router + doorbell** (`moe_into_mix`): `gemv_f32` router logits,
  `router_topk`, `dot_sigmoid`, and a `doorbell_signal` publishing the host's
  mapped-pinned `h_ids`/`h_scales`, so the host spins on `cudaStreamQuery`
  instead of a per-layer `cudaStreamSynchronize`.
- **Device DeltaNet** (`delta_net_into_mix` / `delta_layer_graph`): the linked
  `native_gdn` conv/L2/gate kernels, `native_gdn_step`, and a SiLU closing norm
  (`cuda::gdn_out_norm_silu`; Qwen3.6 uses SiLU where Strata uses sigmoid). The
  recurrent state uses the native `(i*h_v+h)*S+j` layout.
- **Per-layer CUDA graph** for the DeltaNet pre-MoE sequence (input norm,
  recurrence, residual, post norm) after a warm-up call, removing ~16 small
  kernel launches per layer.
- **Grouped MMVQ** (`native_mmvq_f32_grouped`) with by-value pointer arrays and
  dedicated block-hoisted kernels for Q4_K (`native_mmvq_f32_q4k_grouped_kernel`),
  Q5_K (`native_mmvq_f32_q5k_grouped_kernel`) and Q8_0
  (`native_mmvq_f32_q8_grouped_kernel`), each with independent accumulators.
  These are exact (same arithmetic as the per-element path) and replaced the
  ~29-kernel-per-layer MoE with ~6.
- **Weight management**: dense and shared weights (`expert < 0`) are pinned and
  never evicted; evicted expert blocks go to a size-keyed free list instead of
  `cudaMalloc`/`cudaFree` per miss.
- **Non-fast-math elementwise ops** (`src/model/cuda_kernels.cu`): swiglu, RMS
  norm, accumulate, add, gemv, attention norm/RoPE/decode, router top-k,
  dot-sigmoid, doorbell, moe combine. This keeps `expf` precise enough for the
  parity gate.

### Correctness bug fixed

`CudaProjection::weight()` uploaded quantized blocks with a **synchronous
pageable `cudaMemcpy` on the legacy stream** while projections ran on a
`cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)` stream. A queued kernel
could read a partially-uploaded weight; this silently corrupted one expert per
layer (found as a half-zero `down[4]`). Uploads now use `cudaMemcpyAsync` on the
kernel stream.

## Validation

```sh
cmake -S . -B build-cuda -DLAMINA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --config Release --target lamina-infer lamina-cuda-projection-check lamina-cuda-elementwise-check
python -m tools.reference_prefix --layers 40 --engine build-cuda/Release/lamina-infer.exe --cuda --max-diff 0.01
python -m unittest discover -s tests/lamina
python -m tools.benchmark --engine build-cuda/Release/lamina-infer.exe --cuda --tokens 120
```

`LAMINA_PROFILE=1` prints `device gpu_busy_ms` (CUDA-event time across the whole
device chain) and cache statistics; `LAMINA_DEV_PROFILE=1` splits the MoE into
router and kernels. `LAMINA_CUDA_CACHE_MB` is unchanged.

## What still has to be done

The step is now bound by the **accurate direct-FP32-activation quantized decode**
(~16 ms of the ~26 ms GPU-busy step is the MoE). It decodes one quantized element
at a time and stalls; at ~30 cycles/element it is ~4x off the memory floor
(~7 ms/token, ~140 tok/s, given ~2.1 GB of active weights read per token on a
288 GB/s card). Reaching Strata-class throughput is a kernel/architecture
problem, and there are two levers with very different accuracy cost:

1. **Exact FP32-activation decode, optimized** (keeps the 1e-6 gate):
   - Multi-row tiling (2–4 rows per warp) for more ILP at fixed occupancy.
   - 128-bit (vectorized) weight/activation loads and shared-memory staging to
     cut load-instruction count.
   - Finish the exact grouped kernels for the remaining types (Q6_K, and the
     Q8_0 path used by DeltaNet/attention could share the same block structure).
   Expected ceiling roughly 30–40 tok/s.

2. **Relax activation precision to Q8_1/dp4a** (the ~4x lever every other engine
   uses). Measured: `native_mmvq` (Q8_1) is ~4x cheaper per element but moves the
   40-layer hidden error to **0.127** (same top-1 token here; logit 9.601 vs
   9.544). `docs/DEVELOPER_HANDOFF.md` says not to weaken the parity gate, so
   this is a product decision. `LAMINA_Q8_MMVQ` is **not** wired into the tree;
   the per-matrix Q8_1 experiment was removed. If adopted, it should be a grouped
   dp4a kernel (not the 27 per-matrix launches) and probably limited to the
   routed experts, with the dense/DeltaNet path kept exact.

3. **Capture the MoE in a graph.** The MoE cannot be captured while its expert
   weights change each token. This needs Strata's design: a fixed resident expert
   set, per-layer graphs, and a CPU pool for misses (with the doorbell already in
   place), so graph replay and streaming experts coexist.

4. **Capture attention.** Attention is not yet graphed because its position and
   KV-append offset are host-computed. Read both from device memory and it can be
   captured like the DeltaNet layers.

5. **Overlap CPU and GPU experts.** The cache already holds ~93% of requested
   experts on this card; split each layer at the router and compute misses on a
   CPU pool concurrently for larger cards or a smaller VRAM budget. Measure hit
   rate, bytes transferred and peak VRAM.

6. **Batched prompt prefill.** Decode-only token loops make prompt processing
   slow; preserve causal DeltaNet/attention state while batching prefill.

7. **Broader correctness.** All checks here use tokens 42/43. Compare several
   prompts and generated tokens against an official Transformers reference before
   claiming generation quality.
