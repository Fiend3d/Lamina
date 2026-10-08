# Ornith-1.5-35B-A3B bring-up (RTX 4060 Ti 16 GB)

Date: 9 October 2026. Source base: `38285f0` (plus the changes below).

Ornith-1.5-35B-A3B (`ornith-ai/Ornith-1.5-35B-A3B-GGUF`, Q4_K_M, MIT) uses the
same `qwen35moe` architecture as Lamina's pinned Qwen3.6-35B-A3B, so Lamina can
run it after a small, general set of changes. This is a bring-up, not a pinned
second artifact, and it makes **no quality or speed claim**.

## Compatibility (header, from a 64 MiB range read)

- `general.architecture: qwen35moe`; all **733 expected text tensors present with
  byte-identical shapes**.
- One difference: `block_count` is 41 with `nextn_predict_layers: 1` and 20 extra
  `blk.40.*` tensors - an **in-file MTP head** (the pinned Qwen3.6 keeps MTP in a
  separate file and has `block_count` 40).
- 348 of 733 tensors use a **different quantization** than the pinned artifact:
  dense projections `Q4_K`/`Q6_K` (pinned `Q8_0`), expert-down `Q6_K` (pinned
  `Q5_K`), and `ssm_alpha/beta` `Q4_K` (pinned **F32**).

## What blocked it, and the fix

Compute-sanitizer (`compute-sanitizer --tool memcheck --target-processes all`)
pinned the crash:

```
Invalid __global__ write of size 4 bytes
  at lamina::model::cuda::moe_combine_kernel(...)+0xe10
  Access to 0x0 is out of bounds
```

`moe_combine` wrote to a null output because the engine fell back to the scalar
path (`device_chain_supported()` false) and `CudaProjection::moe()` passed
`accum_dev`, which `moe_core` only allocates after its `moe_pipeline` early
return.

`device_chain_supported()` was false because `supports_gdn`/`supports_attention`
required the grouped weights to share one encoding:

- `supports_gdn`: `qkv->type == gate->type`
- `supports_attention`: `q->type == k->type == v->type`

Ornith mixes encodings (e.g. `attn_qkv` Q6_K with `attn_gate` Q4_K; `attn_v`
Q6_K with `attn_q/k` Q4_K), so the whole layer was declared unsupported.

Changes (`src/model/cuda_projection.cpp`, `src/model/inference.cpp`,
`include/lamina/model/qwen36.hpp`, `include/strata/artifact/gguf_reader.hpp`):

1. Allow an optional in-file MTP layer: `check_architecture` accepts
   `block_count == 40 + nextn_predict_layers`; `check_qwen36_tensors` skips
   `blk.40.*` and verifies the text tensors are present; the redundant byte-size
   pin is dropped.
2. `supports_gdn`/`supports_attention` no longer require equal encodings;
   `delta_net_core` and `attention_into_mix` dispatch qkv/gate and q/k/v as one
   group when they match, or per tensor with each tensor's own type when they do
   not (the arithmetic per matrix is unchanged).
3. `CudaProjection::moe()` allocates `accum_dev` up front, so the scalar fallback
   can never write through a null pointer.

Additionally, the engine's DeltaNet path requires `ssm_alpha/beta` to be F32, so
the Q4_K Ornith tensors were re-encoded to F32 (a small in-place GGUF patch:
append the F32 payload, fix the directory; ~15 MB growth, no second copy).

## Result

- `Ornith-1.5-35B-A3B` (patched) generates with the local CUDA engine
  (`--compute-mode fast`, FP16 device KV). Examples:
  - "Name three primary colors" -> coherent answer.
  - "Write a Python one-liner that squares a number" -> `square = lambda x: x ** 2`.
- Qwen3.6-35B-A3B is unchanged: it still generates, and the independent 40-layer
  FP32 reference is `3.59125275e-06`, next token 369.
- `python -m unittest discover -s tests/lamina` passes (49 tests).

## MTP (speculative decoding)

`tools/lamina_inline_mtp.py` packs the in-file nextn layer (`blk.40.*`) into a
`mtp.*` side GGUF that `lamina-infer --mtp` already reads:

| Lamina MTP tensor | Ornith in-file tensor |
| --- | --- |
| `mtp.fc.weight` | `blk.40.nextn.eh_proj.weight` |
| `mtp.pre_fc_norm_embedding.weight` | `blk.40.nextn.enorm.weight` |
| `mtp.pre_fc_norm_hidden.weight` | `blk.40.nextn.hnorm.weight` |
| `mtp.output_norm.weight` | `blk.40.nextn.shared_head_norm.weight` |
| `mtp.attn_*`, `mtp.post_attention_norm`, `mtp.ffn_*` | `blk.40.*` |

Quantized matrices are copied verbatim; `--norms raw` is the correct convention
(the in-file GGUF already stores the effective gamma - `plus1` gave 0.56-0.65
draft acceptance, `raw` gives 0.85-0.89, matching Qwen3.6). The pair-verification
path also assumed grouped qkv/gate and q/k/v share one encoding; those dispatches
were split per type as well (`pair_delta`, `pair_attention`).

Speculation is correct (output identical to non-speculative) with a ~0.87 draw
acceptance. On this machine the packed head is Q4_K/Q6_K, so the two-token verify
cost roughly the same per token as ordinary decode - MTP did not speed up the
measured 64-token run; a longer/steady measurement is still needed.

## Model selection

`START-HERE.bat --model ornith` (or `python -m tools.quickstart --model ornith`)
selects the model. `tools/lamina_ornith.py setup` downloads Ornith + its
tokenizer/template, re-encodes the SSM tensors to F32, and packs the MTP head.
`--model qwen3.6` (default) is unchanged. Ornith needs the qwen35moe engine built
from source (`--build-source`); the selection is stored in `quickstart.json`.
The server still advertises the id `Qwen3.6-35B-A3B-UD-Q4_K_M`, so a client
configured for that id works for either model.

## Limitations

- Ornith needs the engine built from source; the packaged release does not carry
  the qwen35moe changes yet.
- The stock Ornith GGUF must have `ssm_alpha/beta` re-encoded to F32; the engine
  does not yet dequantize those two tensors at load (`lamina_ornith.py` does it).
- No reference/quality/perplexity comparison and no sustained speed measurement
  for Ornith; the vendor's coding benchmarks are not reproduced here.
- Single machine (RTX 4060 Ti 16 GB / i5-12400F). 8 GB was not tested for Ornith.
