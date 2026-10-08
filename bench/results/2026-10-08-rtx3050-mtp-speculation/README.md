# Greedy MTP speculative decoding on the RTX 3050

This record covers the first engine implementation of speculative decoding
with the model's own MTP head. It is written for developers who continue the
performance work; it documents what was built, how it was checked, what it
measured, and what remains open.

## Result

With the packed MTP head, greedy decode measured a **42.94 tokens/s nine-run
median, against 37.15 tokens/s** for single-token decode with the same binary,
a 16% gain. Drafting from the first 98,304 token ids (`LAMINA_MTP_VOCAB=98304`)
raised that to **45.21 tokens/s**. All three prompts improved; prose gains
least because its drafts are accepted least often.

| Mode | Prose | Code | Math | Nine-run median | Warm first token | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-token decode | 39.99 | 37.15 | 35.37 | **37.15** | 2.18 s | 6800 MiB |
| Speculative (`--mtp`) | 42.58 | 42.91 | 43.97 | **42.94** | 2.61 s | 6900 MiB |
| Speculative, 98,304-id drafts | 45.43 | 44.64 | 45.41 | **45.21** | 2.57 s | 6899 MiB |

Prompt columns are three-run medians of later tokens/s; "warm first token" is
the median loaded first-token time without the first, cold request. 1,038 of
1,248 drafts were accepted (83.2%); the acceptance guard never backed off.
The draft took 3.54 ms (2.23 ms with the reduced vocabulary) and the two-token
verification pass 38.9 ms, against about 27 ms for one single-token step.

Hardware and software: RTX 3050 8 GB (sm_86, PCIe Gen3 x8, WDDM), Ryzen 7
5700X, 64 GiB DDR4, Windows 11. Engine SHA-256 prefix `7af05695828a638a`,
built with `python -m tools.build_windows --cuda-arch 86`.

```powershell
python -m tools.compare_lamina --repeats 3 --report-dir <out>/single
python -m tools.compare_lamina --repeats 3 --mtp ..\Lamina-data\mtp\qwen36-mtp-q8_0.gguf --report-dir <out>/speculative
$env:LAMINA_MTP_VOCAB='98304'
python -m tools.compare_lamina --repeats 3 --mtp ..\Lamina-data\mtp\qwen36-mtp-q8_0.gguf --report-dir <out>/speculative-vocab98304
```

The run directories hold every report JSON and the engine's stderr.

## How it works

The engine generates through a new `GENERATE N TOKEN` protocol command:
it feeds the last returned token and returns the next N greedy tokens on one
line. `tools.compare_lamina --mtp` and `lamina.py --mtp` use it for greedy
requests. Each speculative step:

1. Drafts one token with the MTP head from the last token and the hidden
   state that predicted it (`Inference::mtp_draft`). The head's KV cache holds
   only its own drafts; rotary positions are the main sequence's.
2. Runs one two-token pass over the current token and the draft
   (`Inference::step_pair_hidden`). Every weight is read once for both
   columns: two-column Q8 MMVQ kernels for dense projections and experts
   routed by both tokens, one routing round trip for both tokens, the union of
   their experts, and a CPU batch that reads each missed expert row once for
   both tokens. Column 1 sees column 0 causally in attention and DeltaNet.
3. Takes both greedy tokens from one two-column LM-head read. If the first
   equals the draft, both are emitted; otherwise only the first is, and each
   DeltaNet state is restored from the snapshot taken after column 0. Attention
   KV beyond the accepted position is simply overwritten later.

An acceptance guard keeps a slow running average of acceptance; below 55%,
roughly the break-even point of a 42 ms pass against a 27 ms step, it runs 32
single steps before probing again.

## Correctness checks

- `lamina-cuda-ncols-check` (new): two-column and per-column-pointer Q8 table
  kernels are bitwise equal to single-column launches on real Q8_0, Q4_K,
  Q5_K and Q6_K weights; one pair launch costs about one single launch.
- With every expert on the GPU (`LAMINA_EXPERT_POLICY=stream`,
  `LAMINA_EXPERT_PIPELINE=0`, `LAMINA_PREFETCH=0`), speculative and
  single-token greedy output are identical for 96 tokens of math and code
  (`gpu-only-*` directories). The pair path therefore reproduces single-token
  decode exactly when the same kernels run.
- In the default CPU-miss mode the outputs diverge after 56-79 tokens. A pair
  routes different expert sets than single steps, so some experts run on the
  CPU in one mode and on the GPU in the other, and the two differ in their last
  bits; near-tied logits then flip. Speculative output is deterministic across
  repeats, and single-token output still equals the committed
  `2026-10-07-rtx3050-greedy-keep` tokens.
- Unit tests, the elementwise, attention, projection, prefill and KV checks,
  and `python -m tools.reference_prefix --layers 40 ... --batched` pass
  (next token 369, logit 12.448531 against 12.448534 in the reference).

## Changes that also help single-token decode

- `moe_combine_mixed` reads CPU-expert results straight from mapped memory,
  replacing one copy launch per CPU expert per layer.
- The `dense_q8` scratch is sized for two columns from the start; growing it
  later would have freed a pointer held by the captured DeltaNet graphs.

## Open issues

- **First-token time rises by about 0.45 s** once the MTP head has drafted
  (2.18 s to 2.61 s warm). The cause is unknown. It persists after the first
  draft even through later non-speculative requests, and reproduces with only
  the head's `fc` projection. The prefill uploads the same bytes (8.65 GB),
  makes the same 39 allocations with no free-list trims, and spends the same
  host time enqueueing work; its MoE stage simply runs at about 3.5 instead of
  4.4 GB/s. Ruled out: the pageable weight stager, registering the head's file
  for direct DMA, allocator headroom and cache budget, the shared input
  buffer, and stream creation order.
- The verification pass is bound by the CPU experts. A pair routes about 13.2
  distinct experts per layer, of which 6.6 miss the GPU cache and stream about
  13 MB from DDR4 per layer, roughly 18 ms per pass. Per emitted token this is
  the same RAM traffic as single-token decode; speculation saves GPU reads and
  launch overhead, not CPU expert bandwidth.
- Pair attention layers still launch eagerly (positions change per pass).
- `LAMINA_MTP_VOCAB` is opt-in: tokens above the bound (notably multilingual
  text) can never be drafted, which lowers acceptance there.
- Only greedy decoding speculates; sampled requests take single steps.
