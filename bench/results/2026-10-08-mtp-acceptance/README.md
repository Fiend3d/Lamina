# Qwen3.6 MTP head: fetch, pack and draft acceptance

This record covers the first steps toward speculative decoding with the
model's own multi-token-prediction (MTP) head: obtaining the head, packing it
for Lamina, and measuring how often its drafts would be accepted. It is
written for developers; no engine speculation exists yet.

## Result

On real greedy generations from the three benchmark prompts, the MTP head's
one-token draft equals the main model's next greedy token in **87.8%** of
positions (672 of 765).

| Prompt | BF16 head | Packed Q8_0 head |
| --- | ---: | ---: |
| Prose | 75.3% | 75.7% |
| Code | 92.2% | 91.8% |
| Math | 96.1% | 96.1% |
| **Overall** | **87.8%** | **87.8%** |

With one draft per step this would yield about 1.88 tokens per main-model
pass, before the verification and drafting costs. Packing to Q8_0 does not
reduce acceptance.

## Where the head comes from

The pinned UD-Q4_K_M GGUF contains no MTP tensors. The official BF16
checkpoint `Qwen/Qwen3.6-35B-A3B` at revision
`995ad96eacd98c81ed38be0c5b274b04031597b0` does (`mtp_num_hidden_layers: 1`):
19 `mtp.*` tensors, 1.57 GiB, in shards 25 and 26 of 26. They form an input
fusion (`fc` over the normalized next-token embedding and the normalized
hidden state), one gated full-attention decoder layer with a 256-expert,
top-8 MoE and a gated shared expert, and a final norm. The main model's
embedding and LM head are shared.

```powershell
python -m tools.lamina_mtp fetch    # range requests for only the 19 tensors
python -m tools.lamina_mtp verify
python -m tools.lamina_mtp pack     # ../Lamina-data/mtp/qwen36-mtp-q8_0.gguf
python -m tools.mtp_acceptance --engine build-cuda/lamina-infer.exe
python -m tools.mtp_acceptance --mtp-gguf ../Lamina-data/mtp/qwen36-mtp-q8_0.gguf
```

The packed file is 857 MiB, SHA-256
`1c768a3998e10b7cbd0f9515052abc19c059c3bf00a2d704df2cf45b6461170c`.
Matrices and experts are Q8_0 (gguf-py cannot write K-quants), norms and the
router F32.

## Conventions established

- **Norm weights.** Checkpoint RMSNorm weights are zero-centred: the GGUF
  stores exactly 1 + w (difference 0.0 for `blk.0.attn_norm`,
  `blk.3.attn_q_norm` and `output_norm`). The pack adds 1 to every MTP norm.
- **Fused experts.** `gate_up_proj` is gate-first. The other order drops
  acceptance to 52-56%.
- **Hidden state.** The head accepts the main model's hidden state before or
  after its final norm with nearly the same result (87.2% and 87.8%), because
  it applies its own norm. The pack records `post_final_norm`.
- Equations follow vLLM's `qwen3_5_mtp.py`. Attention reuses the validated
  reference math of `tools/reference_prefix.py`.

## Method

`lamina-infer --trace OUT N tokens...` (new) decodes the prompt one token at a
time on the CUDA fast path, generates N greedy tokens, and writes every token
id and every hidden state. `tools/mtp_acceptance.py` runs an independent
NumPy implementation of the head causally over the sequence and compares
each draft for x(t+2) with the generated token. Only positions whose target is
a generated token are scored. The engine SHA-256 begins with
`5e3a2291ab04f350`.

## Limits and next steps

- Only one-token drafts were measured. Chained drafts (feeding the head its
  own output) would accept less per extra token.
- Fast-mode generations were used; FP32 was not measured.
- Engine work remains: running the head on the GPU with its own KV cache,
  verifying draft and current token in one batched pass, checkpointing and
  rolling back DeltaNet recurrent state on rejection, and greedy acceptance
  first, then rejection sampling. The verification pass routes two tokens, so
  more distinct experts are touched; the payoff must be measured, not
  inferred from acceptance alone.
