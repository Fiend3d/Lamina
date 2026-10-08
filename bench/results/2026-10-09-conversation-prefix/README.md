# Advancing conversation prefix checkpoint (agent turns)

Date: 9 October 2026. Source base: `524c92d` (Python server only).

## Problem

Coding agents (pi, Claude Code, opencode) resend the whole conversation every
turn. The v0.1.1 prefix cache held a single checkpoint for the *immutable
system/tools block* only and restored it only when the new prefix was byte
identical, so a growing conversation was always re-prefilled from scratch.

Observed with pi on the RTX 4060 Ti 16 GB at `--max-context 131072`:

```
00:21:37  request: 38022 prompt tokens
00:22:24    prompt read in 47.0 s (809 tokens/s)      # 2485 reused, ~35.5K prefilled
00:23:28  request: 38879 prompt tokens
00:24:09    prompt read in 41.3 s (940 tokens/s)      # every turn
```

Time to first token was 41-47 s on **every** request, because only the 2485-token
system/tools block was reused.

## Change

`tools/lamina_chat.py`:

- `_prepare` now caches the whole message history up to the generation prompt
  (the last `<|im_start|>assistant`) instead of only the system/tools block.
- `events` advances the checkpoint when the conversation grows by an append:
  `RESTORE_PREFIX`, prefill only the new messages, then `CACHE_PREFIX`. It only
  does this after verifying the previous checkpoint is an exact BPE token prefix
  of the new prompt; non-monotonic conversations (compaction, edits) still
  `RESET`. The first turn is unchanged in cost.
- The request log now separates new vs reused tokens and the prefill rate:
  `prompt read in X s (N new of M tokens, R tok/s, reused U)`.

The checkpoint itself is one GDN/conv recurrent-state snapshot plus the in-place
attention KV, so advancing it is safe: attention KV for the reused prefix is
never rewritten and the new messages overwrite only positions past it.

## Results

In-process `Engine` (release v0.1.1 engine):

| turn | prompt tokens | TTFT+gen | note |
| --- | ---: | ---: | --- |
| 1 | 1593 | 4.91 s | cold, caches prefix |
| 2 | 1625 | 0.86 s | reused ~1570 |
| 2 (cold engine) | 1625 | 5.04 s | 5.9x slower than warm |

Warm turn-2 output is **byte-identical** to the cold turn-2 output.

Through the HTTP server (release engine, one growing conversation):

```
request: 2183 prompt tokens
  prompt read in 5.2 s (2183 new of 2183 tokens, 422 tok/s, reused 0)
request: 2211 prompt tokens
  extended prefix 2176 -> 2204 tokens (+28)
  prompt read in 0.5 s (35 new of 2211 tokens, 77 tok/s, reused 2176)
```

Per-turn TTFT falls by the size of the reused history (roughly 10x here; ~40x at
pi's 38K tokens, minus the new tool/user content).

## Correctness / safety

- Warm vs cold greedy output identical (in-process test above).
- `python -m unittest discover -s tests/lamina` passes, including a new
  `test_prefix_checkpoint_advances_when_conversation_grows`.
- Falls back to RESET whenever the previous checkpoint is not an exact prefix
  (verified against full tokenization) - same behavior as before.
- CUDA + device KV only (unchanged guard); images and `LAMINA_PREFIX_CACHE=0`
  bypass it; `_stop_native`/RESET clear the checkpoint.

## Limitations

- The first turn of a conversation still pays the full prompt prefill (~42 s for
  pi's 38K). Only the append-only tail is saved afterward.
- Tool results are part of the append, so a turn whose new content is very large
  still prefills that new content.
- Works with the shipped v0.1.1 engine (uses the existing CACHE/RESTORE protocol),
  so no engine rebuild is required for this change.

## Commands

```powershell
& "C:\projects\lamina\Lamina-data\venv\Scripts\python.exe" lamina.py serve `
  --engine C:\projects\lamina\Lamina-data\engine\v0.1.1\lamina-infer.exe `
  --cuda --compute-mode fast --kv-type f16 --kv-cache device --max-context 131072 `
  --mtp C:\projects\lamina\Lamina-data\mtp\qwen36-mtp-q8_0.gguf --port 8000
```
