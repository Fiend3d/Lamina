# Recommended sampling, presence penalty and llama.cpp comparison

This record covers the change that makes the model cards' sampling settings the
server defaults and adds `presence_penalty` to the native sampler, with a
matched speed comparison against CUDA llama.cpp b11474. It is written for
developers who need to know what the defaults cost, how they were checked, and
how Lamina compares with llama.cpp under the same sampling settings. No layer
math changed.

Hardware: NVIDIA GeForce RTX 3050 (8192 MiB, PCIe Gen3 running at x8), driver
616.92, AMD Ryzen 7 5700X (8 cores/16 threads), 64 GiB DDR4, Windows 11
Enterprise LTSC 2024. Lamina was built from `9ad6a8a` plus this change with
`tools.build_windows --cuda-arch 86`; the engine SHA-256 begins with
`99f72c6eeb6787d4`. llama.cpp is the prebuilt CUDA b11474 release
(`b9acf138a`), as in the earlier comparisons. Model: pinned Qwen3.6-35B-A3B
UD-Q4_K_M; Lamina's MTP runs use its Q8_0 head.

## What changed

The server fills each sampling field a request omits from the loaded model's
card, by thinking mode (`tools.lamina_chat.RECOMMENDED`):

| Model | Mode | temperature | top_p | top_k | presence_penalty |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6 | thinking | 1.0 | 0.95 | 20 | 1.5 |
| Qwen3.6 | non-thinking | 0.7 | 0.80 | 20 | 1.5 |
| Ornith | both | 0.6 | 0.95 | 20 | 0 |

Before this change, an omitted temperature meant greedy decoding. An explicit
`temperature: 0` still gives plain greedy decoding without the default penalty,
and `LAMINA_SAMPLING_DEFAULTS=greedy` restores the old defaults. pi sends no
sampling settings unless configured (checked in its OpenAI-completions
provider), so pi requests now use these defaults.

`presence_penalty` follows OpenAI/vLLM semantics: every token generated since
the request's `SAMPLE` command loses the penalty from its logit once,
independent of how often it occurred; prompt tokens do not count. `SAMPLE`
takes it as an optional fifth field, and engines advertise the
`presence_penalty` capability; the server drops the penalty, with a log line,
for engines without it. In sampled MTP the first verified token joins the
history before the bonus token is drawn, so a penalized speculative pass draws
from the same distributions as two single steps. Greedy selection on the GPU
and greedy `GENERATE` cannot apply a penalty, so penalized greedy requests
step on the logits path.

## Lamina against llama.cpp

Both engines ran the same token IDs for the prose/code/math prompts of
`tools.performance_check`, three repetitions each, 256 generated tokens with
EOS ignored, 32768 context, FP16 KV, no prompt cache. Sampled runs used Qwen's
non-thinking defaults (temperature 0.7, top_p 0.8, top_k 20, presence penalty
1.5) with seed 0, 1 and 2 for the three repetitions. llama.cpp ran with `min_p`
0 (its default is 0.05) and `repeat_last_n` 32768, so its penalty also covers
the whole answer (its default window is 64 tokens).

| Engine and mode | Prose | Code | Math | Nine-run median | Warm first token |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama.cpp, greedy | 28.79 | 28.80 | 28.86 | **28.80** | 1.42 s |
| Lamina, greedy | 41.84 | 39.21 | 38.12 | **39.21** | 0.57 s |
| Lamina, greedy, MTP | 45.52 | 47.22 | 47.00 | **46.86** | 0.56 s |
| llama.cpp, sampled | 26.68 | 28.22 | 28.35 | **28.22** | 1.45 s |
| Lamina, sampled | 39.92 | 37.24 | 36.61 | **37.24** | 0.58 s |
| Lamina, sampled, MTP | 41.58 | 43.90 | 44.24 | **43.90** | 0.58 s |

Rates are later tokens/s, excluding the first token; first-token times are
medians of the eight warm requests. With the same sampling settings, Lamina
decodes 32% faster than llama.cpp without speculation (37.24 against 28.22
tokens/s) and 56% faster with its MTP head (43.90). Greedy decoding is 36% and
63% faster. Lamina's warm first token takes 0.58 s against 1.45 s.

Sampling with the penalty costs Lamina about 5% against greedy decoding without
MTP (39.21 -> 37.24) and 6% with MTP (46.86 -> 43.90; acceptance 83.3% against
83.7%). llama.cpp lost 2% (28.80 -> 28.22).

Peak sampled total GPU memory was 6877-7085 MiB for Lamina and 7351-7519 MiB
for llama.cpp, against about 1.1-1.25 GiB idle desktop use. llama.cpp started
in 11.2 s; Lamina's first request after startup took 4.8-4.9 s because it
initializes CUDA and registers model RAM then.

Limits of this comparison:

- llama.cpp ran without speculation. Its `--spec-type draft-mtp` needs the MTP
  layer inside the model file; the pinned GGUF has none, and Lamina's packed
  head is not in llama.cpp's format. The fair comparison is therefore the two
  rows without MTP.
- The engines filter in a different order. llama.cpp applies top-p before
  temperature (`penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`);
  Lamina applies temperature before top-p, as Hugging Face does. With
  temperature 0.7 the nucleus differs, so the token streams differ even with
  equal seeds. This compares speed, not output quality.
- llama.cpp b11474 is an older release, matching the earlier comparisons on this
  machine. Newer releases may be faster.

## Correctness

Greedy output is unchanged: all nine greedy and all nine greedy-MTP token
streams equal those of the [11 October record](../2026-10-11-rtx3050-current/README.md).
Sampled repetitions differ by seed, as intended.

| Check | Result |
| --- | --- |
| `python -m unittest discover -s tests/lamina` | 78 tests pass, including new default, override, environment and capability tests |
| `lamina-sampling-check` (CUDA and default CPU build) | pass, including penalized greedy, history reset, exact penalized distribution, 100000-draw frequencies, negative penalty and invalid values |
| `lamina-sampled-mtp-check`, Qwen3.6 | pass; with presence 2.0 and -2.0 at top-k=1, speculative output equals single steps and differs from unpenalized greedy |
| `lamina-sampled-mtp-check`, Ornith | pass; presence -2.0 changes the output, 2.0 does not (no repeated greedy token to penalize) |
| `tools.check_sampled_mtp --presence-penalty 1.5`, Qwen3.6, one round | ordinary sampling and sampled MTP give identical answers for all three prompts through the real server |

The end-to-end report is `e2e-sampled-mtp-presence.json`.

## Exact commands

Run from the repository root with the sibling virtual environment. GPU
workloads ran serially.

```powershell
../Lamina-data/venv/Scripts/python.exe -m tools.build_windows --cuda-arch 86 --jobs 6
../Lamina-data/venv/Scripts/python.exe -m tools.check_sampled_mtp --model qwen3.6 --runs 1 --presence-penalty 1.5 --output <out>/e2e-qwen-presence
../Lamina-data/venv/Scripts/python.exe -m tools.compare_llama --load-mode none --report-dir <out>/llama-greedy
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --report-dir <out>/lamina-greedy
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --mtp ../Lamina-data/mtp/qwen36-mtp-q8_0.gguf --report-dir <out>/lamina-greedy-mtp
../Lamina-data/venv/Scripts/python.exe -m tools.compare_llama --load-mode none --temperature 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 --report-dir <out>/llama-sampled
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --temperature 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 --report-dir <out>/lamina-sampled
../Lamina-data/venv/Scripts/python.exe -m tools.compare_lamina --temperature 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 --mtp ../Lamina-data/mtp/qwen36-mtp-q8_0.gguf --report-dir <out>/lamina-sampled-mtp
```

`compare_lamina` and `compare_llama` gained `--temperature`, `--top-p`,
`--top-k`, `--presence-penalty` and `--seed` for this record; without them they
measure greedy decoding as before. Each run directory holds the per-request
JSON (with token IDs, text and the exact request or sampling settings) and a
summary; llama.cpp server logs are excluded by `.gitignore`.
