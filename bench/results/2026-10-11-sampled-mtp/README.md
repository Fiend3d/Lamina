# MTP at nonzero temperature

Hardware: NVIDIA RTX 4060 8 GB (8188 MiB), AMD Ryzen 7 1700X, 64 GB RAM,
Windows. Source: b1552e4 plus this change. Fast compute, device F16 KV, context
capacity 131072, temperature 0.6, top_p 0.95, top_k 20, MTP head loaded in both
variants. Each variant has one warm-up round then three rounds of prose/code/math,
128 output tokens, seeds 43–45: nine measured requests per variant. Prompts
are short (32–36 tokens); these are desktop measurements, not a 128K benchmark
or a quality evaluation.

```powershell
..\Lamina-data\venv\Scripts\python.exe -m tools.build_windows --jobs 8
..\Lamina-data\venv\Scripts\python.exe -m tools.check_sampled_mtp --model qwen3.6 --runs 3 --output ..\Lamina-data\validation\sampled-mtp-qwen-greedy-proposal
..\Lamina-data\venv\Scripts\python.exe -m tools.check_sampled_mtp --model ornith --runs 3 --output ..\Lamina-data\validation\sampled-mtp-ornith
.\build-cuda\lamina-sampled-mtp-check.exe ..\Lamina-data\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf ..\Lamina-data\mtp\qwen36-mtp-q8_0.gguf
.\build-cuda\lamina-sampled-mtp-check.exe ..\Lamina-data\models\Ornith-1.5-35B-Q4_K_M.gguf ..\Lamina-data\mtp\ornith-mtp.gguf
.\build-cuda\lamina-sampling-check.exe
```

| Qwen measurement | Ordinary sampling | Sampled MTP |
| --- | ---: | ---: |
| Median later tokens/s, nine requests | 27.26 | 28.81 |
| Median first-token latency | 0.608 s | 0.612 s |
| Prose median tokens/s | 27.19 | 25.96 |
| Code median tokens/s | 27.68 | 29.57 |
| Math median tokens/s | 27.07 | 28.81 |

Qwen peak total GPU memory sampled with nvidia-smi: 6758 MiB. All nine paired
measured token sequences matched ordinary sampling with the same seed. This
is fixture evidence, not a universal byte-equality guarantee for all fast-mode
CPU/GPU expert placements. Speed gain is modest and prose regressed.

An Ornith pilot measured median later rates 27.41 -> 28.20 tokens/s, first
token 0.589 -> 0.596 s, sampled peak total GPU memory 6870 MiB. Some unrelated
small CUDA checks overlapped that pilot, so it is not used as a controlled
performance claim. Its native correctness tests passed independently.

The initial full-distribution proposal variant was slower on Qwen:
27.83 -> 24.49 tokens/s, first token 0.607 -> 0.631 s, peak 6775 MiB.
It remains an opt-in experiment (`LAMINA_MTP_PROPOSAL=sampled`); default proposals
are greedy point masses. The target is still sampled at the requested nonzero
temperature. Speculation is not allowed to turn it into greedy decoding.

For a point-mass proposal at draft d, drawing x from filtered target p and
accepting iff x=d gives acceptance p[d]. On rejection, x already follows p
conditioned on x!=d, the required residual. If accepted, the bonus draw uses
the second main-model column. If rejected, recurrent state rolls back to the
first column and the rejected second attention row lies beyond the active
position. This preserves the sampling distribution and ordinary RNG draws
when the verified numerical logits agree. The optional sampled-proposal path
uses min(1,p/q) acceptance and normalized max(0,p-q) residual correction, as in
[speculative decoding](https://proceedings.mlr.press/v202/leviathan23a.html).

Checks: 100000-draw tests against known target distributions, including identical
and disjoint supports; top-k=1 sampled/greedy parity; seeded native replay;
position/single-tail accounting; sampled-versus-serial continuation state on the
same emitted tokens (max_abs_diff=0 for Qwen and Ornith); Python server capability
fallback and stop-token handling in speculative chunks. Default/CPU and CUDA
builds passed, as did native sampling and CUDA numerical checks. No layer
formulas changed.

`LAMINA_SAMPLED_MTP=0` selects ordinary sampling. Older v0.2.0 binaries do not
advertise `sampled_mtp` and retain their previous fallback. New source servers
log "decoding with sampled MTP" for supported nonzero-temperature requests.
Existing pi temperature settings are not changed automatically. This addresses
sampling support, not an agent's tendency to retry incorrect tool calls.

Raw records: [Qwen default](qwen.json), [Ornith pilot](ornith.json),
[Qwen full-proposal prototype](qwen-full-proposal.json).
