# Lamina

Lamina runs **Qwen3.6-35B-A3B**, a 35-billion-parameter mixture-of-experts
language model, on an ordinary gaming PC. The 22 GB model does not fit in an
8 GB graphics card, so Lamina keeps the frequently used parts on the GPU and
computes the rest on the CPU from system RAM. It is a native C++/CUDA port of
[Strata](https://github.com/Niko1221/Strata) for this model.

On an RTX 3050 8 GB with a Ryzen 7 5700X and 64 GB of RAM it generates about
**44 tokens per second** with speculative decoding (37 without) and starts
answering a short question in about **0.6 seconds**. You can chat with it in
the terminal or use it as an OpenAI-compatible server.

## What you need

The setup script installs everything else (Python packages, the CUDA compiler,
the model). These must already be on the machine:

| Requirement | Details |
| --- | --- |
| Windows 10 or 11, 64-bit | Other systems need the manual build in [docs/ADVANCED.md](docs/ADVANCED.md). |
| NVIDIA GPU, RTX 30 series or newer | 8 GB of VRAM or more, with a current driver. |
| 64 GB of RAM recommended | With 48 GB or more Lamina pins the model in RAM for faster GPU transfers. Less RAM is untested and will be slower. |
| About 40 GB of free disk space | Model 22 GB, MTP head 2.5 GB during packing, CUDA compiler and build files. |
| [Python 3.11 or newer](https://www.python.org/downloads/) | Tick "Add python.exe to PATH" in the installer. |
| [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/) | Select the "Desktop development with C++" workload; it includes CMake and Ninja. |
| [Git](https://git-scm.com/download/win) | Used to fetch one pinned dependency during the build. |

## Quick start

Download or clone this repository, then double-click `START-HERE.bat` or run
it from a terminal in the repository folder:

```powershell
.\START-HERE.bat
```

The first run asks two questions (see below), then sets everything up and opens
a chat. All downloaded and generated files go to the sibling folder
`..\Lamina-data`, never into the repository. The first run takes a while:

1. It creates a Python environment and installs packages (a few minutes).
2. It downloads the 22 GB model; an interrupted download resumes.
3. It installs the CUDA compiler into `..\Lamina-data` and builds the engine for
   your GPU, which it detects automatically (the first build can take 10-30 minutes).
4. If you enabled MTP, it downloads and packs the MTP head (about 1.6 GB).

If a step fails, fix the cause shown in the message and run the same command
again; finished steps are skipped. Later starts take only a few seconds.

In the chat, type a message and press Enter. Type `/new` to start a new
conversation and `/exit` to quit. Each answer ends with its length and speed.

```text
You: What is the capital of France? Answer in one sentence.
Lamina: The capital of France is Paris.
```

## The two setup questions

Lamina asks these once and stores the answers in `..\Lamina-data\quickstart.json`.
Run `.\START-HERE.bat --reconfigure` to answer them again.

**Context length** is how much text one conversation can hold: your messages,
the model's answers and any pasted documents together. A token is roughly
three quarters of an English word.

| Choice | Holds about | When to pick it |
| --- | --- | --- |
| 8K | 6,000 words | Short questions; leaves the most VRAM for speed. |
| **32K (recommended)** | 24,000 words | Normal chats and medium documents. |
| 64K | 48,000 words | Long documents. |
| 128K | 96,000 words | Very long documents. The context memory grows with the conversation to about 2.7 GB of VRAM and generation slows down: on an RTX 3050 a 64K-token prompt took about 3 minutes to read, then 18 tokens per second. Prompts beyond 64K are untested on 8 GB GPUs. |

**MTP speculative decoding** uses a small extra part of the model (its
multi-token-prediction head) to guess the next token in advance. The model
checks each guess in the same step it would compute anyway, so answers stay
exactly the model's own and generation is about 20% faster. It applies to the
default deterministic answers (temperature 0); it needs a one-time download of
about 1.6 GB and about 100 MB of VRAM. Saying yes is recommended.

## Everyday commands

| Command | What it does |
| --- | --- |
| `.\START-HERE.bat` | Chat in the terminal (sets up first if needed). |
| `.\START-HERE.bat serve` | Starts the server on `http://127.0.0.1:8000`. |
| `.\START-HERE.bat setup` | Installs, downloads and builds without starting anything; run it after updating the repository. |
| `.\START-HERE.bat --reconfigure` | Asks the two setup questions again. |
| `.\START-HERE.bat --max-context 65536` | Uses another context length for this run only. |
| `.\START-HERE.bat --thinking` | Lets the model reason before answering (slower, often better on hard problems); the reasoning is printed separately. |
| `.\START-HERE.bat --max-tokens 4096` | Allows longer answers (default 2048 tokens). |
| `.\START-HERE.bat serve --port 9000` | Uses another port; `--host 0.0.0.0` exposes the server to your network, without any authentication. |

## Using the server

`.\START-HERE.bat serve` starts an OpenAI-compatible HTTP server. Any client
that supports a custom OpenAI base URL can use it, with base URL
`http://127.0.0.1:8000/v1`, any API key, and the model name
`Qwen3.6-35B-A3B-UD-Q4_K_M`. A quick test from PowerShell:

```powershell
$body = @{ model = "Qwen3.6-35B-A3B-UD-Q4_K_M"; messages = @(@{ role = "user"; content = "Name three primary colors." }) } | ConvertTo-Json -Depth 4
(Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType "application/json" -Body $body).choices[0].message.content
```

The same request with the official Python client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
reply = client.chat.completions.create(model="Qwen3.6-35B-A3B-UD-Q4_K_M",
                                       messages=[{"role": "user", "content": "Name three primary colors."}])
print(reply.choices[0].message.content)
```

The server supports streaming, `temperature`, `top_p`, `top_k`, `seed`,
`stop`, `max_tokens`, reasoning ("thinking") controls, function tools and JSON
output. It handles one request at a time. Speculative decoding speeds up
requests with temperature 0; other requests run at the normal speed.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| "Python 3.11 or newer is required" | Install Python from python.org with "Add to PATH" ticked, then open a new terminal. |
| "No NVIDIA GPU found" | Install or update the NVIDIA driver; `nvidia-smi` must work in a terminal. |
| The build fails | Check that Visual Studio 2022 with "Desktop development with C++" and Git are installed, then run `.\START-HERE.bat setup` again. |
| A download stopped | Run the same command again; downloads resume. |
| Out of GPU memory, or the engine exits | Close other programs that use the GPU (browsers and video players take GPU memory), or run `.\START-HERE.bat --reconfigure` and pick a shorter context. When memory runs low Lamina now releases cached experts and keeps going, slower, with one "GPU memory is low" note in its log. |
| Generation is slower than expected | With less than 48 GB of RAM the model cannot be pinned, which costs speed. Other GPU work (games, video) also competes. |
| "prompt and response exceed the context" | Type `/new` to start a fresh conversation, or choose a longer context. |

## How fast is it

Measured on an RTX 3050 8 GB, Ryzen 7 5700X and 64 GB of DDR4, generating 256
tokens for three test prompts (prose, code and math):

| Setting | Generated tokens per second | First token |
| --- | ---: | ---: |
| Lamina with MTP speculative decoding | 44-46 | 0.57 s |
| Lamina without MTP | 37 | 0.57 s |
| llama.cpp b11474 (CUDA), same machine | 28.6 | not compared |

The setup uses Lamina's fast mode: GPU computation with 8-bit activations and
an FP16 context cache, which passed the quality check against full precision
(mean KL divergence 0.003 nats on 1,024 tokens). Exact commands and results are
in the [hybrid prefill](bench/results/2026-10-08-rtx3050-hybrid-prefill/README.md),
[speculation](bench/results/2026-10-08-rtx3050-mtp-speculation/README.md) and
[tuning](bench/results/2026-10-08-rtx3050-decode-tuning/README.md) records.
A GPU with more VRAM keeps more of the model on the GPU and should be
considerably faster; this has not been measured.

## More

- [docs/ADVANCED.md](docs/ADVANCED.md): manual setup, all `lamina.py` options
  (image input, full-precision mode, 128K options, environment variables) and
  the measurement history.
- [docs/DEVELOPER_HANDOFF.md](docs/DEVELOPER_HANDOFF.md): how the engine works,
  checks to run after changes, and the performance work so far.
- [docs/LAMINA_PORT.md](docs/LAMINA_PORT.md): port status and validation limits.

Lamina is pinned to Strata commit `6f32ec070f23ced9f50e704d854d775da52591ab`;
upstream MIT licenses and attribution are preserved. The model files and
downloads are pinned in `tools/lamina_model.py` and `tools/lamina_assets.py`.
