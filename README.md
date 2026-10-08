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

The first run asks two questions (see below), then sets everything up and starts
the model server. All downloaded and generated files go to the sibling folder
`..\Lamina-data`, never into the repository. The first run takes a while:

1. It creates a Python environment and installs packages (a few minutes).
2. It downloads the 22 GB model; an interrupted download resumes.
3. It installs the CUDA compiler into `..\Lamina-data` and builds the engine for
   your GPU, which it detects automatically (the first build can take 10-30 minutes).
4. If you enabled MTP, it downloads and packs the MTP head (about 1.6 GB).

If a step fails, fix the cause shown in the message and run the same command
again; finished steps are skipped. Later starts skip setup.

When everything is loaded, the window shows the address to use. Loading the
model takes up to a minute or two, during which the PC can be slow:

```text
 Lamina is ready.

   Base URL : http://127.0.0.1:8000/v1
   Model    : Qwen3.6-35B-A3B-UD-Q4_K_M
   API key  : anything (it is not checked)
```

Leave the window open while you use the model, and close it (or press Ctrl+C)
to stop the server. Add the base URL as an "OpenAI-compatible" provider in a
coding agent such as [pi](#connecting-pi-and-other-apps) or in any other app.

To talk to the model right in the terminal instead, run
`.\START-HERE.bat chat`. Type a message and press Enter; `/new` starts a new
conversation and `/exit` quits. Each answer ends with its length and speed.

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
| `.\START-HERE.bat` | Starts the server on `http://127.0.0.1:8000/v1` and loads the model (sets up first if needed). |
| `.\START-HERE.bat chat` | Talks to the model in the terminal instead of starting the server. |
| `.\START-HERE.bat setup` | Installs, downloads and builds without starting anything; run it after updating the repository. |
| `.\START-HERE.bat --reconfigure` | Asks the two setup questions again. |
| `.\START-HERE.bat --max-context 65536` | Uses another context length for this run only. |
| `.\START-HERE.bat chat --thinking` | Lets the model reason before answering (slower, often better on hard problems); the reasoning is printed separately. |
| `.\START-HERE.bat chat --max-tokens 4096` | Allows longer answers in the terminal chat (default 2048 tokens). |
| `.\START-HERE.bat --port 9000` | Uses another port; `--host 0.0.0.0` exposes the server to your network, without any authentication. |

## Using the server

`.\START-HERE.bat` starts an OpenAI-compatible HTTP server. Any client
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

## Connecting pi and other apps

Start the server with `.\START-HERE.bat` and leave its window open. Then add
Lamina to the app as an OpenAI-compatible provider. For the
[pi coding agent](https://pi.dev) (needs Node.js 22.19 or newer and Git for
Windows), put this in `~\.pi\agent\models.json`, creating the file if needed.
Set `contextWindow` to the context length you chose. `maxTokens` is the longest
answer pi asks for; the server rejects a budget that does not fit beside the
prompt, so keep it well below the context (8192 for 32K, 32768 for 128K). A
whole file is written in one answer, so a small value cuts big files off:

```json
{
  "providers": {
    "lamina": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "lamina",
      "models": [{
        "id": "Qwen3.6-35B-A3B-UD-Q4_K_M",
        "name": "Qwen3.6-35B-A3B (Lamina)",
        "reasoning": true,
        "input": ["text"],
        "contextWindow": 32768,
        "maxTokens": 8192,
        "compat": {
          "supportsDeveloperRole": false,
          "supportsReasoningEffort": false,
          "supportsUsageInStreaming": true,
          "maxTokensField": "max_tokens",
          "thinkingFormat": "qwen"
        }
      }]
    }
  }
}
```

Then run `pi --model lamina/Qwen3.6-35B-A3B-UD-Q4_K_M` in your project folder
and pick the model with `/model` if needed. The `compat` lines matter: the
server accepts only the system, user, assistant and tool message roles, and
answers `max_tokens`, not `max_completion_tokens`. With `"reasoning": true` and
`"thinkingFormat": "qwen"` pi sends `enable_thinking`, so the model's reasoning
appears in pi while it is generated; switch it off with `/thinking` for faster
answers. If pi's bash tool reports
that no shell is available, set `"shellPath"` in `~\.pi\agent\settings.json` to
your Git Bash, for example `"C:\\Git\\bin\\bash.exe"`.

While the model works, pi shows its reasoning, its text and the file it is
writing as they are generated, and the server window logs every request with
its speed, for example `done: 556 answer tokens in 12.6 s = 43.9 tokens/s`. If
an answer reaches `maxTokens` in the middle of a tool call, pi shows the error
"The answer reached the max_tokens limit while the model was writing a tool
call": raise `maxTokens`, or ask for a smaller step (a big page in several
files, or one part at a time).

What to expect from an agent: pi's own instructions take about 1,700 tokens
and Lamina starts every request from scratch, so each step costs about 7
seconds on an RTX 3050 even when the answer is short, and more as the
conversation grows (reading a prompt runs at roughly 300-400 tokens per
second). The model is a 4-bit 35B-parameter model, so check what it reports
and what it changes.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| "Python 3.11 or newer is required" | Install Python from python.org with "Add to PATH" ticked, then open a new terminal. |
| "No NVIDIA GPU found" | Install or update the NVIDIA driver; `nvidia-smi` must work in a terminal. |
| The build fails | Check that Visual Studio 2022 with "Desktop development with C++" and Git are installed, then run `.\START-HERE.bat setup` again. |
| A download stopped | Run the same command again; downloads resume. |
| Out of GPU memory, or the engine exits | Close other programs that use the GPU (browsers and video players take GPU memory), or run `.\START-HERE.bat --reconfigure` and pick a shorter context. When memory runs low Lamina now releases cached experts and keeps going, slower, with one "GPU memory is low" note in its log. |
| Generation is slower than expected | With less than 48 GB of RAM the model cannot be pinned, which costs speed. Other GPU work (games, video) also competes. |
| "prompt and response exceed the context" | In the terminal chat type `/new` to start a fresh conversation. In an app, start a new conversation, or choose a longer context. |
| Ctrl+C does not stop the server | Older versions waited for the answer in progress; update, or close the window instead. Windows may ask "Terminate batch job (Y/N)?": answer Y. |
| "Something already answers on port 8000" | A Lamina server is still running in another window. Close it, or start this one with `--port 8001`. |

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
