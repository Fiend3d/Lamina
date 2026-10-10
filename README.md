# Lamina

Lamina runs an AI model on your own Windows PC. You can chat in a terminal or
connect a coding app such as pi. After the initial downloads, inference runs
locally.

It supports **Qwen3.6-35B-A3B** and
**Ornith-1.5-35B-A3B**. Each model is about 22 GB. Lamina keeps frequently used
weights on the graphics card and computes the rest from system RAM, allowing
these models to run with 8 GB of VRAM. It is a native C++/CUDA port of
[Strata](https://github.com/Niko1221/Strata).

## What you need

The portable release includes Python, the engine and its runtime libraries.
You need Windows 10/11 x64, an AVX2-capable CPU (with FMA/F16C/BMI2), an NVIDIA
RTX 30/40/50 GPU with at least 8 GB VRAM and a CUDA 13.3-compatible driver,
64 GB RAM recommended, and about 40 GB of free disk space. Internet is needed
for the initial model download. No Visual Studio, Git, Python installation or
CUDA toolkit is needed to use the portable ZIP.

## Quick start

Download **`lamina-windows-x64-portable.zip`** from
[GitHub Releases](https://github.com/Fiend3d/Lamina/releases), extract it into a
writable folder, then double-click **`START-HERE.bat`**. Extract the whole ZIP;
do not run the launcher inside it.

On the first run:

1. Choose a model: Qwen3.6 or Ornith. Both work in the portable ZIP.
2. Choose how much conversation history to support: 8K, 32K, 64K or 128K.
3. Choose whether to enable MTP, an optional generation speedup.
4. Choose whether to enable CPU image input and sampled video input.
5. Let setup download the model and load it. The first download is large;
   wait until you see **Lamina is ready**.

Your choices and downloads are saved in the neighboring `Lamina-data` folder.
Later launches reuse them and start the server without asking again. The
portable ZIP does not require compiling anything.

```text
 Lamina is ready.

   Base URL : http://127.0.0.1:8000/v1
   Model    : Qwen3.6-35B-A3B-UD-Q4_K_M
   API key  : anything (it is not checked)
```

Leave the window open while using the model; close it or press Ctrl+C to stop.
Connect a coding agent such as [pi](#connecting-pi-and-other-apps) or another
OpenAI-compatible app. If you simply want to chat, open a terminal in the
extracted folder and run:

```powershell
.\START-HERE.bat chat
```

Type your message at `You:`. Use `/new` to clear the conversation and `/exit`
to quit. To open a terminal in the folder, right-click its empty space in File
Explorer and choose **Open in Terminal**, or open PowerShell and `cd` there.
The portable release includes text, tools, reasoning and image input for both
models. With image input enabled, setup downloads the selected model's image projector
(about 900 MB). Apps can send screenshots or photos through the API using
OpenAI-compatible `image_url` content, including base64 data URLs. The default image
encoder can use the CPU or GPU; both are included in v0.2.0 as described below.
On older CPUs, an image response can take a minute or more. Text-only requests
do not start the image encoder.

To enable images or sampled video explicitly (saved for later launches):

```powershell
.\START-HERE.bat --vision on
.\START-HERE.bat --video-input on
```

Video input also enables images. `--vision off` disables both; `--video-input off`
preserves the image setting. In terminal chat, use `/image "C:\media\photo.jpg" Describe it`
or `/video "C:\media\clip.mp4" Describe what happens`. For a single answer:

```powershell
.\START-HERE.bat chat --image "C:\media\photo.jpg" --prompt "Describe this image"
.\START-HERE.bat chat --video "C:\media\clip.mp4" --video-frames 4 --video-interval 1 --prompt "What happens in order?"
```

**Video formats:** the bundled FFmpeg decoder accepts common MP4, MOV, MKV,
AVI and WebM files, depending on the codec inside. MP4 with H.264 is the recommended
choice. Video is sampled into timestamped images, starting at 0 seconds: the default
reads at most four frames at 0, 1, 2 and 3 seconds, stopping at the end of the file.
It does not scan the whole clip automatically. Increase `--video-interval` to cover
a longer clip and `--video-frames` (1?16) for more samples. Each frame consumes image
context and encoder time; sampled frames can miss brief events. Audio and native
video temporal positions are not processed. The API still accepts `image_url` items;
API clients must extract and submit frames themselves (the local helper is
`tools.lamina_media.video_content`). It does not accept `video_url` or video data URLs.
Images are decoded by Pillow; common PNG, JPEG and WebP inputs work, and animated
images use their first frame.

If using a **source checkout**, install Python 3.11 or newer and run
`START-HERE.bat`. Setup installs runtime Python packages and downloads the pinned
prebuilt engine. To compile locally, use `START-HERE.bat setup --build-source`;
that requires the C++ build tools described in [the manual setup](docs/ADVANCED.md).
See [Windows releases](docs/WINDOWS_RELEASE.md) for packaging and validation.

To switch to Ornith, run:

```powershell
.\START-HERE.bat --model ornith
```

Setup downloads Ornith, its tokenizer and image projector, and prepares its MTP
head automatically. No compiler is needed.

Setup saves Ornith as your model choice. You can switch back later without
deleting either model's downloaded files.

## Choosing and changing your settings

Lamina saves **model, context length, MTP, image input and sampled video input** in
`..\Lamina-data\quickstart.json`. You do not need to edit that file.
To answer the settings questions again, stop the running server and run:

```powershell
.\START-HERE.bat --reconfigure
```

Press Enter at a question to keep its current setting. Your new choices are
saved for future launches; setup downloads anything missing, then starts the
server. To change settings without starting it, use
`.\START-HERE.bat setup --reconfigure`.

**Model:** choose Qwen3.6 or Ornith. You can also switch directly with
`.\START-HERE.bat --model ornith` or `.\START-HERE.bat --model qwen3.6`.
An explicit `--model` skips the model question, even with `--reconfigure`.

**Context length** is how much text one conversation can hold: your messages,
the model's answers and any pasted documents together. A token is roughly
three quarters of an English word.

| Choice | Holds about | When to pick it |
| --- | --- | --- |
| 8K | 6,000 words | Short questions; leaves the most VRAM for speed. |
| **32K (recommended)** | 24,000 words | Normal chats and medium documents. |
| 64K | 48,000 words | Long documents. |
| 128K | 96,000 words | Long coding sessions and large documents. Full 128K history is tested on RTX 4060 8 GB. Reading a fresh full prompt can take minutes, and generation slows as history grows. |

The context budget includes the answer you ask for. Choosing 128K does not
make every small request ingest 128K tokens: memory and work grow with the
actual conversation. The server reuses unchanged history on follow-up turns,
so it usually needs to read only the added messages and tool results.

**MTP speculative decoding** uses a small extra part of the model (its
multi-token-prediction head) to guess the next token in advance. The model
checks the guesses before accepting them. This can make generation faster;
the gain varies by model, hardware and context length. It applies to requests
with temperature 0, the default. Setup prepares the model's prediction head,
which uses some extra GPU memory. Saying yes is a reasonable starting choice; you
can turn it off with `--reconfigure`.

## Everyday commands

| Command | What it does |
| --- | --- |
| `.\START-HERE.bat` | Starts the server on `http://127.0.0.1:8000/v1` and loads the model (sets up first if needed). |
| `.\START-HERE.bat chat` | Talks to the model in the terminal instead of starting the server. |
| `.\START-HERE.bat setup` | Downloads and sets up without starting the server. |
| `.\START-HERE.bat setup --build-source` | Developer source build; requires Visual Studio C++ tools. |
| `.\START-HERE.bat --reconfigure` | Asks about model, context length and MTP again; saves your choices. |
| `.\START-HERE.bat --model ornith` | Switches to Ornith, downloads missing files and saves the choice. |
| `.\START-HERE.bat --model qwen3.6` | Switches to Qwen3.6 and saves the choice. |
| `.\START-HERE.bat --max-context 65536` | Uses another context length for this run only. |
| `.\START-HERE.bat chat --thinking` | Lets the model reason before answering (slower, often better on hard problems); the reasoning is printed separately. |
| `.\START-HERE.bat chat --max-tokens 4096` | Allows longer answers in the terminal chat (default 2048 tokens). |
| `.\START-HERE.bat --port 9000` | Uses another port; `--host 0.0.0.0` exposes the server to your network, without any authentication. |

## Using the server

`.\START-HERE.bat` starts an OpenAI-compatible HTTP server. Any client
that supports a custom OpenAI base URL can use it, with base URL
`http://127.0.0.1:8000/v1`, any API key, and the model name
`Qwen3.6-35B-A3B-UD-Q4_K_M`. Copy the connection details printed under
**Lamina is ready** into your app. The current API uses that same model ID
when serving Ornith; your saved model choice determines the loaded weights.
A quick test from a second PowerShell window:

```powershell
$body = @{ model = "Qwen3.6-35B-A3B-UD-Q4_K_M"; messages = @(@{ role = "user"; content = "Name three primary colors." }) } | ConvertTo-Json -Depth 4
(Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType "application/json" -Body $body).choices[0].message.content
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
The example is for the 131072-token (128K) context and lets one answer use up
to 32768 tokens:

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
        "input": ["text", "image"],
        "contextWindow": 131072,
        "maxTokens": 32768,
        "compat": {
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

`contextWindow` must match the context length chosen at setup. If you chose
32K, use `"contextWindow": 32768` and `"maxTokens": 8192`; run
`.\START-HERE.bat --reconfigure` to switch the server to 128K. `maxTokens` is
the longest answer pi asks for. The server rejects a request whose prompt and
`maxTokens` together exceed the context, so keep it well below
`contextWindow`. A whole file is written in one answer, so a small value cuts
big files off. The `id` stays `Qwen3.6-35B-A3B-UD-Q4_K_M` when the server runs
Ornith, because that is the only model name the server reports and accepts;
`name` is only the label pi shows.

Then run `pi --model lamina/Qwen3.6-35B-A3B-UD-Q4_K_M` in your project folder
and pick the model with `/model` if needed. With `"reasoning": true` and
`"thinkingFormat": "qwen"` pi sends `enable_thinking`, so the model's reasoning
appears in pi while it is generated; switch it off with `/thinking` for faster
answers. If pi's bash tool reports
that no shell is available, set `"shellPath"` in `~\.pi\agent\settings.json` to
your Git Bash, for example `"C:\\Git\\bin\\bash.exe"`.

**GPU images and video:** use your NVIDIA GPU for the image
encoder instead of the CPU:

```powershell
.\START-HERE.bat --vision-device gpu
```

This saves the device choice. The v0.2.0 portable package uses its bundled GPU
encoder and needs no compiler. In a source checkout, a missing GPU encoder is
built on first use (requires Visual Studio C++ tools and the pinned CUDA toolkit).
The separate `build-vision-gpu` directory preserves the CPU encoder.
For an explicit build: `python -m tools.build_windows --vision-gpu --vision-only
--cuda-arch 89` on RTX 4060; use the architecture detected for your own GPU.
The v0.2.0 portable release includes both CPU and GPU encoders and the video decoder.

GPU mode gives the encoder and language model separate turns on the GPU. For
new images/video frames, it unloads the native text model, encodes the missing
frames on the GPU, releases the encoder, and reloads the text model to answer.
Previously encoded images are cached in server RAM (up to 256 MiB), so repeating
them keeps the text model resident. With CUDA device KV, follow-up requests
reuse the unchanged conversation, including images, and prefill only new content.
Changed image bytes, image order or edited history invalidate prefix reuse.
New images, evicted embeddings and server restarts can still require full prefill;
the first image request in a long conversation can still take minutes. `/health` reports
`"vision_device": "gpu"`, and the server logs each frame's encoding time.
On RTX 4060, a repeated 102K-token image request reached its first answer in
1.47 s versus 261 s for the initial pass; see the
[image cache validation](bench/results/2026-10-10-image-cache/README.md).
Long text sections around images also use the optimized prefill path when
CUDA device-KV prefix caching is available. A 32K-token, two-image test on
RTX 4060 reached its first answer in 63 s versus 90 s previously; see
[image prefill measurements](bench/results/2026-10-10-image-prefill/README.md).
Stop an existing server before switching; its encoder cannot change mid-request.
Use `.\START-HERE.bat --vision-device cpu` to switch back. `--reconfigure` also
asks for the encoder device. Images and sampled video in pi use whichever device
the running server has selected; no further pi setting is required.
The initial resident-encoder implementation failed during a 109K-token pi
conversation on the RTX 4060 8 GB. Encoder release is required before text
inference. The corrected lifecycle passed GPU video, 109K text, an append to
129K, and an image after the long conversation. See the
[GPU memory regression record](bench/results/2026-10-10-gpu-vision-memory/README.md)
for commands, timing, peak VRAM and limitations.

**Images in pi:** the Lamina model entry must contain `"input": ["text", "image"]`
as shown above, and the server must start with image input enabled
(`.\START-HERE.bat --vision on`). Attach a PNG, JPEG or WebP image with pi's
`@path` input, or ask pi to read a local image file and describe it. After editing
`models.json`, restart pi or run `/reload`. Check `pi --list-models lamina`:
the Lamina row should show `yes` in the `images` column. With the CPU encoder, processing can take
a minute or more per image.

**Videos in pi:** use the included [lamina-video skill](.pi/skills/lamina-video/SKILL.md).
It samples timestamped frames and submits them to the existing Lamina server,
without loading a second model. Start pi from this installation folder to discover
the project skill, or add its absolute directory to the `skills` array in
`~\.pi\agent\settings.json` to make it available in other projects:

```json
{
  "skills": ["D:/Projects/lamina/Lamina/.pi/skills/lamina-video"]
}
```

Use your own installation path and merge this field with your existing settings.
Restart pi or run `/reload`, then ask:

```text
/skill:lamina-video Look at C:/media/clip.mp4 and describe what happens in time order.
```

You can also ask pi to inspect a video normally; the skill's description makes
it available for automatic selection. The explicit command ensures it is loaded.
The default samples at most four frames at 0, 1, 2 and 3 seconds. Ask for a
different interval or up to 16 frames to cover a longer clip. MP4/H.264 is
recommended; tested combinations also include MOV/MKV with H.264, AVI with
MPEG-4 and WebM with VP9. Audio and direct MP4 attachments are unsupported:
keep pi's model `input` as `["text", "image"]`, without adding `"video"`.

To use the same running-server video client directly:

```powershell
..\Lamina-data\venv\Scripts\python.exe -m tools.lamina_media --video "C:\media\clip.mp4" --prompt "Describe what happens" --frames 4 --interval 1
```

In a portable installation, use `.\python\python.exe` instead. `--base-url`
overrides the default `http://127.0.0.1:8000/v1`. Use this client while the server
is running; the earlier `START-HERE.bat chat --video` command is for standalone
terminal chat when the server is stopped.

While the model works, pi shows its reasoning, its text and the file it is
writing as they are generated, and the server window logs every request with
its speed, for example `done: 556 answer tokens in 12.6 s = 43.9 tokens/s`. If
an answer reaches `maxTokens` in the middle of a tool call, pi shows the error
"The answer reached the max_tokens limit while the model was writing a tool
call": raise `maxTokens`, or ask for a smaller step (a big page in several
files, or one part at a time).

The first request in a long conversation takes longer because Lamina must
read its history. Later requests reuse the unchanged conversation prefix and
process only the new text. Restarting the server, editing earlier messages or
compacting the conversation can require reading the history again. Check the
server log's `reused` token count to see whether reuse is working.

Review generated code and changes before using them.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| "Python 3.11 or newer is required" | Install Python from python.org with "Add to PATH" ticked, then open a new terminal. |
| "No NVIDIA GPU found" | Install or update the NVIDIA driver; `nvidia-smi` must work in a terminal. |
| The source build fails | Check that Visual Studio 2022 with "Desktop development with C++" and Git are installed, then run `.\START-HERE.bat setup --build-source` again. |
| I want a different model or context length | Stop the server, then run `.\START-HERE.bat --reconfigure`. Press Enter to keep any setting you do not want to change. |
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
able to reduce CPU work. Actual speed also depends on CPU, RAM, context length
and model; the numbers above describe that specific machine.

On RTX 4060 8 GB / Ryzen 7 1700X, the latest source build measures **39.17
tokens/s for Ornith, or 41.27 with MTP**, on short prompts with 128K capacity
enabled. Full-history rates are lower. See the
[decode and context measurements](bench/results/2026-10-09-decode-admission-128k/README.md)
for exact commands, memory use, full-context limits and cached 80K conversation
timings, and the [initial Ornith benchmark](bench/results/2026-10-09-ornith-rtx4060/README.md)
for setup and output-comparison limitations.

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
