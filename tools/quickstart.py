"""One-command setup and launcher for Lamina on Windows.

    python tools/quickstart.py            set up if needed, then start the server
    python tools/quickstart.py chat       talk to the model in this window instead
    python tools/quickstart.py setup      install, download and build only
    python tools/quickstart.py --reconfigure   ask the questions again

The server speaks the OpenAI API on http://127.0.0.1:8000/v1, so coding agents such as
pi and other apps can use it. It loads the model at startup and stops when this window
is closed or Ctrl+C is pressed.

Every setup step is skipped when its result is already present, so re-running
is cheap. Answers to the two questions (context length, MTP speculation) are
stored in ../Lamina-data/quickstart.json.
"""
import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "Lamina-data"
VENV_PYTHON = DATA / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
CONFIG = DATA / "quickstart.json"
ENGINE = ROOT / "build-cuda" / ("lamina-infer.exe" if sys.platform == "win32" else "lamina-infer")
MTP_FILE = DATA / "mtp" / "qwen36-mtp-q8_0.gguf"
REQUIREMENTS = ["requirements.txt", "requirements-build.txt", "requirements-reference.txt", "requirements-benchmark.txt"]

# Context lengths offered on first run. The KV cache is FP16 on the GPU in all
# of them: that is the measured fast configuration, and MTP speculation needs
# the cache on the GPU. Longer contexts leave less VRAM for the expert cache.
CONTEXTS = [
    (8192, "8K    short chats, most VRAM left for experts"),
    (32768, "32K   recommended"),
    (65536, "64K   long documents"),
    (131072, "128K  very long documents; the context memory alone grows to about 2.7 GiB, decode is slower"),
]
SMALL_GPU_MIB = 9000  # up to 8 GB cards: GPU memory gets tight at 128K


def context_menu(vram_mib):
    lines = []
    for i, (context, text) in enumerate(CONTEXTS, 1):
        if vram_mib <= SMALL_GPU_MIB and context == 131072:
            text += ("\n       works on %.0f GB GPUs but slowly: reading a 64K-token prompt takes about 3 minutes;"
                     "\n       prompts beyond 64K are untested" % (vram_mib / 1024))
        lines.append(f"  {i}) {text}")
    return lines


def step(title):
    print(f"\n== {title}", flush=True)


def run(command, **kwargs):
    print("   $ " + " ".join(str(c) for c in command), flush=True)
    subprocess.run([str(c) for c in command], cwd=ROOT, check=True, **kwargs)


def ensure_venv(argv):
    """Creates the virtual environment and re-runs this script inside it."""
    if Path(sys.prefix).resolve() == VENV_PYTHON.parents[1].resolve():
        return
    if sys.version_info < (3, 11):
        sys.exit("Lamina needs Python 3.11 or newer: https://www.python.org/downloads/")
    if not VENV_PYTHON.is_file():
        step(f"Creating the Python environment in {VENV_PYTHON.parents[1]}")
        DATA.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", str(VENV_PYTHON.parents[1])], check=True)
    sys.exit(subprocess.run([str(VENV_PYTHON), "-m", "tools.quickstart", *argv], cwd=ROOT).returncode)


def total_ram_gib():
    if sys.platform != "win32":
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30

    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                    ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                    ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
    status = Status(); status.length = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.total / 2**30


def gpu():
    """Returns (name, compute capability as '86', VRAM MiB) of the first NVIDIA GPU."""
    try:
        line = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        sys.exit("No NVIDIA GPU found (nvidia-smi failed). Lamina's fast path needs an NVIDIA GPU and driver.")
    name, capability, memory = (part.strip() for part in line.split(","))
    return name, capability.replace(".", ""), int(float(memory))


def load_config():
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def prompt(text):
    try:
        return input(text).strip()
    except EOFError:  # closed input: take the default
        print()
        return ""


def ask(config, reconfigure):
    """Asks for the context length and MTP once; non-interactive runs take the defaults."""
    interactive = sys.stdin.isatty()
    if reconfigure or "max_context" not in config:
        choice = 2
        if interactive:
            print("\nHow much context (prompt plus answer) should Lamina support?")
            print("\n".join(context_menu(gpu()[2])))
            answer = prompt("Choose 1-4 [2]: ")
            choice = int(answer) if answer in ("1", "2", "3", "4") else 2
        config["max_context"] = CONTEXTS[choice - 1][0]
    if reconfigure or "mtp" not in config:
        enable = True
        if interactive:
            print("\nEnable MTP speculative decoding? It drafts one token ahead with the model's own")
            print("multi-token-prediction head and verifies it in the same pass: about 20% faster")
            print("generation at temperature 0 (the default), identical quality. It needs a one-time")
            print("download of about 1.6 GB, packed to 857 MB, and about 100 MB of extra VRAM.")
            enable = prompt("Enable MTP? [Y/n]: ").lower() not in ("n", "no")
        config["mtp"] = enable
    CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def requirements_digest():
    digest = hashlib.sha256()
    for name in REQUIREMENTS:
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def setup(config, cuda_arch=None):
    from tools.lamina_model import FILENAME
    name, capability, memory = gpu()
    arch = cuda_arch or capability
    print(f"GPU: {name}, compute capability {capability[0]}.{capability[1:]}, {memory} MiB; RAM: {total_ram_gib():.0f} GiB")

    marker = DATA / ".quickstart-requirements"
    if not marker.is_file() or marker.read_text() != requirements_digest():
        step("Installing Python packages")
        run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q",
             *[arg for name in REQUIREMENTS for arg in ("-r", name)]])
        marker.write_text(requirements_digest())

    if not (DATA / "models" / FILENAME).is_file():
        step("Downloading the model (22 GB, resumable)")
        run([sys.executable, "setup.py", "--download"])
    if not (DATA / "tokenizer" / "tokenizer.json").is_file():
        step("Downloading the tokenizer")
        run([sys.executable, "setup.py", "--tokenizer"])
    step("Checking the chat template and assets")
    run([sys.executable, "-m", "tools.lamina_assets"])
    if not (DATA / "toolchains" / "cuda" / "bin" / ("nvcc.exe" if sys.platform == "win32" else "nvcc")).is_file():
        step("Installing the pinned CUDA compiler into Lamina-data")
        run([sys.executable, "-m", "tools.bootstrap_cuda"])

    step(f"Building the engine for compute capability {arch} (incremental)")
    run([sys.executable, "-m", "tools.build_windows", "--cuda-arch", arch])
    config["cuda_arch"] = arch

    if config.get("mtp") and not MTP_FILE.is_file():
        step("Fetching and packing the MTP head (about 1.6 GB download)")
        run([sys.executable, "-m", "tools.lamina_mtp", "fetch"])
        run([sys.executable, "-m", "tools.lamina_mtp", "pack"])
    CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("\nSetup complete.")


def engine_environment():
    env = dict(os.environ)
    # Pinning the mapped model (about 21 GiB) enables direct expert uploads and
    # expert admission: 44.8-46.2 instead of 36.1-41.0 tokens/s on the RTX 3050
    # machine. Only with enough RAM to spare.
    if "LAMINA_HOST_REGISTER" not in env and total_ram_gib() >= 48:
        env["LAMINA_HOST_REGISTER"] = "1"
    return env


def engine_options(config, context):
    options = ["--engine", ENGINE, "--cuda", "--compute-mode", "fast", "--kv-type", "f16", "--kv-cache", "device",
               "--max-context", str(context)]
    if config.get("mtp") and MTP_FILE.is_file():
        options += ["--mtp", MTP_FILE]
    return options


def chat(config, context, thinking, max_tokens):
    os.environ.update(engine_environment())
    from tools.lamina_chat import Engine
    from tools.lamina_model import FILENAME
    engine = Engine(DATA / "models" / FILENAME, DATA / "tokenizer" / "tokenizer.json", ENGINE, cuda=True,
                    max_context=context, kv_cache="device", kv_type="f16", compute_mode="fast",
                    mtp=MTP_FILE if config.get("mtp") and MTP_FILE.is_file() else None)
    print(f"\nLamina chat, context {context // 1024}K, MTP {'on' if engine.mtp else 'off'}. "
          "Commands: /new starts a new conversation, /exit quits.")
    print("The first answer takes longer while the model loads.\n")
    history = []
    try:
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if text in ("/exit", "/quit"): break
            if text == "/new": history = []; print("(new conversation)"); continue
            if not text: continue
            history.append({"role": "user", "content": text})
            reply = ""
            print("Lamina: ", end="", flush=True)
            try:
                for event in engine.events(history, max_tokens, temperature=0, enable_thinking=thinking):
                    delta = event.get("delta", {})
                    if delta.get("reasoning_content"): print(delta["reasoning_content"], end="", file=sys.stderr, flush=True)
                    if delta.get("content"): print(delta["content"], end="", flush=True); reply += delta["content"]
                    if "timings" in event and event.get("usage"):
                        seconds = event["timings"]["total_seconds"] - event["timings"]["first_token_seconds"]
                        tokens = event["usage"]["completion_tokens"]
                        if tokens > 1 and seconds > 0:
                            print(f"\n[{tokens} tokens, {(tokens - 1) / seconds:.1f} tokens/s]", end="")
            except ValueError as error:  # context full and similar request errors
                print(f"\n[{error}; use /new to start over]")
                history.pop()
                continue
            print("\n")
            history.append({"role": "assistant", "content": reply})
    finally:
        engine.close()


def server_ready(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
            return True
    except OSError:
        return False


def run_server(config, context, host, port, preload=True):
    """Starts lamina.py serve, loads the model, prints how to connect, and waits."""
    from tools.lamina_chat import MODEL_NAME
    if server_ready(port):
        sys.exit(f"Something already answers on port {port}, probably a Lamina server that is still running.\n"
                 f"Close it first, or start this one on another port: START-HERE.bat --port {port + 1}")
    mtp = bool(config.get("mtp") and MTP_FILE.is_file())
    print(f"\nStarting the Lamina server (context {context // 1024}K, MTP {'on' if mtp else 'off'}) ...")
    command = [sys.executable, "lamina.py", "serve", *engine_options(config, context), "--host", host, "--port", str(port)]
    server = subprocess.Popen([str(c) for c in command], cwd=ROOT, env=engine_environment())
    try:
        for _ in range(240):
            if server_ready(port) or server.poll() is not None:
                break
            time.sleep(0.5)
        if server.poll() is not None:
            sys.exit("The server stopped right after starting; see the messages above.")
        if preload:
            print("Loading the model into memory. Your PC can be slow for a minute or two, and longest the first time.", flush=True)
            # The engine starts with the first request; send a tiny one now so that the first real request is fast.
            body = json.dumps({"model": MODEL_NAME, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}).encode()
            request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(request, timeout=900).read()
            except OSError as error:
                sys.exit(f"The model did not load: {error}\nSee the messages above.")
        shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        print(f"""
================================================================
 Lamina is ready.

   Base URL : http://{shown}:{port}/v1
   Model    : {MODEL_NAME}
   API key  : anything (it is not checked)

 In pi or another OpenAI-compatible app, add a provider with this
 base URL. Each request is logged below, with its speed.
 To stop the server press Ctrl+C (then Y if Windows asks), or
 close this window.
================================================================""", flush=True)
        if host not in ("127.0.0.1", "localhost", "::1"):
            print(f" WARNING: listening on {host} without any password; everyone who can reach this PC can use the model.", flush=True)
        return server.wait()
    finally:
        if server.poll() is None:
            subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"], capture_output=True)
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            print("Lamina stopped.")


def main():
    argv = sys.argv[1:]
    ensure_venv(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="serve", choices=("serve", "chat", "setup"))
    parser.add_argument("--reconfigure", action="store_true", help="ask for context length and MTP again")
    parser.add_argument("--max-context", type=int, help="override the stored context length for this run")
    parser.add_argument("--cuda-arch", help="override the detected compute capability, e.g. 86 or 89")
    parser.add_argument("--thinking", action="store_true", help="chat: let the model think before answering")
    parser.add_argument("--max-tokens", type=int, default=2048, help="chat: longest answer in tokens (default 2048)")
    parser.add_argument("--host", default="127.0.0.1", help="serve: listening address")
    parser.add_argument("--port", type=int, default=8000, help="serve: listening port")
    parser.add_argument("--no-preload", action="store_true", help="serve: do not load the model at startup (the first request does)")
    args = parser.parse_args(argv)

    config = ask(load_config(), args.reconfigure)
    needs_setup = (args.command == "setup" or not ENGINE.is_file() or args.cuda_arch
                   or (config.get("mtp") and not MTP_FILE.is_file()))
    if needs_setup:
        setup(config, args.cuda_arch)
    if args.command == "setup":
        return 0
    context = args.max_context or config["max_context"]
    if not 1024 <= context <= 131072:
        parser.error("--max-context must be 1024..131072")
    if args.command == "serve":
        return run_server(config, context, args.host, args.port, preload=not args.no_preload)
    chat(config, context, args.thinking, min(args.max_tokens, context // 2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        sys.exit(f"\nA setup step failed (exit code {error.returncode}). Fix the error above and run the same command again;"
                 " finished steps are skipped.")
    except KeyboardInterrupt:  # Ctrl+C is the normal way to stop the server or leave the chat
        sys.exit(0)
