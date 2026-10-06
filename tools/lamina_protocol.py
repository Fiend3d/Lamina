"""Resident native process transport, with drained stderr and bounded waits."""
import collections
import queue
import subprocess
import threading
import time


class NativeProcess:
    def __init__(self, command, timeout=1800):
        self.timeout = timeout
        self.cancelled = None
        self.errors = collections.deque(maxlen=64)
        self.lines = queue.Queue()
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.readers = []
        for source, output in ((self.process.stdout, self.lines.put), (self.process.stderr, self.errors.append)):
            def drain(source=source, output=output):
                try:
                    for line in source:
                        output(line.rstrip("\r\n"))
                finally:
                    if source is self.process.stdout:
                        self.lines.put(None)
            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            self.readers.append(reader)

    def read(self):
        deadline = time.monotonic() + self.timeout
        while True:
            if self.cancelled is not None and self.cancelled():
                self.close()
                raise ConnectionResetError("client disconnected during inference")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise RuntimeError("native engine timed out")
            try:
                result = self.lines.get(timeout=min(remaining, 0.2) if self.cancelled else remaining)
                break
            except queue.Empty:
                continue
        if result is None:
            for reader in self.readers:
                reader.join(timeout=1)
            raise RuntimeError("native engine exited: " + "\n".join(self.errors))
        return result

    def command(self, line, acknowledgement=False):
        if "\n" in line or "\r" in line:
            raise ValueError("native command contains a newline")
        try:
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError("native engine pipe closed") from error
        result = self.read()
        if acknowledgement:
            if result != ".":
                raise RuntimeError(f"unexpected native acknowledgement: {result}")
            return None
        try:
            token = int(result)
        except ValueError as error:
            raise RuntimeError(f"unexpected native token: {result}") from error
        if not 0 <= token < 248320:
            raise RuntimeError("native token outside vocabulary")
        return token

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        for source in (self.process.stdin, self.process.stdout, self.process.stderr):
            if source:
                source.close()
        for reader in self.readers:
            reader.join(timeout=1)
