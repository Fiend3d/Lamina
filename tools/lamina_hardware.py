"""Describe the measuring machine for benchmark reports (GPU / CPU / RAM)."""
import ctypes
import os
import platform
import subprocess
import sys


def gpu():
    try:
        reply = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=30)
        name, mib = reply.stdout.strip().splitlines()[0].split(", ")
        return f"{name} {int(mib) / 1024:.0f} GB"
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return "unknown GPU"


def cpu():
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as info:
            for line in info:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def ram_gib():
    if sys.platform == "win32":
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                        ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                        ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
        status = Status(length=ctypes.sizeof(Status))
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.total / 2**30
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return None


def describe():
    """For example "NVIDIA GeForce RTX 3050 8 GB / AMD Ryzen 7 5700X 8-Core Processor / 64 GB RAM"."""
    ram = ram_gib()
    return f"{gpu()} / {cpu()} / " + (f"{ram:.0f} GB RAM" if ram else "unknown RAM")


if __name__ == "__main__":
    print(describe())
