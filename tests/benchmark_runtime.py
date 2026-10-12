"""Measure bundled inference without writing output files or downloading models."""

import argparse
import asyncio
import ctypes
import json
from pathlib import Path
import sys
import time

from starlette.concurrency import run_in_threadpool

from ocr_service.application import create_app
from ocr_service.core.settings import Settings


def peak_memory_mb():
    if sys.platform != "win32":
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage / (1024 * 1024 if sys.platform == "darwin" else 1024)
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile"
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.peak / (1024 * 1024)


async def benchmark(project_root, input_path):
    app = create_app(Settings(project_root))
    start = time.perf_counter()
    async with app.router.lifespan_context(app):
        report = {
            "platform": sys.platform,
            "input": str(input_path),
            "startup_seconds": round(time.perf_counter() - start, 3),
            "startup_peak_mb": round(peak_memory_mb(), 1),
            "modes": {},
        }
        for mode in ("none", "invoice", "document"):
            start = time.perf_counter()
            with input_path.open("rb") as stream:
                result = await run_in_threadpool(app.state.ocr_service.process, stream, input_path.name, "", mode)
            report["modes"][mode] = {
                "seconds": round(time.perf_counter() - start, 3),
                "cumulative_peak_mb": round(peak_memory_mb(), 1),
                "text_length": len(result["text"]),
            }
        report["torch_loaded"] = "torch" in sys.modules
    return report


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "img" / "hoadon2.png")
    parser.add_argument("--project-root", type=Path, default=root)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(benchmark(args.project_root, args.input)), indent=2))


if __name__ == "__main__":
    main()
