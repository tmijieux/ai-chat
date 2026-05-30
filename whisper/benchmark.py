"""
Benchmark whisper-tiny on every OpenVINO device.
Each device runs in a subprocess so an NPU crash doesn't kill the whole run.

Run:  python benchmark.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import openvino as ov
import numpy as np

MODEL_ID = "openai/whisper-tiny"
CACHE_DIR = Path(__file__).parent / "model_cache"
SAMPLE_RATE = 16_000
DURATION_S = 5

WORKER = Path(__file__).parent / "_bench_worker.py"


def run_on_device(device: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(WORKER), device, MODEL_ID, str(CACHE_DIR), str(SAMPLE_RATE), str(DURATION_S)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        return json.loads(result.stdout.strip().split("\n")[-1])
    except Exception:
        return {"error": (result.stderr or result.stdout or "no output")[-300:]}


if __name__ == "__main__":
    core = ov.Core()
    devices = core.available_devices
    print(f"Devices: {devices}\n")

    results = {}
    for d in devices:
        print(f"  Testing {d}...", end=" ", flush=True)
        t0 = time.perf_counter()
        r = run_on_device(d)
        elapsed = time.perf_counter() - t0
        if "error" in r:
            print(f"FAILED ({elapsed:.0f}s)")
            print(f"    {r['error'][:120]}")
        else:
            print(f"load={r['load_s']:.1f}s  infer={r['infer_ms']:.0f}ms  ({elapsed:.0f}s total)")
        results[d] = r

    print("\n=== Summary ===")
    for dev, r in results.items():
        if "error" in r:
            first_line = r["error"].splitlines()[-1][:80]
            print(f"  {dev:8s}  ERROR: {first_line}")
        else:
            print(f"  {dev:8s}  load={r['load_s']:.1f}s  infer={r['infer_ms']:.0f}ms")
