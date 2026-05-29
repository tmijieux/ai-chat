"""
Download whisper-tiny as an OpenVINO model and benchmark inference
on every available device (CPU, GPU, NPU).

Run:  python benchmark.py
The model (~40MB) is cached in ./model_cache/ after the first run.
"""

import time
import numpy as np
from pathlib import Path

MODEL_ID = "openai/whisper-tiny"
CACHE_DIR = Path(__file__).parent / "model_cache"
SAMPLE_RATE = 16000
DURATION_S = 5  # seconds of silence/noise used for benchmarking

# ── build a dummy audio input ─────────────────────────────────────────────────
# Real silence: the model will output nothing useful, but latency is real.
dummy_audio = np.zeros(SAMPLE_RATE * DURATION_S, dtype=np.float32)

print(f"Benchmarking {MODEL_ID} with {DURATION_S}s dummy audio\n")

# ── detect available OpenVINO devices ─────────────────────────────────────────
import openvino as ov
core = ov.Core()
ov_devices = core.available_devices
print(f"OpenVINO devices: {ov_devices}\n")

# ── load processor once (device-agnostic) ────────────────────────────────────
from transformers import AutoProcessor
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
inputs = processor(dummy_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
print("Processor ready.\n")

# ── benchmark each device ────────────────────────────────────────────────────
from optimum.intel import OVModelForSpeechSeq2Seq

results = {}

for device in ov_devices:
    print(f"─── {device} ───────────────────────────────")
    try:
        t0 = time.perf_counter()
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            device=device,
            cache_dir=CACHE_DIR,
            export=True,          # convert on first run, cached after
        )
        load_time = time.perf_counter() - t0
        print(f"  load: {load_time:.1f}s")

        # warm-up
        _ = model.generate(**inputs)

        # timed run
        runs = 3
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model.generate(**inputs)
        infer_time = (time.perf_counter() - t0) / runs

        print(f"  inference (avg {runs} runs): {infer_time*1000:.0f}ms")
        results[device] = {"load_s": load_time, "infer_ms": infer_time * 1000}

    except Exception as e:
        print(f"  ✗ {e}")
        results[device] = {"error": str(e)}

# ── summary ───────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
for dev, r in results.items():
    if "error" in r:
        print(f"  {dev:8s}  ERROR: {r['error'][:80]}")
    else:
        print(f"  {dev:8s}  load={r['load_s']:.1f}s  infer={r['infer_ms']:.0f}ms")
