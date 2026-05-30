"""
Worker process: load whisper on one device, time it, print JSON result.
Called by benchmark.py per-device.
"""

import json
import sys
import time
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

device, model_id, cache_dir, sample_rate, duration_s = sys.argv[1:]
sample_rate = int(sample_rate)
duration_s = int(duration_s)

import numpy as np
from transformers import AutoProcessor
from optimum.intel import OVModelForSpeechSeq2Seq

dummy_audio = np.zeros(sample_rate * duration_s, dtype=np.float32)
processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
inputs = processor(dummy_audio, sampling_rate=sample_rate, return_tensors="pt")

t0 = time.perf_counter()
model = OVModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    device=device,
    cache_dir=cache_dir,
    export=True,
)
load_time = time.perf_counter() - t0

# warm-up
model.generate(**inputs)

# timed runs
runs = 3
t0 = time.perf_counter()
for _ in range(runs):
    model.generate(**inputs)
infer_ms = (time.perf_counter() - t0) / runs * 1000

print(json.dumps({"device": device, "load_s": round(load_time, 2), "infer_ms": round(infer_ms, 1)}))
