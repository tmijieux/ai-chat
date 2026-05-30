"""
Reshape whisper-tiny OV IR to static shapes and compile on NPU.
Uses raw OpenVINO API to reshape each submodel, bypassing the optimum wrapper.

Whisper encoder:  input_features [batch, n_mels=80, time=3000]
Whisper decoder:  input_ids [batch, seq], encoder_hidden_states [batch, 1500, 384]

Run:  python npu_static.py
"""

import warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import time
import numpy as np
from pathlib import Path
from transformers import AutoProcessor
import openvino as ov

MODEL_ID    = "openai/whisper-tiny"
CACHE_DIR   = Path(__file__).parent / "model_cache"
OV_DIR      = Path(__file__).parent / "ov_model"
SAMPLE_RATE = 16_000

assert OV_DIR.exists(), "Run npu_static.py once with OV_DIR missing to export first"

core = ov.Core()

# ── Step 1: reshape encoder to [1, 80, 3000] ─────────────────────────────────
print("Step 1: Reshape encoder")
enc = core.read_model(OV_DIR / "openvino_encoder_model.xml")
enc.reshape({"input_features": [1, 80, 3000]})
for inp in enc.inputs:
    print(f"  {inp.any_name}: {inp.partial_shape}")
for out in enc.outputs:
    print(f"  → {out.any_name}: {out.partial_shape}")

# ── Step 2: reshape decoder ───────────────────────────────────────────────────
print("\nStep 2: Reshape decoder")
dec = core.read_model(OV_DIR / "openvino_decoder_model.xml")
# Decoder inputs: input_ids [1,?], encoder_hidden_states [1,1500,384], beam_idx [1]
# Pin everything to batch=1, decoder seq=1 (autoregressive: one token at a time)
shapes = {}
for inp in dec.inputs:
    name = inp.any_name
    ps   = inp.partial_shape
    if name == "input_ids":
        shapes[name] = [1, 1]          # batch=1, one token at a time
    elif name == "encoder_hidden_states":
        shapes[name] = [1, 1500, 384]  # fixed encoder output
    elif name == "beam_idx":
        shapes[name] = [1]
    else:
        # any other input: pin batch dim to 1 if dynamic
        concrete = [1 if str(d) == "?" else int(str(d)) for d in ps]
        shapes[name] = concrete
    print(f"  {name}: {ps}  →  {shapes[name]}")

dec.reshape(shapes)

# Audit shapes AFTER reshape
print("  Post-reshape decoder inputs:")
for inp in dec.inputs:
    print(f"    {inp.any_name}: {inp.partial_shape}")
print("  Post-reshape decoder outputs:")
for out in dec.outputs:
    print(f"    {out.any_name}: {out.partial_shape}")

# ── Step 3: hybrid compile — encoder on NPU, decoder on GPU.0 ────────────────
# The encoder is the heavy part (3000-frame convolutions). The decoder is tiny.
# NPU handles encoder; GPU.0 (Arc iGPU) handles the decoder loop.
print("\nStep 3: Hybrid compile (encoder→NPU, decoder→GPU.0)")
t0 = time.perf_counter()
enc_compiled = core.compile_model(enc, "NPU")
print(f"  encoder on NPU  ({time.perf_counter()-t0:.1f}s)")

# Decoder on GPU.0 (dynamic shapes OK there)
dec_dyn = core.read_model(OV_DIR / "openvino_decoder_model.xml")
dec_compiled = core.compile_model(dec_dyn, "GPU.0")
print(f"  decoder on GPU.0  ({time.perf_counter()-t0:.1f}s total)")

# ── Step 4: run a real inference via the processor + manual loop ──────────────
print("\nStep 4: End-to-end transcription on NPU (5s silent audio)")
processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

dummy = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
feat  = processor(dummy, sampling_rate=SAMPLE_RATE, return_tensors="np")
input_features = feat["input_features"].astype(np.float32)  # [1, 80, 3000]

# encoder forward
enc_out = enc_compiled({"input_features": input_features})
hidden  = list(enc_out.values())[0]  # [1, 1500, 384]

# simple greedy decoder loop
SOT   = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
EOT   = processor.tokenizer.eos_token_id
token = SOT
generated = []
MAX_NEW = 50

t0 = time.perf_counter()
for _ in range(MAX_NEW):
    dec_in = {
        "input_ids":             np.array([[token]], dtype=np.int64),
        "encoder_hidden_states": hidden,
        "beam_idx":              np.array([0], dtype=np.int32),
    }
    logits = list(dec_compiled(dec_in).values())[0]  # [1, 1, vocab]
    token  = int(np.argmax(logits[0, -1]))
    if token == EOT:
        break
    generated.append(token)

elapsed = time.perf_counter() - t0
text = processor.tokenizer.decode(generated, skip_special_tokens=True)
print(f"  text : '{text}'")
print(f"  time : {elapsed*1000:.0f}ms  ({len(generated)} tokens)")
