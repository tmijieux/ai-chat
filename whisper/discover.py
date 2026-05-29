"""
Discovery script: probe what's available for local Whisper inference.
Run with: python discover.py
"""

import sys

print("=== Python ===")
print(f"  {sys.version}")

# ── OpenVINO ──────────────────────────────────────────────────────────────────
print("\n=== OpenVINO ===")
try:
    import openvino as ov
    print(f"  version : {ov.__version__}")
    core = ov.Core()
    devices = core.available_devices
    print(f"  devices : {devices}")
    for d in devices:
        try:
            name = core.get_property(d, "FULL_DEVICE_NAME")
            print(f"    {d}: {name}")
        except Exception:
            print(f"    {d}: (no name)")
    if "NPU" in devices:
        print("  ✓ NPU is available!")
    else:
        print("  ✗ NPU not found in OpenVINO devices")
except ImportError:
    print("  ✗ openvino not installed  →  pip install openvino")

# ── optimum-intel ─────────────────────────────────────────────────────────────
print("\n=== optimum-intel (Hugging Face) ===")
try:
    from optimum.intel import OVModelForSpeechSeq2Seq
    from importlib.metadata import version
    print(f"  optimum-intel version: {version('optimum-intel')}")
    print("  ✓ OVModelForSpeechSeq2Seq available")
except ImportError as e:
    print(f"  ✗ {e}  →  pip install optimum[openvino]")

# ── transformers ──────────────────────────────────────────────────────────────
print("\n=== transformers ===")
try:
    import transformers
    print(f"  version: {transformers.__version__}")
except ImportError:
    print("  ✗ not installed  →  pip install transformers")

# ── audio ─────────────────────────────────────────────────────────────────────
print("\n=== Audio (sounddevice) ===")
try:
    import sounddevice as sd
    devices = sd.query_devices()
    inputs = [d for d in devices if d["max_input_channels"] > 0]
    print(f"  ✓ sounddevice available — {len(inputs)} input device(s):")
    for d in inputs:
        print(f"    [{d['index']}] {d['name']}")
except ImportError:
    print("  ✗ sounddevice not installed  →  pip install sounddevice")

print("\n=== Done ===")
