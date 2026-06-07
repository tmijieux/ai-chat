"""
Benchmark NPU vs GPU.0 for the Whisper encoder + full transcribe.
Run from backend/:  python benchmark_encoder.py [tiny|small|large_fr] [audio.webm]

Variant defaults to ACTIVE_VARIANT in whisper_pipeline.py.
Audio defaults to last_recording.webm.
"""
import sys
import time
import numpy as np
import openvino as ov
from pathlib import Path

import whisper_pipeline as wp

RUNS = 3
DEC_DEVICE = "GPU.0"

_VARIANTS = {"tiny": wp.WHISPER_TINY, "small": wp.WHISPER_SMALL, "large_fr": wp.WHISPER_LARGE_FR}
args = sys.argv[1:]
variant_arg = args.pop(0) if args and args[0] in _VARIANTS else None
VARIANT = _VARIANTS[variant_arg] if variant_arg else wp.ACTIVE_VARIANT
AUDIO_FILE = Path(args[0]) if args else Path("last_recording.webm")


def bench_encoder(enc: ov.CompiledModel, feat_np: np.ndarray, schema: wp._Schema, label: str) -> None:
    times = []
    for i in range(RUNS):
        enc_req = enc.create_infer_request()
        t0 = time.perf_counter()
        enc_req.infer({schema.enc_in: feat_np})
        times.append(time.perf_counter() - t0)
        print(f"  run {i+1}: {times[-1]:.3f}s")
    print(f"  → {label}: min={min(times):.3f}s  mean={sum(times)/len(times):.3f}s  max={max(times):.3f}s\n")


def main() -> None:
    variant = VARIANT
    print(f"Variant: {variant.model_id}\n")
    variant.blob_dir.mkdir(parents=True, exist_ok=True)
    core = ov.Core()

    enc_xml = variant.ov_dir / "openvino_encoder_model.xml"
    dec_xml = variant.ov_dir / "openvino_decoder_model.xml"
    schema    = wp._introspect(core, enc_xml, dec_xml)
    processor = wp._load_processor(variant)

    audio_bytes = AUDIO_FILE.read_bytes() if AUDIO_FILE.exists() else b""
    audio = wp._decode_audio(audio_bytes) if audio_bytes else np.zeros(wp.SAMPLE_RATE * 5, dtype=np.float32)
    print(f"Audio duration: {len(audio) / wp.SAMPLE_RATE:.2f}s  stateful={schema.stateful}\n")

    feat    = processor(audio, sampling_rate=wp.SAMPLE_RATE, return_tensors="np")
    feat_np = feat["input_features"].astype(np.float32)

    dec_blob = variant.blob_dir / f"decoder_{DEC_DEVICE.lower().replace('.', '')}.blob"
    print(f"Loading decoder ({DEC_DEVICE})...")
    dec = wp._load_compiled_model(core, dec_xml, DEC_DEVICE, dec_blob)
    print("  done\n")

    encoders: dict[str, ov.CompiledModel] = {}
    for device in ["NPU", "GPU.0"]:
        blob = variant.blob_dir / f"encoder_{device.lower().replace('.', '')}.blob"
        print(f"Loading encoder on {device}...")
        t0 = time.perf_counter()
        encoders[device] = wp._load_compiled_model(
            core, enc_xml, device, blob,
            reshape={schema.enc_in: [1, schema.n_mels, 3000]},
        )
        print(f"  loaded in {time.perf_counter() - t0:.2f}s\n")

    print(f"--- Encoder only ({RUNS} runs each) ---\n")
    for device, enc in encoders.items():
        print(f"  {device}:")
        bench_encoder(enc, feat_np, schema, device)

    print(f"--- Full transcribe (encoder + decoder, {RUNS} runs each) ---\n")
    for device, enc in encoders.items():
        times = []
        for i in range(RUNS):
            pipeline = wp.WhisperPipeline(enc=enc, dec=dec, processor=processor, schema=schema)
            t0 = time.perf_counter()
            text = wp.transcribe(pipeline, audio_bytes or bytes(44), "fr")
            times.append(time.perf_counter() - t0)
            print(f"  enc={device} run {i+1}: {times[-1]:.3f}s  → {text!r}")
        print(f"  → enc={device}: min={min(times):.3f}s  mean={sum(times)/len(times):.3f}s\n")


if __name__ == "__main__":
    main()
