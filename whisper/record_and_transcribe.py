"""
Record N seconds from the default microphone, then transcribe with
the hybrid NPU+GPU model.

Run:  python record_and_transcribe.py [seconds=5]
"""

import time as _t
_t0 = _t.perf_counter()

import warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import sys, time, pickle
import numpy as np
from pathlib import Path
from transformers import AutoProcessor
import sounddevice as sd
import openvino as ov

MODEL_ID    = "openai/whisper-base"
SAMPLE_RATE = 16_000
DURATION    = int(sys.argv[1]) if len(sys.argv) > 1 else 5
LANGUAGE    = "fr"

ROOT        = Path(__file__).parent
CACHE_DIR   = ROOT / "model_cache"
OV_DIR      = ROOT / "ov_model_base"
BLOB_DIR    = ROOT / "compiled_blobs"


# ── setup helpers ─────────────────────────────────────────────────────────────

def export_ov_model() -> None:
    """Convert HuggingFace whisper-base to OpenVINO IR. Runs once."""
    from optimum.intel import OVModelForSpeechSeq2Seq
    print(f"Exporting {MODEL_ID} to OpenVINO IR...")
    m = OVModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, device="CPU", cache_dir=CACHE_DIR, export=True)
    m.save_pretrained(OV_DIR)


def load_compiled_model(
    core: ov.Core,
    xml_path: Path,
    device: str,
    blob_path: Path,
    reshape: dict[str, list[int]] | None = None,
) -> ov.CompiledModel:
    """Load a compiled model from blob cache, compiling and saving it on first run."""
    if blob_path.exists():
        with open(blob_path, "rb") as f:
            return core.import_model(f.read(), device)
    model = core.read_model(xml_path)
    if reshape:
        model.reshape(reshape)
    compiled = core.compile_model(model, device)
    with open(blob_path, "wb") as f:
        f.write(compiled.export_model().read())
    return compiled


def load_processor() -> AutoProcessor:
    """Load processor from pickle cache, saving it on first run."""
    pkl = BLOB_DIR / "processor.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    with open(pkl, "wb") as f:
        pickle.dump(processor, f)
    return processor


# ── inference ─────────────────────────────────────────────────────────────────

def transcribe(
    enc_compiled: ov.CompiledModel,
    dec_compiled: ov.CompiledModel,
    processor: AutoProcessor,
    audio: np.ndarray,
) -> str:
    feat   = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
    hidden = list(enc_compiled({"input_features": feat["input_features"].astype(np.float32)}).values())[0]

    tok = processor.tokenizer
    prefix = [tok.convert_tokens_to_ids("<|startoftranscript|>")]
    if LANGUAGE:
        prefix.append(tok.convert_tokens_to_ids(f"<|{LANGUAGE}|>"))
    prefix += [tok.convert_tokens_to_ids("<|transcribe|>"), tok.convert_tokens_to_ids("<|notimestamps|>")]

    token = prefix[0]
    for forced in prefix[1:]:
        dec_compiled({"input_ids": np.array([[token]], dtype=np.int64),
                      "encoder_hidden_states": hidden,
                      "beam_idx": np.array([0], dtype=np.int32)})
        token = forced

    generated: list[int] = []
    for _ in range(200):
        out   = dec_compiled({"input_ids": np.array([[token]], dtype=np.int64),
                              "encoder_hidden_states": hidden,
                              "beam_idx": np.array([0], dtype=np.int32)})
        token = int(np.argmax(list(out.values())[0][0, -1]))
        if token == tok.eos_token_id:
            break
        generated.append(token)

    return tok.decode(generated, skip_special_tokens=True).strip()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not OV_DIR.exists():
        export_ov_model()
    BLOB_DIR.mkdir(exist_ok=True)

    core = ov.Core()
    print(f"  imports:   {_t.perf_counter()-_t0:.2f}s")
    print("Loading model (encoder→NPU, decoder→GPU.0)...")
    t0 = time.perf_counter()

    enc = load_compiled_model(core, OV_DIR / "openvino_encoder_model.xml", "NPU",
                              BLOB_DIR / "encoder_npu.blob",
                              reshape={"input_features": [1, 80, 3000]})
    print(f"  encoder:   {time.perf_counter()-t0:.2f}s")

    t1 = time.perf_counter()
    dec = load_compiled_model(core, OV_DIR / "openvino_decoder_model.xml", "GPU.0",
                              BLOB_DIR / "decoder_gpu0.blob")
    print(f"  decoder:   {time.perf_counter()-t1:.2f}s")

    t1 = time.perf_counter()
    processor = load_processor()
    print(f"  processor: {time.perf_counter()-t1:.2f}s")

    print(f"  total:     {time.perf_counter()-t0:.2f}s")

    print(f"\nRecording {DURATION}s — speak now...")
    audio = sd.rec(DURATION * SAMPLE_RATE, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("Recording done.")

    t0 = time.perf_counter()
    text = transcribe(enc, dec, processor, audio.flatten())
    print(f"\nTranscript : '{text}'")
    print(f"Latency    : {(time.perf_counter()-t0)*1000:.0f}ms")


main()
