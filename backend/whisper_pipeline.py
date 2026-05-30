"""
Local Whisper transcription pipeline using OpenVINO.
Encoder runs on NPU, decoder runs on GPU.0 (Arc iGPU).

Call load_pipeline() once at startup, then pass the result to transcribe().
Tensor names are discovered at load time so any whisper variant works.
"""

import pickle
import logging
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import openvino as ov
from transformers import AutoProcessor

WHISPER_DIR = Path(__file__).parent.parent / "whisper"
SAMPLE_RATE = 16_000


@dataclass
class WhisperVariant:
    model_id: str
    ov_dir:   Path
    blob_dir: Path


WHISPER_TINY         = WhisperVariant("openai/whisper-tiny",                   WHISPER_DIR / "ov_model_tiny",         WHISPER_DIR / "compiled_blobs_tiny")
WHISPER_BASE         = WhisperVariant("openai/whisper-base",                   WHISPER_DIR / "ov_model_base",         WHISPER_DIR / "compiled_blobs_base")
WHISPER_SMALL        = WhisperVariant("openai/whisper-small",                  WHISPER_DIR / "ov_model_small",        WHISPER_DIR / "compiled_blobs_small")
WHISPER_LARGE_FR     = WhisperVariant("bofenghuang/whisper-large-v3-french",   WHISPER_DIR / "ov_model_large_fr",     WHISPER_DIR / "compiled_blobs_large_fr")

ACTIVE_VARIANT = WHISPER_LARGE_FR

logger = logging.getLogger(__name__)


@dataclass
class _Schema:
    enc_in:     str        # encoder input tensor name  (e.g. "input_features")
    enc_out:    str        # encoder output tensor name (e.g. "last_hidden_state")
    dec_ids:    str        # decoder ids input name     (e.g. "input_ids")
    dec_hidden: str        # decoder cross-attn input   (e.g. "encoder_hidden_states")
    dec_out:    str        # decoder output tensor name (e.g. "logits")
    stateful:   bool       # True = one-token-at-a-time with internal KV cache (tiny, base)
    beam_idx:   str | None # name of beam_idx input, present on stateful models
    n_mels:     int        # mel bin count for encoder reshape (80 for tiny/base/small, 128 for large-v3)


def _port_names(node) -> list[str]:
    """Return all non-empty tensor names for an OV input/output node."""
    return [n for n in node.names if n]


def _introspect(core: ov.Core, enc_xml: Path, dec_xml: Path) -> _Schema:
    """Read uncompiled models and 
    discover the tensor names we need."""
    enc_m = core.read_model(enc_xml)
    dec_m = core.read_model(dec_xml)

    # Encoder: one float input, one float output
    enc_in    = _port_names(enc_m.inputs[0])[0]
    enc_out   = _port_names(enc_m.outputs[0])[0]
    # Read mel bin count from the model's input shape (dim 1); large-v3 uses 128, others use 80.
    enc_shape = enc_m.inputs[0].partial_shape
    n_mels    = int(enc_shape[1].get_length()) if enc_shape[1].is_static else 80

    # Decoder: find int input (token ids, not beam_idx) and float input (hidden states)
    dec_ids = dec_hidden = beam_idx = ""
    for inp in dec_m.inputs:
        names = _port_names(inp)
        if not names:
            continue
        if "beam_idx" in names:
            beam_idx = names[0]
        elif inp.element_type == ov.Type.i64 or inp.element_type == ov.Type.i32:
            dec_ids = names[0]
        elif inp.element_type == ov.Type.f32 or inp.element_type == ov.Type.f16:
            dec_hidden = names[0]

    dec_out  = _port_names(dec_m.outputs[0])[0]
    stateful = any("ReadValue" in str(op) for op in dec_m.get_ops())

    schema = _Schema(enc_in=enc_in, enc_out=enc_out, dec_ids=dec_ids,
                     dec_hidden=dec_hidden, dec_out=dec_out,
                     stateful=stateful, beam_idx=beam_idx or None,
                     n_mels=n_mels)
    logger.info("Whisper schema: %s", schema)
    return schema


def _load_compiled_model(
    core: ov.Core,
    xml_path: Path,
    device: str,
    blob_path: Path,
    reshape: dict[str, list[int]] | None = None,
) -> ov.CompiledModel:
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


def _load_processor(variant: WhisperVariant) -> AutoProcessor:
    pkl = variant.blob_dir / "processor.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    processor = AutoProcessor.from_pretrained(variant.model_id, cache_dir=WHISPER_DIR / "model_cache")
    with open(pkl, "wb") as f:
        pickle.dump(processor, f)
    return processor


def load_pipeline(variant: WhisperVariant = ACTIVE_VARIANT) -> tuple[ov.CompiledModel, ov.CompiledModel, AutoProcessor, _Schema]:
    """Load and return (encoder, decoder, processor, schema). Call once at startup."""
    variant.blob_dir.mkdir(parents=True, exist_ok=True)
    core = ov.Core()

    enc_xml = variant.ov_dir / "openvino_encoder_model.xml"
    dec_xml = variant.ov_dir / "openvino_decoder_model.xml"
    schema  = _introspect(core, enc_xml, dec_xml)

    logger.info("Whisper: loading encoder (NPU)...")
    enc = _load_compiled_model(core, enc_xml, "NPU", variant.blob_dir / "encoder_npu.blob",
                               reshape={schema.enc_in: [1, schema.n_mels, 3000]})
    logger.info("Whisper: loading decoder (GPU.0)...")
    dec = _load_compiled_model(core, dec_xml, "GPU.0", variant.blob_dir / "decoder_gpu0.blob")

    logger.info("Whisper: loading processor...")
    processor = _load_processor(variant)
    logger.info("Whisper: pipeline ready.")
    return enc, dec, processor, schema


def _decode_audio(audio_bytes: bytes) -> np.ndarray:
    """Decode audio bytes (WebM, WAV, OGG, etc.) to float32 mono at SAMPLE_RATE via ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"],
        input=audio_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio decode failed: {result.stderr.decode()}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def transcribe(
    enc: ov.CompiledModel,
    dec: ov.CompiledModel,
    processor: AutoProcessor,
    audio_bytes: bytes,
    language: str | None = "fr",
    schema: _Schema | None = None,
) -> str:
    """Transcribe raw audio bytes and return the transcript string."""
    if schema is None:
        raise ValueError("schema is required — pass the value returned by load_pipeline()")

    audio  = _decode_audio(audio_bytes)
    feat   = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
    hidden = enc({schema.enc_in: feat["input_features"].astype(np.float32)})[schema.enc_out]

    tok    = processor.tokenizer
    prefix = [tok.convert_tokens_to_ids("<|startoftranscript|>")]
    if language:
        prefix.append(tok.convert_tokens_to_ids(f"<|{language}|>"))
    prefix += [
        tok.convert_tokens_to_ids("<|transcribe|>"),
        tok.convert_tokens_to_ids("<|notimestamps|>"),
    ]

    dec_req = dec.create_infer_request()
    if schema.stateful:
        generated = _decode_stateful(dec_req, schema, hidden, prefix, tok.eos_token_id)
    else:
        generated = _decode_full_sequence(dec_req, schema, hidden, prefix, tok.eos_token_id)
    return tok.decode(generated, skip_special_tokens=True).strip()


def _decode_stateful(
    dec_req: ov.InferRequest,
    schema: _Schema,
    hidden: np.ndarray,
    prefix: list[int],
    eos_id: int,
) -> list[int]:
    """One-token-at-a-time decode; KV cache accumulates in model state (tiny, base)."""
    def _step(token_id: int):
        inputs = {schema.dec_ids: np.array([[token_id]], dtype=np.int64),
                  schema.dec_hidden: hidden}
        if schema.beam_idx:
            inputs[schema.beam_idx] = np.array([0], dtype=np.int32)
        return dec_req.infer(inputs)

    token = prefix[0]
    for forced in prefix[1:]:
        _step(token)
        token = forced

    generated: list[int] = []
    for _ in range(200):
        out   = _step(token)
        token = int(np.argmax(out[schema.dec_out][0, -1]))
        if token == eos_id:
            break
        generated.append(token)
    return generated


def _decode_full_sequence(
    dec_req: ov.InferRequest,
    schema: _Schema,
    hidden: np.ndarray,
    prefix: list[int],
    eos_id: int,
) -> list[int]:
    """Full-sequence decode; no internal state (small)."""
    all_tokens = list(prefix)
    generated:  list[int] = []
    for _ in range(200):
        out        = dec_req.infer({schema.dec_ids: np.array([all_tokens], dtype=np.int64),
                                    schema.dec_hidden: hidden})
        next_token = int(np.argmax(out[schema.dec_out][0, -1]))
        if next_token == eos_id:
            break
        all_tokens.append(next_token)
        generated.append(next_token)
    return generated
