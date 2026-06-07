"""
Local Whisper transcription pipeline using OpenVINO.
Encoder and decoder run on GPU.0 (Arc iGPU).

Call load_pipeline() once at startup, pass the returned WhisperPipeline to transcribe().
Tensor names are discovered at load time so any Whisper variant works.

Stateful models (tiny, base, large-fr) use _decode_stateful: one token per step,
KV cache maintained internally by the model. Non-stateful (small) use _decode_full_sequence.
"""

import pickle
import logging
import subprocess
import time
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


WHISPER_TINY     = WhisperVariant("openai/whisper-tiny",                 WHISPER_DIR / "ov_model_tiny",     WHISPER_DIR / "compiled_blobs_tiny")
WHISPER_BASE     = WhisperVariant("openai/whisper-base",                 WHISPER_DIR / "ov_model_base",     WHISPER_DIR / "compiled_blobs_base")
WHISPER_SMALL    = WhisperVariant("openai/whisper-small",                WHISPER_DIR / "ov_model_small",    WHISPER_DIR / "compiled_blobs_small")
WHISPER_LARGE_FR = WhisperVariant("bofenghuang/whisper-large-v3-french", WHISPER_DIR / "ov_model_large_fr", WHISPER_DIR / "compiled_blobs_large_fr")

ACTIVE_VARIANT = WHISPER_SMALL

logger = logging.getLogger(__name__)


@dataclass
class _Schema:
    enc_in:     str        # encoder input tensor name
    enc_out:    str        # encoder output tensor name
    dec_ids:    str        # decoder token ids input name
    dec_hidden: str        # decoder cross-attn (encoder hidden states) input name
    dec_out:    str        # decoder logits output name
    stateful:   bool       # True = one-token-at-a-time with internal KV cache
    beam_idx:   str | None # beam_idx input name, present on stateful models
    n_mels:     int        # mel bins (80 for tiny/base/small, 128 for large-v3)


@dataclass
class WhisperPipeline:
    enc:       ov.CompiledModel
    dec:       ov.CompiledModel
    processor: AutoProcessor
    schema:    _Schema


def _port_names(node) -> list[str]:
    return [n for n in node.names if n]


def _introspect(core: ov.Core, enc_xml: Path, dec_xml: Path) -> _Schema:
    enc_m = core.read_model(enc_xml)
    dec_m = core.read_model(dec_xml)

    enc_in    = _port_names(enc_m.inputs[0])[0]
    enc_out   = _port_names(enc_m.outputs[0])[0]
    enc_shape = enc_m.inputs[0].partial_shape
    n_mels    = int(enc_shape[1].get_length()) if enc_shape[1].is_static else 80

    dec_ids = dec_hidden = beam_idx = ""
    for inp in dec_m.inputs:
        names = _port_names(inp)
        if not names:
            continue
        if "beam_idx" in names:
            beam_idx = names[0]
        elif inp.element_type in (ov.Type.i64, ov.Type.i32):
            dec_ids = names[0]
        elif inp.element_type in (ov.Type.f32, ov.Type.f16):
            dec_hidden = names[0]

    dec_out  = _port_names(dec_m.outputs[0])[0]
    stateful = any("ReadValue" in str(op) for op in dec_m.get_ops())

    schema = _Schema(enc_in=enc_in, enc_out=enc_out, dec_ids=dec_ids,
                     dec_hidden=dec_hidden, dec_out=dec_out,
                     stateful=stateful, beam_idx=beam_idx or None, n_mels=n_mels)
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


def load_pipeline(
    variant: WhisperVariant = ACTIVE_VARIANT,
    enc_device: str = "GPU.0",
    dec_device: str = "GPU.0",
) -> WhisperPipeline:
    variant.blob_dir.mkdir(parents=True, exist_ok=True)
    core = ov.Core()

    enc_xml = variant.ov_dir / "openvino_encoder_model.xml"
    dec_xml = variant.ov_dir / "openvino_decoder_model.xml"
    schema  = _introspect(core, enc_xml, dec_xml)

    enc_blob = variant.blob_dir / f"encoder_{enc_device.lower().replace('.', '')}.blob"
    dec_blob = variant.blob_dir / f"decoder_{dec_device.lower().replace('.', '')}.blob"

    logger.info("Whisper: loading encoder (%s)...", enc_device)
    enc = _load_compiled_model(core, enc_xml, enc_device, enc_blob,
                               reshape={schema.enc_in: [1, schema.n_mels, 3000]})
    logger.info("Whisper: loading decoder (%s)...", dec_device)
    dec = _load_compiled_model(core, dec_xml, dec_device, dec_blob)

    logger.info("Whisper: loading processor...")
    processor = _load_processor(variant)
    logger.info("Whisper: pipeline ready (stateful=%s)", schema.stateful)
    return WhisperPipeline(enc=enc, dec=dec, processor=processor, schema=schema)


def _decode_audio(audio_bytes: bytes) -> np.ndarray:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"],
        input=audio_bytes, capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio decode failed: {result.stderr.decode()}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def transcribe(pipeline: WhisperPipeline, audio_bytes: bytes, language: str | None = None) -> str:
    schema = pipeline.schema
    t0 = time.perf_counter()

    audio = _decode_audio(audio_bytes)
    t1 = time.perf_counter()
    logger.info("[whisper] ffmpeg decode: %.3fs  (audio %.2fs)", t1 - t0, len(audio) / SAMPLE_RATE)

    feat = pipeline.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
    t2 = time.perf_counter()
    logger.info("[whisper] feature extraction: %.3fs", t2 - t1)

    enc_req = pipeline.enc.create_infer_request()
    hidden  = enc_req.infer({schema.enc_in: feat["input_features"].astype(np.float32)})[schema.enc_out]
    t3 = time.perf_counter()
    logger.info("[whisper] encoder: %.3fs", t3 - t2)

    tok    = pipeline.processor.tokenizer
    prefix = [tok.convert_tokens_to_ids("<|startoftranscript|>")]
    if language:
        prefix.append(tok.convert_tokens_to_ids(f"<|{language}|>"))
    prefix += [tok.convert_tokens_to_ids("<|transcribe|>"), tok.convert_tokens_to_ids("<|notimestamps|>")]
    logger.info("[whisper] language token: %s", f"<|{language}|>" if language else "none")

    dec_req = pipeline.dec.create_infer_request()
    if schema.stateful:
        generated = _decode_stateful(dec_req, schema, hidden, prefix, tok.eos_token_id)
        mode = "stateful"
    else:
        generated = _decode_full_sequence(dec_req, schema, hidden, prefix, tok.eos_token_id)
        mode = "full_seq"
    t4 = time.perf_counter()
    logger.info("[whisper] decoder (%s): %.3fs  tokens=%d", mode, t4 - t3, len(generated))

    text = tok.decode(generated, skip_special_tokens=True).strip()
    logger.info("[whisper] total: %.3fs  → %r", t4 - t0, text)
    return text


def _decode_stateful(
    dec_req: ov.InferRequest,
    schema: _Schema,
    hidden: np.ndarray,
    prefix: list[int],
    eos_id: int,
) -> list[int]:
    def _step(token_id: int):
        inputs = {schema.dec_ids: np.array([[token_id]], dtype=np.int64), schema.dec_hidden: hidden}
        if schema.beam_idx:
            inputs[schema.beam_idx] = np.array([0], dtype=np.int32)
        return dec_req.infer(inputs)

    token = prefix[0]
    for forced in prefix[1:]:
        _step(token)
        token = forced

    generated: list[int] = []
    for i in range(200):
        out   = _step(token)
        token = int(np.argmax(out[schema.dec_out][0, -1]))
        if i == 0:
            logger.info("[whisper] first generated token id=%d  eos_id=%d", token, eos_id)
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
