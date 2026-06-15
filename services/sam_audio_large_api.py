from __future__ import annotations

import gc
import os
import re
import shutil
import threading
import types
import uuid
import warnings
import zipfile
from pathlib import Path
from typing import Any

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydub import AudioSegment
from sam_audio import SAMAudio, SAMAudioProcessor

from services.common import (
    OUTPUT_DIR,
    exception_detail,
    parse_bool,
    public_file_ref,
    resolve_local_path,
    resolve_output_dir,
)


warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*", category=FutureWarning)

MODEL_ID = os.environ.get("SAM_AUDIO_MODEL_ID", "facebook/sam-audio-large")
DEFAULT_MAX_AUDIO_SECONDS = float(os.environ.get("SAM_AUDIO_MAX_AUDIO_SECONDS", "35"))
DEFAULT_PREDICT_SPANS = parse_bool(os.environ.get("SAM_AUDIO_PREDICT_SPANS"), default=False)
DEFAULT_RERANKING_CANDIDATES = int(os.environ.get("SAM_AUDIO_RERANKING_CANDIDATES", "1"))
MAX_BATCH = max(1, int(os.environ.get("SAM_AUDIO_MAX_BATCH", "4")))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="SAM-Audio Large API",
    version="1.0.0",
    summary="Separate described sounds from local audio files with facebook/sam-audio-large.",
)
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")


class SeparateRequest(BaseModel):
    audio_path: str | None = Field(default=None, description="Local path or file:// URL to an audio file.")
    file_path: str | None = Field(default=None, description="Alias for audio_path.")
    file: str | None = Field(default=None, description="Alias for audio_path.")
    audio_url: str | None = Field(default=None, description="Alias for audio_path; only file:// URLs are supported.")
    prompt: str | None = Field(default=None, description="Target sound description.")
    input: str | None = Field(default=None, description="Alias for prompt.")
    description: str | None = Field(default=None, description="Alias for prompt.")
    anchors: Any | None = Field(default=None, description="Optional SAM-Audio anchors.")
    predict_spans: bool = DEFAULT_PREDICT_SPANS
    reranking_candidates: int = Field(default=DEFAULT_RERANKING_CANDIDATES, ge=1, le=16)
    max_audio_seconds: float = Field(default=DEFAULT_MAX_AUDIO_SECONDS, gt=0)
    output_dir: str | None = Field(default=None, description="Optional local output directory.")
    output_prefix: str | None = Field(default=None, description="Optional output filename prefix.")

    def audio_ref(self) -> str:
        value = self.audio_path or self.file_path or self.file or self.audio_url
        if not value:
            raise ValueError("Provide audio_path, file_path, file, or audio_url.")
        return value

    def description_text(self) -> str:
        value = self.prompt or self.input or self.description
        if not value or not value.strip():
            raise ValueError("Provide prompt, input, or description.")
        return value.strip()


class SeparateResponse(BaseModel):
    model_id: str
    request_id: str
    audio_path: str
    description: str
    duration_seconds: float
    sample_rate: int
    target: dict[str, dict[str, str]]
    residual: dict[str, dict[str, str]]
    zip: dict[str, str]
    peak_cuda_allocated_gb: float | None = None
    peak_cuda_reserved_gb: float | None = None


class SeparateBatchItem(BaseModel):
    audio_path: str | None = Field(default=None, description="Local path or file:// URL to an audio file.")
    file_path: str | None = Field(default=None, description="Alias for audio_path.")
    file: str | None = Field(default=None, description="Alias for audio_path.")
    audio_url: str | None = Field(default=None, description="Alias for audio_path; only file:// URLs are supported.")
    prompt: str | None = Field(default=None, description="Target sound description.")
    input: str | None = Field(default=None, description="Alias for prompt.")
    description: str | None = Field(default=None, description="Alias for prompt.")
    anchors: Any | None = Field(default=None, description="Optional SAM-Audio anchors.")
    max_audio_seconds: float = Field(default=DEFAULT_MAX_AUDIO_SECONDS, gt=0)
    output_prefix: str | None = Field(default=None, description="Optional output filename prefix.")

    def audio_ref(self) -> str:
        value = self.audio_path or self.file_path or self.file or self.audio_url
        if not value:
            raise ValueError("Provide audio_path, file_path, file, or audio_url.")
        return value

    def description_text(self) -> str:
        value = self.prompt or self.input or self.description
        if not value or not value.strip():
            raise ValueError("Provide prompt, input, or description.")
        return value.strip()


class SeparateBatchRequest(BaseModel):
    items: list[SeparateBatchItem] = Field(min_length=1)
    predict_spans: bool = DEFAULT_PREDICT_SPANS
    reranking_candidates: int = Field(default=DEFAULT_RERANKING_CANDIDATES, ge=1, le=16)
    output_dir: str | None = Field(default=None, description="Optional local output directory.")


class SeparateBatchEntry(BaseModel):
    error: str | None = None
    result: SeparateResponse | None = None


class SeparateBatchResponse(BaseModel):
    model_id: str
    results: list[SeparateBatchEntry]


class _ModelState:
    model: Any | None = None
    processor: Any | None = None
    device: str | None = None
    dtype: torch.dtype | None = None


_state = _ModelState()
_load_lock = threading.Lock()
_inference_lock = threading.Lock()


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for MP3 conversion.")


def disable_vision_encoder_for_audio_only(model: Any) -> Any:
    if not hasattr(model, "vision_encoder"):
        raise AttributeError("Expected SAM-Audio model to have a vision_encoder.")

    vision_dim = getattr(model.vision_encoder, "dim", None)
    if vision_dim is None:
        raise AttributeError("Could not read model.vision_encoder.dim.")

    del model.vision_encoder
    model._vision_encoder_dim = vision_dim

    def _get_video_features_audio_only(self: Any, video: Any, audio_features: torch.Tensor) -> torch.Tensor:
        if video is not None:
            raise ValueError("This service is audio-only; video inputs are disabled.")
        batch_size, time_steps, _ = audio_features.shape
        return audio_features.new_zeros(batch_size, self._vision_encoder_dim, time_steps)

    model._get_video_features = types.MethodType(_get_video_features_audio_only, model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def load_model() -> _ModelState:
    if _state.model is not None and _state.processor is not None:
        return _state

    with _load_lock:
        if _state.model is not None and _state.processor is not None:
            return _state
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SAM-Audio Large BF16 inference.")

        ensure_ffmpeg()
        device = "cuda"
        dtype = torch.bfloat16
        model = SAMAudio.from_pretrained(MODEL_ID, proxies=None, resume_download=False)
        model = disable_vision_encoder_for_audio_only(model)
        model = model.to(device, dtype).eval()
        processor = SAMAudioProcessor.from_pretrained(MODEL_ID)

        _state.model = model
        _state.processor = processor
        _state.device = device
        _state.dtype = dtype
        return _state


def tensor_from_result(value: Any) -> torch.Tensor:
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def result_component(value: Any, index: int) -> torch.Tensor:
    if isinstance(value, (list, tuple)):
        return value[index]
    if isinstance(value, torch.Tensor) and value.ndim >= 2:
        return value[index]
    if index != 0:
        raise IndexError(f"Cannot index separation result of type {type(value)!r} at {index}.")
    return value


def save_waveform(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    waveform = waveform.detach().to(torch.float32).cpu()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(str(path), waveform, sample_rate)


def wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    AudioSegment.from_file(str(wav_path)).export(str(mp3_path), format="mp3", bitrate="192k")


def zip_outputs(paths: list[Path], zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in paths:
            zip_file.write(path, arcname=path.name)
    return zip_path


def safe_prefix(value: str | None) -> str:
    if not value:
        return "sam_audio"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return cleaned[:80] or "sam_audio"


def build_separation_response(
    *,
    audio_path: Path,
    description: str,
    duration_seconds: float,
    sample_rate: int,
    target: torch.Tensor,
    residual: torch.Tensor,
    output_dir: Path,
    output_prefix: str,
    peak_allocated: float | None,
    peak_reserved: float | None,
) -> SeparateResponse:
    request_id = uuid.uuid4().hex
    job_dir = output_dir / request_id
    job_dir.mkdir(parents=True, exist_ok=False)

    target_wav = job_dir / f"{output_prefix}_target.wav"
    residual_wav = job_dir / f"{output_prefix}_residual.wav"
    target_mp3 = job_dir / f"{output_prefix}_target.mp3"
    residual_mp3 = job_dir / f"{output_prefix}_residual.mp3"
    zip_path = job_dir / f"{output_prefix}_outputs.zip"

    save_waveform(target_wav, target, sample_rate)
    save_waveform(residual_wav, residual, sample_rate)
    wav_to_mp3(target_wav, target_mp3)
    wav_to_mp3(residual_wav, residual_mp3)
    zip_outputs([target_wav, residual_wav, target_mp3, residual_mp3], zip_path)

    return SeparateResponse(
        model_id=MODEL_ID,
        request_id=request_id,
        audio_path=str(audio_path),
        description=description,
        duration_seconds=float(duration_seconds),
        sample_rate=sample_rate,
        target={"wav": public_file_ref(target_wav), "mp3": public_file_ref(target_mp3)},
        residual={"wav": public_file_ref(residual_wav), "mp3": public_file_ref(residual_mp3)},
        zip=public_file_ref(zip_path),
        peak_cuda_allocated_gb=peak_allocated,
        peak_cuda_reserved_gb=peak_reserved,
    )


@torch.inference_mode()
def run_separation(
    *,
    audio_path: Path,
    description: str,
    anchors: Any | None,
    predict_spans: bool,
    reranking_candidates: int,
    max_audio_seconds: float,
    output_dir: Path,
    output_prefix: str,
) -> SeparateResponse:
    state = load_model()
    assert state.model is not None
    assert state.processor is not None
    assert state.device is not None
    assert state.dtype is not None

    duration_seconds = AudioSegment.from_file(str(audio_path)).duration_seconds
    if duration_seconds > max_audio_seconds:
        raise ValueError(
            f"Audio is {duration_seconds:.1f}s, above max_audio_seconds={max_audio_seconds:.1f}. "
            "Split longer audio before sending it to SAM-Audio."
        )

    processor_kwargs: dict[str, Any] = {
        "audios": [str(audio_path)],
        "descriptions": [description],
    }
    if anchors is not None:
        processor_kwargs["anchors"] = [anchors]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with _inference_lock, torch.autocast(device_type=state.device, dtype=state.dtype):
        inputs = state.processor(**processor_kwargs).to(state.device)
        result = state.model.separate(
            inputs,
            predict_spans=predict_spans,
            reranking_candidates=reranking_candidates,
        )

    sample_rate = int(state.processor.audio_sampling_rate)
    peak_allocated = None
    peak_reserved = None
    if torch.cuda.is_available():
        peak_allocated = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
        peak_reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 4)

    return build_separation_response(
        audio_path=audio_path,
        description=description,
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        target=tensor_from_result(result.target),
        residual=tensor_from_result(result.residual),
        output_dir=output_dir,
        output_prefix=output_prefix,
        peak_allocated=peak_allocated,
        peak_reserved=peak_reserved,
    )


@torch.inference_mode()
def run_separation_batch(
    *,
    items: list[SeparateBatchItem],
    predict_spans: bool,
    reranking_candidates: int,
    output_dir: Path,
) -> list[SeparateBatchEntry]:
    state = load_model()
    assert state.model is not None
    assert state.processor is not None
    assert state.device is not None
    assert state.dtype is not None

    entries: list[SeparateBatchEntry | None] = [None] * len(items)
    valid: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        try:
            audio_path = resolve_local_path(item.audio_ref())
            description = item.description_text()
            duration_seconds = AudioSegment.from_file(str(audio_path)).duration_seconds
            if duration_seconds > item.max_audio_seconds:
                raise ValueError(
                    f"Audio is {duration_seconds:.1f}s, above max_audio_seconds={item.max_audio_seconds:.1f}. "
                    "Split longer audio before sending it to SAM-Audio."
                )
            valid.append(
                {
                    "index": index,
                    "audio_path": audio_path,
                    "description": description,
                    "anchors": item.anchors,
                    "duration_seconds": duration_seconds,
                    "output_prefix": safe_prefix(item.output_prefix),
                }
            )
        except Exception as exc:
            entries[index] = SeparateBatchEntry(error=exception_detail(exc))

    sample_rate = int(state.processor.audio_sampling_rate)
    for start in range(0, len(valid), MAX_BATCH):
        group = valid[start : start + MAX_BATCH]
        try:
            processor_kwargs: dict[str, Any] = {
                "audios": [str(entry["audio_path"]) for entry in group],
                "descriptions": [entry["description"] for entry in group],
            }
            if any(entry["anchors"] is not None for entry in group):
                processor_kwargs["anchors"] = [entry["anchors"] for entry in group]

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            with _inference_lock, torch.autocast(device_type=state.device, dtype=state.dtype):
                inputs = state.processor(**processor_kwargs).to(state.device)
                result = state.model.separate(
                    inputs,
                    predict_spans=predict_spans,
                    reranking_candidates=reranking_candidates,
                )

            peak_allocated = None
            peak_reserved = None
            if torch.cuda.is_available():
                peak_allocated = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
                peak_reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 4)
        except Exception as exc:
            error = exception_detail(exc)
            for entry in group:
                entries[entry["index"]] = SeparateBatchEntry(error=error)
            continue

        for position, entry in enumerate(group):
            try:
                entries[entry["index"]] = SeparateBatchEntry(
                    result=build_separation_response(
                        audio_path=entry["audio_path"],
                        description=entry["description"],
                        duration_seconds=entry["duration_seconds"],
                        sample_rate=sample_rate,
                        target=result_component(result.target, position),
                        residual=result_component(result.residual, position),
                        output_dir=output_dir,
                        output_prefix=entry["output_prefix"],
                        peak_allocated=peak_allocated,
                        peak_reserved=peak_reserved,
                    )
                )
            except Exception as exc:
                entries[entry["index"]] = SeparateBatchEntry(error=exception_detail(exc))

    return [entry if entry is not None else SeparateBatchEntry(error="Item was not processed.") for entry in entries]


@app.on_event("startup")
def maybe_load_on_startup() -> None:
    if parse_bool(os.environ.get("SAM_AUDIO_LOAD_ON_STARTUP")) or parse_bool(os.environ.get("LOAD_MODEL_ON_STARTUP")):
        load_model()


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "model_id": MODEL_ID, "loaded": _state.model is not None}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    return {
        "ready": _state.model is not None,
        "model_id": MODEL_ID,
        "device": _state.device,
        "dtype": str(_state.dtype) if _state.dtype is not None else None,
    }


@app.post("/load")
def load() -> dict[str, Any]:
    try:
        state = load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=exception_detail(exc)) from exc

    return {
        "ready": True,
        "model_id": MODEL_ID,
        "device": state.device,
        "dtype": str(state.dtype),
        "sample_rate": state.processor.audio_sampling_rate if state.processor is not None else None,
    }


@app.post("/v1/sam-audio/separate", response_model=SeparateResponse)
@app.post("/separate", response_model=SeparateResponse)
def separate(request: SeparateRequest) -> SeparateResponse:
    try:
        audio_path = resolve_local_path(request.audio_ref())
        description = request.description_text()
        output_dir = resolve_output_dir(request.output_dir, default_subdir="sam-audio-large")
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        return run_separation(
            audio_path=audio_path,
            description=description,
            anchors=request.anchors,
            predict_spans=request.predict_spans,
            reranking_candidates=request.reranking_candidates,
            max_audio_seconds=request.max_audio_seconds,
            output_dir=output_dir,
            output_prefix=safe_prefix(request.output_prefix),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=exception_detail(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=exception_detail(exc)) from exc


@app.post("/v1/sam-audio/separate_batch", response_model=SeparateBatchResponse)
@app.post("/separate_batch", response_model=SeparateBatchResponse)
def separate_batch(request: SeparateBatchRequest) -> SeparateBatchResponse:
    try:
        output_dir = resolve_output_dir(request.output_dir, default_subdir="sam-audio-large")
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        results = run_separation_batch(
            items=request.items,
            predict_spans=request.predict_spans,
            reranking_candidates=request.reranking_candidates,
            output_dir=output_dir,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=exception_detail(exc)) from exc

    return SeparateBatchResponse(model_id=MODEL_ID, results=results)
