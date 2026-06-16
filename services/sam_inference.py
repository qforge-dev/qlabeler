"""
Minimal SAM-Audio inference API.
Accepts: audio file path + text prompt.
Returns: target + residual file paths.
Nothing else — no stereo, no gates, no pipeline logic.
"""
import os
import uuid
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="SAM-Audio Inference")

MODEL_ID = os.environ.get("SAM_AUDIO_MODEL_ID", "facebook/sam-audio-large")
RERANKING_CANDIDATES = int(os.environ.get("SAM_AUDIO_RERANKING_CANDIDATES", "8"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/outputs/sam"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_model = None
_processor = None


def get_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    from sam_audio import SAMAudio, SAMAudioProcessor
    _model = SAMAudio.from_pretrained(MODEL_ID).eval().cuda()
    _processor = SAMAudioProcessor.from_pretrained(MODEL_ID)
    return _model, _processor


class SeparateRequest(BaseModel):
    audio_path: str = Field(description="Path to audio file (shared volume)")
    prompt: str = Field(description="Text description of sound to extract")
    predict_spans: bool = True
    reranking_candidates: int | None = None


class SeparateResponse(BaseModel):
    target_path: str
    residual_path: str
    sample_rate: int
    duration_seconds: float


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_id": MODEL_ID, "loaded": _model is not None}


@app.post("/load")
def load():
    model, processor = get_model()
    return {
        "ready": True,
        "model_id": MODEL_ID,
        "dtype": str(next(model.parameters()).dtype),
        "sample_rate": processor.audio_sampling_rate,
    }


@app.post("/separate", response_model=SeparateResponse)
def separate(request: SeparateRequest):
    audio_path = Path(request.audio_path)
    if not audio_path.exists():
        raise HTTPException(status_code=422, detail=f"File not found: {audio_path}")

    model, processor = get_model()
    reranking = request.reranking_candidates or RERANKING_CANDIDATES

    # Check duration
    wav, sr = torchaudio.load(str(audio_path))
    duration = wav.shape[-1] / sr
    if duration > 35:
        raise HTTPException(status_code=422, detail=f"Audio too long: {duration:.1f}s (max 35s)")

    # Run separation
    batch = processor(audios=[str(audio_path)], descriptions=[request.prompt]).to("cuda")
    with torch.inference_mode():
        result = model.separate(
            batch,
            predict_spans=request.predict_spans,
            reranking_candidates=reranking,
        )

    # Extract tensors
    target = result.target[0] if isinstance(result.target, list) else result.target
    residual = result.residual[0] if isinstance(result.residual, list) else result.residual
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if residual.ndim == 1:
        residual = residual.unsqueeze(0)

    # Save
    request_id = uuid.uuid4().hex[:12]
    out_dir = OUTPUT_DIR / request_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = processor.audio_sampling_rate
    target_path = out_dir / "target.wav"
    residual_path = out_dir / "residual.wav"
    torchaudio.save(str(target_path), target.float().cpu(), sample_rate)
    torchaudio.save(str(residual_path), residual.float().cpu(), sample_rate)

    return SeparateResponse(
        target_path=str(target_path),
        residual_path=str(residual_path),
        sample_rate=sample_rate,
        duration_seconds=round(duration, 3),
    )
