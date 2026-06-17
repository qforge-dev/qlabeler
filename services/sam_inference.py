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

# Serve output files
from fastapi.staticfiles import StaticFiles

MODEL_ID = os.environ.get("SAM_AUDIO_MODEL_ID", "facebook/sam-audio-large")
RERANKING_CANDIDATES = int(os.environ.get("SAM_AUDIO_RERANKING_CANDIDATES", "8"))
MODEL_DTYPE = os.environ.get("SAM_AUDIO_DTYPE", "fp32")  # fp32, fp16, bf16
RANKER_DTYPE = os.environ.get("SAM_AUDIO_RANKER_DTYPE", "fp32")  # keep fp32 for CLAP compat
DISABLE_VISION = os.environ.get("SAM_AUDIO_DISABLE_VISION", "0") in ("1", "true", "yes")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/outputs/sam"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# Serve files from the shared output directory
SERVE_DIR = Path(os.environ.get("SERVE_DIR", "/app/outputs"))
if SERVE_DIR.exists():
    app.mount("/files", StaticFiles(directory=str(SERVE_DIR)), name="files")

_model = None
_processor = None


def get_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    from sam_audio import SAMAudio, SAMAudioProcessor

    _model = SAMAudio.from_pretrained(MODEL_ID, proxies=None, resume_download=False).eval().cuda()

    # Optionally remove vision encoder to save VRAM (~3GB)
    if DISABLE_VISION and hasattr(_model, "vision_encoder"):
        import types, gc
        vision_dim = _model.vision_encoder.dim
        del _model.vision_encoder
        _model._vision_encoder_dim = vision_dim

        def _get_video_features_audio_only(self, video, audio_features):
            if video is not None:
                raise ValueError("Vision encoder disabled (SAM_AUDIO_DISABLE_VISION=1)")
            batch_size, time_steps, _ = audio_features.shape
            return audio_features.new_zeros(batch_size, self._vision_encoder_dim, time_steps)

        _model._get_video_features = types.MethodType(_get_video_features_audio_only, _model)
        gc.collect()
        torch.cuda.empty_cache()

    # Apply model precision
    if MODEL_DTYPE == "fp16":
        _model = _model.half()
    elif MODEL_DTYPE == "bf16":
        _model = _model.to(torch.bfloat16)

    # Keep ONLY the ranker in fp32 (CLAP spectrogram Conv1d layers require fp32).
    # Everything else (codec, main model) stays in the model dtype.
    if MODEL_DTYPE != "fp32":
        if hasattr(_model, "text_ranker") and _model.text_ranker is not None:
            _model.text_ranker.float()
            # Wrap ranker forward to ensure all inputs are cast to fp32
            import types
            _orig_ranker_forward = _model.text_ranker.forward
            def _fp32_ranker_forward(*args, **kwargs):
                with torch.amp.autocast("cuda", dtype=torch.float32):
                    return _orig_ranker_forward(*args, **kwargs)
            _model.text_ranker.forward = _fp32_ranker_forward
        if hasattr(_model, "visual_ranker") and _model.visual_ranker is not None:
            _model.visual_ranker.float()

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
        "model_dtype": MODEL_DTYPE,
        "ranker_dtype": RANKER_DTYPE,
        "reranking_candidates": RERANKING_CANDIDATES,
        "disable_vision": DISABLE_VISION,
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
    if MODEL_DTYPE != "fp32":
        batch.audios = batch.audios.to(DTYPE_MAP[MODEL_DTYPE])
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


class BatchSeparateRequest(BaseModel):
    audio_path: str = Field(description="Path to audio file (shared volume)")
    prompts: list[str] = Field(description="List of text prompts to separate simultaneously")
    predict_spans: bool = True
    reranking_candidates: int | None = None


class BatchSeparateItem(BaseModel):
    prompt: str
    target_path: str
    residual_path: str


class BatchSeparateResponse(BaseModel):
    results: list[BatchSeparateItem]
    sample_rate: int
    duration_seconds: float


@app.post("/separate_batch", response_model=BatchSeparateResponse)
def separate_batch(request: BatchSeparateRequest):
    """Separate multiple prompts from the same audio file in one batch."""
    audio_path = Path(request.audio_path)
    if not audio_path.exists():
        raise HTTPException(status_code=422, detail=f"File not found: {audio_path}")

    model, processor = get_model()
    reranking = request.reranking_candidates or RERANKING_CANDIDATES

    wav, sr = torchaudio.load(str(audio_path))
    duration = wav.shape[-1] / sr
    if duration > 35:
        raise HTTPException(status_code=422, detail=f"Audio too long: {duration:.1f}s (max 35s)")

    n_prompts = len(request.prompts)
    # Batch: same file repeated for each prompt
    batch = processor(
        audios=[str(audio_path)] * n_prompts,
        descriptions=request.prompts,
    ).to("cuda")
    if MODEL_DTYPE != "fp32":
        batch.audios = batch.audios.to(DTYPE_MAP[MODEL_DTYPE])

    with torch.inference_mode():
        result = model.separate(
            batch,
            predict_spans=request.predict_spans,
            reranking_candidates=reranking,
        )

    sample_rate = processor.audio_sampling_rate
    request_id = uuid.uuid4().hex[:12]
    out_dir = OUTPUT_DIR / request_id
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for i, prompt in enumerate(request.prompts):
        target = result.target[i] if isinstance(result.target, list) else result.target
        residual = result.residual[i] if isinstance(result.residual, list) else result.residual
        if target.ndim == 1:
            target = target.unsqueeze(0)
        if residual.ndim == 1:
            residual = residual.unsqueeze(0)

        target_path = out_dir / f"{i}_{prompt.replace(' ', '_')}_target.wav"
        residual_path = out_dir / f"{i}_{prompt.replace(' ', '_')}_residual.wav"
        torchaudio.save(str(target_path), target.float().cpu(), sample_rate)
        torchaudio.save(str(residual_path), residual.float().cpu(), sample_rate)
        items.append(BatchSeparateItem(
            prompt=prompt,
            target_path=str(target_path),
            residual_path=str(residual_path),
        ))

    return BatchSeparateResponse(
        results=items,
        sample_rate=sample_rate,
        duration_seconds=round(duration, 3),
    )


@app.post("/benchmark")
def benchmark(request: SeparateRequest):
    """Run separation with detailed timing of each stage."""
    import time as _time
    from sam_audio.model.model import DFLT_ODE_OPT
    from torchdiffeq import odeint as _odeint

    audio_path = Path(request.audio_path)
    if not audio_path.exists():
        raise HTTPException(status_code=422, detail=f"File not found: {audio_path}")

    model, processor = get_model()
    reranking = request.reranking_candidates or RERANKING_CANDIDATES
    timings = {}

    wav, sr = torchaudio.load(str(audio_path))
    duration = wav.shape[-1] / sr

    # Prepare batch
    batch = processor(audios=[str(audio_path)], descriptions=[request.prompt]).to("cuda")
    if MODEL_DTYPE != "fp32":
        batch.audios = batch.audios.to(DTYPE_MAP[MODEL_DTYPE])

    with torch.inference_mode():
        torch.cuda.synchronize()

        # 1. Audio encoding + text encoding
        t0 = _time.time()
        forward_args = model._get_forward_args(batch, candidates=reranking)
        torch.cuda.synchronize()
        timings["1_encoding"] = round(_time.time() - t0, 3)

        # 2. Span prediction
        t0 = _time.time()
        if request.predict_spans and hasattr(model, "span_predictor") and batch.anchors is None:
            batch = model.predict_spans(
                batch=batch,
                audio_features=model._unrepeat_from_reranking(forward_args["audio_features"], reranking),
                audio_pad_mask=model._unrepeat_from_reranking(forward_args["audio_pad_mask"], reranking),
            )
            forward_args.update({
                "anchor_ids": model._repeat_for_reranking(batch.anchor_ids, reranking),
                "anchor_alignment": model._repeat_for_reranking(batch.anchor_alignment, reranking),
            })
        torch.cuda.synchronize()
        timings["2_span_prediction"] = round(_time.time() - t0, 3)

        # 3. ODE generation
        audio_features = forward_args["audio_features"]
        B, T, C = audio_features.shape
        C = C // 2
        noise = torch.randn_like(audio_features)

        def vector_field(t, noisy_audio):
            return model.forward(
                noisy_audio=noisy_audio,
                time=t.expand(noisy_audio.size(0)),
                **forward_args,
            )

        t0 = _time.time()
        states = _odeint(
            vector_field, noise,
            torch.tensor([0.0, 1.0], device=noise.device),
            **DFLT_ODE_OPT,
        )
        torch.cuda.synchronize()
        timings["3_ode_generation"] = round(_time.time() - t0, 3)

        # 4. Audio decoding
        generated_features = states[-1].transpose(1, 2)
        t0 = _time.time()
        wavs = model.audio_codec.decode(generated_features.reshape(2 * B, C, T)).view(B, 2, -1)
        torch.cuda.synchronize()
        timings["4_audio_decoding"] = round(_time.time() - t0, 3)

        # 5. Reranking
        bsz = wavs.size(0) // reranking
        sizes = model.audio_codec.feature_idx_to_wav_idx(batch.sizes)
        target_wavs = model.unbatch(wavs[:, 0].view(bsz, reranking, -1), sizes)

        input_audio = [
            audio[:, :size].expand(reranking, -1)
            for audio, size in zip(batch.audios, sizes, strict=False)
        ]

        t0 = _time.time()
        with torch.amp.autocast("cuda", dtype=torch.float32):
            scores = model.text_ranker(
                extracted_audio=target_wavs,
                input_audio=input_audio,
                descriptions=batch.descriptions,
                sample_rate=model.audio_codec.sample_rate,
            )
        torch.cuda.synchronize()
        timings["5_reranking"] = round(_time.time() - t0, 3)

    total = sum(timings.values())
    timings["total"] = round(total, 3)
    timings["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    # Percentages
    pct = {k: f"{v/total*100:.1f}%" for k, v in timings.items() if k != "total" and k != "peak_vram_gb"}

    return {
        "model_id": MODEL_ID,
        "model_dtype": MODEL_DTYPE,
        "reranking_candidates": reranking,
        "predict_spans": request.predict_spans,
        "audio_duration_s": round(duration, 1),
        "timings_seconds": timings,
        "percentages": pct,
    }
