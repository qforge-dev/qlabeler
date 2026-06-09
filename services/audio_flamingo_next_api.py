from __future__ import annotations

import os
import threading
from contextlib import nullcontext
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def patch_torch_float8_symbols_for_transformers() -> None:
    # Transformers 5.x imports FP8 integration symbols eagerly. Older torch builds
    # can lack a few dtype aliases even when FP8 is not used by this model.
    for name in ("float8_e8m0fnu", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        if not hasattr(torch, name):
            setattr(torch, name, torch.uint8)


patch_torch_float8_symbols_for_transformers()

from transformers import AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoProcessor

from services.common import exception_detail, parse_bool, resolve_local_path


MODEL_ID = os.environ.get("AFNEXT_MODEL_ID", "nvidia/audio-flamingo-next-think-hf")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("AFNEXT_MAX_NEW_TOKENS", "1024"))
DEFAULT_REPETITION_PENALTY = float(os.environ.get("AFNEXT_REPETITION_PENALTY", "1.2"))

app = FastAPI(
    title="Audio Flamingo Next API",
    version="1.0.0",
    summary="Ask nvidia/audio-flamingo-next-think-hf questions about local audio files.",
)


class AskRequest(BaseModel):
    audio_path: str | None = Field(default=None, description="Local path or file:// URL to an audio file.")
    file_path: str | None = Field(default=None, description="Alias for audio_path.")
    file: str | None = Field(default=None, description="Alias for audio_path.")
    audio_url: str | None = Field(default=None, description="Alias for audio_path; only file:// URLs are supported.")
    prompt: str | None = Field(default=None, description="Question or instruction for the model.")
    input: str | None = Field(default=None, description="Alias for prompt.")
    question: str | None = Field(default=None, description="Alias for prompt.")
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=8192)
    repetition_penalty: float = Field(default=DEFAULT_REPETITION_PENALTY, ge=0.1, le=10.0)

    def audio_ref(self) -> str:
        value = self.audio_path or self.file_path or self.file or self.audio_url
        if not value:
            raise ValueError("Provide audio_path, file_path, file, or audio_url.")
        return value

    def prompt_text(self) -> str:
        value = self.prompt or self.input or self.question
        if not value or not value.strip():
            raise ValueError("Provide prompt, input, or question.")
        return value.strip()


class AskResponse(BaseModel):
    model_id: str
    audio_path: str
    prompt: str
    text: str


class _ModelState:
    processor: Any | None = None
    model: Any | None = None
    model_device: torch.device | None = None
    model_dtype: torch.dtype | None = None
    runtime: str | None = None


_state = _ModelState()
_load_lock = threading.Lock()
_inference_lock = threading.Lock()


def preferred_runtime() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_input_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_model() -> _ModelState:
    if _state.model is not None and _state.processor is not None:
        return _state

    with _load_lock:
        if _state.model is not None and _state.processor is not None:
            return _state

        runtime = preferred_runtime()
        dtype = torch.bfloat16 if runtime in {"cuda", "mps"} else torch.float32
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        config = AutoConfig.from_pretrained(MODEL_ID)

        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if runtime == "cuda":
            model_kwargs["device_map"] = "auto"
        elif runtime == "mps":
            model_kwargs["device_map"] = {"": "mps"}

        try:
            auto_model_cls = AutoModel._model_mapping[type(config)]
            auto_model_cls_name = auto_model_cls.__name__
        except Exception:
            auto_model_cls_name = ""

        if str(auto_model_cls_name).endswith("ForConditionalGeneration"):
            model = AutoModel.from_pretrained(MODEL_ID, **model_kwargs).eval()
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, **model_kwargs).eval()

        _state.processor = processor
        _state.model = model
        _state.model_device = model_input_device(model)
        _state.model_dtype = next(model.parameters()).dtype
        _state.runtime = runtime
        return _state


def ask_audio(audio_path: str, prompt: str, *, max_new_tokens: int, repetition_penalty: float) -> str:
    state = load_model()
    assert state.processor is not None
    assert state.model is not None
    assert state.model_device is not None
    assert state.model_dtype is not None

    conversation = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": audio_path},
                ],
            }
        ]
    ]

    batch = state.processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    batch = batch.to(state.model_device)

    if "input_features" in batch:
        batch["input_features"] = batch["input_features"].to(state.model_dtype)

    use_cuda_amp = state.model_device.type == "cuda" and state.model_dtype in {torch.float16, torch.bfloat16}
    amp_context = torch.autocast("cuda", dtype=state.model_dtype) if use_cuda_amp else nullcontext()

    with _inference_lock, torch.inference_mode(), amp_context:
        generated = state.model.generate(
            **batch,
            max_new_tokens=int(max_new_tokens),
            repetition_penalty=float(repetition_penalty),
        )

    prompt_len = batch["input_ids"].shape[1]
    completion = generated[:, prompt_len:] if generated.shape[1] > prompt_len else generated
    return state.processor.batch_decode(
        completion,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


@app.on_event("startup")
def maybe_load_on_startup() -> None:
    if parse_bool(os.environ.get("AFNEXT_LOAD_ON_STARTUP")) or parse_bool(os.environ.get("LOAD_MODEL_ON_STARTUP")):
        load_model()


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "model_id": MODEL_ID, "loaded": _state.model is not None}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    return {
        "ready": _state.model is not None,
        "model_id": MODEL_ID,
        "runtime": _state.runtime,
        "device": str(_state.model_device) if _state.model_device is not None else None,
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
        "runtime": state.runtime,
        "device": str(state.model_device),
        "dtype": str(state.model_dtype),
    }


@app.post("/v1/audio-flamingo/ask", response_model=AskResponse)
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        audio_path = resolve_local_path(request.audio_ref())
        prompt = request.prompt_text()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        text = ask_audio(
            str(audio_path),
            prompt,
            max_new_tokens=request.max_new_tokens,
            repetition_penalty=request.repetition_penalty,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=exception_detail(exc)) from exc

    return AskResponse(model_id=MODEL_ID, audio_path=str(audio_path), prompt=prompt, text=text)
