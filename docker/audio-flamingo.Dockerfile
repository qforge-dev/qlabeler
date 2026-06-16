FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_DISABLE_XET=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    ffmpeg libsndfile1 git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install Python deps
RUN python3 -m pip install --upgrade pip wheel setuptools

# Install PyTorch (CUDA 12.4)
RUN python3 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Uninstall torchvision to save space (not needed for audio)
RUN python3 -m pip uninstall -y torchvision || true

# Install Audio Flamingo deps
RUN python3 -m pip install --upgrade \
    accelerate \
    fastapi \
    hf_transfer \
    librosa \
    pydub \
    safetensors \
    soundfile \
    transformers \
    "uvicorn[standard]" \
    audioop-lts

# Copy application code
COPY services/ /app/services/

EXPOSE 8001

CMD ["uvicorn", "services.audio_flamingo_next_api:app", "--host", "0.0.0.0", "--port", "8001"]
