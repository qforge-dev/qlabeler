FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_DISABLE_XET=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
RUN pip install --upgrade pip wheel setuptools

# Install PyTorch (from PyPI, bundles CUDA)
RUN pip install torch torchaudio

# Uninstall torchvision to save space (not needed for audio)
RUN pip uninstall -y torchvision || true

# Install Audio Flamingo deps
RUN pip install --upgrade \
    accelerate \
    fastapi \
    hf_transfer \
    librosa \
    pydub \
    safetensors \
    soundfile \
    transformers \
    "uvicorn[standard]"

# Copy application code
COPY services/ /app/services/

EXPOSE 8001

CMD ["uvicorn", "services.audio_flamingo_next_api:app", "--host", "0.0.0.0", "--port", "8001"]
