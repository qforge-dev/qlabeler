FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV USE_TF=0
ENV USE_FLAX=0
ENV TRANSFORMERS_NO_TF=1
ENV TRANSFORMERS_NO_FLAX=1
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
RUN pip install torch torchaudio torchvision

# Install xformers (needed by perception-models, a transitive dep of sam_audio)
RUN pip install xformers

# Install SAM-Audio from git
RUN pip install --no-warn-conflicts \
    "sam_audio @ git+https://github.com/facebookresearch/sam-audio.git"

# Install API deps
RUN pip install --no-warn-conflicts \
    fastapi \
    "transformers>=4.54,<5" \
    "huggingface_hub>=0.34,<1.0" \
    hf_transfer \
    pydub \
    "uvicorn[standard]"

# Patch SAM-Audio audio loader
COPY docker/patch_sam_audio.py /tmp/patch_sam_audio.py
RUN python /tmp/patch_sam_audio.py && rm /tmp/patch_sam_audio.py

# Copy application code
COPY services/ /app/services/

EXPOSE 8002

CMD ["uvicorn", "services.sam_audio_large_api:app", "--host", "0.0.0.0", "--port", "8002"]
