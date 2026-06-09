ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    JUPYTER_TOKEN=qlabeler \
    TOKENIZERS_PARALLELISM=false \
    USE_TF=0 \
    USE_FLAX=0 \
    TRANSFORMERS_NO_TF=1 \
    TRANSFORMERS_NO_FLAX=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      accelerate \
      hf_transfer \
      huggingface_hub \
      ipywidgets \
      jupyterlab \
      librosa \
      pydub \
      soundfile \
      transformers

WORKDIR /workspace

EXPOSE 8888

CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token=\"$JUPYTER_TOKEN\" --ServerApp.password=''"]
