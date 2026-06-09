ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    JUPYTER_TOKEN=qlabeler \
    TOKENIZERS_PARALLELISM=false \
    USE_TF=0 \
    USE_FLAX=0 \
    TRANSFORMERS_NO_TF=1 \
    TRANSFORMERS_NO_FLAX=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    python -m pip uninstall -y \
      tensorflow tensorflow-cpu tensorflow-text tensorflow-datasets \
      tensorflow-metadata tf-keras keras jax jaxlib || true && \
    python -m pip install --no-warn-conflicts \
      "sam_audio @ git+https://github.com/facebookresearch/sam-audio.git" && \
    python -m pip install --upgrade --no-warn-conflicts \
      "transformers>=4.54,<5" \
      "huggingface_hub>=0.34,<1.0" \
      hf_transfer \
      ipywidgets \
      jupyterlab \
      pydub

WORKDIR /workspace

EXPOSE 8889

CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8889 --no-browser --allow-root --ServerApp.token=\"$JUPYTER_TOKEN\" --ServerApp.password=''"]
