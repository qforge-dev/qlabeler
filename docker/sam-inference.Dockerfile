FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV USE_TF=0
ENV USE_FLAX=0
ENV TRANSFORMERS_NO_TF=1
ENV TRANSFORMERS_NO_FLAX=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_DISABLE_XET=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip wheel setuptools
RUN pip install torch torchaudio torchvision xformers
RUN pip install --no-warn-conflicts "sam_audio @ git+https://github.com/facebookresearch/sam-audio.git"
RUN pip install fastapi "uvicorn[standard]" "huggingface_hub>=0.34,<0.37" "transformers>=4.54,<5"

# Patch torchcodec AudioDecoder -> torchaudio.load
COPY docker/patch_sam_audio.py /tmp/patch_sam_audio.py
RUN python /tmp/patch_sam_audio.py && rm /tmp/patch_sam_audio.py

COPY services/sam_inference.py /app/services/sam_inference.py

EXPOSE 8002

CMD ["uvicorn", "services.sam_inference:app", "--host", "0.0.0.0", "--port", "8002"]
