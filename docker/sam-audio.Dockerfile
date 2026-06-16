FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

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
    python3.11 python3.11-venv python3.11-dev python3-pip \
    ffmpeg libsndfile1 git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install Python deps
RUN python3 -m pip install --upgrade pip wheel setuptools

# Install PyTorch (CUDA 12.4) - this is the big one
RUN python3 -m pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124

# Install SAM-Audio from git
RUN python3 -m pip install --no-warn-conflicts \
    "sam_audio @ git+https://github.com/facebookresearch/sam-audio.git"

# Install API deps
RUN python3 -m pip install --no-warn-conflicts \
    fastapi \
    "transformers>=4.54,<5" \
    "huggingface_hub>=0.34,<1.0" \
    hf_transfer \
    pydub \
    "uvicorn[standard]" \
    audioop-lts

# Copy application code
COPY services/ /app/services/

# Patch SAM-Audio audio loader (replace torchcodec AudioDecoder with torchaudio.load)
RUN python3 -c "
from pathlib import Path
import site

site_paths = [Path(base) for base in site.getsitepackages()]

for path in [base / 'core' / 'audio_visual_encoder' / 'transforms.py' for base in site_paths]:
    if path.exists():
        break
else:
    raise SystemExit('Could not find core/audio_visual_encoder/transforms.py')

text = path.read_text()
if 'AudioDecoder' in text:
    text = text.replace(
        'from torchcodec.decoders import AudioDecoder, VideoDecoder\n',
        'from torchcodec.decoders import VideoDecoder\nimport torchaudio\n',
    )
    text = text.replace(
        '''    def _load_audio(self, path: str):
        ad = AudioDecoder(path, sample_rate=self.sampling_rate, num_channels=1)
        return ad.get_all_samples().data
''',
        '''    def _load_audio(self, path: str):
        wav, sample_rate = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sample_rate != self.sampling_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sampling_rate)
        return wav
''',
    )
    path.write_text(text)
    print(f'patched {path}')

for path in [base / 'sam_audio' / 'processor.py' for base in site_paths]:
    if path.exists():
        break
else:
    raise SystemExit('Could not find sam_audio/processor.py')

text = path.read_text()
if 'AudioDecoder' in text:
    text = text.replace(
        'from torchcodec.decoders import AudioDecoder, VideoDecoder\n',
        'from torchcodec.decoders import VideoDecoder\n',
    )
    path.write_text(text)
    print(f'patched {path}')
"

EXPOSE 8002

CMD ["uvicorn", "services.sam_audio_large_api:app", "--host", "0.0.0.0", "--port", "8002"]
