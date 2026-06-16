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

# Copy application code
COPY services/ /app/services/

# Patch SAM-Audio audio loader (replace torchcodec AudioDecoder with torchaudio.load)
RUN python -c "
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
