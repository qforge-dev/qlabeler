"""Patch SAM-Audio to replace torchcodec AudioDecoder with torchaudio.load"""
from pathlib import Path
import site

site_paths = [Path(base) for base in site.getsitepackages()]

for path in [base / "core" / "audio_visual_encoder" / "transforms.py" for base in site_paths]:
    if path.exists():
        break
else:
    raise SystemExit("Could not find core/audio_visual_encoder/transforms.py")

text = path.read_text()
if "AudioDecoder" in text:
    text = text.replace(
        "from torchcodec.decoders import AudioDecoder, VideoDecoder\n",
        "from torchcodec.decoders import VideoDecoder\nimport torchaudio\n",
    )
    old_load = """    def _load_audio(self, path: str):
        ad = AudioDecoder(path, sample_rate=self.sampling_rate, num_channels=1)
        return ad.get_all_samples().data
"""
    new_load = """    def _load_audio(self, path: str):
        wav, sample_rate = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sample_rate != self.sampling_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sampling_rate)
        return wav
"""
    text = text.replace(old_load, new_load)
    path.write_text(text)
    print(f"patched {path}")

for path in [base / "sam_audio" / "processor.py" for base in site_paths]:
    if path.exists():
        break
else:
    raise SystemExit("Could not find sam_audio/processor.py")

text = path.read_text()
if "AudioDecoder" in text:
    text = text.replace(
        "from torchcodec.decoders import AudioDecoder, VideoDecoder\n",
        "from torchcodec.decoders import VideoDecoder\n",
    )
    path.write_text(text)
    print(f"patched {path}")
