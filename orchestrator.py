"""
Audio separation orchestrator.

Takes a stereo audio file (<=30s), separates into stems (music, voice, sfx),
applies stereo transfer from the original, saves results.

Usage:
    python orchestrator.py <audio_file_or_url> [--sam-url http://localhost:8002]
"""
import argparse
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torchaudio
import requests

SAM_URL = os.environ.get("SAM_URL", "http://localhost:8002")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/app/outputs"))

# Sound gate thresholds
GATE_MIN_DBFS = -40.0
GATE_MIN_PEAK_DBFS = -35.0
GATE_MIN_ACTIVE_MS = 1000
GATE_MIN_ACTIVE_RATIO = 0.05
GATE_WINDOW_MS = 100


def amplitude_dbfs(amplitude: float, max_possible: float) -> float:
    if amplitude <= 0 or max_possible <= 0:
        return float("-inf")
    return 20.0 * math.log10(amplitude / max_possible)


def sound_gate(audio_path: Path) -> bool:
    """Returns True if the audio has meaningful content."""
    from pydub import AudioSegment
    audio = AudioSegment.from_file(str(audio_path)).set_channels(1)
    duration_ms = len(audio)
    if duration_ms == 0:
        return False

    max_possible = float(audio.max_possible_amplitude)
    overall_dbfs = amplitude_dbfs(float(audio.rms), max_possible)
    peak_dbfs = amplitude_dbfs(float(audio.max), max_possible)

    window_ms = GATE_WINDOW_MS
    active_ms = 0
    loudest_window_dbfs = float("-inf")

    for start_ms in range(0, duration_ms, window_ms):
        window = audio[start_ms:min(start_ms + window_ms, duration_ms)]
        if len(window) <= 0:
            continue
        w_dbfs = amplitude_dbfs(float(window.rms), max_possible)
        w_peak = amplitude_dbfs(float(window.max), max_possible)
        loudest_window_dbfs = max(loudest_window_dbfs, w_dbfs)
        if w_dbfs >= GATE_MIN_DBFS and w_peak >= GATE_MIN_PEAK_DBFS:
            active_ms += len(window)

    active_ratio = active_ms / duration_ms
    has_peak = peak_dbfs >= GATE_MIN_PEAK_DBFS
    has_level = overall_dbfs >= GATE_MIN_DBFS or loudest_window_dbfs >= GATE_MIN_DBFS
    has_active = active_ms >= GATE_MIN_ACTIVE_MS and active_ratio >= GATE_MIN_ACTIVE_RATIO

    return has_peak and has_level and has_active


def apply_stereo_transfer(separated_mono: torch.Tensor, model_sr: int, source_path: Path) -> tuple[torch.Tensor, int]:
    """Apply stereo panning + volume from original onto mono separated signal."""
    source_waveform, source_sr = torchaudio.load(str(source_path))
    source = source_waveform.to(torch.float32)
    n_channels = source.shape[0]

    sep_mono = separated_mono.squeeze(0) if separated_mono.ndim > 1 else separated_mono
    if model_sr != source_sr:
        sep_mono = torchaudio.functional.resample(sep_mono, model_sr, source_sr)

    if n_channels == 1:
        og_peak = source.abs().max().item()
        sep_peak = sep_mono.abs().max().item()
        if sep_peak > 1e-8:
            sep_mono = sep_mono * (og_peak / sep_peak)
        return sep_mono.unsqueeze(0), source_sr

    n_samples = source.shape[-1]
    if sep_mono.shape[-1] > n_samples:
        sep_mono = sep_mono[:n_samples]
    elif sep_mono.shape[-1] < n_samples:
        sep_mono = torch.nn.functional.pad(sep_mono, (0, n_samples - sep_mono.shape[-1]))

    n_fft = 4096
    hop = 1024
    window = torch.hann_window(n_fft)

    orig_L = torch.stft(source[0], n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    orig_R = torch.stft(source[1], n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    sep_stft = torch.stft(sep_mono, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)

    eps = 1e-10
    sep_energy = sep_stft.abs().pow(2)
    orig_L_mag = orig_L.abs()
    orig_R_mag = orig_R.abs()

    frame_max = sep_energy.max(dim=0, keepdim=True).values
    dominant_mask = (sep_energy > 0.01 * frame_max).float()
    masked_energy = sep_energy * dominant_mask

    raw_pan_L = orig_L_mag / (orig_L_mag + orig_R_mag + eps)
    weighted_pan_L = (raw_pan_L * masked_energy).sum(dim=0) / (masked_energy.sum(dim=0) + eps)

    # Bidirectional EMA smoothing
    alpha = 0.03
    n_frames = weighted_pan_L.shape[0]
    ema_fwd = torch.zeros(n_frames)
    ema_fwd[0] = weighted_pan_L[0]
    for i in range(1, n_frames):
        ema_fwd[i] = alpha * weighted_pan_L[i] + (1 - alpha) * ema_fwd[i - 1]
    ema_bwd = torch.zeros(n_frames)
    ema_bwd[-1] = weighted_pan_L[-1]
    for i in range(n_frames - 2, -1, -1):
        ema_bwd[i] = alpha * weighted_pan_L[i] + (1 - alpha) * ema_bwd[i + 1]
    pan_L_smooth = (ema_fwd + ema_bwd) / 2.0
    pan_R_smooth = 1.0 - pan_L_smooth

    # Peak-match volume
    og_peak = source.abs().max().item()
    sep_peak = sep_mono.abs().max().item()
    global_gain = og_peak / (sep_peak + eps) if sep_peak > 1e-8 else 1.0

    out_L_stft = sep_stft * global_gain * pan_L_smooth.unsqueeze(0)
    out_R_stft = sep_stft * global_gain * pan_R_smooth.unsqueeze(0)

    out_L = torch.istft(out_L_stft, n_fft=n_fft, hop_length=hop, window=window, length=n_samples)
    out_R = torch.istft(out_R_stft, n_fft=n_fft, hop_length=hop, window=window, length=n_samples)

    return torch.stack([out_L, out_R], dim=0).clamp(-1.0, 1.0), source_sr


def call_sam(audio_path: str, prompt: str, sam_url: str = SAM_URL) -> dict:
    """Call SAM-Audio API. Returns dict with target_path, residual_path, sample_rate."""
    resp = requests.post(
        f"{sam_url}/separate",
        json={"audio_path": audio_path, "prompt": prompt},
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"SAM API error {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def download_file(url_or_path: str, dest: Path) -> Path:
    """Download URL to dest, or just return the path if it's a local file."""
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        print(f"  Downloading {url_or_path}...")
        urllib.request.urlretrieve(url_or_path, str(dest))
        return dest
    # Local file — copy to shared dir if not already there
    src = Path(url_or_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")
    if str(src).startswith(str(SHARED_DIR)):
        return src
    import shutil
    shutil.copy2(str(src), str(dest))
    return dest


def main():
    parser = argparse.ArgumentParser(description="Audio separation orchestrator")
    parser.add_argument("audio", help="Audio file path or URL (must be <=30s)")
    parser.add_argument("--sam-url", default=SAM_URL, help="SAM-Audio API URL")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    sam_url = args.sam_url
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    job_id = f"job_{int(time.time())}_{os.urandom(4).hex()}"
    job_dir = SHARED_DIR / "orchestrator" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Audio Separation ===")
    print(f"Input: {args.audio}")
    print(f"Job: {job_id}")
    print(f"SAM URL: {sam_url}")

    # 1. Download/copy file
    og_path = job_dir / "input.wav"
    og_path = download_file(args.audio, og_path)
    print(f"  Source: {og_path}")

    # 2. Check duration
    wav, sr = torchaudio.load(str(og_path))
    duration = wav.shape[-1] / sr
    print(f"  Duration: {duration:.1f}s, Channels: {wav.shape[0]}, Rate: {sr}")
    if duration > 30:
        print(f"  ERROR: Audio is {duration:.1f}s, max 30s")
        sys.exit(1)

    # 2.5. Sound gate
    if not sound_gate(og_path):
        print("  Sound gate: EMPTY — skipping all")
        print("\n=== Results: 0 files (empty input) ===")
        sys.exit(0)
    print("  Sound gate: PASSED")

    results = []

    # 3. Separate music
    print(f"\n--- Step 1: Separate 'music' ---")
    t0 = time.time()
    music_resp = call_sam(str(og_path), "music", sam_url)
    print(f"  Done in {time.time()-t0:.1f}s")

    music_target = Path(music_resp["target_path"])
    music_residual = Path(music_resp["residual_path"])
    model_sr = music_resp["sample_rate"]

    # 5. Gate the music target
    music_has_sound = sound_gate(music_target)
    print(f"  Music target gate: {'PASSED' if music_has_sound else 'EMPTY'}")

    if music_has_sound:
        # Apply stereo transfer to music
        music_wav, _ = torchaudio.load(str(music_target))
        music_stereo, out_sr = apply_stereo_transfer(music_wav, model_sr, og_path)
        music_out = output_dir / f"{job_id}_music.wav"
        torchaudio.save(str(music_out), music_stereo, out_sr)
        results.append(("music", music_out))
        print(f"  Saved: {music_out}")

    # 6. Separate voice
    # If music was extracted, use residual. If not, use original.
    voice_input = str(music_residual) if music_has_sound else str(og_path)
    print(f"\n--- Step 2: Separate 'human voice' from {'residual' if music_has_sound else 'original'} ---")
    t0 = time.time()
    voice_resp = call_sam(voice_input, "human voice", sam_url)
    print(f"  Done in {time.time()-t0:.1f}s")

    voice_target = Path(voice_resp["target_path"])
    voice_residual = Path(voice_resp["residual_path"])

    # 8. Gate voice target
    voice_has_sound = sound_gate(voice_target)
    print(f"  Voice target gate: {'PASSED' if voice_has_sound else 'EMPTY'}")

    if voice_has_sound:
        voice_wav, _ = torchaudio.load(str(voice_target))
        voice_stereo, out_sr = apply_stereo_transfer(voice_wav, model_sr, og_path)
        voice_out = output_dir / f"{job_id}_voice.wav"
        torchaudio.save(str(voice_out), voice_stereo, out_sr)
        results.append(("voice", voice_out))
        print(f"  Saved: {voice_out}")

    # SFX is whatever remains after voice separation
    sfx_path = voice_residual
    sfx_has_sound = sound_gate(sfx_path)
    print(f"\n--- SFX residual gate: {'PASSED' if sfx_has_sound else 'EMPTY'} ---")

    if sfx_has_sound:
        sfx_wav, _ = torchaudio.load(str(sfx_path))
        sfx_stereo, out_sr = apply_stereo_transfer(sfx_wav, model_sr, og_path)
        sfx_out = output_dir / f"{job_id}_sfx.wav"
        torchaudio.save(str(sfx_out), sfx_stereo, out_sr)
        results.append(("sfx", sfx_out))
        print(f"  Saved: {sfx_out}")

    # 10. Summary
    print(f"\n=== Results: {len(results)} file(s) ===")
    for stem_name, path in results:
        print(f"  {stem_name}: {path}")
    if not results:
        print("  (no stems extracted)")

    return results


if __name__ == "__main__":
    main()
