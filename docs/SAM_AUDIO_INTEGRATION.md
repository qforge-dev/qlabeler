# SAM-Audio Integration

## Model Overview

SAM-Audio (Segment Anything Model for Audio) by Meta/Facebook. We use `facebook/sam-audio-large`.
It isolates target sounds from audio mixtures using text descriptions. It's a generative model
(flow-matching ODE solver) that produces new waveforms, not just masks.

## Critical Configuration

### fp32, NOT bf16

**This is the single most important thing.** The model MUST run in fp32 (float32).

When `reranking_candidates > 1`, the model uses a CLAP model internally to score and
pick the best separation candidate. CLAP's spectrogram extractor (`torchlibrosa`)
has Conv1d layers that require fp32 input tensors. If you load the model in bf16:

- With `reranking_candidates > 1`: crashes with `RuntimeError: Input type (FloatTensor) and weight type (BFloat16Type) should be the same`
- With `reranking_candidates = 1`: "works" but produces garbage separation because there's no quality selection

We wasted days debugging this. The bf16 model appeared to work (no crash with reranking=1)
but produced terrible separation with massive spillover. The fix was loading in fp32.

**Memory cost**: fp32 model uses ~33GB VRAM (vs ~16GB for bf16). On an H100 80GB this
is fine. On smaller GPUs you'd need to choose: fp32 with reranking (better quality)
vs bf16 without reranking (worse quality but fits in less VRAM).

### predict_spans=True

The model can optionally predict the temporal span (time region) where the target
sound occurs. This significantly improves separation for non-ambient sounds because
the model focuses on the relevant time region rather than trying to extract from the
entire clip.

### reranking_candidates=4-8

The model generates N candidate separations and uses the CLAP reranker to pick the
best one. Higher is better quality but more memory and latency:

| Candidates | Quality | Peak VRAM (30s audio, fp32) | Latency |
|------------|---------|----------------------------|---------|
| 1 | Poor | ~35GB | ~4s |
| 4 | Good | ~58GB | ~20s |
| 8 | Best | ~75GB | ~35s |

We use 4 in production (to coexist with Audio Flamingo on the same GPU) and 8 for
offline quality testing.

### Simple NP/VP Prompts

From Meta's README: "To match training, please use lowercase noun-phrase/verb-phrase
(NP/VP) format for text."

**Good prompts**: `"music"`, `"human voice"`, `"thunder"`, `"car engine"`

**Bad prompts**: `"upbeat electronic dance music with synth pads and four-on-the-floor drums"`,
`"A man is speaking in the background while birds chirp"`

We tried using Audio Flamingo to generate detailed music descriptions for SAM. It
made things worse. The text encoder was trained on short NP/VP labels, not sentences.

### No torch.autocast

We initially wrapped inference in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`.
This interfered with the ODE solver's numerical precision. The model is already fp32
after loading, no autocast needed.

### The ODE Solver is Stochastic

`model.separate()` starts from random noise (`torch.randn_like(audio_features)`) and
integrates an ODE to generate the separated waveform. This means:

- Every run produces slightly different results
- You can't expect bit-identical output to Meta's demo
- But quality should be comparable (same distribution)

There's no way to set a seed externally because the noise is generated inside the
model's `separate()` method.

## Model Loading

```python
from sam_audio import SAMAudio, SAMAudioProcessor

model = SAMAudio.from_pretrained("facebook/sam-audio-large")
processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
model = model.eval().cuda()  # fp32, NO .to(torch.bfloat16)
```

### Vision Encoder Optimization

SAM-Audio has a vision encoder for video-prompted separation. For audio-only use we
delete it to save ~3GB VRAM:

```python
vision_dim = model.vision_encoder.dim
del model.vision_encoder
model._vision_encoder_dim = vision_dim

def _get_video_features_audio_only(self, video, audio_features):
    batch_size, time_steps, _ = audio_features.shape
    return audio_features.new_zeros(batch_size, self._vision_encoder_dim, time_steps)

model._get_video_features = types.MethodType(_get_video_features_audio_only, model)
```

This was validated to work correctly in fp32. The model returns zeros for video
features which is functionally identical to not providing video input.

## Inference

```python
batch = processor(audios=[audio_path], descriptions=["music"]).to("cuda")
with torch.inference_mode():
    result = model.separate(batch, predict_spans=True, reranking_candidates=8)

target = result.target[0]   # mono waveform tensor
residual = result.residual[0]  # mono waveform tensor
sample_rate = processor.audio_sampling_rate  # 48000
```

The processor internally:
- Loads the audio via `torchaudio.load`
- Downmixes to mono (`wav.mean(0)`)
- Resamples to 48kHz

Output is always mono at 48kHz regardless of input format.

## Package Notes

- `sam_audio==0.1.0` installed from `git+https://github.com/facebookresearch/sam-audio.git`
- Requires patching `core/audio_visual_encoder/transforms.py` to replace `AudioDecoder`
  (from torchcodec) with `torchaudio.load` — newer torchaudio versions removed the
  `AudioDecoder` API
- Similarly patch `sam_audio/processor.py` to remove the `AudioDecoder` import
- These patches are applied by `scripts/setup_model_apis.sh` during installation

## Memory Management

On H100 80GB with both SAM-Audio and Audio Flamingo loaded:
- Audio Flamingo (bf16): ~16GB
- SAM-Audio (fp32, loaded): ~33GB
- Total at rest: ~49GB
- Peak during separation (reranking=4): ~74GB
- Available headroom: ~6GB

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.

If you get OOM, reduce `reranking_candidates` to 2-3, or kill Audio Flamingo during
SAM inference and restart it after.
