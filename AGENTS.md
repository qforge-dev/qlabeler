# AGENTS.md — Lessons Learned

This file captures hard-won knowledge from building and debugging the audio
separation pipeline. It's intended for AI agents and developers working on this
codebase in the future.

## Model Configuration

### Always validate quality in fp32 first

We spent days debugging terrible separation quality. The root cause was loading the
SAM-Audio model in bf16, which silently broke the CLAP reranker (used when
`reranking_candidates > 1`). The model appeared to "work" but produced garbage.

**Lesson**: When a model produces unexpectedly bad results, check if precision
reduction is causing silent failures in subcomponents. Test in fp32 first to establish
a quality baseline, then optimize precision only after confirming quality matches.

### The CLAP reranker is the key quality component

Without reranking (`reranking_candidates=1`), SAM-Audio produces mediocre separation.
With reranking=4-8, it's dramatically better. The CLAP model scores each candidate
and picks the best one. This is where most of the quality comes from.

The reranker requires fp32 input. This is a non-negotiable constraint that determines
the entire model loading strategy.

### Simple prompts beat detailed descriptions

Meta's README explicitly says: "use lowercase noun-phrase/verb-phrase format."
We tried generating detailed music descriptions with Audio Flamingo and passing them
to SAM. It made things worse. The text encoder was trained on short labels.

Good: `"music"`, `"human voice"`, `"thunder"`
Bad: `"cinematic orchestral score with strings, brass, and timpani"`

### torch.autocast interferes with ODE solvers

We wrapped inference in `torch.autocast(dtype=bf16)`. This caused subtle numerical
issues in the torchdiffeq ODE solver (flow matching). The solver needs consistent
precision throughout. Don't use autocast with generative ODE-based models.

## Stereo Reconstruction

### You cannot mask the original without destroying quality

The intuitive approach (STFT mask from mono separation applied to stereo original)
sounds terrible. It causes "mushy" artifacts because:
- ISTFT reconstruction from masked complex spectrogram introduces phase
  discontinuities
- The mask is binary-ish, causing sudden jumps in the frequency domain
- High frequencies suffer most (more bins, more discontinuities)

### Build stereo from scratch, don't filter the original

The working approach: take the clean mono separation (which has correct phase from
the ODE solver) and place it in the stereo field using panning information extracted
from the original. The separated signal is the source of truth for content; the
original is only the source of truth for spatial position.

### Smoothing signals: bidirectional EMA, not box filters

Box filters (moving average via convolution) have two problems:
1. They shrink peaks toward the mean (a signal that goes from 0.7 to 0.3 will be
   smoothed to ~0.5, losing the extremes)
2. Zero padding at boundaries pulls the start/end toward 0.5

Bidirectional EMA (forward pass + backward pass, averaged):
- Preserves peaks (exponential decay doesn't shift the center)
- No boundary artifacts (starts from the actual first/last value)
- Smooth, natural-feeling transitions

### Always reference the ORIGINAL for stereo, never intermediates

Residuals are mono and have compounding errors from the ODE solver. If you use a
residual as the stereo reference for the next separation, you're building on
corrupted data. Always go back to the original stereo mix.

## Pipeline Design

### Gate the target, not just the residual

In iterative separation (the SFX loop), we were only checking if the RESIDUAL had
sound before continuing. This meant the loop kept running even when every extraction
produced empty targets — because the residual always has sound (it's the input minus
nothing useful).

The fix: sound-gate the TARGET immediately after separation. If empty, stop.

### When a separation fails, discard both target AND residual

If the target is empty (separation produced nothing), the residual is also unreliable
(it's the input + ODE solver noise). Don't continue the pipeline from a corrupted
residual. Fall back to the input that was fed into the failed separation.

### Conditional branching saves time and prevents errors

If Audio Flamingo says there's no music, don't run music separation. It will just
produce garbage (the model tries to extract "music" from audio with no music,
generating noise). Same for voice.

## Development Process

### Always have a known-good reference

We compared our output against Meta's official demo output for the same file.
This made quality issues immediately obvious. Without a reference, you're guessing
whether the output is "good enough."

### Test model changes in isolation with a standalone script

The `test_sam.py` approach was essential: load the model, run 4 variants (fp32
vanilla, fp32 fast, bf16, bf16+hacks), compare results. This isolated the exact
configuration that caused quality degradation (bf16 breaking CLAP).

Don't debug quality issues through the full pipeline. Strip away all layers and
test the model directly.

### Kill GPU processes before testing

Multiple leaked uvicorn workers can fill 80GB VRAM. Always check `nvidia-smi` before
running tests and kill orphaned processes. We hit OOM multiple times because old
processes were still holding GPU memory.

### SSH/SCP bulk transfer issues

Some networks (ISP or AWS security groups) kill SSH connections during bulk data
transfer. Workarounds:
- Serve files via HTTP through port forwards
- Use the pipeline's static file serving at `/files/`
- If file is already on the machine from a previous pipeline run, use that

### The stochastic nature of generative models

SAM-Audio's ODE solver starts from random noise. Every run produces different output.
You can't compare bit-for-bit against a reference. Compare quality characteristics:
- Is the separation clean? (no spillover)
- Is the right content in the right track?
- Is the volume reasonable?
- Is the stereo image natural?

## GPU Memory Budget (H100 80GB)

| Component | VRAM | Notes |
|-----------|------|-------|
| SAM-Audio fp32 (loaded) | 33 GB | Non-negotiable for quality |
| Audio Flamingo bf16 | 16 GB | Can use bf16 fine |
| SAM peak (reranking=4) | +25 GB | During inference only |
| SAM peak (reranking=8) | +42 GB | Doesn't fit with Flamingo |
| Total at rest | 49 GB | Both loaded |
| Total peak (reranking=4) | 74 GB | Tight but works |

If you need reranking=8: kill Audio Flamingo, run SAM, restart Flamingo after.

## File Locations on EC2

- Code: `/home/ubuntu/qlabeler/`
- Venvs: `/home/ubuntu/venvs/{audio-flamingo-next,sam-audio-large,pipeline}/`
- Outputs: `/home/ubuntu/outputs/`
- Logs: `/home/ubuntu/logs/qlabeler/`
- HF cache: `/home/ubuntu/.cache/huggingface/`
- SQLite DB: `/home/ubuntu/pipeline.sqlite3`

## Service Management

```bash
# Start all (after ensuring GPU is free)
ROOT_DIR=/home/ubuntu/qlabeler PYTHON_BIN=python3.13 /home/ubuntu/qlabeler/scripts/setup_model_apis.sh start

# Check what's using GPU
nvidia-smi

# Kill everything
kill $(ps aux | grep uvicorn | grep -v grep | awk '{print $2}')

# Check health
curl http://127.0.0.1:8000/healthz  # pipeline
curl http://127.0.0.1:8001/healthz  # flamingo
curl http://127.0.0.1:8002/healthz  # sam

# Load models
curl -X POST http://127.0.0.1:8001/load
curl -X POST http://127.0.0.1:8002/load
```
