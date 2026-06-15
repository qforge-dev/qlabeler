# Pipeline Architecture

## Separation Chain

The pipeline peels layers from audio one at a time:

```
Source (stereo) 
  --> SAM "music" 
      target: music (mono) --> stereo transfer from OG --> final music stem
      residual: everything-except-music (mono)
        --> SAM "human voice"
            target: voice (mono) --> stereo transfer from OG --> final voice stem
            residual: everything-except-music-and-voice (mono)
              --> SAM "[sfx description]"
                  target: sfx (mono) --> stereo transfer from OG --> final sfx stem
                  residual: remaining (mono) --> loop or stop
```

### Key Principle: Always Peel from Mono Residual, Always Stereo from OG

- **Residuals stay mono**. They are inputs to the next separation stage. SAM downmixes
  to mono internally anyway, so making them stereo is wasted work.
- **Stereo transfer always references the ORIGINAL source audio** (the chunk), never
  the residual. Errors compound if you use an intermediate.
- **The pipeline passes `source_audio_path` through** so downstream separations know
  where the original stereo lives.

## Chunking

- Source audio is split into 30s chunks with 5s overlap
- **Minimum chunk length: 10 seconds** — shorter chunks are discarded (last chunk
  of a file is often too short to separate meaningfully)
- Chunks are exported as WAV (preserving original sample rate and channels)

## Scene Description (Audio Flamingo)

Before any separation, Audio Flamingo analyzes the full source audio and returns:

```
HAS_MUSIC: true/false
HAS_VOICES: true/false
MUSIC_DESCRIPTION: <short phrase about the music style>
DESCRIPTION: <one sentence about the scene>
```

This is used for conditional branching — if no music, skip music separation entirely.
If no voices, skip voice separation.

Note: the `MUSIC_DESCRIPTION` is stored for reference but NOT used as the SAM prompt.
SAM gets simple NP/VP prompts ("music", "human voice") regardless of what Flamingo says.

## Sound Gate

The sound gate determines if a separated track has meaningful audio content.

### Thresholds (stricter than defaults)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| min_dbfs | -40 dB | Overall RMS must exceed this |
| min_peak_dbfs | -35 dB | Peak must exceed this |
| min_active_ms | 1000ms | At least 1s of active audio required |
| min_active_ratio | 5% | At least 5% of duration must be active |
| Logic | AND | ALL conditions must pass (not OR) |

A window is "active" if its RMS >= min_dbfs AND its peak >= min_peak_dbfs.

### Gate Behavior

- **Chunk-level gate**: If chunk fails, skip entirely (STAGE_SKIPPED_SILENT)
- **Music target gate**: If music extraction is empty, skip music separation and
  use the original chunk audio for voice separation (not the residual)
- **Voice target gate**: If voice extraction is empty, use the parent audio for SFX
  listing (not the empty residual)
- **SFX target gate**: After each SFX extraction, immediately gate the target. If
  empty, stop the SFX loop. Don't even store the empty artifact.

### The "Fallback to Original" Pattern

When a separation produces an empty target, the residual is also unreliable (it's
just the input signal with noise from the ODE solver). Instead of continuing with
that corrupted residual, fall back to the input that was fed into the separation:

- Music target empty? Use original chunk audio for next stage.
- Voice target empty? Use the parent track (sfx+voice or original) for SFX.

This prevents compounding errors from garbage residuals.

## SFX Loop

After voice separation, Audio Flamingo is asked to list sound effects in the
remaining audio. For each identified SFX:

1. SAM separates it using the SFX label as prompt
2. The isolated target is immediately sound-gated
3. If the target is empty: **stop the loop** (model failed to extract anything useful)
4. If the target has sound: store it, then gate the residual
5. If residual has sound and iterations < max (8): loop back to step 1
6. If residual is empty: loop exhausted, done

The key fix here was gating the TARGET, not just the residual. Previously the loop
would continue 6-8 times producing empty target after empty target because the
residual always had sound (it was the original minus nothing).

## Conditional Branching

Based on the scene description:

```
has_music=true, has_voices=true:
  chunk -> music separation -> voice separation -> SFX loop (full pipeline)

has_music=false, has_voices=true:
  chunk -> voice separation -> SFX loop (skip music entirely)

has_music=true, has_voices=false:
  chunk -> music separation -> SFX loop (skip voice)

has_music=false, has_voices=false:
  chunk -> SFX loop (just extract effects)
```

## DAW-Style UI

The chunk detail view shows all stems as a multitrack DAW:
- One transport (play/pause/stop) controls all tracks simultaneously
- Per-track mute/unmute buttons
- Source track starts unmuted, stems start muted
- Click on waveform to seek
- Synced playhead across all tracks
- Download button per track
- Only shows targets (extracted), not residuals
- Shows the SAM prompt used for each separation
