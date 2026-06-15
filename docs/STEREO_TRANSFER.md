# Stereo Transfer

## The Problem

SAM-Audio produces mono separation output. Our source audio is stereo. We need to
reconstruct stereo from the mono separated signal without introducing artifacts,
spillover, or destroying the spatial image.

## What We Tried (and Why It Failed)

### 1. Simple Channel Duplication

Duplicate mono to both L/R channels. Result: technically "stereo" but sounds 100%
mono. No spatial image at all.

### 2. STFT Wiener Masking on Original

Compute a Wiener mask from the mono separation, apply it to the original stereo
signal's STFT per channel. The mask filters the original.

**Why it failed**: Operating on the original's phase and then reconstructing via
ISTFT introduces phase discontinuities between masked and unmasked regions. This
causes "mushy" artifacts, especially at high frequencies. The audio sounds like it's
underwater. The separation bleeds too because you're still using the original signal
(which contains everything).

### 3. Per-Bin Panning Transfer

For each time-frequency bin, compute the L/R ratio from the original and apply it to
the separated signal.

**Why it failed**: At any given frequency bin, the L/R ratio in the original mix
reflects ALL sources at that frequency, not just the target. Music panned left
influences the panning estimate at frequencies where voice also exists. Result:
massive spillover, sounds jumping around.

### 4. Per-Frame Panning with Box Filter Smoothing

Weight the per-bin panning by voice energy, then average across frequency to get one
pan value per frame. Smooth with a moving average (convolution with a box kernel).

**Why it failed**: The box kernel uses zero-padding at signal boundaries. This pulls
the first and last frames toward 0.5 (center), regardless of the actual content.
Result: the voice that should start panned left instead starts centered/right, then
jumps left.

## What Works: Bidirectional EMA Panning + Peak Volume Match

### The Algorithm

1. **Compute per-bin panning from the original stereo mix**:
   ```
   pan_L(freq, time) = |Orig_L(freq, time)| / (|Orig_L| + |Orig_R| + eps)
   ```

2. **Weight by dominant bins of the separated signal**:
   Only frequency bins where the separated signal is strong (>1% of frame peak)
   contribute to the panning estimate. This filters out bins where other sources
   dominate the L/R ratio.

3. **Compute per-frame weighted average panning**:
   ```
   frame_pan_L(time) = sum(pan_L * masked_energy, dim=freq) / sum(masked_energy, dim=freq)
   ```
   This gives one panning value per time frame, representing where the target source
   is positioned in the stereo field at that moment.

4. **Smooth with bidirectional exponential moving average (EMA)**:
   ```
   Forward:  ema_fwd[i] = alpha * raw[i] + (1-alpha) * ema_fwd[i-1]
   Backward: ema_bwd[i] = alpha * raw[i] + (1-alpha) * ema_bwd[i+1]
   Result:   smooth[i] = (ema_fwd[i] + ema_bwd[i]) / 2
   ```
   With `alpha = 0.03` (~700ms time constant). This:
   - Preserves peaks (unlike box filter which shrinks toward the mean)
   - Has no boundary artifacts (no zero padding)
   - Produces smooth, natural-sounding panning movement
   - Is bidirectional so there's no lag

5. **Peak-match the volume**:
   ```
   global_gain = og_peak / separated_peak
   ```
   A single scalar that brings the separated signal to the same peak level as the
   original mix. The separated signal's own dynamics (loud parts loud, quiet parts
   quiet) are preserved.

6. **Apply in STFT domain**:
   ```
   Output_L = STFT(separated) * global_gain * pan_L_smooth
   Output_R = STFT(separated) * global_gain * pan_R_smooth
   ```
   Then ISTFT back to time domain. The separated signal's phase is preserved because
   we're multiplying (scaling) the complex STFT, not replacing it.

### Why This Works

- **Content/separation quality**: comes from SAM's mono output (correct, no spillover)
- **Stereo position**: comes from the original mix's L/R balance at the target's
  frequency bins (correct spatial placement)
- **Volume**: peak-matched to original (correct loudness)
- **Phase**: from the separated signal's own ODE-generated output (coherent, no
  artifacts, no mushiness)
- **No spillover**: we never touch the original signal's content. We only borrow its
  spatial characteristics.

### Key Design Decisions

- **Always use the ORIGINAL source audio for panning reference**, even for downstream
  separations (voice from residual, SFX from voice residual). The residual is mono
  and has compounding errors. The original stereo mix is the only reliable source of
  spatial information.

- **Only apply stereo transfer to TARGETS (extracted tracks)**. Residuals stay mono
  because they're inputs to further separation stages. If you make a residual stereo,
  the next SAM call will downmix it to mono anyway, losing the stereo work.

- **STFT parameters**: n_fft=4096, hop=1024. Larger FFT gives better frequency
  resolution for the panning estimate. Hop of 1024 at 48kHz gives ~21ms per frame.

- **EMA alpha=0.03**: About 33 frames time constant = ~700ms. Fast enough to track
  real panning movements (a sound moving from left to center over 2-3 seconds), slow
  enough to not follow per-frame noise from interfering sources.

## STFT Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| n_fft | 4096 | Good frequency resolution for panning estimation |
| hop_length | 1024 | ~21ms per frame, smooth enough for EMA |
| window | Hann | Standard, good sidelobe suppression |
| alpha (EMA) | 0.03 | ~700ms time constant, natural movement |
