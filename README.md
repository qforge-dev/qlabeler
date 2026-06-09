# Audio Analysis Notebooks And Docker Images

This repo contains three small audio notebooks and two GPU Docker images:

- `audio_flamingo_next_mp3_qa.ipynb`: ask `nvidia/audio-flamingo-next-think-hf`
  questions about uploaded MP3/audio files.
- `split_mp3_30s_overlap.ipynb`: split one MP3 into 30-second chunks with
  5 seconds of overlap, then zip the chunks.
- `sam_audio_large_mp3_separation.ipynb`: isolate a described sound from an MP3
  with `facebook/sam-audio-large`.

The Docker setup is intended for a rented H100 machine where you SSH in and run
`docker compose`.

## Docker On H100

The repo defines two separate images:

- `qlabeler/audio-flamingo-next-think:cuda12.8`
- `qlabeler/sam-audio-large:cuda12.8`

Both images use a PyTorch CUDA 12.8 base image and start JupyterLab. The SAM
image is a clean dedicated environment, so it avoids the TensorFlow/JAX/protobuf
conflicts that showed up in hosted notebooks.

Host prerequisites:

- NVIDIA driver installed on the rented machine
- Docker and Docker Compose v2
- NVIDIA Container Toolkit configured
- Hugging Face access token with access to gated models, especially
  `facebook/sam-audio-large`

On the remote machine:

```bash
cp .env.example .env
nano .env
```

Set at least:

```text
HF_TOKEN=hf_your_token_here
JUPYTER_TOKEN=some-long-token
```

Build both images:

```bash
docker compose build audio-flamingo sam-audio
```

Start Audio Flamingo Next Think:

```bash
docker compose up audio-flamingo
```

Start SAM-Audio Large:

```bash
docker compose up sam-audio
```

You can also start both services, but each notebook can load a large model, so
it is usually cleaner to run only the one you are actively using:

```bash
docker compose up
```

Ports:

- Audio Flamingo JupyterLab: `http://localhost:8888`
- SAM-Audio JupyterLab: `http://localhost:8889`

From your laptop, use an SSH tunnel:

```bash
ssh -L 8888:localhost:8888 -L 8889:localhost:8889 user@your-h100-host
```

Then open the local URLs above and use `JUPYTER_TOKEN` from `.env`.

Shared paths inside both containers:

```text
/workspace
/workspace/data
/workspace/outputs
/workspace/.cache/huggingface
```

Put input audio in `data/` and write outputs to `outputs/`. Hugging Face model
downloads are stored in the shared `hf-cache` Docker volume.

When using the Docker images, skip notebook setup/restart cells. The
dependencies are already baked into the image. Start from login/import/model
loading cells.

To verify GPU visibility inside either container:

```bash
docker compose run --rm audio-flamingo nvidia-smi
docker compose run --rm sam-audio nvidia-smi
```

If a PyTorch base image tag ever disappears, override it during build:

```bash
docker compose build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime \
  audio-flamingo sam-audio
```

## Q&A Notebook

Open:

```text
audio_flamingo_next_mp3_qa.ipynb
```

The notebook uses only the Think checkpoint:

```text
nvidia/audio-flamingo-next-think-hf
```

The Q&A notebook is focused on sound analysis: SFX, environment sounds,
ambience, music, voices, and timestamped sound events.

## Recommended Runtime

Use a GPU runtime if possible.

- Practical minimum: 24 GB VRAM for shorter clips.
- Safer target: 48 GB VRAM, especially for longer audio.
- CPU-only inference is not recommended.

The model is an 8B BF16 checkpoint. The first load downloads the model weights
and can take a while.

## How To Run Q&A

1. Run the setup cell.

   The setup cell installs the notebook dependencies:

   ```python
   %pip install -q --upgrade transformers accelerate librosa soundfile
   ```

   It intentionally does not upgrade `torch`, because Colab/Kaggle runtimes
   usually already have a matching PyTorch/CUDA stack.

2. Run the import/runtime cell.

   This selects CUDA, MPS, or CPU and sets:

   ```python
   MODEL_ID = "nvidia/audio-flamingo-next-think-hf"
   ```

3. Run the model loading cell.

   The notebook checks whether `AutoModel` resolves to a generative class. If
   the installed `transformers` maps `AutoModel` to a non-generative base class,
   it falls back to `AutoModelForSeq2SeqLM` so `model.generate(...)` works.

4. Upload or choose an audio file.

   - In Google Colab, the notebook opens a file picker.
   - In local Jupyter, paste a local MP3/audio path.

5. Edit the `## Question` cell.

   This cell contains the prompt. The default prompt asks the model to detect:

   - footsteps, movement, impacts, Foley
   - cars, engines, horns, brakes, sirens
   - wind, rain, thunder, water, insects
   - animal sounds
   - doors, keys, dishes, appliances, electronics
   - traffic beds, crowds, construction, machinery, alarms
   - music, singing, rhythm, score, source music
   - speech, narration, announcements, laughter, shouting

   It asks for a timestamped timeline with confidence and evidence.

6. Adjust generation settings if needed.

   ```python
   MAX_NEW_TOKENS = 4096
   REPETITION_PENALTY = 1.2
   ```

7. Run the final answer cell.

## MP3 Chunk Splitter Notebook

Open:

```text
split_mp3_30s_overlap.ipynb
```

This notebook takes one MP3 and writes overlapping chunks:

- chunk length: 30 seconds
- overlap: 5 seconds
- step between chunk starts: 25 seconds

For example, the chunks start at:

```text
00:00, 00:25, 00:50, 01:15, ...
```

Each chunk filename includes its index and timestamp range:

```text
input_chunk_0001_00h00m00s-00h00m30s.mp3
input_chunk_0002_00h00m25s-00h00m55s.mp3
```

The notebook saves chunks into:

```text
mp3_chunks/
```

and creates:

```text
mp3_chunks.zip
```

In Google Colab, the final cell downloads the zip. In local Jupyter, it prints
the zip path.

The splitter uses `pydub` and `ffmpeg`. If `ffmpeg` is missing and `apt-get` is
available, the notebook attempts to install it.

## SAM-Audio Large Separation Notebook

Open:

```text
sam_audio_large_mp3_separation.ipynb
```

This notebook uses `facebook/sam-audio-large` to separate one described target
sound from an MP3. It saves:

```text
sam_audio_outputs/target.wav
sam_audio_outputs/residual.wav
sam_audio_outputs/target.mp3
sam_audio_outputs/residual.mp3
sam_audio_outputs.zip
```

The model is gated on Hugging Face. Before loading it, request access to the
model repo and log in from the notebook.

The notebook loads SAM-Audio Large in audio-only CUDA BF16 mode. It deletes the
vision encoder before moving the model to GPU, which avoids spending VRAM on
video features you are not using:

```python
DEVICE = "cuda"
DTYPE = torch.bfloat16
model = SAMAudio.from_pretrained(MODEL_ID, proxies=None, resume_download=False)
model = disable_vision_encoder_for_audio_only(model)
model = model.to(DEVICE, DTYPE).eval()
```

Rankers and the span predictor are kept. The default settings still use:

```python
PREDICT_SPANS = False
RERANKING_CANDIDATES = 1
```

It wraps preprocessing and separation in:

```python
with torch.autocast(device_type=DEVICE, dtype=DTYPE):
    inputs = processor(...).to(DEVICE)
    result = model.separate(inputs)
```

This notebook raises early if CUDA is not available. It is configured for
30-second clips and does not do chunking; use the splitter notebook first for
longer MP3s.

The editable prompt cell uses:

```python
DESCRIPTION = "footsteps"
MAX_AUDIO_SECONDS = 35
PREDICT_SPANS = False
RERANKING_CANDIDATES = 1
```

Use short noun phrases or verb phrases for `DESCRIPTION`, such as `footsteps`,
`car engine`, `wind`, `dog barking`, `piano`, or `man speaking`.

`predict_spans=True` and higher reranking candidates can improve results, but
they increase latency and memory use.

SAM-Audio pulls a broad dependency tree. Use a fresh dedicated runtime for this
notebook and do not mix it with JAX, TensorFlow, or Google SDK work.

Some pip warnings are expected in hosted notebook images:

- SAM-Audio's codec stack uses `descript-audiotools`, which requires
  `protobuf<3.20`.
- Many Google/JAX/TensorFlow packages in hosted runtimes want newer protobuf.
- CLAP pins NumPy `<2`, while JAX/TensorFlow often require NumPy `>=2`.

For this notebook, prioritize SAM-Audio's audio stack. Do not upgrade protobuf
after installing SAM-Audio.

The setup cell uninstalls TensorFlow/JAX packages because this notebook does not
use them, and they can crash on import after SAM-Audio installs the protobuf
version required by its audio stack:

```text
ImportError: cannot import name 'runtime_version' from 'google.protobuf'
```

The import cell also sets `USE_TF=0` and `USE_FLAX=0` before importing
`sam_audio`.

The setup cell uses `--no-warn-conflicts` to suppress those known hosted-runtime
warnings. Actual install failures still show. To inspect dependency conflicts,
turn on `RUN_PIP_CHECK` in the optional diagnostics cell.

After the SAM-Audio setup cell, run the notebook's restart cell before importing
`sam_audio`. If you see:

```text
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

the runtime was not restarted after pip replaced NumPy. Restart the runtime, skip
the setup/restart cells, and continue from the dependency diagnostics or Hugging
Face login cell.

The notebook also pins the model-loading stack to the compatible API family:

```python
%pip install -q --upgrade --no-warn-conflicts "transformers>=4.54,<5" "huggingface_hub>=0.34,<1.0"
```

and loads SAM-Audio with:

```python
SAMAudio.from_pretrained(MODEL_ID, proxies=None, resume_download=False)
```

This avoids a compatibility error where SAM-Audio expects `proxies` and
`resume_download` but newer `huggingface_hub` mixin code does not pass them.

## Troubleshooting

If `torchvision::nms does not exist` appears, restart the runtime and rerun the
setup cell. The notebook uninstalls `torchvision` because a mismatched optional
`torchvision` build can break `transformers` imports.

If pip reports CUDA package conflicts after upgrading `torch`, start a fresh
runtime. The notebook no longer upgrades `torch`; it leaves the runtime's
existing PyTorch/CUDA stack alone.

If dtype errors appear during generation, rerun the latest import/config cell
and the `ask_audio` definition cell. The notebook uses CUDA autocast during
generation so BF16 model weights and the audio frontend can run together.
