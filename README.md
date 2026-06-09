# Audio Model APIs

This repo contains the original working notebooks plus production-style FastAPI
wrappers for two audio models:

- `nvidia/audio-flamingo-next-think-hf` for audio question answering.
- `facebook/sam-audio-large` for separating one described sound from a short
  audio clip.

The two services run in separate Python virtual environments. This avoids the
dependency conflicts that showed up when both model stacks were installed into
one notebook runtime.

## Fresh Machine Setup

Use this path on a fresh RunPod or Ubuntu GPU machine.

1. Clone the repo and enter it:

   ```bash
   cd /workspace
   git clone <this-repo-url> qlabeler-simpler
   cd /workspace/qlabeler-simpler
   ```

2. Create `.env` and add your Hugging Face token:

   ```bash
   cp .env.example .env
   nano .env
   ```

   `facebook/sam-audio-large` is gated, so `HF_TOKEN` must belong to a Hugging
   Face account with access to that model.

3. Bootstrap everything:

   ```bash
   ./scripts/setup_model_apis.sh bootstrap
   ```

That single command installs OS packages, creates two isolated virtual
environments, installs model dependencies, patches the SAM-Audio loader to use
`torchaudio`, and starts both FastAPI services.

To download and load model weights during setup, set this in `.env` or inline:

```bash
LOAD_MODELS=1 ./scripts/setup_model_apis.sh bootstrap
```

## Service Commands

The setup script is also the service control script:

```bash
./scripts/setup_model_apis.sh doctor
./scripts/setup_model_apis.sh status
./scripts/setup_model_apis.sh load
./scripts/setup_model_apis.sh logs audio-flamingo-next
./scripts/setup_model_apis.sh logs sam-audio-large
./scripts/setup_model_apis.sh restart
./scripts/setup_model_apis.sh stop
```

Default paths:

```text
repo:       /workspace/qlabeler-simpler
venvs:      /workspace/venvs
outputs:    /workspace/outputs
logs:       /workspace/logs/qlabeler-simpler
HF cache:   /workspace/.cache/huggingface
```

Default ports:

```text
Audio Flamingo: http://127.0.0.1:8001
SAM-Audio:      http://127.0.0.1:8002
```

All paths and ports can be changed in `.env`; see [.env.example](.env.example).

## Health And Readiness

Health checks confirm that the API process is running:

```bash
curl http://127.0.0.1:8001/healthz
curl http://127.0.0.1:8002/healthz
```

Readiness shows whether model weights are loaded:

```bash
curl http://127.0.0.1:8001/readyz
curl http://127.0.0.1:8002/readyz
```

Models lazy-load on first inference. To preload explicitly:

```bash
./scripts/setup_model_apis.sh load
```

## Audio Flamingo API

Endpoint:

```text
POST /v1/audio-flamingo/ask
POST /ask
```

Example:

```bash
curl -X POST http://127.0.0.1:8001/v1/audio-flamingo/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "audio_path": "/workspace/data/chunk_001.mp3",
    "input": "List the audible sound sources. Then suggest one concise SAM-Audio target prompt.",
    "max_new_tokens": 256,
    "repetition_penalty": 1.2
  }'
```

Response:

```json
{
  "model_id": "nvidia/audio-flamingo-next-think-hf",
  "audio_path": "/workspace/data/chunk_001.mp3",
  "prompt": "List the audible sound sources. Then suggest one concise SAM-Audio target prompt.",
  "text": "SOUNDS: horse hooves, cinematic strings\nSAM_PROMPT: horse hooves"
}
```

Accepted audio fields: `audio_path`, `file_path`, `file`, or `audio_url`.
Accepted prompt fields: `prompt`, `input`, or `question`.

Only local filesystem paths and `file://` URLs are supported today.

## SAM-Audio API

Endpoint:

```text
POST /v1/sam-audio/separate
POST /separate
```

Pass exactly one target sound description. For example, use `horse hooves`, not
`horse hooves from background strings`.

Example:

```bash
curl -X POST http://127.0.0.1:8002/v1/sam-audio/separate \
  -H 'Content-Type: application/json' \
  -d '{
    "audio_path": "/workspace/data/chunk_001.mp3",
    "input": "horse hooves",
    "output_prefix": "chunk_001_horse_hooves",
    "max_audio_seconds": 35,
    "predict_spans": false,
    "reranking_candidates": 1
  }'
```

Response includes target, residual, and zip refs:

```json
{
  "model_id": "facebook/sam-audio-large",
  "request_id": "c0ffee...",
  "audio_path": "/workspace/data/chunk_001.mp3",
  "description": "horse hooves",
  "duration_seconds": 30.0,
  "sample_rate": 48000,
  "target": {
    "wav": {"path": "/workspace/outputs/sam-audio-large/.../chunk_001_horse_hooves_target.wav", "url": "/files/sam-audio-large/.../chunk_001_horse_hooves_target.wav"},
    "mp3": {"path": "/workspace/outputs/sam-audio-large/.../chunk_001_horse_hooves_target.mp3", "url": "/files/sam-audio-large/.../chunk_001_horse_hooves_target.mp3"}
  },
  "residual": {
    "wav": {"path": "/workspace/outputs/sam-audio-large/.../chunk_001_horse_hooves_residual.wav", "url": "/files/sam-audio-large/.../chunk_001_horse_hooves_residual.wav"},
    "mp3": {"path": "/workspace/outputs/sam-audio-large/.../chunk_001_horse_hooves_residual.mp3", "url": "/files/sam-audio-large/.../chunk_001_horse_hooves_residual.mp3"}
  },
  "zip": {"path": "/workspace/outputs/sam-audio-large/.../chunk_001_horse_hooves_outputs.zip", "url": "/files/sam-audio-large/.../chunk_001_horse_hooves_outputs.zip"}
}
```

Accepted target fields: `prompt`, `input`, or `description`.

SAM-Audio is configured for short clips by default. Split long files into
30-second chunks before calling the endpoint.

## Example Pipeline Call

Ask Audio Flamingo for a one-sound target prompt:

```bash
curl -sS -X POST http://127.0.0.1:8001/v1/audio-flamingo/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "audio_path": "/workspace/data/example_chunk.mp3",
    "input": "Identify audible sources and return exactly two lines: SOUNDS: <sources>; SAM_PROMPT: <one target sound only>.",
    "max_new_tokens": 256
  }'
```

Then pass only the `SAM_PROMPT` value to SAM-Audio:

```bash
curl -sS -X POST http://127.0.0.1:8002/v1/sam-audio/separate \
  -H 'Content-Type: application/json' \
  -d '{
    "audio_path": "/workspace/data/example_chunk.mp3",
    "input": "horse hooves",
    "output_prefix": "example_chunk_horse_hooves"
  }'
```

## Moving Files From RunPod

RunPod's SSH proxy may not support SCP or port forwarding on all pods. The
reliable fallback is `runpodctl send/receive`.

From the pod:

```bash
runpodctl send --code qlabeler-output /workspace/outputs/sam-audio-large/<request_id>/<prefix>_outputs.zip
```

RunPod may append a suffix to the code. Use the exact code it prints, then run
this on your local machine:

```bash
runpodctl receive qlabeler-output-10
```

Install local `runpodctl` on macOS:

```bash
brew install runpod/runpodctl/runpodctl
```

## Notebook References

The notebooks remain useful as reference workflows and for experiments:

- `audio_flamingo_next_mp3_qa.ipynb`
- `sam_audio_large_mp3_separation.ipynb`
- `split_mp3_30s_overlap.ipynb`

The service installer follows the notebook setup that worked on the tested A100
RunPod machine, including isolated dependencies and the SAM-Audio audio loader
patch needed for MP3/WAV handling.
