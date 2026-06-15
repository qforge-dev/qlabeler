#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
  set +a
fi

if [[ -d /workspace ]]; then
  DEFAULT_WORKSPACE_DIR="/workspace"
else
  DEFAULT_WORKSPACE_DIR="$ROOT_DIR"
fi

WORKSPACE_DIR="${WORKSPACE_DIR:-$DEFAULT_WORKSPACE_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE_DIR/outputs}"
VENV_DIR="${VENV_DIR:-$WORKSPACE_DIR/venvs}"
LOG_DIR="${LOG_DIR:-$WORKSPACE_DIR/logs/qlabeler}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-$WORKSPACE_DIR/.cache/huggingface}}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

AFNEXT_PORT="${AFNEXT_PORT:-8001}"
SAM_AUDIO_PORT="${SAM_AUDIO_PORT:-8002}"
PIPELINE_PORT="${PIPELINE_PORT:-8000}"
AFNEXT_MODEL_ID="${AFNEXT_MODEL_ID:-nvidia/audio-flamingo-next-think-hf}"
SAM_AUDIO_MODEL_ID="${SAM_AUDIO_MODEL_ID:-facebook/sam-audio-large}"
PIPELINE_BACKEND="${PIPELINE_BACKEND:-real}"
PIPELINE_STORAGE_BACKEND="${PIPELINE_STORAGE_BACKEND:-local}"
AFNEXT_ENDPOINT="${AFNEXT_ENDPOINT:-http://127.0.0.1:${AFNEXT_PORT}/v1/audio-flamingo/ask}"
SAM_AUDIO_ENDPOINT="${SAM_AUDIO_ENDPOINT:-http://127.0.0.1:${SAM_AUDIO_PORT}/v1/sam-audio/separate}"
S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-qlabeler}"
S3_REGION="${S3_REGION:-${AWS_REGION:-}}"
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-}"
S3_PUBLIC_BASE_URL="${S3_PUBLIC_BASE_URL:-}"
S3_PRESIGN_SECONDS="${S3_PRESIGN_SECONDS:-0}"

PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
LOAD_MODELS="${LOAD_MODELS:-${RUNPOD_LOAD_MODELS:-1}}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

AFNEXT_VENV="$VENV_DIR/audio-flamingo-next"
SAM_AUDIO_VENV="$VENV_DIR/sam-audio-large"
PIPELINE_VENV="$VENV_DIR/pipeline"

log() {
  printf '\n== %s ==\n' "$*"
}

usage() {
  cat <<EOF
Usage:
  $0 [bootstrap|install|start|restart|stop|status|load|doctor|logs] [service]

Commands:
  bootstrap   Install system packages, create venvs, start all APIs. Default.
  install     Install system packages and Python dependencies only.
  start       Start all FastAPI services from existing venvs.
  restart     Stop, then start all services.
  stop        Stop services started by this script.
  status      Show health/readiness and local process ids.
  load        Call /load on both services to load model weights.
  doctor      Check common machine prerequisites.
  logs        Tail logs. Optional service: pipeline, audio-flamingo-next, or sam-audio-large.

Environment:
  ROOT_DIR=$ROOT_DIR
  WORKSPACE_DIR=$WORKSPACE_DIR
  VENV_DIR=$VENV_DIR
  OUTPUT_DIR=$OUTPUT_DIR
  LOG_DIR=$LOG_DIR
  HF_CACHE_DIR=$HF_CACHE_DIR
  PYTHON_BIN=$PYTHON_BIN
  AFNEXT_PORT=$AFNEXT_PORT
  SAM_AUDIO_PORT=$SAM_AUDIO_PORT
  PIPELINE_PORT=$PIPELINE_PORT
  PIPELINE_BACKEND=$PIPELINE_BACKEND
  PIPELINE_STORAGE_BACKEND=$PIPELINE_STORAGE_BACKEND
  PYTORCH_INDEX_URL=$PYTORCH_INDEX_URL
  LOAD_MODELS=$LOAD_MODELS

  Put HF_TOKEN in $ROOT_DIR/.env or export it before running this script.
EOF
}

require_root_for_install() {
  if [[ "$(id -u)" != "0" ]]; then
    echo "Install/bootstrap needs root for apt packages. Re-run with sudo or as root." >&2
    exit 1
  fi
}

require_services_dir() {
  if [[ ! -d "$ROOT_DIR/services" ]]; then
    echo "Could not find services under ROOT_DIR=$ROOT_DIR" >&2
    exit 1
  fi
}

install_base_packages() {
  require_root_for_install
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "This installer expects an Ubuntu/Debian machine with apt-get." >&2
    exit 1
  fi

  log "Installing base packages"
  apt-get update
  apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    iproute2 \
    jq \
    libsndfile1 \
    procps \
    software-properties-common \
    python3 \
    python3-pip \
    python3-venv \
    unzip

  if [[ "$PYTHON_BIN" == "python3.11" ]] && ! command -v python3.11 >/dev/null 2>&1; then
    if ! apt-cache show python3.11-venv >/dev/null 2>&1; then
      add-apt-repository -y ppa:deadsnakes/ppa
      apt-get update
    fi
  fi

  if [[ "$PYTHON_BIN" == "python3.11" ]]; then
    apt-get install -y python3.11 python3.11-dev python3.11-venv
  fi
}

create_venv() {
  local path="$1"
  if [[ ! -x "$path/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$path"
  fi
  "$path/bin/python" -m pip install --upgrade pip wheel "setuptools<81"
}

install_torch() {
  local venv="$1"
  if "$venv/bin/python" - <<PY >/dev/null 2>&1
import torch
import torchaudio
if "$REQUIRE_CUDA" == "1" and not torch.cuda.is_available():
    raise SystemExit(1)
print(torch.__version__, torchaudio.__version__)
PY
  then
    "$venv/bin/python" - <<'PY'
import torch
print(f"torch {torch.__version__}, cuda_available={torch.cuda.is_available()}")
PY
    return
  fi

  "$venv/bin/python" -m pip install --upgrade \
    torch \
    torchaudio \
    --index-url "$PYTORCH_INDEX_URL"
}

install_audioop_lts_if_needed() {
  local venv="$1"
  if "$venv/bin/python" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 13) else 1)
PY
  then
    "$venv/bin/python" -m pip install --upgrade audioop-lts
  fi
}

patch_sam_audio_audio_loader() {
  local venv="$1"
  "$venv/bin/python" - <<'PY'
from pathlib import Path
import site

site_paths = [Path(base) for base in site.getsitepackages()]

for path in [base / "core" / "audio_visual_encoder" / "transforms.py" for base in site_paths]:
    if path.exists():
        break
else:
    raise SystemExit("Could not find core/audio_visual_encoder/transforms.py")

text = path.read_text()
text = text.replace(
    "from torchcodec.decoders import AudioDecoder, VideoDecoder\n",
    "from torchcodec.decoders import VideoDecoder\nimport torchaudio\n",
)
text = text.replace(
    """    def _load_audio(self, path: str):
        ad = AudioDecoder(path, sample_rate=self.sampling_rate, num_channels=1)
        return ad.get_all_samples().data
""",
    """    def _load_audio(self, path: str):
        wav, sample_rate = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sample_rate != self.sampling_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sampling_rate)
        return wav
""",
)
path.write_text(text)
print(f"patched {path}")

for path in [base / "sam_audio" / "processor.py" for base in site_paths]:
    if path.exists():
        break
else:
    raise SystemExit("Could not find sam_audio/processor.py")

text = path.read_text()
text = text.replace(
    "from torchcodec.decoders import AudioDecoder, VideoDecoder\n",
    "from torchcodec.decoders import VideoDecoder\n",
)
path.write_text(text)
print(f"patched {path}")
PY
}

install_audio_flamingo_env() {
  log "Installing Audio Flamingo Next environment"
  create_venv "$AFNEXT_VENV"
  install_torch "$AFNEXT_VENV"
  "$AFNEXT_VENV/bin/python" -m pip uninstall -y torchvision || true
  "$AFNEXT_VENV/bin/python" -m pip install --upgrade \
    accelerate \
    fastapi \
    hf_transfer \
    librosa \
    pydub \
    safetensors \
    soundfile \
    transformers \
    "uvicorn[standard]"
  install_audioop_lts_if_needed "$AFNEXT_VENV"
}

install_sam_audio_env() {
  log "Installing SAM-Audio Large environment"
  create_venv "$SAM_AUDIO_VENV"
  install_torch "$SAM_AUDIO_VENV"
  # xformers needs torch at build time; install with --no-build-isolation
  # so it finds the already-installed torch in the venv.
  "$SAM_AUDIO_VENV/bin/python" -m pip install --upgrade --no-build-isolation \
    "xformers>=0.0.28" || true
  "$SAM_AUDIO_VENV/bin/python" -m pip uninstall -y \
    tensorflow tensorflow-cpu tensorflow-text tensorflow-datasets \
    tensorflow-metadata tf-keras keras jax jaxlib || true
  "$SAM_AUDIO_VENV/bin/python" -m pip install --upgrade --no-warn-conflicts \
    "sam_audio @ git+https://github.com/facebookresearch/sam-audio.git"
  "$SAM_AUDIO_VENV/bin/python" -m pip install --force-reinstall --no-deps \
    --index-url "$PYTORCH_INDEX_URL" \
    "torchcodec>=0.2,<0.3"
  "$SAM_AUDIO_VENV/bin/python" -m pip install --upgrade "nvidia-npp-cu12"
  patch_sam_audio_audio_loader "$SAM_AUDIO_VENV"
  "$SAM_AUDIO_VENV/bin/python" -m pip install --upgrade --no-warn-conflicts \
    fastapi \
    "transformers>=4.54,<5" \
    "huggingface_hub>=0.34,<1.0" \
    hf_transfer \
    pydub \
    "uvicorn[standard]"
  install_audioop_lts_if_needed "$SAM_AUDIO_VENV"
}

install_pipeline_env() {
  log "Installing pipeline environment"
  create_venv "$PIPELINE_VENV"
  "$PIPELINE_VENV/bin/python" -m pip install --upgrade \
    fastapi \
    boto3 \
    pydub \
    python-multipart \
    "uvicorn[standard]"
  install_audioop_lts_if_needed "$PIPELINE_VENV"
}

install_all() {
  require_services_dir
  install_base_packages
  mkdir -p "$VENV_DIR" "$LOG_DIR" "$OUTPUT_DIR"
  install_audio_flamingo_env
  install_sam_audio_env
  install_pipeline_env
}

pid_file_for() {
  printf '%s/%s.pid' "$LOG_DIR" "$1"
}

service_pid() {
  local name="$1"
  local pid_file
  pid_file="$(pid_file_for "$name")"
  if [[ ! -f "$pid_file" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$pid_file")"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    printf '%s' "$pid"
    return 0
  fi
  return 1
}

stop_service() {
  local name="$1"
  local pid
  local pid_file
  pid_file="$(pid_file_for "$name")"
  if pid="$(service_pid "$name")"; then
    log "Stopping $name ($pid)"
    kill "$pid" || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        break
      fi
      sleep 0.5
    done
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" || true
    fi
  fi
  rm -f "$pid_file"
}

stop_all() {
  stop_service pipeline
  stop_service audio-flamingo-next
  stop_service sam-audio-large
}

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "( sport = :$port )" | tail -n +2 | grep -q .
  else
    return 1
  fi
}

start_service() {
  local name="$1"
  local venv="$2"
  local module="$3"
  local port="$4"
  shift 4

  if [[ ! -x "$venv/bin/uvicorn" ]]; then
    echo "$name venv is missing uvicorn: $venv" >&2
    echo "Run: $0 install" >&2
    exit 1
  fi

  stop_service "$name"
  if port_in_use "$port"; then
    echo "Port $port is already listening; refusing to start $name over it." >&2
    exit 1
  fi

  log "Starting $name on port $port"
  mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$HF_CACHE_DIR"
  local cuda_lib_path=""
  local runtime_library_path="${LD_LIBRARY_PATH:-}"
  cuda_lib_path="$(find "$venv/lib" -path '*/site-packages/nvidia/*/lib' -type d 2>/dev/null | paste -sd: - || true)"
  if [[ -n "$cuda_lib_path" && -n "$runtime_library_path" ]]; then
    runtime_library_path="$cuda_lib_path:$runtime_library_path"
  elif [[ -n "$cuda_lib_path" ]]; then
    runtime_library_path="$cuda_lib_path"
  fi

  (
    cd "$ROOT_DIR"
    nohup env \
      LD_LIBRARY_PATH="$runtime_library_path" \
      PYTHONPATH="$ROOT_DIR" \
      HF_HOME="$HF_CACHE_DIR" \
      HF_HUB_DISABLE_XET=1 \
      HF_HUB_ENABLE_HF_TRANSFER=1 \
      WORKSPACE_DIR="$WORKSPACE_DIR" \
      OUTPUT_DIR="$OUTPUT_DIR" \
      TOKENIZERS_PARALLELISM=false \
      USE_TF=0 \
      USE_FLAX=0 \
      TRANSFORMERS_NO_TF=1 \
      TRANSFORMERS_NO_FLAX=1 \
      "$@" \
      "$venv/bin/uvicorn" "$module:app" --host 0.0.0.0 --port "$port" \
      >"$LOG_DIR/$name.log" 2>&1 &
    echo $! >"$(pid_file_for "$name")"
  )
}

start_all() {
  require_services_dir
  start_service \
    audio-flamingo-next \
    "$AFNEXT_VENV" \
    services.audio_flamingo_next_api \
    "$AFNEXT_PORT" \
    AFNEXT_MODEL_ID="$AFNEXT_MODEL_ID"

  start_service \
    sam-audio-large \
    "$SAM_AUDIO_VENV" \
    services.sam_audio_large_api \
    "$SAM_AUDIO_PORT" \
    SAM_AUDIO_MODEL_ID="$SAM_AUDIO_MODEL_ID"

  wait_for_health audio-flamingo-next "$AFNEXT_PORT"
  wait_for_health sam-audio-large "$SAM_AUDIO_PORT"
  if [[ "$LOAD_MODELS" == "1" ]]; then
    load_all
  fi
  start_service \
    pipeline \
    "$PIPELINE_VENV" \
    services.pipeline_app \
    "$PIPELINE_PORT" \
    PIPELINE_BACKEND="$PIPELINE_BACKEND" \
    PIPELINE_STORAGE_BACKEND="$PIPELINE_STORAGE_BACKEND" \
    S3_BUCKET="$S3_BUCKET" \
    S3_PREFIX="$S3_PREFIX" \
    S3_REGION="$S3_REGION" \
    S3_ENDPOINT_URL="$S3_ENDPOINT_URL" \
    S3_PUBLIC_BASE_URL="$S3_PUBLIC_BASE_URL" \
    S3_PRESIGN_SECONDS="$S3_PRESIGN_SECONDS" \
    AFNEXT_ENDPOINT="$AFNEXT_ENDPOINT" \
    SAM_AUDIO_ENDPOINT="$SAM_AUDIO_ENDPOINT"

  wait_for_health pipeline "$PIPELINE_PORT"
  print_access_notes
}

wait_for_health() {
  local name="$1"
  local port="$2"
  local url="http://127.0.0.1:$port/healthz"

  log "Checking $name health"
  for _ in $(seq 1 60); do
    if curl -fsS "$url"; then
      printf '\n'
      return
    fi
    sleep 1
  done

  echo "$name did not become healthy. Last log lines:" >&2
  tail -120 "$LOG_DIR/$name.log" >&2 || true
  exit 1
}

load_service() {
  local name="$1"
  local port="$2"
  log "Loading $name model"
  if ! curl -fsS -X POST "http://127.0.0.1:$port/load"; then
    printf '\n%s\n' "$name model load failed. Last log lines:" >&2
    tail -160 "$LOG_DIR/$name.log" >&2 || true
    exit 1
  fi
  printf '\n'
}

load_all() {
  load_service audio-flamingo-next "$AFNEXT_PORT"
  load_service sam-audio-large "$SAM_AUDIO_PORT"
}

show_service_status() {
  local name="$1"
  local port="$2"
  local pid="not running"
  if service_pid "$name" >/dev/null; then
    pid="$(service_pid "$name")"
  fi

  printf '\n%s\n' "$name"
  printf '  pid: %s\n' "$pid"
  printf '  health: '
  curl -fsS "http://127.0.0.1:$port/healthz" || true
  printf '\n  ready:  '
  curl -fsS "http://127.0.0.1:$port/readyz" || true
  printf '\n'
}

show_status() {
  show_service_status pipeline "$PIPELINE_PORT"
  show_service_status audio-flamingo-next "$AFNEXT_PORT"
  show_service_status sam-audio-large "$SAM_AUDIO_PORT"
}

show_logs() {
  local name="${1:-}"
  case "$name" in
    ""|"all")
      tail -n 120 "$LOG_DIR/pipeline.log" "$LOG_DIR/audio-flamingo-next.log" "$LOG_DIR/sam-audio-large.log"
      ;;
    "pipeline"|"audio-flamingo-next"|"sam-audio-large")
      tail -n 160 "$LOG_DIR/$name.log"
      ;;
    *)
      echo "Unknown service for logs: $name" >&2
      exit 1
      ;;
  esac
}

doctor() {
  log "Machine"
  uname -a
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "nvidia-smi not found"
  fi

  log "Paths"
  printf 'ROOT_DIR=%s\nWORKSPACE_DIR=%s\nVENV_DIR=%s\nOUTPUT_DIR=%s\nLOG_DIR=%s\nPIPELINE_BACKEND=%s\nPIPELINE_STORAGE_BACKEND=%s\n' \
    "$ROOT_DIR" "$WORKSPACE_DIR" "$VENV_DIR" "$OUTPUT_DIR" "$LOG_DIR" "$PIPELINE_BACKEND" "$PIPELINE_STORAGE_BACKEND"

  log "Python"
  command -v "$PYTHON_BIN" || true
  "$PYTHON_BIN" --version || true

  log "FFmpeg"
  command -v ffmpeg || true
  ffmpeg -version 2>/dev/null | head -n 1 || true

  log "Hugging Face token"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is set"
  else
    echo "HF_TOKEN is not set. facebook/sam-audio-large requires gated model access."
  fi
}

print_access_notes() {
  log "Services running"
  cat <<EOF
Pipeline Dashboard:
  dashboard: http://127.0.0.1:${PIPELINE_PORT}/
  api:       http://127.0.0.1:${PIPELINE_PORT}/api/dashboard
  backend:   ${PIPELINE_BACKEND}
  storage:   ${PIPELINE_STORAGE_BACKEND}
  log:       ${LOG_DIR}/pipeline.log

Audio Flamingo Next:
  health: http://127.0.0.1:${AFNEXT_PORT}/healthz
  ask:    POST http://127.0.0.1:${AFNEXT_PORT}/v1/audio-flamingo/ask
  log:    ${LOG_DIR}/audio-flamingo-next.log

SAM-Audio Large:
  health: http://127.0.0.1:${SAM_AUDIO_PORT}/healthz
  split:  POST http://127.0.0.1:${SAM_AUDIO_PORT}/v1/sam-audio/separate
  log:    ${LOG_DIR}/sam-audio-large.log

Useful commands:
  $0 status
  $0 load
  $0 logs pipeline
  $0 logs audio-flamingo-next
  $0 logs sam-audio-large
EOF
}

main() {
  local action="${1:-bootstrap}"
  local service="${2:-}"

  case "$action" in
    "-h"|"--help"|"help")
      usage
      ;;
    "bootstrap")
      install_all
      start_all
      ;;
    "install")
      install_all
      ;;
    "start")
      start_all
      ;;
    "restart")
      stop_all
      start_all
      ;;
    "stop")
      stop_all
      ;;
    "status")
      show_status
      ;;
    "load")
      load_all
      ;;
    "doctor")
      doctor
      ;;
    "logs")
      show_logs "$service"
      ;;
    *)
      echo "Unknown command: $action" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
