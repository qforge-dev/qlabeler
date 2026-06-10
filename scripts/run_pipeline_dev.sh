#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
  set +a
fi

PIPELINE_PORT="${PIPELINE_PORT:-8000}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$ROOT_DIR/.local/pipeline}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE_DIR/outputs}"
PIPELINE_DB_PATH="${PIPELINE_DB_PATH:-$WORKSPACE_DIR/pipeline.sqlite3}"
PIPELINE_DEV_VENV="${PIPELINE_DEV_VENV:-$ROOT_DIR/.venvs/pipeline-dev}"

log() {
  printf '\n== %s ==\n' "$*"
}

if [[ ! -x "$PIPELINE_DEV_VENV/bin/python" ]]; then
  log "Creating local pipeline venv"
  python3 -m venv "$PIPELINE_DEV_VENV"
fi

log "Installing local pipeline dependencies"
"$PIPELINE_DEV_VENV/bin/python" -m pip install --upgrade \
  pip \
  wheel \
  fastapi \
  audioop-lts \
  pydub \
  python-multipart \
  "uvicorn[standard]"

mkdir -p "$WORKSPACE_DIR" "$OUTPUT_DIR"

log "Starting mock pipeline dashboard"
cat <<EOF
Dashboard: http://127.0.0.1:${PIPELINE_PORT}/
Backend:   mock
DB:        ${PIPELINE_DB_PATH}
Outputs:   ${OUTPUT_DIR}
EOF

cd "$ROOT_DIR"
exec env \
  PYTHONPATH="$ROOT_DIR" \
  WORKSPACE_DIR="$WORKSPACE_DIR" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  PIPELINE_DB_PATH="$PIPELINE_DB_PATH" \
  PIPELINE_BACKEND=mock \
  PIPELINE_WORKER_ENABLED="${PIPELINE_WORKER_ENABLED:-1}" \
  "$PIPELINE_DEV_VENV/bin/uvicorn" services.pipeline_app:app --host 127.0.0.1 --port "$PIPELINE_PORT"
