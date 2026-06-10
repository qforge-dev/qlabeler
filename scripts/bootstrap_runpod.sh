#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

REPO_DIR="${REPO_DIR:-/workspace/qlabeler}"
REPO_URL="${REPO_URL:-https://github.com/qforge-dev/qlabeler.git}"
REPO_REF="${REPO_REF:-main}"
UPDATE_REPO="${UPDATE_REPO:-1}"
LOAD_MODELS="${LOAD_MODELS:-0}"

usage() {
  cat <<EOF
Usage:
  $0

This is the one-shot fresh RunPod bootstrap. It installs clone prerequisites,
clones or updates the repo, prompts for HF_TOKEN when needed, writes .env, then
installs and starts both model APIs.

Environment:
  REPO_URL      Git URL to clone. Default: https://github.com/qforge-dev/qlabeler.git
  REPO_DIR      Checkout path. Default: /workspace/qlabeler
  REPO_REF      Branch/tag to clone. Default: main
  UPDATE_REPO   Pull clean existing checkouts. Default: 1
  LOAD_MODELS   Load model weights during setup. Default: 0

Override REPO_URL only when testing a fork or private mirror.
If HF_TOKEN is not in .env or the environment, the script prompts securely.
EOF
}

log() {
  printf '\n== %s ==\n' "$*"
}

done_step() {
  printf '[done] %s\n' "$*"
}

run_step() {
  printf '[run ] %s\n' "$*"
}

warn_step() {
  printf '[warn] %s\n' "$*" >&2
}

need_sudo() {
  if [[ "$(id -u)" == "0" ]]; then
    SUDO=()
  elif command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo)
  else
    echo "This bootstrap needs root privileges or sudo." >&2
    exit 1
  fi
}

require_apt_machine() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "This bootstrap expects an Ubuntu/Debian RunPod-style machine with apt-get." >&2
    exit 1
  fi
}

install_clone_prereqs() {
  log "Prerequisites"
  local missing=()
  for cmd in ca-certificates curl git; do
    if ! dpkg -s "$cmd" >/dev/null 2>&1; then
      missing+=("$cmd")
    fi
  done

  if [[ "${#missing[@]}" == "0" ]]; then
    done_step "clone prerequisites are installed"
    return
  fi

  run_step "installing clone prerequisites: ${missing[*]}"
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y ca-certificates curl git
}

repo_has_expected_files() {
  [[ -f "$REPO_DIR/scripts/setup_model_apis.sh" && -d "$REPO_DIR/services" ]]
}

clone_or_update_repo() {
  log "Repository"

  if [[ -d "$REPO_DIR/.git" ]]; then
    done_step "repository exists at $REPO_DIR"
    if [[ "$UPDATE_REPO" != "1" ]]; then
      done_step "repo update skipped because UPDATE_REPO=$UPDATE_REPO"
      return
    fi

    if [[ -n "$(git -C "$REPO_DIR" status --short)" ]]; then
      warn_step "repository has local changes; skipping git pull"
      return
    fi

    local current_branch
    current_branch="$(git -C "$REPO_DIR" branch --show-current || true)"
    if [[ -n "$current_branch" ]]; then
      run_step "updating $current_branch with git pull --ff-only"
      git -C "$REPO_DIR" pull --ff-only || warn_step "git pull failed; continuing with existing checkout"
    else
      warn_step "repository is detached; skipping git pull"
    fi
    return
  fi

  if [[ -d "$REPO_DIR" ]] && repo_has_expected_files; then
    done_step "repo-like directory exists at $REPO_DIR"
    return
  fi

  if [[ -d "$REPO_DIR" ]] && [[ -n "$(find "$REPO_DIR" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]]; then
    echo "$REPO_DIR exists but is not this repo and is not empty." >&2
    exit 1
  fi

  if [[ -z "$REPO_URL" ]]; then
    read -r -p "Git repo URL to clone into $REPO_DIR: " REPO_URL
  fi
  if [[ -z "$REPO_URL" ]]; then
    echo "REPO_URL is required when $REPO_DIR does not exist." >&2
    exit 1
  fi

  run_step "cloning $REPO_URL"
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone --branch "$REPO_REF" "$REPO_URL" "$REPO_DIR"
}

env_has_real_token() {
  local env_file="$REPO_DIR/.env"
  [[ -f "$env_file" ]] || return 1
  local value
  value="$(grep -E '^HF_TOKEN=' "$env_file" | tail -n 1 | cut -d= -f2- || true)"
  [[ -n "$value" && "$value" != "hf_your_token_here" ]]
}

write_env_token() {
  local env_file="$REPO_DIR/.env"
  local token="$1"

  umask 077
  if [[ ! -f "$env_file" ]]; then
    if [[ -f "$REPO_DIR/.env.example" ]]; then
      cp "$REPO_DIR/.env.example" "$env_file"
    else
      touch "$env_file"
    fi
  fi

  local tmp_file
  tmp_file="$(mktemp)"
  awk -v token="$token" '
    BEGIN { seen = 0 }
    /^HF_TOKEN=/ {
      if (!seen) {
        print "HF_TOKEN=" token
        seen = 1
      }
      next
    }
    { print }
    END {
      if (!seen) {
        print "HF_TOKEN=" token
      }
    }
  ' "$env_file" >"$tmp_file"
  mv "$tmp_file" "$env_file"
  chmod 600 "$env_file"
}

configure_env() {
  log "Environment"

  if env_has_real_token; then
    done_step ".env exists and HF_TOKEN is set"
    return
  fi

  local token="${HF_TOKEN:-}"
  if [[ -z "$token" ]]; then
    read -r -s -p "HF_TOKEN: " token
    printf '\n'
  fi
  if [[ -z "$token" ]]; then
    echo "HF_TOKEN cannot be empty." >&2
    exit 1
  fi

  run_step "writing HF_TOKEN to $REPO_DIR/.env"
  write_env_token "$token"
}

load_env() {
  if [[ -f "$REPO_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_DIR/.env"
    set +a
  fi
}

api_health_ok() {
  local port="$1"
  curl -fsS "http://127.0.0.1:$port/healthz" >/dev/null 2>&1
}

venvs_ready() {
  local workspace_dir="${WORKSPACE_DIR:-/workspace}"
  local venv_dir="${VENV_DIR:-$workspace_dir/venvs}"
  [[ -x "$venv_dir/audio-flamingo-next/bin/uvicorn" && -x "$venv_dir/sam-audio-large/bin/uvicorn" ]]
}

bootstrap_or_start_services() {
  log "Model APIs"
  chmod +x "$REPO_DIR/scripts/setup_model_apis.sh"
  load_env

  local af_port="${AFNEXT_PORT:-8001}"
  local sam_port="${SAM_AUDIO_PORT:-8002}"

  if api_health_ok "$af_port" && api_health_ok "$sam_port"; then
    done_step "both APIs are already healthy"
    ROOT_DIR="$REPO_DIR" "$REPO_DIR/scripts/setup_model_apis.sh" status
    return
  fi

  if venvs_ready; then
    run_step "venvs exist; starting services"
    ROOT_DIR="$REPO_DIR" LOAD_MODELS="$LOAD_MODELS" "$REPO_DIR/scripts/setup_model_apis.sh" start
  else
    run_step "installing dependencies and starting services"
    "${SUDO[@]}" env ROOT_DIR="$REPO_DIR" LOAD_MODELS="$LOAD_MODELS" "$REPO_DIR/scripts/setup_model_apis.sh" bootstrap
  fi
}

print_summary() {
  log "Complete"
  cat <<EOF
Repo:
  $REPO_DIR

Useful commands:
  cd $REPO_DIR
  ./scripts/setup_model_apis.sh status
  ./scripts/setup_model_apis.sh load
  ./scripts/setup_model_apis.sh logs audio-flamingo-next
  ./scripts/setup_model_apis.sh logs sam-audio-large

APIs:
  Audio Flamingo: http://127.0.0.1:${AFNEXT_PORT:-8001}/v1/audio-flamingo/ask
  SAM-Audio:      http://127.0.0.1:${SAM_AUDIO_PORT:-8002}/v1/sam-audio/separate
EOF
}

main() {
  case "${1:-}" in
    "-h"|"--help"|"help")
      usage
      exit 0
      ;;
  esac

  need_sudo
  require_apt_machine
  install_clone_prereqs
  clone_or_update_repo
  configure_env
  bootstrap_or_start_services
  print_summary
}

main "$@"
