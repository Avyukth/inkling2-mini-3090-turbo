#!/usr/bin/env bash
# Reproducible setup & launch for Inkling-2-Mini / Inkling-2-Mini-Derisked on 1x RTX 3090 (sm_86).
# Clones lulakien/inkling2-mini-3090-recipe at its pinned commit, installs the W8/wo_ud overlays,
# and launches the tuned 128K BF16 KV server (MAXSEQS=8, FAR_ATTN=1, W8+=wo, VLLM_W8_FAST=1).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-$REPO_ROOT/.upstream/inkling2-mini-3090-recipe}"
UPSTREAM_COMMIT="75761b0eaaf2a80aeb76ffa9dff0768300236e2a"

MODEL_DIR="${MODEL_DIR:-$HOME/models/Inkling-2-Mini-Derisked}"
SERVED_NAME="${SERVED_NAME:-Inkling-2-Mini-Derisked}"
NAME="${NAME:-inkling2-turbo}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PROFILE="${PROFILE:-128k}"
GPU="${GPU:-0}"
RESTART_POLICY="${RESTART_POLICY:---restart unless-stopped}"

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "Error: MODEL_DIR=$MODEL_DIR does not contain config.json." >&2
  echo "Download weights first, e.g.:" >&2
  echo "  hf download Blackfrost-AI/Inkling-2-Mini-Derisked --local-dir \"$MODEL_DIR\" --exclude 'DWM_RECIPE/*' --exclude 'ASSETS/*'" >&2
  exit 1
fi

if [ ! -d "$UPSTREAM_DIR/.git" ]; then
  mkdir -p "$(dirname "$UPSTREAM_DIR")"
  git clone https://github.com/lulakien/inkling2-mini-3090-recipe "$UPSTREAM_DIR"
fi

git -C "$UPSTREAM_DIR" fetch --quiet origin || true
git -C "$UPSTREAM_DIR" checkout --quiet "$UPSTREAM_COMMIT"
git -C "$UPSTREAM_DIR" checkout -- patches/w8_gemm.py patches/inkling2_attention.py

cp "$REPO_ROOT/patches/w8_gemm.py" "$UPSTREAM_DIR/patches/w8_gemm.py"
cp "$REPO_ROOT/patches/inkling2_attention.py" "$UPSTREAM_DIR/patches/inkling2_attention.py"

. "$UPSTREAM_DIR/launch/image.env"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Pulling pinned vLLM image ($IMAGE)..."
  docker pull "$IMAGE"
fi

EXTRA_PORT=""
if [ "$HOST" != "127.0.0.1" ] && [ "$HOST" != "localhost" ]; then
  EXTRA_PORT="-p ${HOST}:${PORT}:8000"
fi

export MODEL_DIR SERVED_NAME NAME PORT PROFILE GPU
export FAR_ATTN="${FAR_ATTN:-1}"
export MAXSEQS="${MAXSEQS:-8}"
export W8="${W8:-attn,wo,head,markov,draft}"
export DOCKER_EXTRA="-e VLLM_W8_FAST=${VLLM_W8_FAST:-1} ${EXTRA_PORT} ${RESTART_POLICY} ${DOCKER_EXTRA:-}"

exec bash "$UPSTREAM_DIR/launch/1gpu.sh"
