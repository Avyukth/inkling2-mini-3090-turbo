#!/usr/bin/env bash
# Runs the standalone numerical isomorphism verification inside the pinned vLLM container.
# Note: Stop any running vLLM server first so ~1 GiB of GPU memory is free.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e}"

exec docker run --rm -i --gpus all \
  -v "$REPO_ROOT/patches/w8_gemm.py:/workspace/w8_gemm.py:ro" \
  -v "$REPO_ROOT/scripts/verify_isomorphism.py:/workspace/verify_isomorphism.py:ro" \
  --entrypoint python3 "$IMAGE" /workspace/verify_isomorphism.py
