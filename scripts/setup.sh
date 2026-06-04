#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${LITERT_LM_MODEL_DIR:-$HOME/.litert-lm/models/gemma4-12b}"
MODEL_PATH="$MODEL_DIR/model.litertlm"

echo "Installing Python dependencies with uv..."
uv sync

if ! command -v litert-lm >/dev/null 2>&1; then
  echo
  echo "litert-lm CLI was not found on PATH."
  echo "Install it with:"
  echo "  uv tool install litert-lm"
fi

if [[ -f "$MODEL_PATH" ]]; then
  echo
  echo "Model already present:"
  echo "  $MODEL_PATH"
  exit 0
fi

echo
echo "Model not found at:"
echo "  $MODEL_PATH"
echo
echo "Import Gemma 4 12B with:"
echo "  litert-lm import --from-huggingface-repo=litert-community/gemma-4-12B-it-litert-lm gemma-4-12B-it.litertlm gemma4-12b"
