#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-google/gemma-4-12B-it}"

echo "Installing Elixir dependencies..."
mix deps.get

echo
echo "Building CLI escript..."
mix escript.build

echo
echo "Default direct-audio model:"
echo "  $MODEL_NAME"
echo
echo "Current runtime status:"
echo "  PCM WAV normalization and Gemma 4 Unified input construction are implemented."
echo "  Gemma4UnifiedForConditionalGeneration still needs a Bumblebee/Nx backbone implementation."
echo "  Until that lands, inference fails clearly instead of falling back to Python, LiteRT, or Whisper."
