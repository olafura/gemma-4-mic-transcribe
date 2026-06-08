#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-google/gemma-4-12B-it}"

echo "Installing Elixir dependencies..."
mix deps.get

echo
echo "Compiling Elixir app and native dependencies..."
mix compile

echo
echo "Default direct-audio model:"
echo "  $MODEL_NAME"
echo
echo "Current runtime status:"
echo "  PCM WAV normalization, Gemma 4 Unified input construction, and a local"
echo "  Bumblebee/Axon Gemma4Unified audio runtime are implemented."
echo "  First inference downloads and loads the full google/gemma-4-12B-it checkpoint."
