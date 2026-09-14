#!/usr/bin/env bash
# Stages the language-ID slice of the project and uploads it to a Docker
# Space, which builds the image (the same Dockerfile as a local build).
# Docker Spaces need a PRO subscription; without one, build the image
# locally and push it to Docker Hub instead (see the README).
#
#   hf-space/upload.sh [olafura/gemma-language-detection]
#
# Create the Space once with
#   hf repos create olafura/gemma-language-detection --type space --space-sdk docker
set -euo pipefail

space="${1:-olafura/gemma-language-detection}"
root="$(cd "$(dirname "$0")/.." && pwd)"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

cd "$root"
mkdir -p "$stage/config" "$stage/lib/gemma_4_mic_transcribe" "$stage/vendor" "$stage/artifacts/language-id"
cp Dockerfile .dockerignore mix.exs mix.lock "$stage/"
cp hf-space/README.md "$stage/README.md"
cp config/config.exs "$stage/config/"
cp lib/gemma_4_mic_transcribe/audio.ex lib/gemma_4_mic_transcribe/language_id_cli.ex lib/gemma_4_mic_transcribe/rocm_preflight.ex "$stage/lib/gemma_4_mic_transcribe/"
cp -r lib/gemma_4_mic_transcribe/gemma4_e4b lib/gemma_4_mic_transcribe/language_id "$stage/lib/gemma_4_mic_transcribe/"
for dep in exla xla ex_libsrt; do
  mkdir -p "$stage/vendor/$dep"
  cp "vendor/$dep/mix.exs" "$stage/vendor/$dep/"
done
cp -r artifacts/language-id/ft-depth5-1s-e2 "$stage/artifacts/language-id/"

hf upload "$space" "$stage" . --repo-type space --commit-message "language_id image $(git rev-parse --short HEAD)"
