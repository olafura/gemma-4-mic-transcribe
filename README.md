# Gemma 4 Mic Transcribe

Local microphone transcription with Gemma 4 12B and LiteRT-LM.

## Setup

```bash
scripts/setup.sh
```

If the model is not already present, import it with:

```bash
litert-lm import --from-huggingface-repo=litert-community/gemma-4-12B-it-litert-lm gemma-4-12B-it.litertlm gemma4-12b
```

The CLI defaults to:

```text
~/.litert-lm/models/gemma4-12b/model.litertlm
```

## Usage

List input devices:

```bash
uv run gemma-4-mic-transcribe --list-devices
```

Transcribe from a selected microphone:

```bash
uv run gemma-4-mic-transcribe \
  --device 1 \
  --system-message "You are a precise transcription engine. Return verbatim text only." \
  --window-seconds 5 \
  --stride-seconds 2.5
```

Useful options:

```text
--model PATH              model.litertlm path
--window-seconds FLOAT    rolling audio window sent to Gemma
--stride-seconds FLOAT    interval between transcription requests
--sample-rate INT         requested microphone sample rate, default 16000
--backend cpu|gpu         LiteRT-LM text backend, default gpu
--audio-backend cpu|gpu   LiteRT-LM audio backend, default cpu
```
