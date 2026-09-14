---
title: Gemma 4 spoken language detector
emoji: 🗣️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: 1 s spoken language ID from the Gemma 4 E2B audio tower
---

# Gemma 4 spoken language detector

The first five conformer blocks of the Gemma 4 E2B audio tower with a 34-way
head, fine-tuned on Common Voice single words and answering from a 1 s window
that starts at the first sound in the clip. Built with Elixir, Nx and Torchx on
CPU; source in the `gemma-4-mic-transcribe` project.

`POST /detect` takes an audio file in any format ffmpeg reads and returns every
language with its probability:

```sh
curl --data-binary @clip.mp3 https://<space-host>/detect
```

The same image validates a detector against Common Voice 17 parquet shards
mounted from the `olafura/gemma_language_detection` bucket:

```sh
hf jobs run --flavor cpu-upgrade --timeout 3h \
  -v hf://buckets/olafura/gemma_language_detection:/data \
  IMAGE \
  language_id validate --artifact /data/artifacts/ft-depth5-1s-e2 \
  --per-language 30 --output /data/validation/ft-depth5-1s-e2.json
```

where IMAGE is this Space's image or any public registry copy of the
Dockerfile build; the project README shows the same job run from the public
hexpm Elixir image with the built runtime synced into the bucket, which needs
neither a Space nor a public image.
