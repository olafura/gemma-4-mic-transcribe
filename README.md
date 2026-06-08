# Gemma 4 Mic Transcribe

Elixir CLI for direct Gemma 4 12B Unified audio transcription.

The app reads PCM WAV audio, normalizes it to mono 16 kHz float samples, and
builds Gemma 4 Unified audio inputs directly: raw 640-sample audio frames,
`<boa>/<eoa>` prompt markers, and text prompts. It intentionally does not use
Python, LiteRT, or Whisper as a fallback.

The remaining runtime gap is Gemma4UnifiedForConditionalGeneration support in
Bumblebee/Nx. Until that model backbone is implemented, inference exits with a
clear unsupported-runtime error after the Elixir audio/prompt path is prepared.

## Setup

```bash
scripts/setup.sh
```

Or run the steps directly:

```bash
mix deps.get
mix test
mix escript.build
```

The default model is:

```text
google/gemma-4-12B-it
```

Other Gemma 4 12B variants are listed for reference:

```text
google/gemma-4-12B-it-qat-q4_0-gguf   non-Bumblebee GGUF runtimes
google/gemma-4-12B-it-qat-w4a16-ct    Transformers/vLLM compressed-tensors workflows
```

## Usage

List supported model variants:

```bash
mix run -e 'System.halt(Gemma4MicTranscribe.CLI.main(["--list-models"]))'
```

Run against a PCM WAV file:

```bash
mix run -e 'System.halt(Gemma4MicTranscribe.CLI.main([
  "--wav", "journal1.wav",
  "--window-seconds", "2",
  "--stride-seconds", "2",
  "--skip-windows", "1",
  "--max-windows", "1",
  "--system-message-file", "system-message-drive-thru.txt"
]))'
```

After `mix escript.build`:

```bash
./gemma_4_mic_transcribe --wav journal1.wav --max-windows 1
```

Useful options:

```text
--list-models                  show supported Gemma 4 12B model variants
--wav PATH                     read PCM WAV audio from a file
--skip-windows INT             skip leading windows
--max-windows INT              stop after N selected windows
--system-message TEXT          system instruction for every window
--system-message-file PATH     read system instruction from a file
--prompt TEXT                  user prompt paired with every audio window
--window-seconds FLOAT         audio window duration, default 5.0
--stride-seconds FLOAT         seconds between windows, default 2.5
--sample-rate INT              target sample rate, default 16000
--model-name NAME              Hugging Face or local model name, default google/gemma-4-12B-it
--max-response-tokens INT      maximum generated tokens, default 512
--request-timeout-seconds FLOAT
                               maximum seconds for one generation
```

Microphone input is intentionally not advertised yet. The CLI currently supports
PCM WAV file input only.

## Implementation Status

Implemented:

- Mix/escript CLI.
- PCM16 and float32 WAV normalization to mono 16 kHz samples.
- Windowing, timestamps, prompt construction, and Gemma 4 Unified raw-audio features.
- Explicit unsupported-runtime error at the Bumblebee model boundary.

Not implemented yet:

- Full Bumblebee/Nx Gemma4UnifiedForConditionalGeneration text backbone.
- Multimodal embedding injection into the decoder token sequence.
- Live microphone/WebRTC CLI mode.
