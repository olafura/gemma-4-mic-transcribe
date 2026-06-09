# Gemma 4 Mic Transcribe

Elixir CLI for direct Gemma 4 12B Unified audio transcription.

The app reads PCM WAV audio, normalizes it to mono 16 kHz float samples, and
builds Gemma 4 Unified audio inputs directly: raw 640-sample audio frames,
`<|audio>/<audio|>` prompt markers, repeated `<|audio|>` soft-token
placeholders, and text prompts. It intentionally does not use Python, LiteRT, or
Whisper as a fallback.

The runtime includes a local Bumblebee/Axon implementation of the audio-only
Gemma4UnifiedForConditionalGeneration path. Generation currently runs a
static-shape full-context greedy loop rather than a KV-cached loop. This avoids
per-token EXLA recompilation, but still reruns the 12B checkpoint over the whole
fixed context for every generated token.

The CLI gates windows with a cheap local speech detector before loading or
running the model. This is intentionally separate from the Gemma prompt: silent,
too-short, or high-noise windows are skipped without generation, and empty
transcripts are not printed. Very small windows such as 10 ms are not useful
transcription units; Gemma 4 Unified consumes 640-sample raw PCM audio tokens
at 16 kHz, so one audio token already spans 40 ms before any text decoding cost.

This is local model inference, not a hosted API call. The app does not base64
encode audio and send it to Gemma; it injects raw audio frame tensors into the
local model graph. Torchx downloads CPU LibTorch by default, so CUDA execution
requires compiling Torchx with a CUDA LibTorch target. ROCm users should try the
EXLA backend, which is backed by XLA.

## Setup

```bash
scripts/setup.sh
```

Or run the steps directly:

```bash
mix deps.get
mix test
mix compile
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

After `scripts/setup.sh` or `mix compile`:

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
--backend host|torchx|torchx:cpu|torchx:cuda|exla|exla:host|exla:cuda|exla:rocm
                               Nx/Bumblebee backend label, default torchx
--no-speech-gate               disable cheap local speech gating before model generation
--min-speech-seconds FLOAT     minimum likely speech duration before generation, default 0.25
--speech-threshold FLOAT       RMS threshold for active audio frames, default 0.01
--speech-min-active-ratio FLOAT
                               required active-frame ratio per window, default 0.2
--speech-max-zero-crossing-rate FLOAT
                               reject very noisy windows above this zero-crossing ratio, default 0.35
--debug                        emit progress logs to stderr
--request-timeout-seconds FLOAT
                               maximum seconds for one generation
```

For a long first run, add `--debug` to see whether the process is loading the
checkpoint, building the Axon graph, tokenizing, or generating a specific token.
It also logs the selected Torchx device and CUDA availability, or the selected
EXLA client and `XLA_TARGET`.

To force CUDA and fail fast if Torchx cannot see a GPU:

```bash
./gemma_4_mic_transcribe --wav journal1.wav --backend torchx:cuda --debug
```

To install a CUDA LibTorch build, recompile Torchx with the CUDA target matching
your local driver/runtime, for example:

```bash
LIBTORCH_TARGET=cu129 mix deps.clean torchx
LIBTORCH_TARGET=cu129 mix deps.compile torchx
```

To try EXLA on ROCm, build/install XLA for ROCm and select the ROCm client:

```bash
mise install
XLA_TARGET=rocm XLA_BUILD=true BAZEL="$(mise which bazel)" mix deps.clean xla --build
XLA_TARGET=rocm XLA_BUILD=true BAZEL="$(mise which bazel)" mix deps.compile xla exla
./gemma_4_mic_transcribe --wav journal1.wav --backend exla:rocm --debug
```

If the ROCm XLA archive already exists, reuse it while still setting
`XLA_TARGET=rocm` so EXLA does not compile CUDA helper objects when `nvcc` is on
`PATH`:

```bash
XLA_TARGET=rocm XLA_ARCHIVE_PATH=/home/olafura/.cache/xla/0.10.0/build/xla_extension-0.10.0-x86_64-linux-gnu-rocm.tar.gz EXLA_FORCE_REBUILD=partial mix deps.compile exla --force
```

The vendored XLA build is pinned to Bazel 7.7.0. Bazel 9.x fails this OpenXLA
snapshot during package loading with `CcInfo symbol has been removed`. Use
`BAZEL="$(mise which bazel)"` so the XLA build gets the pinned Bazel binary even
when Bazel runs from the OpenXLA cache directory.

The vendored XLA patches also avoid OpenXLA's hardcoded LLVM executable path.
If you need to force a specific compiler, set `CLANG_COMPILER_PATH` or `CC`
before compiling instead of editing distro-specific paths into the build.

The ROCm build target list starts with MI200/MI300, RDNA2/RDNA3, and RDNA4
targets, then appends local GPU targets reported by `rocm_agent_enumerator`.
This catches Strix/Radeon 8060S `gfx1151`, which otherwise reaches XLA and can
abort in ROCm while loading HIP code objects. See
https://github.com/ROCm/rocm-jax/issues/234 for the same `gfx1151` failure mode.

The CLI preflights `exla:rocm` before starting EXLA. If the installed GPU ISA is
missing from the compiled `libxla_extension.so` offload bundles, it exits with a
rebuild message instead of letting BEAM core dump.

If your ROCm install is not under the default path, XLA may also need matching
`ROCM_PATH` and `LD_LIBRARY_PATH` values before compiling/running.

If the ROCm XLA build fails with `Cannot find rocm library rocprofiler-sdk`,
install the ROCm profiler SDK package for your distro or point `ROCM_PATH` and
`LD_LIBRARY_PATH` at the ROCm installation that provides it.

If the ROCm XLA build fails with `xxd: command not found`, install `xxd` or the
package that provides it for your distro and make sure it is on `PATH`.

The vendored EXLA build disables EXLA's CUDA helper objects automatically when
`XLA_TARGET=rocm`, even if `nvcc` is installed. ROCm support still comes from the
locally built ROCm `libxla_extension`.

Microphone input is intentionally not advertised yet. The CLI currently supports
PCM WAV file input only.

## Implementation Status

Implemented:

- Mix CLI with a local launcher script.
- PCM16 and float32 WAV normalization to mono 16 kHz samples.
- Windowing, timestamps, prompt construction, and Gemma 4 Unified raw-audio features.
- Local Bumblebee/Axon Gemma4Unified audio model loader and static-shape full-context greedy generation.

Not implemented yet:

- KV-cached generation for Gemma 4's mixed sliding/full attention head sizes.
- Vision/video inputs.
- Live microphone/WebRTC CLI mode.
