# Gemma 4 Mic Transcribe

Elixir CLI for direct Gemma 4 12B Unified audio transcription.

The app reads PCM WAV audio, normalizes it to mono 16 kHz float samples, and
builds Gemma 4 Unified audio inputs directly: raw 640-sample audio frames,
`<|audio>/<audio|>` prompt markers, repeated `<|audio|>` soft-token
placeholders, and text prompts. It does not use LiteRT or Whisper as a fallback.
The runtime is pure Elixir/Nx/Bumblebee. The compressed-tensors QAT variant is
loaded by unpacking its packed int4 Linear weights during Bumblebee parameter
conversion.

The runtime includes a local Bumblebee/Axon implementation of the audio-only
Gemma4UnifiedForConditionalGeneration path. Generation runs one prefill pass over
the prompt and audio tokens, then uses a KV-cached greedy decode loop so each
subsequent decode step processes one generated token against the cached context.

On EXLA backends the whole forward pass is JIT-compiled into one fused XLA
executable per input shape instead of dispatching each Nx op eagerly. Input
shapes are kept static so executables compile once and are then reused:

- audio windows/utterances are padded to fixed audio-token buckets,
- KV cache lengths are rounded up to a fixed step (512), so all buckets share
  one compiled decode step,
- prefill computes logits only for the final prompt position instead of the
  whole sequence,
- parameters load as bf16 by default (`--param-type`), halving the memory
  bandwidth per decoded token versus f32.

The first generation per shape pays the XLA compile cost. The CLI and
streaming sessions warm these executables at startup by generating a couple of
tokens over silence (disable with `--no-warmup`), so live audio never hits a
compile stall.

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

The default model for this Elixir Bumblebee/Axon runtime is:

```text
google/gemma-4-12B-it
```

The QAT model repos use different artifact formats and runtimes:

```text
google/gemma-4-12B-it-qat-q4_0-gguf      GGUF for llama.cpp/Ollama/LM Studio
google/gemma-4-12B-it-qat-w4a16-ct       compressed-tensors safetensors with local unpacking
```

The local `Gemma4Unified.Runtime` can load the w4a16 compressed-tensors
safetensors by unpacking packed int4 Linear weights during Bumblebee parameter
conversion. GGUF repos still require a GGUF runtime such as llama.cpp/Ollama/LM
Studio and fail before backend allocation in this CLI.

## Usage

List known model variants and their required runtimes:

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

For LLM-bound events, use streaming WAV mode. It forms speech utterances first
and only marks committed final transcripts as safe to send downstream:

```bash
./gemma_4_mic_transcribe \
  --wav journal1.wav \
  --stream-wav \
  --output jsonl \
  --backend exla:rocm \
  --model-name gemma4-12b-qat-w4a16-ct \
  --max-response-tokens 32
```

Useful options:

```text
--list-models                  show known Gemma 4 12B model variants and required runtimes
--wav PATH                     read PCM WAV audio from a file
--skip-windows INT             skip leading windows
--max-windows INT              stop after N selected windows
--stream-wav                   process WAV audio as timed streaming chunks
--output text|jsonl            output format for streaming events, default text
--chunk-ms FLOAT               streaming WAV chunk duration, default 100.0
--system-message TEXT          system instruction for every window
--system-message-file PATH     read system instruction from a file
--prompt TEXT                  user prompt paired with every audio window
--window-seconds FLOAT         audio window duration, default 5.0
--stride-seconds FLOAT         seconds between windows, default 2.5
--sample-rate INT              target sample rate, default 16000
--model-name NAME              model alias or Hugging Face repo; selects the required runtime
--max-response-tokens INT      maximum generated tokens, default 512
--backend host|torchx|torchx:cpu|torchx:cuda|exla|exla:host|exla:cuda|exla:rocm
                               Nx/Bumblebee backend label, default torchx
--param-type bf16|f16|f32      model parameter/compute precision, default bf16
--no-warmup                    skip startup warmup; JIT compiles on the first real utterance
--no-speech-gate               disable cheap local speech gating before model generation
--min-speech-seconds FLOAT     minimum likely speech duration before generation, default 0.25
--speech-threshold FLOAT       RMS threshold for active audio frames, default 0.01
--speech-min-active-ratio FLOAT
                               required active-frame ratio per window, default 0.2
--speech-max-zero-crossing-rate FLOAT
                               reject very noisy windows above this zero-crossing ratio, default 0.35
--speech-start-ms FLOAT        active speech needed to start an utterance, default 120
--speech-end-silence-ms FLOAT  silence needed to commit an utterance, default 500
--min-utterance-ms FLOAT       suppress shorter utterances, default 350
--max-utterance-ms FLOAT       force-commit long utterances, default 8000
--partial-interval-ms FLOAT    minimum time between partial transcript events, default 1000
--no-partials                  disable unstable partial transcript events
--tts-text TEXT                recent TTS text to suppress as echo in stream mode
--tts-timestamp-ms FLOAT       timestamp for --tts-text, default 0
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

The `Gemma4MicTranscribe.StreamingSession` API is the reusable path for live
audio integrations. It accepts timestamped sample chunks and explicit TTS text
events, then emits `partial`, `final`, and `suppressed` events. Only `final`
events have `send_to_llm: true`. `Gemma4MicTranscribe.WebRTC.TestHarness`
contains the Elixir WebRTC-facing adapter used for browser/WebRTC testing; see
https://elixir-webrtc.org/ for the WebRTC project.

## Tracing

Two tracing paths are available for performance investigation:

BEAM call tracing needs no root. It uses `:dbg` from `runtime_tools` (see
[A guide to tracing in Elixir](https://www.erlang-solutions.com/blog/a-guide-to-tracing-in-elixir/))
and logs per-call wall durations for the pipeline modules to stderr:

```bash
./gemma_4_mic_transcribe --wav journal1.wav --trace
mise run trace:elixir -- --wav journal1.wav --backend exla:rocm
```

GPU-side tracing uses [bpftrace](https://bpftrace.org/) uprobes on the HIP
runtime to histogram `hipLaunchKernel`, `hipMemcpy*`, and
`hipStreamSynchronize` latencies for a running transcription. bpftrace needs
root, so it is wrapped in a mise task that attaches to the running `beam.smp`
(start a transcription first, then in another terminal):

```bash
mise run trace:bpf            # oldest beam.smp
mise run trace:bpf -- <pid>   # specific pid
```

Set `ROCM_PATH` or `HIP_LIB` if the HIP runtime is not under `/opt/rocm`.

[OBI](https://opentelemetry.io/docs/zero-code/obi/setup/standalone/)
(OpenTelemetry eBPF Instrumentation) covers a third angle: zero-code
protocol-level spans (HTTP/S, gRPC, SQL, ...) for a target process, exported
as OpenTelemetry data. It cannot hook arbitrary native functions, so it does
not replace the HIP histograms; it becomes useful for the WebRTC/signaling
path and any outbound HTTP (for example Hugging Face downloads). If the
installed `obi` binary carries file capabilities (`getcap $(which obi)`), no
root is needed:

```bash
mise run trace:obi   # attaches to beam.smp processes, prints spans to stdout
```

Example span for a BEAM HTTP client call:

```text
2026-07-20 04:08:36 (11.5ms) HTTPClient 200 GET / [beam.smp:35280]->[104.20.23.154:80] traceparent=[00-dc61...]
```

BEAM TLS is implemented in Erlang rather than libssl, so OBI reports HTTPS
connections at the TCP level without decoding request contents; plain HTTP,
gRPC, and proxied traffic decode fully.

## Implementation Status

Implemented:

- Mix CLI with a local launcher script.
- PCM16 and float32 WAV normalization to mono 16 kHz samples.
- Windowing, timestamps, prompt construction, and Gemma 4 Unified raw-audio features.
- Local Bumblebee/Axon Gemma4Unified audio model loader and KV-cached greedy generation.
- Streaming WAV utterance segmentation with partial/final/suppressed event output.
- Explicit TTS echo text suppression for streaming sessions.
- Elixir WebRTC test-harness adapter for feeding decoded f32 audio into streaming sessions.

Not implemented yet:

- Vision/video inputs.
- Production live microphone/WebRTC CLI mode.
