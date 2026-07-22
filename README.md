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

## Inspecting Gemma 4 experts

`Gemma4MicTranscribe.Gemma4.Experts.list/2` lists the shared and routed FFN
experts described by a Gemma 4 MoE config without loading its weights. Each
descriptor contains its checkpoint tensor names and the expert-axis slice that
can later be used for extraction or replacement. Dense Gemma variants return an
empty list.

```elixir
{:ok, config} = File.read!("config.json") |> Jason.decode()
experts = Gemma4MicTranscribe.Gemma4.Experts.list(config)

# Gemma 4 26B-A4B: one shared + 128 routed experts in each of 30 layers
length(experts) #=> 3_870

first_routed = Enum.find(experts, &(&1.kind == :routed))
first_routed.id
#=> "language_model.layer.0.expert.0"
```

This change only catalogs the experts. The current Axon runtime still rejects
MoE inference until routing and the expert forward pass are implemented.

Dense models expose their always-active feed-forward networks separately:

```elixir
ffns = Gemma4MicTranscribe.Gemma4.list_ffns(config)
first_ffn = hd(ffns)

first_ffn.operation
#=> "down(activation(gate(x)) * up(x))"

first_ffn.weights.gate
#=> %{
#=>   checkpoint_tensor: "model.language_model.layers.0.mlp.gate_proj.weight",
#=>   checkpoint_shape: {15360, 3840},
#=>   axon_parameter: "decoder.blocks.0.ffn.gate.kernel",
#=>   axon_shape: {3840, 15360}
#=> }
```

The descriptor treats the pre/post RMS norms and residual connection as the
FFN's surrounding context. They are intentionally not included in the three
weights needed to run the extracted FFN itself.

## Probing layer behavior

An already-loaded runtime can run one instrumented prefill pass over the same
input used for generation. The probe copies only selected token positions out
of the compiled graph:

```elixir
{:ok, report} =
  Gemma4MicTranscribe.Gemma4Unified.Runtime.probe(runtime, input,
    layers: [0, 5, 11, runtime.model_info.spec.num_blocks - 1],
    positions: [:audio_begin, :first_audio, :last_audio, :audio_end, :last],
    capture: [:attention, :ffn, :per_layer_input, :hidden_state],
    top_k_logits: 10
  )
```

`:attention` and `:ffn` are the normalized contributions added to the residual
stream, not raw attention probabilities. `:per_layer_input` captures E4B's
actual gated and projected per-layer embedding contribution; it is reported as
unavailable on 12B. For each position the report includes norm, RMS, maximum
absolute value, post-`layer_scalar` contribution-to-hidden-state norm ratio,
adjacent captured-layer cosine similarity, and optional logit-lens candidates.
The raw pre-scalar ratio is retained as `pre_scalar_hidden_norm_ratio`.

Use `include_activations: true` to retain the selected hidden vectors. The first
call for a new combination of input shape and probe outputs compiles an
instrumented graph, and `top_k_logits` adds a vocabulary projection for every
captured layer, so start with a small layer set.

On the current gfx1151 ROCm stack, exposing intermediate graph outputs crashes
the XLA autotuner. Layer probing therefore fails fast on `exla:rocm`; load a
separate runtime with `backend: "torchx:cpu"` for probes. Ordinary EXLA/ROCm
generation is unaffected.

## Extracting a decoder block

A loaded unified 12B runtime can isolate one complete decoder block and run it
with only that block's parameters. Capture the full hidden-state sequence that
enters the block, then pass it to the extracted block:

```elixir
{:ok, report} =
  Gemma4MicTranscribe.Gemma4Unified.Runtime.probe(runtime, input,
    layers: [46],
    positions: :all,
    capture: [:block_input, :hidden_state],
    include_activations: true
  )

{:ok, block} = Gemma4MicTranscribe.Gemma4.extract_decoder_block(runtime, 46)

standalone_output =
  Gemma4MicTranscribe.Gemma4.DecoderBlocks.run!(
    block,
    report.activations["46:block_input"]
  )
```

`run/3` generates contiguous position ids and an all-visible attention mask by
default. Pass `position_ids:` and `attention_mask:` when the source sequence
contains padding or custom positions. The extracted model state contains only
the selected block's weights. E4B blocks are not yet supported because their
KV sharing and per-layer embedding inputs require a different boundary.

Materializing an internal bf16 activation can change tensor layout and reduction
order. Compare standalone and in-model results with a numerical tolerance, not
bit equality.

To persist a genuinely independent block, build the dedicated escript and use
its separate extract/run subcommands:

```bash
GEMMA4_ESCRIPT=decoder_block mix escript.build

./decoder_block extract \
  --artifact artifacts/gemma4-12b-layer-45 \
  --layer 45 \
  --backend torchx:cpu

./decoder_block run \
  --artifact artifacts/gemma4-12b-layer-45 \
  --backend exla:rocm \
  --runs 3
```

The real 12B layer-45 artifact contains 224,148,993 parameters in a
448,299,060-byte safetensors file. It is about 1.7% of the complete 25.8 GB
pipeline artifact and contains no embeddings, neighboring layers, output norm,
or vocabulary head. The included eight-position verification fixture compares
the fresh GPU result to the output recorded from the source block. It measured
a maximum absolute CPU/GPU difference of `2.36e-5`; cold XLA compile plus
execution took 1.20 seconds and warm executions took 16-17 ms.

Block outputs use the same safetensors schema as block inputs (`hidden_state`,
`position_ids`, and `attention_mask`), so independently loaded processes can be
chained:

```bash
./decoder_block run \
  --artifact artifacts/gemma4-12b-layer-45 \
  --backend exla:rocm \
  --runs 1 \
  --output artifacts/layer-45-output.safetensors

./decoder_block run \
  --artifact artifacts/gemma4-12b-layer-46 \
  --backend exla:rocm \
  --input artifacts/layer-45-output.safetensors \
  --output artifacts/layer-46-output.safetensors
```

Each runner loads only its roughly 448 MB block. A single block cannot accept
audio or emit text by itself: it transforms the hidden-state stream supplied by
the prefix, while a final tail still supplies the output norm and vocabulary
head.

Adjacent blocks can be extracted as one graph and run without materializing the
hidden state between them:

```elixir
{:ok, chain} = Gemma4MicTranscribe.Gemma4.extract_decoder_chain(runtime, 45..47)

final_hidden_state =
  Gemma4MicTranscribe.Gemma4.DecoderBlocks.run!(
    chain,
    report.activations["45:block_input"]
  )
```

Chains must use contiguous ascending layers. Skipping from layer 6 directly to
layer 29 is not compositionally valid because layers 7 through 28 produce the
representation expected by layer 29.

To produce vocabulary scores without retaining the full model, extract a chain
that ends at the final decoder layer together with its final norm and tied
output head:

```elixir
{:ok, tail} = Gemma4MicTranscribe.Gemma4.extract_decoder_tail(runtime, 45..47)

candidates =
  Gemma4MicTranscribe.Gemma4.DecoderBlocks.top_k!(
    tail,
    report.activations["45:block_input"],
    5
  )
```

The tail projects only the final sequence position. For 12B, layers 45–47 plus
the vocabulary head contain about 1.697 billion parameters (3.39 GB in bf16);
the tied 262k-token vocabulary matrix accounts for roughly 2 GB of that total.
The returned tail retains only its 43 parameter nodes, compiled graph, backend,
and tokenizer, so the original runtime can be released afterward.

The tail can also be persisted and run against a real prefix boundary in a
different process:

```bash
./decoder_block extract-tail \
  --artifact artifacts/gemma4-12b-tail-45-47 \
  --tail-start 45 \
  --backend torchx:cpu

./decoder_block capture-prefix \
  --pipeline-artifact artifacts/gemma4-12b-baseline \
  --output artifacts/journal1-layer-45-input.safetensors \
  --wav journal1.wav \
  --seconds 5 \
  --backend exla:rocm

./decoder_block run-tail \
  --artifact artifacts/gemma4-12b-tail-45-47 \
  --input artifacts/journal1-layer-45-input.safetensors \
  --backend exla:rocm \
  --top-k 10
```

For the five-second reference, the independent tail's highest-scoring token
was `all` (id 712, score 20.06), exactly matching the full model's first output
token. The 3.39 GB tail loaded in 2.34 seconds, compiled and ran cold in 3.82
seconds, and ran warm in 146 ms. Prefix capture is a one-time boundary export;
the tail process does not load the other 11.22 billion parameters. Continuing
past the first token requires handing the per-layer KV cache across the same
boundary as the hidden state.

For full cache-aware generation, `run-split` keeps the prefix and external tail
in one orchestrating process while passing the KV cache across their graph
boundary on every token:

```bash
./decoder_block extract-prefix \
  --artifact artifacts/gemma4-12b-prefix-0-44 \
  --tail-start 45 \
  --backend torchx:cpu

./decoder_block run-split \
  --prefix-artifact artifacts/gemma4-12b-prefix-0-44 \
  --artifact artifacts/gemma4-12b-tail-45-47 \
  --wav journal1.wav \
  --seconds 5 \
  --backend exla:rocm \
  --max-new-tokens 32 \
  --runs 2
```

The dedicated prefix contains embeddings, audio projection, and layers 0–44:
11.220 billion parameters in a 22,440,161,051-byte tensor file. The runner loads
that prefix and the 3.39 GB tail directly; it never loads a complete-model
artifact. It reproduced the complete reference text and all nine token ids
exactly through the split cache path:
`"all cavalry today feelingly fresh the morning light"`. Cold split compilation
plus generation took 17.23 seconds and the warm run took 7.27 seconds.

For a long-running 12B service, extract the trained W4A16 checkpoint instead.
The artifacts retain the checkpoint's `packed` int4 matrices and group scales,
so XLA can use its q4 GEMM/GEMV kernels without loading or reconstructing the
bf16 model:

```bash
./decoder_block extract-prefix \
  --artifact artifacts/gemma4-12b-packed-prefix-0-44 \
  --tail-start 45 \
  --backend torchx:cpu \
  --model-name gemma4-12b-qat-w4a16-ct

./decoder_block extract-tail \
  --artifact artifacts/gemma4-12b-packed-tail-45-47 \
  --tail-start 45 \
  --backend torchx:cpu \
  --model-name gemma4-12b-qat-w4a16-ct

./decoder_block run-split \
  --prefix-artifact artifacts/gemma4-12b-packed-prefix-0-44 \
  --artifact artifacts/gemma4-12b-packed-tail-45-47 \
  --wav journal1.wav \
  --seconds 5 \
  --backend exla:rocm \
  --max-new-tokens 32 \
  --runs 3
```

On the reference AMD Radeon 8060S, the packed artifacts occupy 14.20 GB total
instead of 25.83 GB for bf16. Warm split generation took 2.12–2.16 seconds and
produced `"I woke up today feeling refreshed. The morning light"`, exactly
matching the resident packed runtime. The resident baseline took 2.02–2.05
seconds, leaving only about a 4.8% median separation overhead. A one-token run
measured 1.18 seconds of prefill; the remaining nine tokens averaged roughly
106 ms/token. Artifact and checkpoint load times are intentionally excluded
because they occur before the long-running service accepts work.

## Splitting raw-audio inference

The model can also be partitioned at the tail boundary. The prefix owns text
and audio embeddings plus layers 0–44; the replaceable tail owns layers 45–47
and the vocabulary head:

```elixir
{:ok, pipeline} =
  Gemma4MicTranscribe.Gemma4.extract_decoder_pipeline(runtime, 45..47)

# `runtime` may now be released.
{:ok, candidates} =
  Gemma4MicTranscribe.Gemma4.DecoderPipeline.top_k_samples(
    pipeline,
    samples_16khz,
    5
  )

{:ok, %{text: text, token_ids: token_ids}} =
  Gemma4MicTranscribe.Gemma4.DecoderPipeline.generate_samples(
    pipeline,
    samples_16khz,
    max_new_tokens: 32
  )
```

`generate_samples/3` uses one global KV cache and the already-built full-model
predictor. The extracted prefix and tail remain independently runnable, while
normal generation executes their recomposed matrix graph as one compiled unit.
Pass `execution: :split` only when explicitly inspecting the boundary. Generation
applies the same channel-aware token suppression, EOS handling, and no-repeat
n-gram rule as the full runtime.

On the real 12B bf16 checkpoint the prefix retains 11.220 billion parameter
references (22.44 GB) and the tail 1.697 billion (3.39 GB). The split is therefore
a swappable model boundary, not compression: every intervening decoder layer
remains necessary.

Build and run the dedicated compiled benchmark with GPU XLA:

```bash
mix escript.build
./decoder_pipeline_bench --backend exla:rocm --wav journal1.wav --runs 2
```

The benchmark defaults to the CLI's five-second window, ordinary EOS handling,
and a 32-token ceiling. On `journal1.wav`, both the normal CLI and a separately
loaded baseline artifact produced
`"all cavalry today feelingly fresh the morning light"` with token ids
`[712, 81686, 3124, 8178, 586, 5756, 506, 5597, 2214]`. The escript emits JSON
records so later runs can be compared without editing inline Elixir commands.

### Frankenstein layer transplants

A pipeline can install one compatible decoder layer's weights into another
slot without changing or recompiling the graph:

```elixir
frankenstein =
  Gemma4MicTranscribe.Gemma4.DecoderPipeline.transplant_layer!(pipeline, 44, 45)
```

Source and target must have the same attention type and identical parameter
layouts. The target retains its position, cache slot, and residual path; all of
its learned parameter tensors come from the source. Both `execution: :composed`
and `execution: :split` use the transplanted weights.

The benchmark compares baseline and mutant in one loaded, compiled process:

```bash
./decoder_pipeline_bench --backend exla:rocm --wav journal1.wav \
  --transplant 44:45 --runs 2
```

For the five-second `journal1.wav` reference, replacing layer 45 with layer 44
changes the fluent baseline into a repeating `"e/e/e/..."` sequence. This does
not by itself identify an audio-specific layer, but it shows that adjacent
shape-compatible late layers are not functionally interchangeable.

Fractional blends provide a less destructive insertion point and can be swept
in one loaded process:

```bash
./decoder_pipeline_bench --backend exla:rocm --wav journal1.wav --runs 1 \
  --blend 44:45:0.05 --blend 44:45:0.1 --blend 44:45:0.25
```

On the same reference, 1%, 2.5%, 5%, and 10% donor weight retained the exact
baseline text and token ids. At 25%, only the second word degraded, producing
`"all'arbori today feelingly fresh the morning light"`. Blending copies donor
tensors across backends without consuming them, so every candidate starts from
the same unmodified pipeline.

Extraction and execution can also happen in separate processes. The extractor
loads the source checkpoint once and writes a self-contained directory with the
recomposed safetensors, model/generation manifest, and tokenizer files:

```bash
./decoder_pipeline_bench extract \
  --artifact artifacts/gemma4-12b-44-to-45 \
  --backend torchx:cpu \
  --tail-start 45 \
  --transplant 44:45
```

Use `--blend 44:45:0.1` instead of `--transplant 44:45` to persist the exact-
output 10% blend.

The runner loads only that directory and compiles it for GPU XLA; it does not
load the Hugging Face model checkpoint or donor runtime:

```bash
./decoder_pipeline_bench run \
  --artifact artifacts/gemma4-12b-44-to-45 \
  --backend exla:rocm \
  --wav journal1.wav \
  --runs 2
```

The real 12B baseline and transplanted tensor files are each 25,833,722,613
bytes: transplanting replaces values without adding parameters. In a fresh
process the transplanted artifact deterministically filled the 32-token limit
with alternating token ids `[236744, 236786, ...]`, decoded as `"e/e/e/..."`.
Artifact manifests use Erlang terms and must only be loaded from trusted
sources.

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
and only marks committed final transcripts as safe to send downstream.

**Pass `--no-partials` when only final transcripts are consumed.** Partials are
throwaway feedback that nothing downstream reads, and each one costs a full
generation. Because a partial costs more than the partial interval, the queue
grows and delays the final behind it. Measured on `journal1.wav` with
`--realtime`: final lag 2.0s/3.5s without partials versus 4.7s/11.9s with them,
for identical transcripts. If partials are needed for a UI, raise
`--partial-interval-ms` until a partial costs less than the interval.

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
--realtime                     pace chunks to the wall clock (preloads the model first) and
                               annotate events with lag_ms; prints a lag summary to stderr
--no-repeat-ngram INT          ban repeating generated n-grams of this size, default 4, 0 disables
--partial-max-response-tokens INT
                               token cap for partial transcripts, default 16
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
./gemma_4_mic_transcribe --wav journal1.wav --trace 2>&1 | tee trace.log
mise run trace:elixir -- --wav journal1.wav --backend exla:rocm 2>&1 | tee trace.log
```

GPU-side tracing uses [bpftrace](https://bpftrace.org/) uprobes on the HIP
runtime to histogram `hipLaunchKernel`, `hipMemcpy*`, and
`hipStreamSynchronize` latencies for a running transcription. bpftrace needs
root, so it is wrapped in a mise task that attaches to the running `beam.smp`
(start a transcription first, then in another terminal):

```bash
# terminal 1: the run under test, both streams captured
./gemma_4_mic_transcribe --wav journal1.wav --stream-wav --realtime --repeat 8 \
  --output jsonl --backend exla:rocm --model-name gemma4-12b-qat-w4a16-ct \
  --max-response-tokens 32 --no-partials --debug 2>&1 | tee run.log

# terminal 2: once "generation prefill start" appears in run.log
mise run trace:bpf | tee hip-trace.txt          # oldest beam.smp
mise run trace:bpf -- <pid> | tee hip-trace.txt # specific pid
```

Pipe through `tee`, not `tail`: `tail` buffers until its input closes, so
interrupting the tracer discards everything it had collected. `tee` writes
each histogram as it prints.

Read the histograms as: many `@kernel_launch_us` entries at tens of
microseconds means per-call dispatch dominates and batching work differently
would help; `@sync_wait_us` dominating means the GPU is genuinely busy and the
per-call cost is real compute.

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
mise run trace:obi | tee obi-spans.txt   # attaches to beam.smp processes
```

Example span for a BEAM HTTP client call:

```text
2026-07-20 04:08:36 (11.5ms) HTTPClient 200 GET / [beam.smp:35280]->[104.20.23.154:80] traceparent=[00-dc61...]
```

BEAM TLS is implemented in Erlang rather than libssl, so OBI reports HTTPS
connections at the TCP level without decoding request contents; plain HTTP,
gRPC, and proxied traffic decode fully.

## Measurement reliability

`--repeat N` replays the audio N times against the loaded model, advancing the
audio timeline so lag stays comparable across passes. Four passes on
`journal1.wav` with `--no-partials`: a 2s utterance measured 2081/2083/2061/2087
ms and a 5s utterance 3561/3620/3505/3516 ms, so run to run spread is 26 ms and
115 ms. Per-utterance cost is stable once the model is loaded and every shape is
compiled, and differences above roughly 150 ms in the comparisons below are
signal rather than noise.

### Erlang VM tuning: measured, nothing to gain

The BEAM is not on the critical path, so its tuning flags have nothing to
give. Measured per-decode-step time (E4B, GPU, 35 warm steps per config):

```text
baseline (32 schedulers, JIT on)                        p50 72 ms  p90 77 ms
+sbwt/+sbwtdcpu/+sbwtdio very_long (spin-wait hard)     p50 74 ms  p90 78 ms
+S 8:8 +SDcpu 8:8 +SDio 4 (fewer BEAM threads)          p50 72 ms  p90 77 ms
```

A decode step is one dirty-NIF call into XLA; the GPU kernel is the whole
72 ms, and scheduler wakeup costs sit in the microseconds the percentiles
cannot see. Spin-waiting burned ~4 cores of idle CPU for a within-noise
regression - actively harmful on a shared machine. beamasm (the JIT) is
already on by default on this OTP. The one flag worth adopting is
`+JPperf true`, which lets `perf`/Parca resolve JIT-compiled Elixir frames
when profiling - observability, not speed.

## Latency budget

Streaming ASR services report two separate numbers, and mixing them hides which
one is slow:

- **EOT latency** — speaker stops to end-of-turn event. Ours is
  `--speech-end-silence-ms`, default 500 ms.
- **Transcript latency** — audio available to transcript emitted, excluding the
  endpoint wait. This is transcription speed.

`--realtime` reports both for finals (`bench: final ... eot_ms=... transcript_ms
...`). Measured on `journal1.wav` with `--no-partials`: EOT 500 ms, transcript
1.5-3.0 s, total 2.0-3.5 s.

For reference, Deepgram publishes 100-500 ms EOT and 150-300 ms transcript
latency for their purpose-built streaming models. Our endpointing is in that
range; transcription is roughly 5-10x slower, which is the cost of a 12B
multimodal LLM generating tokens autoregressively rather than a dedicated
streaming recognizer. Transcript latency here scales with generated tokens
(~90 ms each), so `--max-response-tokens` bounds the worst case.

Measured per final transcript on `journal1.wav` with `--realtime --no-partials`
(steady state; model load and device transfer excluded, since they are startup
costs):

```text
end-of-speech silence   500 ms   --speech-end-silence-ms
prefill               ~1100 ms   one pass over the utterance audio
decode                  90 ms    per generated token
audio -> tensor         7-25 ms  all WAV reading, resampling and framing
```

Audio ingestion is under 1.5% of the cost, so faster decoding of the input
format (GPU FLAC decode, for example) cannot move the total meaningfully. The
levers that matter are, in order: how many tokens a final generates, prefill,
and the end-of-speech wait.

Prefill is the one place where packed int4 loses: it gives up rocBLAS matrix
cores for a hand kernel (~1100 ms versus ~240 ms measured on the older
`Axon.dense` path).

`--weights hybrid` loads both representations (~31 GB) so prefill can use the
dequantized kernel while decode keeps packed int4. **Measured slower than
packed alone**: 2.6s/4.2s versus 2.0s/3.5s, with the dispatch verified correct
(no `exla_q4_gemm` calls, so prefill really did leave the hand kernel) and the
same result whether the dequantized kernel is f32 or bf16. The 240 ms figure
came from `Axon.dense`, which XLA fuses; the plain `Nx.dot` in the hybrid layer
does not reach the same path. Recovering it would mean matching that lowering,
not just holding a dequantized copy.

### Gemma 4 E4B versus 12B (`--model-name gemma4-e4b`)

Both models measured back-to-back on the same day, same harness
(`--backend exla:rocm --stream-wav --realtime --no-partials --debug`,
`journal1.wav`), E4B in its best configuration (exact incremental
prefill, bf16) against the 12B in its best (packed int4):

```text
                        E4B              12B packed
final lag               1.24-1.78 s      2.20-3.77 s
transcript latency      0.74-1.28 s      1.70-3.27 s
prefill (warm, live)    128-133 ms       1146-1689 ms
decode per token (p50)  72 ms            104 ms
decode per token (p90)  77 ms            110 ms
resident weights        16 GB bf16       ~7 GB packed
load + transfer         ~18 s            ~203 s
```

E4B's advantage compounds: ~10x faster prefill, ~1.4x faster decode, and
finals that are both shorter and less variable. The 12B wins only on
resident memory.

Quality on the same two utterances, against the HF reference
implementation's transcript of the same audio:

```text
reference (E4B, HF): "Okay, Bali today, feeling refreshed. The morning
                      light" / "Tomorrow I write a <garbled> and I
                      enjoyed a nice cup of coffee."
ours (E4B):          "How are you today? Feeling refreshed?" /
                     "Morning lights, biripo and i enjoy the nice cup
                      of coffee."
ours (12B):          "I'm feeling fresh." /
                     "The morning light is beautiful, and I enjoy a
                      nice cup of coffee."
```

The patterns are characteristic: E4B transcribes more literally and
garbles the hard word (so does the HF reference E4B - it is the model,
not this port), while the 12B produces the most fluent English but
compresses or paraphrases ("Okay, Bali today, feeling refreshed" became
"I'm feeling fresh."). For feeding a downstream LLM, E4B's
literal-but-occasionally-garbled output at half the latency is the
better trade; the 12B reads better to a human but silently drops
content, which a transcription pipeline cannot detect after the fact.

Transcript quality on this clip matches the HF reference implementation
run on the same audio (one mishear each). Caveats found on the way, all
fixed in the runtime: the conformer encoder and mel front end are verified
against `transformers` layer-by-layer to float rounding; convolutions are
computed as dots because MIOpen segfaults building conv kernels on gfx1151;
mel extraction is pinned to libtorch because the launcher does not load Mix
config and would otherwise run FFTs on `Nx.BinaryBackend`; and mel shapes
follow the audio token buckets so streaming never compiles mid-utterance.

**`--incremental-prefill` is worth turning on for E4B** (unlike the 12B,
where it measured worse). Audio prefills into the KV cache during speech,
so a final pays only the sub-chunk flush, the prompt suffix and decode:

```text
final lag           1.1-1.6 s   (E4B full path: 1.2-1.7 s)
transcript latency  0.6-1.1 s   (E4B full path: 0.7-1.2 s)
```

The gap grows with utterance length, since the full path re-prefills the
whole utterance while the incremental final's cost stays flat.

Chunked audio encoding is **exact**, not approximate. Mel frames are
extracted continuously across chunks (the utterance carries the unconsumed
tail of the padded sample stream), so chunked extraction is
bitwise-identical to whole-utterance extraction. The conformer encoder is
causal at token granularity with a bounded receptive field (~183 mel
frames: 11 frames of attention reach plus 4 of causal convolution per
block, over 12 blocks, plus the subsampling overhang), so each chunk is
encoded with the previous 200 mel frames prepended as lookback and only
the last placeholder-count encoder tokens spliced - reproducing
whole-utterance encoding exactly. Measured: incremental transcripts are
verbatim identical to the full path's. The exactness costs ~150 ms over
the approximate variant (lookback doubles encoder input per append, and
flushes pad to one warmed size instead of decomposing), which is the
right trade: deterministic parity with the reference-verified full path.

## Incremental prefill

Without it, streaming re-transcribes the whole utterance for every partial, so
each partial pays a full prefill over all audio so far. `Gemma4Unified.Runtime`
exposes an utterance cache (`start_utterance/2`, `append_audio/3`,
`transcribe_utterance/2`) that prefills the prompt prefix once, appends new
audio in fixed 50-token chunks so a single compiled executable serves every
append, and prefills only the short prompt suffix per transcript. The cache
returned by a transcript is discarded, because the suffix and generated tokens
must not be visible to the next audio append; Nx tensors being immutable makes
that rollback free. Disable with `incremental_prefill: false`.

This depends on appending several tokens to the KV cache at a non-zero offset,
which the single-shot path never does. That path was broken by a rotary
position bug (see below) and is covered by a test asserting chunked prefill
and one-token-at-a-time decode both match a single-shot prefill exactly.

Enable with `--incremental-prefill`. Transcripts match the one-shot path
exactly. Latency is a trade rather than a win: measured on `journal1.wav` with
`--no-partials`, a 2s utterance costs 2635ms against 2093ms one-shot, while a
5s utterance costs 3261ms against 3523ms. Prefill is hidden behind arriving
audio, so lag stops scaling with utterance length (spread 626ms versus
1430ms) at the price of a fixed prefix/flush/suffix overhead per utterance.
It pays off above roughly 4-5 seconds of speech.

Leftover audio is prefilled at its real size (decomposed into 50/25/10/5/2/1
token chunks) rather than padded up to a whole chunk. That helps short
utterances (2s: 2400ms versus 2635ms padded) and hurts longer ones (5s: 3719ms
versus 3261ms padded), because a decomposed remainder costs several prefill
calls instead of one and each call carries fixed dispatch overhead. Averaged
over both, one-shot prefill is still ahead on this file (2808ms).

The remaining idea worth testing is prefilling on a timer during speech, so
whole 50-token chunks are consumed as they complete and the flush is at most
one small chunk. That keeps the call count near the one-shot count while still
hiding prefill behind arriving audio.

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
