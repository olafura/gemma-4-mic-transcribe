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

Generation now explicitly releases each consumed logits buffer, the final
request-local K/V cache, and copied audio tensors after token selection has
synchronized the XLA result. Intermediate caches are not manually released:
XLA donates them forward into the next decode step, so only the final returned
cache has unambiguous ownership.

After a streaming utterance is reset, its long-running owner performs one minor
BEAM collection. This releases any remaining unreachable NIF references without
major-collecting the old-generation model runtime on every token. Final and
suppressed events report the pause as `metrics.cleanup_gc_us`; programmatic
sessions can disable it with `post_utterance_gc: false` for comparison. A real
two-window packed-12B ROCm run completed after deterministic cleanup with the
same transcripts as before, confirming that persistent parameters and compiled
executables remain valid across requests.

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

The `expert_tool` escript can range-extract one routed expert directly from the
official 26B-A4B safetensors checkpoint. The leading expert axis is contiguous,
so it downloads only the two selected slices and writes separate gate, up, and
down matrices:

```bash
GEMMA4_ESCRIPT=expert mix escript.build

./expert_tool extract \
  --artifact artifacts/gemma4-26b-layer0-expert0 \
  --layer 0 --expert 0

./expert_tool inspect \
  --artifact artifacts/gemma4-26b-layer0-expert0

mix gemma.expert run \
  --artifact artifacts/gemma4-26b-layer0-expert0 \
  --backend exla:rocm --tokens 1 --runs 20
```

Extraction and inspection are pure BEAM escript operations. Native execution
runs as a started Mix application so EXLA resolves its NIF from the real
application directory through `:code.priv_dir(:exla)`. A production service
should use an OTP release for the same reason; it should not discover native
libraries from a hard-coded `_build` path.

The real layer-0/expert-0 artifact contains 5,947,392 BF16 parameters
(11,894,784 bytes). Extraction transferred 12,024,592 bytes including the
safetensors header instead of the complete 51,611,872,412-byte checkpoint. A
warmed one-token standalone expert call measured 170 microseconds median
(150 microseconds minimum) on the local ROCm GPU and returned a nonzero
`{1, 2816}` hidden-state matrix.

The standalone operation exactly matches the official expert body:

```text
down(gelu_tanh(gate(x)) * up(x))
```

Its input must be the pre-FFN-normalized layer state. Router selection and
weighting, the always-on shared FFN, surrounding norms, and the residual
connection remain outside the artifact. Consequently its output is a useful
matrix-level building block, not meaningful text by itself. The current full
Axon runtime still rejects complete MoE inference until those surrounding MoE
operations are implemented.

To preserve those surrounding operations, extract a complete MoE feed-forward
layer instead:

```bash
./expert_tool extract-layer \
  --artifact artifacts/gemma4-26b-layer0-moe \
  --layer 0

./expert_tool inspect-layer \
  --artifact artifacts/gemma4-26b-layer0-moe

mix gemma.expert run-layer \
  --artifact artifacts/gemma4-26b-layer0-moe \
  --backend exla:rocm --tokens 1 --runs 5
```

`Gemma4MicTranscribe.Gemma4.ExtractedMoeLayer.run/2` accepts the residual stream
immediately after attention as a `{tokens, 2816}` tensor. It executes the
always-on shared FFN and the router, selects and renormalizes eight of the 128
routed experts for each token, applies the per-expert scales, and combines both
paths with the five feed-forward RMS norms, residual connection, and layer
scalar. It returns the output plus router probabilities, selected expert
indices, selected weights, and the separate shared/routed outputs.

The layer-0 artifact has 779,485,825 loaded BF16 parameters. Its contiguous
source range is 1,558,982,914 bytes, compared with the complete 51,611,872,412
byte checkpoint. A one-token ROCm run over a constant `0.01` residual selected
experts `[126, 34, 101, 84, 79, 114, 56, 55]` and measured 1.18 ms median after
warmup. Negating that input selected a disjoint set
`[53, 122, 90, 74, 117, 2, 124, 41]`, demonstrating that the extracted router
is active rather than replaying fixed experts.

This is specifically a language-model MoE experiment. The 26B-A4B checkpoint
does not contain Gemma 4's audio encoder, so the artifact cannot isolate or
transcribe audio. It is useful for studying, replacing, and eventually
recombining the router and expert mechanism. The CLI benchmark uses synthetic
residual states; meaningful text still requires embeddings, attention, every
decoder layer, final normalization, and the language-model head.

The first specialization probe compares real token embeddings from curated math
and control corpora against the extracted layer-0 router:

```bash
mix gemma.expert profile-math \
  --artifact artifacts/gemma4-26b-layer0-moe \
  --backend exla:rocm --limit 10
```

This range-fetches only the embedding rows used by the corpora and loads only
the router's projection and scale tensors. Expert 112 was the strongest stable
math candidate: it was selected for 29/75 math tokens versus 9/68 controls
(38.7% versus 13.2%, 2.72x enrichment). On a held-out set of complete math and
ordinary-language questions it remained enriched at 21/91 versus 9/92 (23.1%
versus 9.8%, 2.22x). Expert 32 was second on the discovery vocabulary, while
experts 9 and 16 showed narrower spikes on symbolic problem tokens.

These are candidates, not semantic labels embedded in the checkpoint. Layer 0
uses post-attention residual states in real inference, while this inexpensive
probe substitutes token embeddings. A definitive result requires capturing
router decisions across all 30 layers on real math and control prompts, then
ablating the consistently enriched experts and measuring math-specific quality
loss.

The layer-0 caller closes the token-embedding approximation. It separately
extracts the eight attention and norm tensors needed to produce the real router
input:

```bash
./expert_tool extract-caller \
  --artifact artifacts/gemma4-26b-layer0-caller

mix gemma.expert call-expert \
  --artifact artifacts/gemma4-26b-layer0-moe \
  --caller-artifact artifacts/gemma4-26b-layer0-caller \
  --expert-artifact artifacts/gemma4-26b-layer0-expert112 \
  --text "Solve the quadratic equation and prove the theorem using a matrix determinant." \
  --backend exla:rocm

mix gemma.expert call-layer \
  --artifact artifacts/gemma4-26b-layer0-moe \
  --caller-artifact artifacts/gemma4-26b-layer0-caller \
  --expert-artifact artifacts/gemma4-26b-layer0-expert112 \
  --text "Solve the quadratic equation and prove the theorem using a matrix determinant." \
  --expert-scale 1.0 --backend exla:rocm

./expert_tool extract-caller \
  --artifact artifacts/gemma4-26b-layer1-caller --layer 1

./expert_tool extract-layer \
  --artifact artifacts/gemma4-26b-layer1-moe --layer 1

mix gemma.expert call-chain \
  --artifact artifacts/gemma4-26b-layer0-moe \
  --caller-artifact artifacts/gemma4-26b-layer0-caller \
  --expert-artifact artifacts/gemma4-26b-layer0-expert112 \
  --next-artifact artifacts/gemma4-26b-layer1-moe \
  --next-caller-artifact artifacts/gemma4-26b-layer1-caller \
  --next-artifact artifacts/gemma4-26b-layer2-moe \
  --next-caller-artifact artifacts/gemma4-26b-layer2-caller \
  --text "Solve the quadratic equation and prove the theorem using a matrix determinant." \
  --expert-scale 0.0 --backend exla:rocm

mix gemma.expert call-prefix \
  --artifact-prefix artifacts/gemma4-26b \
  --expert-artifact artifacts/gemma4-26b-layer0-expert112 \
  --last-layer 5 \
  --text "Solve the quadratic equation and prove the theorem using a matrix determinant." \
  --expert-scale 0.0 --backend exla:rocm

./expert_tool extract-head \
  --artifact artifacts/gemma4-26b-output-head

mix gemma.expert call-prefix \
  --artifact-prefix artifacts/gemma4-26b \
  --expert-artifact artifacts/gemma4-26b-layer0-expert112 \
  --head-artifact artifacts/gemma4-26b-output-head \
  --text "Solve the quadratic equation and prove the theorem using a matrix determinant." \
  --chat --expert-scale 0.0 --backend exla:rocm

mix gemma.expert generate-prefix \
  --artifact-prefix artifacts/gemma4-26b \
  --expert-artifact artifacts/gemma4-26b-layer0-expert112 \
  --head-artifact artifacts/gemma4-26b-output-head \
  --text "Solve the quadratic equation and prove the theorem using a matrix determinant." \
  --chat --expert-scale 0.0 --max-new-tokens 8 \
  --expert-cache-gb 16.0 --backend exla:rocm
```

The caller loads only the router and routed-input norm from the MoE artifact.
It fetches the prompt's embedding rows, prepends the model's `<bos>` token, and
runs the following path on XLA:

```text
token embeddings
  -> sqrt(hidden_size) scaling
  -> layer-0 input RMS norm
  -> Q/K/V projections, Q/K norms, RoPE, causal sliding attention
  -> output projection and post-attention RMS norm
  -> attention residual
  -> router top-8 selection
  -> pre_feedforward_layernorm_2
  -> selected standalone expert
```

The caller artifact contains 34,609,152 BF16 parameters (69,218,942 bytes),
rather than another copy of the 1.56 GB MoE layer or the 51.6 GB checkpoint.
On the real prompt above, expert 112 was selected for `equation` as route 0
(router probability 0.2531, routed weight 0.5905) and `determinant` as route 7
(0.0200, 0.0559). Its standalone output was a nonzero `{2, 2816}` matrix with
mean absolute value 0.2005. The matrix is useful as an intermediate activation;
`call-layer` now inserts it into the selected top-8 routes, combines the other
routed experts and shared FFN, and returns the complete `{14, 2816}` layer-0
output. With scale `1.0`, the standalone replacement matched the original
expert bank to a mean absolute layer-output difference of `1.70e-8` and a
maximum difference of `1.53e-5` (BF16 rounding).

`--expert-scale` is a controlled replacement hook. At `0.0`, expert 112 was
ablated only on its two selected routes; the complete layer output moved by
`0.02149` mean absolute and `16.2946` maximum while all routing decisions and
other experts remained unchanged. A newly trained artifact with the same
`2816 -> 704 -> 2816` contract can use the same slot. The validation path also
computes the original bank output so it can report a baseline; removing that
comparison is a later runtime optimization.

Attention callers and `call-chain` are layer-generic, so the complete layer-0 output can now
remain on the ROCm device and feed layer 1 without repeating the token-embedding
scale or crossing through host memory. Repeated `--next-artifact` and
`--next-caller-artifact` pairs form a validated contiguous list; later layer
weights load sequentially, so extending the chain does not require every layer
to reside on the GPU simultaneously. Sliding layers use 256-wide attention
heads. Full-attention layers use the checkpoint's 512-wide global heads, two
shared K/V heads, and proportional partial RoPE. The real layer-5 caller
validated that path across both checkpoint shards: 49,027,584 BF16 parameters
(98,055,722 artifact bytes), with no duplicate value projection.

The initial three-layer chain produced `{14, 2816}` at every boundary, with fresh
top-8 routing decisions per token and layer. Ablating layer-0 expert 112
propagated mean absolute output deltas of `0.02313` at layer 1 and `0.01902` at
layer 2; the corresponding maxima were `7.5370` and `5.7120`. The effect is
observable across an arbitrary extracted decoder-layer chain.

The chain has since been extended through layer 5, the first full-attention
block. The layer-0 ablation remained measurable there with mean absolute output
delta `0.02374` and maximum delta `2.7935`, against mean absolute activation
`0.53446`. The full six-layer run took 47.6 seconds including application
startup, loading six MoE artifacts, and first-time XLA compilation. Use
`call-prefix` to derive a validated ordered chain from the
`PREFIX-layerN-{moe,caller}` names instead of repeating two flags per layer.
The separately extracted output head contains the final RMS norm and tied token
embedding projection. When supplied to a complete 30-layer prefix, it reports
the top next-token logits for both the expert-modified and baseline paths.
`--chat` wraps the text in the canonical one-turn Gemma 4 template with thinking
disabled; omit it when raw token activation probes are intentional.

A complete extracted 30-layer chat pass now reaches the independently extracted
output head. For a 26-token math prompt, expert 112 was selected twice. Ablating
it left the top next-token candidate as `To`, but changed the top-10 set:
baseline candidate `Solving` was replaced by `While`. The mean absolute
activation delta grew through the decoder, peaked at `0.03873` in layer 25, and
ended at `0.02181` after layer 29. The run took 126.9 seconds end to end,
including BEAM startup, artifact verification and loading, and first-time XLA
compilation. This is a useful next-token distribution, not autoregressive text
generation yet. `generate-prefix` closes that loop: it greedily selects a token,
reads the corresponding row from the extracted tied embedding matrix, appends
it to the input, and runs the independently assembled model again. Generation
uses output-only compiled layer entry points, so it does not execute the
diagnostic baseline path or return every layer's routing tensors.

The first full-prefix generator produced `To solve` in two real decoder passes.
Its steps took 97.8 and 108.7 seconds. A scale-1 run selected the same first
token, `To`, confirming parity at the greedy decision boundary.

Generation now prefills a fixed-size K/V cache for each extracted layer and
feeds only the preceding token through subsequent decode passes. On the same
ablated prompt it generated `To solve a`. Prefill took 108.7 seconds, the first
one-token decode took 66.8 seconds, and the next same-shape decode took 55.3
seconds. Relative to the 108.7-second full-prefix second pass, those decode
steps were 38.5% and 49.1% faster.

Artifact checksums are now verified once during prefill rather than rereading
every file only to hash it for every token. Validation remains the default for
all public artifact loaders; only already-validated artifacts use the explicit
unchecked reload path. This reduced the two cached decode steps to 27.0 and
14.0 seconds.

One-token decoding now also loads only the eight experts selected by each
router. Attention, shared FFN, router, and normalization shells for layers 1–29
remain resident on the GPU, occupying 3,208,272,860 bytes. Per-token matrix
streaming fell from 47,362,026,460 bytes for those complete layers to
2,759,589,888 bytes of selected experts—a 94.2% reduction. A four-token run
produced `To solve a quadratic`: prefill took 107.2 seconds, the shell-loading
decode took 13.2 seconds, and subsequent steady-state tokens took 1.87 and 1.52
seconds. The latter is 98.6% faster than the original 108.7-second
full-prefix decode.

A bounded GPU LRU now retains exact BF16 routed experts across tokens. The
default 16 GiB limit leaves headroom for the output head, resident decoder
shells, transient route banks, and XLA. It is configurable with
`--expert-cache-gb`; evicted entries are explicitly deallocated.

An eight-token run produced `To solve a quadratic equation using a matrix`,
preserving the uncached prefix exactly. The cache recorded 866 hits and 758
misses—a 53.3% hit rate—using 9,016,246,272 bytes across 758 entries with no
evictions. Populating the cache made the first sparse decode slower at 24.8
seconds, but later tokens settled at 1.32–1.43 seconds instead of the uncached
1.52–1.87 seconds. This is an exact long-running-service optimization rather
than quantization, so it introduces no numerical approximation.

The extracted generator can now be loaded once and owned by a long-running
process:

```elixir
{:ok, model} =
  Gemma4MicTranscribe.Gemma4.ExtractedGeneratorServer.start_link(
    artifact_prefix: "artifacts/gemma4-26b",
    expert_artifact: "artifacts/gemma4-26b-layer0-expert112",
    head_artifact: "artifacts/gemma4-26b-output-head",
    backend: {EXLA.Backend, client: :rocm},
    expert_cache_bytes: 16 * 1024 * 1024 * 1024
  )

result =
  Gemma4MicTranscribe.Gemma4.ExtractedGeneratorServer.generate(model,
    input_text: prompt,
    expert_scale: 0.0,
    max_new_tokens: 4
  )

GenServer.stop(model)
```

The output head, complete override layer, compiled XLA functions, sparse
decoder shells, and routed-expert LRU survive across requests. Token state and
fixed-shape K/V caches remain request-local. `generate-prefix --runs N`
exercises this lifecycle and reports both decoder `processing_us` and complete
call `wall_us`, excluding model startup from every request measurement.

Prompt prefill now uses the same sparse path as one-token decode. Each layer
loads one compact copy of every expert selected anywhere in the prompt and
remaps its global router IDs into that bank. These prompt banks are deliberately
ephemeral: admitting a one-pass prompt scan filled the 16 GiB LRU and caused
cyclic eviction of the much smaller recurring decode working set.

For two identical four-token requests over the 26-token math prompt, the first
request took 113.04 seconds while compiling the new shapes. The warm request
took 13.62 seconds of processing: 9.77 seconds for prefill and 1.07–1.52 seconds
for each decode token. Before sparse prefill, the same persistent warm request
took 50.54 seconds, including a 47.44-second prefill. This is a 73.1% reduction
in warm request processing time and a 79.4% reduction in warm prefill. It
preserved the greedy output `To solve a quadratic`. The decode-only cache used
5.63 GiB across 473 experts with zero evictions.

The decode cache now stores those experts as one contiguous GPU table per
layer. Router IDs map directly to resident table slots inside the sparse XLA
finish graph; the old path copied and concatenated eight individual matrices
for every layer and token. On the same warm request, decode fell from
1.07–1.52 seconds per token to 78–84 milliseconds per token. Total four-token
processing fell again from 13.62 to 10.18 seconds, of which 9.93 seconds was
prefill. The 29 tables occupied 5.61 GiB across 472 exact BF16 experts with no
evictions, survived repeated XLA calls without buffer donation, and preserved
`To solve a quadratic`.

Transient expert banks, request K/V caches, and terminated model resources are
explicitly released with `Nx.backend_deallocate/1`. Erlang's
[per-process generational collector](https://www.erlang.org/doc/apps/erts/garbagecollection.html)
tracks BEAM heap, stack, and referenced off-heap binaries; it does not use ROCm
device-memory pressure as a collection trigger. Waiting for a later BEAM
collection therefore allowed already-finished XLA buffers to exhaust the GPU
during the complete chain.

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
instead of 25.83 GB for bf16. `run-split` keeps the files independently
replaceable but merges their parameter maps into one XLA graph after loading.
Warm fused generation took 2.019–2.022 seconds and produced
`"I woke up today feeling refreshed. The morning light"`, exactly matching the
resident packed runtime at 2.020–2.054 seconds. A one-token run measured about
1.19 seconds of prefill; the remaining nine tokens averaged roughly 92
ms/token. Artifact and checkpoint load times are intentionally excluded because
they occur before the long-running service accepts work.

The gfx1151 ROCm prefill kernel now uses the GPU's packed BF16 dot-product
instruction and a 16-token reuse tile. On the same 12B journal input, warm
processing fell to 1.415–1.429 seconds (about 30% faster) with identical token
ids. The seeded 33-language single-word gate also produced zero changed outputs
and identical CER while mean processing fell from 1.251 seconds to 0.675
seconds, a 1.85x speedup. Other ROCm architectures retain the portable scalar
dot-product loop.

Decode uses the same gfx1151 packed BF16 instruction while retaining the
scalar kernel's accumulate-then-scale order. In a forced 32-token run, this
reduced warm processing from a 3.591-second scalar mean to 3.131 seconds
(12.8%) with all 32 token ids unchanged. The natural ten-token journal result
remained identical at roughly 1.40 seconds; its smaller gain is expected
because prefill is unchanged. The 33-language gate again reported zero changed
outputs and identical CER.

`--fused-ffn` aliases each layer's existing packed gate/up tensors into one
dual-projection custom call for composed decode; the independently stored
artifacts and prefill graph are unchanged. A forced 64-token run fell from a
5.916-second warm mean to 5.675 seconds (4.1%, about 3.8 ms/token), with all 64
token ids unchanged. On the eight-token multilingual gate it reduced mean
processing from 0.694 to 0.676 seconds (2.6%) with zero changed outputs and
identical CER.

The normal long-running transcription runtime exposes the same optimization:

```bash
./gemma_4_mic_transcribe --wav journal1.wav --stream-wav --realtime \
  --backend exla:rocm --model-name gemma4-12b-qat-w4a16-ct \
  --max-response-tokens 32 --no-partials --fused-ffn
```

It builds a second decode graph over aliases to the already-loaded parameters,
so resident weights are not duplicated and the original graph still handles
prefill. On 42 warm decode steps from two `journal1.wav` passes, p50/p90 fell
from the previously measured 104/110 ms to 94/101 ms. Both transcripts remained
verbatim identical. The flag requires packed or hybrid weights and has no effect
on E4B.

Use `--execution split` when an observable runtime boundary is required. It
dispatches the prefix and tail as separate XLA executables and measured
2.12–2.16 seconds warm. The default `--execution composed` avoids that dispatch
and materialization overhead without changing how the artifacts are stored or
swapped.

### Multilingual single-word regression gate

Build the corpus benchmark as a dedicated escript and take a seeded random
sample from every language with a non-empty test split:

```bash
GEMMA4_ESCRIPT=single_word_bench mix escript.build

./single_word_bench \
  --corpus /path/to/cv-corpus-7.0-singleword \
  --prefix-artifact artifacts/gemma4-12b-packed-prefix-0-44 \
  --tail-artifact artifacts/gemma4-12b-packed-tail-45-47 \
  --output artifacts/single-word-packed-native-seed42.json \
  --per-language 1 \
  --seed 42 \
  --seconds 3 \
  --max-new-tokens 8 \
  --backend exla:rocm
```

The runner decodes MP3 with `ffmpeg`, pads or truncates every clip to one fixed
shape, warms XLA before measuring, and records normalized exact match, character
error rate, per-language results, and latency percentiles. A candidate run can
compare the exact same seeded clips with `--baseline`:

The same gate can run a complete catalog model without extracting it first.
This is useful for measuring E4B on the identical multilingual sample used for
12B topology experiments:

```bash
./single_word_bench \
  --corpus /path/to/cv-corpus-7.0-singleword \
  --model-name gemma4-e4b \
  --baseline artifacts/single-word-packed-native-seed42.json \
  --output artifacts/single-word-e4b-seed42.json \
  --per-language 1 --seed 42 --backend exla:rocm --allow-regression
```

On the seed-42 sample, E4B averaged 268 ms per three-second clip versus
1,251 ms for the packed 12B baseline: a 4.67x processing speedup. It improved
exact matches from 5/33 to 8/33 and CER from 0.742 to 0.583. The outputs were
not a strict quality superset: E4B gained five exact matches (including Tamil
and Thai) but lost Catalan `sí` and Polish `jeden` to small transcription
errors. Script- or output-shape routing cannot identify those subtle losses;
the next useful routing signal is decoder token confidence.

```bash
./single_word_bench \
  --corpus /path/to/cv-corpus-7.0-singleword \
  --prefix-artifact artifacts/candidate-prefix \
  --tail-artifact artifacts/candidate-tail \
  --baseline artifacts/single-word-packed-native-seed42.json \
  --output artifacts/candidate-seed42.json \
  --per-language 1 --seed 42
```

The comparison reports changed outputs, exact matches lost and gained, accuracy
and character-error deltas, and processing speedup. The seed-42 packed baseline
covered 33 languages and scored 5/33 normalized exact matches with 0.742 CER;
its mean warm processing time was 1.251 seconds per three-second clip. The low
absolute exact score reflects the model's frequent transliteration of less
common native scripts, so topology experiments should guard both exact-match
transitions and aggregate CER. A baseline comparison exits non-zero when it
loses any exact match or worsens CER; pass `--allow-regression` only for an
exploratory run whose rejected result still needs to be recorded.

Initial identity-bypass smoke tests rejected every untrained layer deletion:
layer 11 lost all five known-correct words, layers 24 and 25 each lost the
Catalan accent, and layers 36 and 42 each lost two of five exact matches. No
shortened graph is selected by default. The `--bypass-layers` option remains an
experimental mechanism for evaluating a distilled or fine-tuned shortened
checkpoint against the same gate.

FFNs can be tested separately with `--bypass-ffn-layers`. Use
`--bypass-phase decode` to run the complete graph for audio/prompt prefill and
the shortened graph only for autoregressive token steps. This keeps the audio
representation intact and allows the prefill KV cache to flow directly into a
different compiled decode graph:

```bash
./single_word_bench \
  --corpus /path/to/cv-corpus-7.0-singleword \
  --prefix-artifact artifacts/gemma4-12b-packed-prefix-0-44 \
  --tail-artifact artifacts/gemma4-12b-packed-tail-45-47 \
  --baseline artifacts/single-word-packed-native-seed42.json \
  --output artifacts/single-word-no-ffn-25-43-decode-seed42.json \
  --bypass-ffn-layers 25,43 \
  --bypass-phase decode
```

On the seed-42 33-language gate, decode-only FFN layers 25 and 43 lost no exact
matches, improved exact matches from 5 to 6, and reduced CER from 0.742 to
0.717. They also preserved all ten baseline tokens for the first five seconds
of `journal1.wav`. A fixed 32-step run measured 3.830 seconds warm for the full
graph and 3.767 seconds for the candidate: 1.7% end-to-end, or approximately
3.1% for the repeated decode portion after subtracting their shared prefill
cost. This is a useful experimental candidate, not a production-quality proof:
the regression corpus has only one seeded clip per language. Skipping six FFNs
passed that small gate but truncated the journal transcript, demonstrating why
both checks are required.

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

For gated Hugging Face checkpoints, set `HF_TOKEN` in the service environment.
The runtime also accepts an `:auth_token` option when it is embedded directly.

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

Call `StreamingSession.subscribe_speech_end/2` to receive
`{StreamingSession, :speech_end, session, event, monotonic_ms}` as soon as the
endpointer commits an utterance. The notification is sent before input building,
prefill, or decode begins, so a voice pipeline can react to end-of-turn without
waiting for the final transcript call to return.

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

- **EOT event latency** — speaker stops to the delivered end-of-turn event.
- **Transcript latency** — the wall-clock audio cursor when a transcript is
  delivered minus the transcript's audio cursor.

`--realtime` subscribes to the immediate speech-end notification, reports both
event lags, and splits final transcript latency into
`endpoint_detection_ms` (audio consumed after the final active speech frame)
and `post_endpoint_ms` (input build, prefill, and decode). The split is measured
from event timestamps; it no longer subtracts the configured silence threshold
and calls the remainder transcript latency. The regular `push_audio/3` return
still contains both events for compatibility, while subscribers receive
`speech_end` before generation. Consequently the benchmark's speech-end lag is
normally the configured 500 ms and final lag includes model processing.

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
endpoint detection      500 ms   --speech-end-silence-ms
prefill               ~1100 ms   one pass over the utterance audio
decode                  90 ms    per generated token
audio -> tensor         7-25 ms  all WAV reading, resampling and framing
```

Audio ingestion is under 1.5% of the cost, so faster decoding of the input
format (GPU FLAC decode, for example) cannot move the total meaningfully. The
levers that matter are, in order: how many tokens a final generates, prefill,
and the end-of-speech wait.

Streaming prefill uses warmed audio-token buckets at 25-token intervals from
25 through 200. This bounds masked padding to under one second instead of
rounding directly from 100 to 200 tokens. On the two `journal1.wav` utterances,
the selected buckets fell from 100/200 to 75/150 with verbatim-identical
transcripts. Combined with `--fused-ffn`, average post-endpoint processing fell
from 1709 to 1577 ms (7.7%), and total transcript latency fell from 2209 to
2077 ms (6.0%). The additional shapes increase startup warmup only; they do not
duplicate weights.

### Learned E2B-to-12B cascade experiment

The [Cactus Hybrid checkpoint](https://huggingface.co/Cactus-Compute/gemma-4-e2b-it-hybrid)
adds a 64,833-parameter correctness probe to Gemma 4 E2B. It reads the output
of decoder layer 28 for each generated token, then applies layer normalization,
a 1536-to-32 projection, learned attention pooling, and a small binary
classifier. The probe predicts whether the fast answer is wrong; this runtime
reports `1 - p_wrong` as handoff confidence.

The extractor downloads only the safetensors header and contiguous probe byte
range from the Cactus shard (329,372 bytes for the current release), verifies
the tensor names, shapes, dtype, and checksum, and writes a separate 255 KiB
artifact. The E2B audio model remains the official audio-capable checkpoint:

```bash
GEMMA4_ESCRIPT=handoff_probe mix escript.build
./handoff_probe extract --artifact artifacts/cactus-e2b-handoff-probe
./handoff_probe inspect --artifact artifacts/cactus-e2b-handoff-probe
```

`--model-cascade` keeps the fast E2B model, probe, and selected accurate model
loaded. It accepts sufficiently confident E2B output without running 12B and
escalates low confidence, empty output, refusals, malformed control tags,
replacement characters, or fast-model errors:

```bash
./gemma_4_mic_transcribe --wav journal1.wav --stream-wav --realtime \
  --backend exla:rocm --model-name gemma4-12b-qat-w4a16-ct \
  --weights packed --fused-ffn --model-cascade \
  --handoff-probe-artifact artifacts/cactus-e2b-handoff-probe \
  --cascade-min-handoff-confidence 0.9 --no-partials
```

The probe scorer uses a fixed 1,024-row masked input. XLA therefore compiles it
once during startup warmup instead of compiling a new executable for every
generated-token count. Model loading, device transfer, and probe compilation
remain outside measured request processing in a long-running service.

An initial `journal1.wav` run proved the learned route works: the second
utterance fell below `0.9`, escalated, and 12B returned `"The morning light was
beautiful, and I enjoyed a nice cup of coffee."` The first utterance was
incorrect but scored above `0.9` and was accepted. This is useful machinery,
not yet a production threshold: Cactus reports zero audio examples in the
probe's training mix, so its confidence must be calibrated on our multilingual
audio regression corpus.

With the file replayed twice in one loaded process and startup warmup disabled,
the first pass compiled the required shapes and the identical second pass
measured 385 ms for the accepted E2B route. The escalated utterance took 2,849
ms total because it necessarily paid for both E2B and 12B. Its learned scores
were stable across passes: 0.9200 for the incorrect accepted transcript and
0.8506 for the correctly escalated one. Load, device transfer, and first-shape
compilation are excluded from those second-pass numbers.

The density threshold also defaults to zero (disabled), because characters per
second is not language-independent confidence. The cascade deliberately does
not expose incremental prefill yet; accepted requests use the fast model's full
path, while escalated requests use the already-loaded optimized 12B path.

`--cascade-min-logit-margin` collects the top-two decoder-logit margin for each
fast-model output token and escalates when the minimum is below the threshold.
It is also disabled by default. On the 33-language seed-42 gate, `0.125` escalated
only the zero-margin Catalan error, raised the combined exact score from 8 to 9,
and estimated 299 ms mean cascade processing versus 1,251 ms for 12B alone.
The Polish `jeden`/`jedem` error remained confidently wrong, so this signal is
useful but not sufficient and the threshold should be validated on more data.
The same `0.125` threshold also accepted the journal transcript containing
`biripo`; logit margin measures decisiveness, not correctness, and cannot be
the cascade's only semantic quality signal.

`--self-review` is a separate prompt-guided experiment. It asks the selected
model for a draft, then sends the same audio and draft back to the model with a
correction instruction. It is off by default: it runs inference twice and the
draft makes prompt lengths variable, so a service may compile more prefill
shapes. Use it to measure whether revision quality justifies that cost before
enabling it in production. On the second `journal1.wav` utterance, E4B review
changed the official-prompt draft's invented `birapul` to the real but still
incorrect word `purple`; it improved lexical plausibility without recovering
`is beautiful`, so review is not enabled as a quality policy.

The cascade emits `[:gemma_4_mic_transcribe, :cascade, :route]` telemetry with
the selected route, escalation reason, confidence values, and per-model
processing time. Streaming benchmark runs also print accepted/escalated counts
and average model times, so router changes can be evaluated independently from
model load and warmup. `--e4b-cascade` remains an alias for the original
E4B-first configuration; `--cascade-fast-model` can select another fast model.

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

The same configuration-driven conformer/decoder runtime also accepts the
smaller `--model-name gemma4-e2b`. E2B is especially useful for learned
handoff experiments: Cactus Hybrid's released correctness probe targets E2B
decoder layer 28 at width 1536, whereas E4B's layer width is 2560 and cannot
reuse those probe weights directly.

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
