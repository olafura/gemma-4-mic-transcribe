defmodule Gemma4MicTranscribe.Gemma4Unified.Runtime do
  @moduledoc false

  require Logger

  alias Gemma4MicTranscribe.Config
  alias Gemma4MicTranscribe.Gemma4Unified.AudioFeatureExtractor
  alias Gemma4MicTranscribe.Gemma4Unified.ChannelState
  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Model
  alias Gemma4MicTranscribe.Gemma4Unified.Prompt
  alias Gemma4MicTranscribe.Gemma4Unified.TokenSelection
  alias Gemma4MicTranscribe.Gemma4Unified.Transcript
  alias Gemma4MicTranscribe.ModelCatalog
  alias Gemma4MicTranscribe.RocmPreflight

  # Cache lengths are rounded up to this step so every generation with the
  # same audio bucket reuses one compiled prefill executable and all buckets
  # share one compiled decode executable.
  @cache_length_step 512

  # Audio is prefilled in fixed-size chunks so every append reuses one compiled
  # executable (2 seconds at 640 samples per token).
  @audio_chunk_tokens 50
  # Leftover audio is decomposed into these sizes instead of being padded up to
  # a whole chunk, so a short utterance prefills the tokens it actually has.
  # Each size is one more compiled executable, all warmed at startup.
  @audio_chunk_sizes [50, 25, 10, 5, 2, 1]
  @max_utterance_audio_tokens 750
  @prompt_headroom_tokens 128

  defstruct [
    :model_name,
    :repo_id,
    :backend,
    :param_type,
    :max_response_tokens,
    :model_info,
    :tokenizer,
    :generation_config,
    :suppressed_token_ids,
    :suppression_mask,
    :channel_token_ids,
    :inside_channel_suppression_mask,
    :content_suppression_mask,
    :no_repeat_ngram_size,
    :e4b?,
    :predict_fun,
    :debug,
    :debug_top_k
  ]

  def load(opts \\ []) do
    debug? = Keyword.get(opts, :debug, false)
    model_name = Keyword.fetch!(opts, :model_name)

    with :ok <- verify_supported_model(model_name),
         :ok <- verify_bumblebee_available(),
         {:ok, backend} <- backend(Keyword.get(opts, :backend, Config.backend())),
         {:ok, param_type} <- param_type(Keyword.get(opts, :param_type)) do
      repo_id = ModelCatalog.resolve(model_name)
      repo = {:hf, repo_id}

      # E4B is a different architecture: conformer audio encoder, per-layer
      # input embeddings, KV sharing. It gets its own spec module.
      spec_module = spec_module_for(repo_id)

      spec_opts = [
        module: spec_module,
        architecture: :for_conditional_generation
      ]

      model_opts = if backend, do: [backend: backend], else: []

      log_debug(debug?, fn ->
        "runtime: resolved model #{inspect(model_name)} -> #{repo_id}; backend=#{inspect(backend)} " <>
          "param_type=#{inspect(param_type)}"
      end)

      log_backend_details(debug?, backend)

      with {:ok, spec} <-
             timed_debug(debug?, "runtime: Bumblebee.load_spec #{repo_id}", fn ->
               Bumblebee.load_spec(repo, spec_opts)
             end),
           spec <- configure_spec(spec, param_type, opts),
           {:ok, model_info} <-
             timed_debug(
               debug?,
               "runtime: Bumblebee.load_model #{repo_id} (checkpoint download/load)",
               fn ->
                 model_opts
                 |> Keyword.put(:spec, spec)
                 |> maybe_put_param_type(spec, param_type)
                 |> Keyword.replace_lazy(:backend, &param_load_backend/1)
                 |> then(&Bumblebee.load_model(repo, &1))
               end
             ),
           {:ok, tokenizer} <-
             timed_debug(debug?, "runtime: Bumblebee.load_tokenizer #{repo_id}", fn ->
               Bumblebee.load_tokenizer(repo, type: :gemma)
             end),
           {:ok, generation_config} <-
             timed_debug(debug?, "runtime: Bumblebee.load_generation_config #{repo_id}", fn ->
               Bumblebee.load_generation_config(repo, spec_module: spec_module)
             end) do
        {_init_fun, predict_fun} =
          timed_debug(debug?, "runtime: Axon.build predict function", fn ->
            Axon.build(model_info.model, build_opts(backend))
          end)

        model_info =
          timed_debug(debug?, "runtime: wrap model params (device transfer)", fn ->
            params =
              model_info.params
              |> transfer_params(backend)
              |> Axon.ModelState.new()

            %{model_info | params: params}
          end)

        generation_config =
          Bumblebee.configure(generation_config,
            max_new_tokens: Keyword.get(opts, :max_response_tokens, 512),
            strategy: %{type: :greedy_search}
          )

        suppressed_token_ids =
          transcription_suppressed_token_ids(tokenizer, generation_config, model_info.spec)

        suppression_mask =
          TokenSelection.suppression_mask(
            suppressed_token_ids,
            model_info.spec.vocab_size,
            runtime_backend_from_backend(backend)
          )

        channel_token_ids = channel_token_ids(tokenizer)

        inside_channel_suppression_mask =
          suppression_mask(
            suppressed_token_ids ++ [channel_token_ids.start],
            model_info.spec.vocab_size,
            runtime_backend_from_backend(backend)
          )

        content_suppression_mask =
          suppression_mask(
            suppressed_token_ids ++ [channel_token_ids.start, channel_token_ids.end],
            model_info.spec.vocab_size,
            runtime_backend_from_backend(backend)
          )

        log_debug(debug?, fn ->
          "runtime: loaded spec hidden_size=#{model_info.spec.hidden_size} layers=#{model_info.spec.num_blocks} " <>
            "boa_token_id=#{model_info.spec.boa_token_id} audio_token_id=#{model_info.spec.audio_token_id} " <>
            "eoa_token_id=#{model_info.spec.eoa_token_id} max_new_tokens=#{generation_config.max_new_tokens}"
        end)

        log_debug(debug?, fn ->
          "runtime: channel token ids=#{inspect(channel_token_ids)}"
        end)

        log_debug(debug?, fn ->
          "runtime: suppressed token ids=#{inspect(suppressed_token_ids)}"
        end)

        {:ok,
         %__MODULE__{
           model_name: model_name,
           repo_id: repo_id,
           backend: backend,
           param_type: param_type,
           max_response_tokens: generation_config.max_new_tokens,
           model_info: model_info,
           tokenizer: tokenizer,
           generation_config: generation_config,
           suppressed_token_ids: suppressed_token_ids,
           suppression_mask: suppression_mask,
           channel_token_ids: channel_token_ids,
           inside_channel_suppression_mask: inside_channel_suppression_mask,
           content_suppression_mask: content_suppression_mask,
           no_repeat_ngram_size: Keyword.get(opts, :no_repeat_ngram_size, 0),
           e4b?: spec_module == Gemma4MicTranscribe.Gemma4E4B.Model,
           predict_fun: predict_fun,
           debug: debug?,
           debug_top_k: Keyword.get(opts, :debug_top_k, 0)
         }}
      end
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  @doc """
  Compiles and warms the generation executables for the given audio token
  buckets by running a tiny generation over silence.

  With a JIT compiler backend the first prefill/decode call per shape pays the
  full XLA compilation cost. Calling this once at startup moves that cost out
  of the first real utterance.
  """
  def warmup(%__MODULE__{} = runtime, opts \\ []) do
    token_counts =
      opts
      |> Keyword.get(:audio_token_counts, [])
      |> Enum.uniq()
      |> Enum.sort()

    prompt = Keyword.get(opts, :prompt, Config.default_prompt())
    system_message = Keyword.get(opts, :system_message)

    result = warmup_incremental(runtime, prompt, system_message)

    if result != :ok do
      result
    else
      warmup_buckets(runtime, token_counts, prompt, system_message)
    end
  end

  # The incremental path has its own shapes (prompt prefix, fixed audio chunk,
  # prompt suffix). Without warming them, each compiles during a live
  # utterance, which costs far more than the prefill it saves.
  defp warmup_incremental(runtime, prompt, system_message) do
    samples_per_chunk = @audio_chunk_tokens * AudioFeatureExtractor.samples_per_token()

    timed_debug(runtime, "runtime: warmup incremental prefill (prefix/append/suffix)", fn ->
      with {:ok, utterance} <-
             start_utterance(runtime, prompt: prompt, system_message: system_message) do
        utterance = append_audio(utterance, List.duplicate(0.0, samples_per_chunk))

        # Every flush size is a separate executable, so exercise them all here
        # rather than compiling one mid-utterance.
        samples_per_token = AudioFeatureExtractor.samples_per_token()

        utterance =
          Enum.reduce(@audio_chunk_sizes, utterance, fn size, acc ->
            %{acc | pending_samples: List.duplicate(0.0, size * samples_per_token)}
            |> transcribe_utterance(max_new_tokens: 1, min_new_tokens: 1)
            |> case do
              {:ok, _} -> acc
              _ -> acc
            end
          end)

        utterance = %{utterance | pending_samples: List.duplicate(0.0, samples_per_token)}

        case transcribe_utterance(utterance, max_new_tokens: 2, min_new_tokens: 2) do
          {:ok, _text} -> :ok
          {:error, reason} -> {:error, reason}
        end
      end
    end)
  end

  defp warmup_buckets(runtime, token_counts, prompt, system_message) do
    Enum.reduce_while(token_counts, :ok, fn token_count, :ok ->
      samples = List.duplicate(0.0, token_count * AudioFeatureExtractor.samples_per_token())

      input =
        Input.build(samples,
          prompt: prompt,
          system_message: system_message,
          audio_token_count: token_count
        )

      result =
        timed_debug(
          runtime,
          "runtime: warmup audio_tokens=#{token_count} (compiling prefill/decode)",
          fn -> generate(runtime, input, max_new_tokens: 2, min_new_tokens: 2) end
        )

      case result do
        {:ok, _text} -> {:cont, :ok}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
  end

  defmodule Utterance do
    @moduledoc false
    # KV cache for one in-progress utterance. `cache` covers the prompt prefix
    # plus every audio chunk appended so far, so a partial transcript only has
    # to prefill the short suffix rather than the whole utterance again.
    # `offset` counts cache slots consumed; `content_length` counts real
    # positions. A padded flush consumes slots without advancing position, so
    # the two diverge and position ids must follow content_length.
    defstruct [:runtime, :cache, :offset, :content_length, :audio_tokens, :pending_samples]
  end

  @doc """
  Starts an utterance and prefills the static prompt prefix.

  Streaming re-transcribes the same growing audio once per partial; without a
  reusable cache each of those pays a full prefill over all audio so far.
  """
  def start_utterance(%__MODULE__{} = runtime, opts \\ []) do
    prompt = Keyword.get(opts, :prompt, Config.default_prompt())
    system_message = Keyword.get(opts, :system_message)

    prefix = Prompt.prefix(system_message, prompt)

    with {:ok, prefix_ids} <- tokenize(runtime.tokenizer, prefix) do
      cache = init_generation_cache(runtime, cache_length(0, max_utterance_cache_tokens(runtime)))
      length = length(prefix_ids)

      {_logits, cache} =
        predict_chunk(runtime, prefix_ids, silent_audio(runtime, 0), 0, cache)

      log_debug(runtime, fn -> "runtime: utterance started prefix_tokens=#{length}" end)

      {:ok,
       %Utterance{
         runtime: runtime,
         cache: cache,
         offset: length,
         content_length: length,
         audio_tokens: 0,
         pending_samples: []
       }}
    end
  end

  @doc """
  Appends audio samples, prefilling whole chunks into the cache.

  Chunks are a fixed number of audio tokens so one compiled executable serves
  every append; leftover samples stay pending until the next call.
  """
  def append_audio(%Utterance{} = utterance, samples, opts \\ []) do
    chunk_tokens = Keyword.get(opts, :chunk_tokens, @audio_chunk_tokens)
    samples_per_chunk = chunk_tokens * AudioFeatureExtractor.samples_per_token()
    runtime = utterance.runtime

    pending = utterance.pending_samples ++ samples
    chunk_count = div(length(pending), samples_per_chunk)
    {ready, leftover} = Enum.split(pending, chunk_count * samples_per_chunk)

    utterance =
      ready
      |> Enum.chunk_every(samples_per_chunk)
      |> Enum.reduce(utterance, fn chunk, acc ->
        features =
          AudioFeatureExtractor.extract(chunk,
            audio_token_count: chunk_tokens,
            max_tokens: chunk_tokens
          )

        {input_features, _mask} =
          prepare_audio_tensors(runtime, features.input_features, features.attention_mask)

        audio_ids = List.duplicate(runtime.model_info.spec.audio_token_id, chunk_tokens)

        {_logits, cache} =
          predict_chunk(runtime, audio_ids, input_features, acc.content_length, acc.cache)

        %{
          acc
          | cache: cache,
            offset: acc.offset + chunk_tokens,
            content_length: acc.content_length + chunk_tokens,
            audio_tokens: acc.audio_tokens + chunk_tokens
        }
      end)

    %{utterance | pending_samples: leftover}
  end

  @doc """
  Produces a transcript from the audio cached so far.

  Prefills the prompt suffix on top of the cached audio and decodes. The
  utterance is returned unchanged, because the suffix and generated tokens
  must not be part of the cache the next chunk of audio appends to.
  """
  def transcribe_utterance(%Utterance{} = utterance, opts \\ []) do
    runtime = utterance.runtime

    with {:ok, suffix_ids} <- tokenize_suffix(runtime) do
      started_at = System.monotonic_time(:millisecond)

      # Audio below a whole chunk has not been prefilled yet. Flushing it into
      # the throwaway cache keeps the utterance tail in the transcript without
      # committing a short chunk to the cache the next append builds on.
      {offset, cache, flushed_tokens} = flush_pending_audio(utterance)

      {logits, cache} =
        predict_chunk(
          runtime,
          suffix_ids,
          silent_audio(runtime, 0),
          offset,
          cache
        )

      limits = generate_limits(runtime, opts)
      channel_state = ChannelState.content()
      suppression_mask = suppression_mask_for_state(runtime, channel_state)
      token_id = TokenSelection.next_token_id_from_sequence(logits, suppression_mask)
      prompt_length = offset + length(suffix_ids)

      log_debug(runtime, fn ->
        "runtime: utterance suffix prefilled audio_tokens=#{utterance.audio_tokens + flushed_tokens} " <>
          "flushed=#{flushed_tokens} context_tokens=#{prompt_length} " <>
          "elapsed_ms=#{System.monotonic_time(:millisecond) - started_at}"
      end)

      token_ids =
        if stop?(eos?(token_id, runtime.generation_config.eos_token_id), false, 1, limits) do
          []
        else
          greedy_decode(
            runtime,
            cache,
            token_id,
            prompt_length,
            [token_id],
            1,
            limits,
            channel_state
          )
        end

      {:ok, Transcript.decode(runtime.tokenizer, token_ids)}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  defp tokenize_suffix(runtime) do
    tokenize(runtime.tokenizer, Prompt.suffix())
  end

  # Prefills leftover audio as a full-size chunk padded with masked positions,
  # so the flush reuses the same compiled executable as a normal append.
  defp flush_pending_audio(%Utterance{pending_samples: []} = utterance) do
    {utterance.content_length, utterance.cache, 0}
  end

  defp flush_pending_audio(%Utterance{} = utterance) do
    runtime = utterance.runtime
    samples_per_token = AudioFeatureExtractor.samples_per_token()
    total_tokens = ceil_div(length(utterance.pending_samples), samples_per_token)

    total_tokens
    |> decompose_tokens()
    |> Enum.reduce({utterance.content_length, utterance.cache, 0}, fn size,
                                                                      {offset, cache, flushed} ->
      chunk =
        Enum.slice(
          utterance.pending_samples,
          flushed * samples_per_token,
          size * samples_per_token
        )

      features =
        AudioFeatureExtractor.extract(chunk, audio_token_count: size, max_tokens: size)

      {input_features, _mask} =
        prepare_audio_tensors(runtime, features.input_features, features.attention_mask)

      audio_ids = List.duplicate(runtime.model_info.spec.audio_token_id, size)

      {_logits, cache} = predict_chunk(runtime, audio_ids, input_features, offset, cache)

      {offset + size, cache, flushed + size}
    end)
  end

  # Largest size first, so a 37 token remainder becomes 25 + 10 + 2 rather than
  # 37 single token prefills.
  @doc false
  def decompose_tokens(0), do: []

  def decompose_tokens(count) do
    size = Enum.find(@audio_chunk_sizes, 1, &(&1 <= count))
    [size | decompose_tokens(count - size)]
  end

  defp ceil_div(0, _denominator), do: 0
  defp ceil_div(numerator, denominator), do: div(numerator + denominator - 1, denominator)

  defp max_utterance_cache_tokens(runtime) do
    @max_utterance_audio_tokens + runtime.max_response_tokens + @prompt_headroom_tokens
  end

  defp silent_audio(runtime, token_count) do
    Nx.broadcast(
      Nx.tensor(0.0, type: {:f, 32}, backend: runtime_backend(runtime)),
      {max(token_count, 1), runtime.model_info.spec.audio_embed_dim}
    )
  end

  # One forward pass over a chunk of tokens starting at `offset`.
  defp predict_chunk(runtime, input_ids, input_features, offset, cache, opts \\ []) do
    backend = runtime_backend(runtime)
    count = length(input_ids)
    attention_mask = Keyword.get(opts, :attention_mask, List.duplicate(1, count))
    audio_token_id = runtime.model_info.spec.audio_token_id
    audio_count = Enum.count(input_ids, &(&1 == audio_token_id))

    input_features =
      if audio_count == 0, do: silent_audio(runtime, 0), else: input_features

    inputs =
      Nx.with_default_backend(backend, fn ->
        %{
          "input_ids" => Nx.tensor([input_ids], type: :s64),
          "attention_mask" => Nx.tensor([attention_mask], type: :s64),
          "position_ids" => Nx.tensor([Enum.to_list(offset..(offset + count - 1))], type: :s64),
          "input_features" => Nx.new_axis(input_features, 0),
          "input_features_mask" =>
            Nx.tensor([List.duplicate(1, Nx.axis_size(input_features, 0))], type: :s64),
          "cache" => cache
        }
      end)

    %{logits: logits, cache: cache} = runtime.predict_fun.(runtime.model_info.params, inputs)
    {logits, cache}
  end

  def generate(%__MODULE__{} = runtime, input, opts \\ []) do
    input = maybe_rebuild_for_e4b(runtime, input)

    log_debug(runtime, fn ->
      "runtime: generate start prompt_bytes=#{byte_size(input.prompt)} audio_tokens=#{input.audio.token_count}"
    end)

    with {:ok, input_ids} <-
           timed_debug(runtime, "runtime: tokenize prompt", fn ->
             tokenize(runtime.tokenizer, input.prompt)
           end),
         {:ok, audio_tokens} <-
           validate_audio_tokens(runtime, input_ids, input.audio.token_count) do
      log_debug(runtime, fn ->
        "runtime: prompt tokenized input_tokens=#{length(input_ids)} " <>
          "audio_tokens=#{audio_tokens.audio} text_control_tokens=#{audio_tokens.text_control} " <>
          "boa=#{audio_tokens.begin} eoa=#{audio_tokens.end} " <>
          "boa_index=#{audio_tokens.begin_index} audio_span=#{format_span(audio_tokens)} " <>
          "eoa_index=#{audio_tokens.end_index}"
      end)

      log_debug(runtime, fn ->
        "runtime: audio features shape=#{inspect(Nx.shape(input.audio.input_features))} " <>
          "mask_shape=#{inspect(Nx.shape(input.audio.attention_mask))} " <>
          "samples_per_token=#{input.audio.samples_per_token}"
      end)

      {input_features, input_features_mask} =
        timed_debug(runtime, "runtime: prepare audio tensors", fn ->
          prepare_audio_tensors(runtime, input.audio.input_features, input.audio.attention_mask)
        end)

      actual_audio_tokens =
        input.audio.attention_mask |> Nx.sum() |> Nx.to_number()

      masks =
        prefill_masks(
          length(input_ids),
          audio_tokens.audio_start_index,
          actual_audio_tokens,
          audio_tokens.audio
        )

      log_debug(runtime, fn ->
        "runtime: audio padding masked_tokens=#{audio_tokens.audio - actual_audio_tokens} " <>
          "content_tokens=#{masks.content_length}"
      end)

      token_ids =
        cached_greedy_generate(
          runtime,
          input_ids,
          input_features,
          input_features_mask,
          masks,
          generate_limits(runtime, opts)
        )

      log_debug(runtime, fn ->
        "runtime: generation finished generated_tokens=#{length(token_ids)}"
      end)

      transcript =
        runtime.tokenizer
        |> Transcript.decode(token_ids)

      {:ok, transcript}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  defp generate_limits(runtime, opts) do
    max_new_tokens =
      opts
      |> Keyword.get(:max_new_tokens, runtime.max_response_tokens)
      |> min(runtime.max_response_tokens)

    %{
      max_new_tokens: max_new_tokens,
      min_new_tokens: Keyword.get(opts, :min_new_tokens, 0)
    }
  end

  @doc """
  Builds the prefill attention mask and position ids for a prompt whose audio
  soft tokens are padded to a bucket.

  Padded audio positions are masked out of attention so the model never
  attends to the zero-padding (which it otherwise transcribes as looping
  hallucinated speech), and positions stay contiguous over the real tokens.
  """
  def prefill_masks(prompt_length, audio_start_index, actual_audio_tokens, prompt_audio_tokens) do
    padding_start = (audio_start_index || 0) + actual_audio_tokens
    padding_count = max(prompt_audio_tokens - actual_audio_tokens, 0)
    padding_range = padding_start..(padding_start + padding_count - 1)//1

    {attention_mask, position_ids, content_length} =
      Enum.reduce(0..(prompt_length - 1), {[], [], 0}, fn index, {mask, positions, position} ->
        if padding_count > 0 and index in padding_range do
          {[0 | mask], [0 | positions], position}
        else
          {[1 | mask], [position | positions], position + 1}
        end
      end)

    %{
      attention_mask: Enum.reverse(attention_mask),
      position_ids: Enum.reverse(position_ids),
      content_length: content_length
    }
  end

  defp cached_greedy_generate(
         _runtime,
         _input_ids,
         _input_features,
         _input_features_mask,
         _masks,
         %{max_new_tokens: max_new_tokens}
       )
       when max_new_tokens <= 0 do
    []
  end

  defp cached_greedy_generate(
         runtime,
         input_ids,
         input_features,
         input_features_mask,
         masks,
         limits
       ) do
    prompt_length = length(input_ids)
    max_cache_length = cache_length(prompt_length, runtime.max_response_tokens)

    cache =
      timed_debug(runtime, "runtime: initialize KV cache", fn ->
        init_generation_cache(runtime, max_cache_length)
      end)

    log_debug(runtime, fn ->
      "runtime: generation prefill start input_tokens=#{prompt_length} cache_tokens=#{max_cache_length}"
    end)

    started_at = System.monotonic_time(:millisecond)

    {logits, cache} =
      predict_prefill_next_logits(
        runtime,
        input_ids,
        input_features,
        input_features_mask,
        masks,
        cache
      )

    channel_state = ChannelState.content()
    suppression_mask = suppression_mask_for_state(runtime, channel_state)

    log_top_token_candidates(runtime, "runtime: prefill", logits, suppression_mask)

    token_id = TokenSelection.next_token_id_from_sequence(logits, suppression_mask)
    eos? = eos?(token_id, runtime.generation_config.eos_token_id)
    pad? = token_id == runtime.model_info.spec.pad_token_id
    elapsed_ms = System.monotonic_time(:millisecond) - started_at

    log_debug(runtime, fn ->
      "runtime: generation step 1/#{limits.max_new_tokens} prefill token_id=#{token_id} " <>
        "eos=#{eos?} pad=#{pad?} elapsed_ms=#{elapsed_ms}"
    end)

    if stop?(eos?, pad?, 1, limits) do
      []
    else
      greedy_decode(
        runtime,
        cache,
        token_id,
        masks.content_length,
        [token_id],
        1,
        limits,
        channel_state
      )
    end
  end

  defp greedy_decode(
         runtime,
         cache,
         previous_token_id,
         prompt_length,
         generated,
         generated_count,
         limits,
         channel_state
       ) do
    if generated_count >= limits.max_new_tokens do
      log_debug(runtime, fn ->
        "runtime: generation reached max_new_tokens=#{limits.max_new_tokens}"
      end)

      Enum.reverse(generated)
    else
      step = generated_count + 1
      previous_token_position = prompt_length + generated_count - 1
      context_length = prompt_length + generated_count

      log_debug(runtime, fn ->
        "runtime: generation step #{step}/#{limits.max_new_tokens} start " <>
          "context_tokens=#{context_length} input_tokens=1 cache_position=#{previous_token_position}"
      end)

      started_at = System.monotonic_time(:millisecond)

      {logits, cache} =
        predict_decode_next_logits(runtime, previous_token_id, previous_token_position, cache)

      suppression_mask = suppression_mask_for_state(runtime, channel_state)

      log_top_token_candidates(
        runtime,
        "runtime: generation step #{step}",
        logits,
        suppression_mask
      )

      banned_ids = banned_ngram_token_ids(generated, runtime.no_repeat_ngram_size)

      token_id =
        TokenSelection.next_allowed_token_id_from_sequence(logits, suppression_mask, banned_ids)

      eos? = eos?(token_id, runtime.generation_config.eos_token_id)
      pad? = token_id == runtime.model_info.spec.pad_token_id
      elapsed_ms = System.monotonic_time(:millisecond) - started_at

      log_debug(runtime, fn ->
        "runtime: generation step #{step}/#{limits.max_new_tokens} token_id=#{token_id} " <>
          "eos=#{eos?} pad=#{pad?} elapsed_ms=#{elapsed_ms}"
      end)

      if stop?(eos?, pad?, step, limits) do
        Enum.reverse(generated)
      else
        greedy_decode(
          runtime,
          cache,
          token_id,
          prompt_length,
          [token_id | generated],
          generated_count + 1,
          limits,
          ChannelState.advance(channel_state, token_id, runtime.channel_token_ids)
        )
      end
    end
  end

  defp stop?(eos?, pad?, step, limits) do
    (eos? or pad?) and step >= limits.min_new_tokens
  end

  @doc """
  Token ids that would complete an n-gram already present in the generated
  sequence (HF-style no_repeat_ngram).

  Greedy decode loops on pauses and silence by re-emitting phrases it has
  already produced; banning exact n-gram repeats breaks the loop. Stop
  tokens are never banned because they never occur inside the generated
  sequence. `generated_reversed` is the generated ids, most recent first.
  """
  def banned_ngram_token_ids(generated_reversed, size)
      when is_integer(size) and size > 1 and length(generated_reversed) >= size - 1 do
    sequence = Enum.reverse(generated_reversed)
    suffix = generated_reversed |> Enum.take(size - 1) |> Enum.reverse()

    sequence
    |> Enum.chunk_every(size, 1, :discard)
    |> Enum.filter(fn chunk -> Enum.take(chunk, size - 1) == suffix end)
    |> Enum.map(&List.last/1)
    |> Enum.uniq()
  end

  def banned_ngram_token_ids(_generated_reversed, _size), do: []

  defp cache_length(prompt_length, max_response_tokens) do
    needed = prompt_length + max_response_tokens
    div(needed + @cache_length_step - 1, @cache_length_step) * @cache_length_step
  end

  defp init_generation_cache(runtime, max_cache_length) do
    backend = runtime_backend(runtime)

    Nx.with_default_backend(backend, fn ->
      Model.init_cache(runtime.model_info.spec, 1, max_cache_length, %{})
    end)
  end

  defp predict_prefill_next_logits(
         runtime,
         input_ids,
         input_features,
         input_features_mask,
         masks,
         cache
       ) do
    backend = runtime_backend(runtime)

    inputs =
      Nx.with_default_backend(backend, fn ->
        %{
          "input_ids" => Nx.tensor([input_ids], type: :s64),
          "attention_mask" => Nx.tensor([masks.attention_mask], type: :s64),
          "position_ids" => Nx.tensor([masks.position_ids], type: :s64),
          "input_features" => Nx.new_axis(input_features, 0),
          "input_features_mask" => Nx.new_axis(input_features_mask, 0),
          "cache" => cache
        }
      end)

    %{logits: logits, cache: cache} = runtime.predict_fun.(runtime.model_info.params, inputs)

    {logits, cache}
  end

  defp predict_decode_next_logits(runtime, previous_token_id, position_id, cache) do
    backend = runtime_backend(runtime)
    audio_embed_dim = runtime.model_info.spec.audio_embed_dim

    inputs =
      Nx.with_default_backend(backend, fn ->
        %{
          "input_ids" => Nx.tensor([[previous_token_id]], type: :s64),
          "attention_mask" => Nx.tensor([[1]], type: :s64),
          "position_ids" => Nx.tensor([[position_id]], type: :s64),
          "input_features" => Nx.broadcast(0.0, {1, 1, audio_embed_dim}),
          "input_features_mask" => Nx.tensor([[0]], type: :s64),
          "cache" => cache
        }
      end)

    %{logits: logits, cache: cache} = runtime.predict_fun.(runtime.model_info.params, inputs)

    {logits, cache}
  end

  defp prepare_audio_tensors(runtime, input_features, input_features_mask) do
    backend = runtime_backend(runtime)

    {
      Nx.backend_copy(input_features, backend),
      Nx.backend_copy(input_features_mask, backend)
    }
  end

  defp runtime_backend(%__MODULE__{backend: nil}), do: Nx.BinaryBackend
  defp runtime_backend(%__MODULE__{backend: backend}), do: backend
  defp runtime_backend_from_backend(nil), do: Nx.BinaryBackend
  defp runtime_backend_from_backend(backend), do: backend

  defp tokenize(tokenizer, prompt) do
    inputs =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        Bumblebee.apply_tokenizer(tokenizer, [prompt])
      end)

    {:ok, Nx.to_flat_list(inputs["input_ids"])}
  end

  defp validate_audio_tokens(runtime, input_ids, expected_count) do
    spec = runtime.model_info.spec
    tokenizer = runtime.tokenizer

    boa_token_id = token_id(tokenizer, "<|audio>", spec.boa_token_id)
    audio_token_id = token_id(tokenizer, "<|audio|>", spec.audio_token_id)
    eoa_token_id = token_id(tokenizer, "<audio|>", spec.eoa_token_id)

    begin_count = Enum.count(input_ids, &(&1 == boa_token_id))
    actual_count = Enum.count(input_ids, &(&1 == audio_token_id))
    end_count = Enum.count(input_ids, &(&1 == eoa_token_id))
    begin_index = Enum.find_index(input_ids, &(&1 == boa_token_id))
    end_index = Enum.find_index(input_ids, &(&1 == eoa_token_id))

    audio_indices =
      input_ids
      |> Enum.with_index()
      |> Enum.flat_map(fn
        {^audio_token_id, index} -> [index]
        _other -> []
      end)

    cond do
      begin_count != 1 ->
        {:error, "prompt has #{begin_count} audio begin tokens, expected 1"}

      actual_count != expected_count ->
        {:error,
         "prompt has #{actual_count} audio soft tokens, but audio features contain #{expected_count} tokens"}

      end_count != 1 ->
        {:error, "prompt has #{end_count} audio end tokens, expected 1"}

      expected_count > 0 and not contiguous?(audio_indices) ->
        {:error, "audio soft tokens are not contiguous in the tokenized prompt"}

      expected_count > 0 and begin_index + 1 != List.first(audio_indices) ->
        {:error, "audio soft tokens must immediately follow the audio begin token"}

      expected_count > 0 and end_index != List.last(audio_indices) + 1 ->
        {:error, "audio end token must immediately follow the audio soft tokens"}

      expected_count == 0 and end_index != begin_index + 1 ->
        {:error, "audio end token must immediately follow the audio begin token"}

      true ->
        {:ok,
         %{
           begin: begin_count,
           audio: actual_count,
           end: end_count,
           text_control: length(input_ids) - actual_count,
           begin_index: begin_index,
           audio_start_index: List.first(audio_indices),
           audio_end_index: List.last(audio_indices),
           end_index: end_index
         }}
    end
  end

  defp contiguous?([]), do: true

  defp contiguous?(indices) do
    indices == Enum.to_list(List.first(indices)..List.last(indices))
  end

  defp format_span(%{audio: 0}), do: "none"

  defp format_span(audio_tokens) do
    "#{audio_tokens.audio_start_index}..#{audio_tokens.audio_end_index}"
  end

  defp token_id(tokenizer, token, fallback) do
    Bumblebee.Tokenizer.token_to_id(tokenizer, token) || fallback
  end

  defp transcription_suppressed_token_ids(tokenizer, generation_config, spec) do
    control_tokens = [
      "<|tool>",
      "<tool|>",
      "<|tool_call>",
      "<tool_call|>",
      "<|tool_response>",
      "<tool_response|>"
    ]

    generation_config.suppressed_token_ids
    |> List.wrap()
    |> Kernel.++([spec.pad_token_id])
    |> Kernel.++(Enum.map(control_tokens, &Bumblebee.Tokenizer.token_to_id(tokenizer, &1)))
    |> Kernel.++(unused_token_ids(tokenizer))
    |> Enum.reject(&is_nil/1)
    |> Enum.uniq()
  end

  defp channel_token_ids(tokenizer) do
    %{
      start: Bumblebee.Tokenizer.token_to_id(tokenizer, "<|channel>"),
      end: Bumblebee.Tokenizer.token_to_id(tokenizer, "<channel|>")
    }
  end

  defp suppression_mask(token_ids, vocab_size, backend) do
    token_ids
    |> Enum.reject(&is_nil/1)
    |> Enum.uniq()
    |> TokenSelection.suppression_mask(vocab_size, backend)
  end

  defp suppression_mask_for_state(runtime, :before_content), do: runtime.suppression_mask

  defp suppression_mask_for_state(runtime, :inside_channel),
    do: runtime.inside_channel_suppression_mask

  defp suppression_mask_for_state(runtime, :content), do: runtime.content_suppression_mask

  defp unused_token_ids(tokenizer) do
    0..255
    |> Enum.map(&Bumblebee.Tokenizer.token_to_id(tokenizer, "<unused#{&1}>"))
  end

  defp eos?(token_id, eos_token_id), do: token_id in List.wrap(eos_token_id)

  defp log_top_token_candidates(
         %__MODULE__{debug_top_k: count},
         _label,
         _logits,
         _suppression_mask
       )
       when not is_integer(count) or count <= 0,
       do: :ok

  defp log_top_token_candidates(%__MODULE__{} = runtime, label, logits, suppression_mask) do
    top_tokens =
      TokenSelection.top_tokens_from_sequence(
        logits,
        suppression_mask,
        runtime.debug_top_k
      )

    log_debug(runtime, fn ->
      candidates =
        top_tokens
        |> Enum.map(fn {token_id, score} ->
          "#{token_id}=#{format_score(score)}:#{token_debug_label(runtime, token_id)}"
        end)
        |> Enum.join(", ")

      "#{label} top#{runtime.debug_top_k} after_suppression=[#{candidates}]"
    end)
  end

  defp token_debug_label(runtime, token_id) do
    case Map.get(special_token_labels(runtime), token_id) do
      nil ->
        runtime.tokenizer
        |> Bumblebee.Tokenizer.decode([token_id])
        |> inspect()

      label ->
        label
    end
  end

  defp special_token_labels(runtime) do
    spec = runtime.model_info.spec

    eos_labels = Enum.map(List.wrap(runtime.generation_config.eos_token_id), &{&1, "<eos>"})

    (eos_labels ++
       [
         {spec.pad_token_id, "<pad>"},
         {spec.bos_token_id, "<bos>"},
         {spec.boa_token_id, "<|audio>"},
         {spec.audio_token_id, "<|audio|>"},
         {spec.eoa_token_id, "<audio|>"},
         {token_id(runtime.tokenizer, "<|turn>", nil), "<|turn>"},
         {token_id(runtime.tokenizer, "<turn|>", nil), "<turn|>"},
         {token_id(runtime.tokenizer, "<|channel>", nil), "<|channel>"},
         {token_id(runtime.tokenizer, "<channel|>", nil), "<channel|>"}
       ])
    |> Enum.reject(fn {token_id, _label} -> is_nil(token_id) end)
    |> Map.new()
  end

  defp format_score(score) when is_float(score) do
    :io_lib.format("~.4f", [score]) |> IO.iodata_to_binary()
  end

  defp format_score(score), do: inspect(score)

  defp timed_debug(%__MODULE__{debug: debug?}, label, fun), do: timed_debug(debug?, label, fun)

  defp timed_debug(true, label, fun) do
    started_at = System.monotonic_time(:millisecond)
    Logger.debug("#{label}: start")

    result = fun.()

    elapsed_ms = System.monotonic_time(:millisecond) - started_at

    case result do
      {:error, reason} ->
        Logger.debug("#{label}: error after #{elapsed_ms}ms reason=#{inspect(reason)}")

      _ ->
        Logger.debug("#{label}: done in #{elapsed_ms}ms")
    end

    result
  end

  defp timed_debug(false, _label, fun), do: fun.()

  defp log_debug(%__MODULE__{debug: debug?}, message_fun), do: log_debug(debug?, message_fun)
  defp log_debug(true, message_fun), do: Logger.debug(message_fun)
  defp log_debug(false, _message_fun), do: :ok

  # A quantized checkpoint keeps its int4 weights packed in integer tensors, and
  # a global mixed-precision policy would rewrite those packed words as floats,
  # destroying them. Such checkpoints keep their stored dtypes instead.
  defp maybe_put_param_type(
         opts,
         %{quantization_config: %{"quant_method" => "compressed-tensors"}},
         _type
       ),
       do: opts

  defp maybe_put_param_type(opts, _spec, type), do: Keyword.put(opts, :type, type)

  # The E4B spec has none of the 12B's quantisation or logit options.
  defp configure_spec(%Gemma4MicTranscribe.Gemma4E4B.Model{} = spec, _param_type, _opts), do: spec

  defp configure_spec(spec, param_type, opts) do
    Bumblebee.configure(spec,
      logits_last_only: true,
      cache_type: param_type,
      packed_linear: Keyword.get(opts, :packed_weights, true),
      hybrid_linear: Keyword.get(opts, :hybrid_weights, false)
    )
  end

  # E4B takes log-mel frames and one placeholder per encoder frame, where the
  # 12B takes raw 640 sample PCM frames and one placeholder per frame.
  defp maybe_rebuild_for_e4b(%__MODULE__{e4b?: true} = runtime, input) do
    alias Gemma4MicTranscribe.Gemma4E4B.MelFeatures

    spec = runtime.model_info.spec
    samples = Map.get(input, :samples, [])

    tokens =
      MelFeatures.audio_token_count(
        length(samples),
        struct(Gemma4MicTranscribe.Gemma4E4B.Spec, Map.from_struct(spec))
      )

    features =
      MelFeatures.extract(
        samples,
        struct(Gemma4MicTranscribe.Gemma4E4B.Spec, Map.from_struct(spec))
      )

    prompt =
      Gemma4MicTranscribe.Gemma4Unified.Prompt.build(
        input[:system_message],
        input[:user_prompt] || Config.default_prompt(),
        tokens
      )

    %{
      input
      | prompt: prompt,
        audio: %{
          input_features: features,
          attention_mask: Nx.broadcast(1, {max(tokens, 1)}),
          token_count: tokens,
          samples_per_token: 640
        }
    }
  end

  defp maybe_rebuild_for_e4b(_runtime, input), do: input

  defp spec_module_for(repo_id) do
    if String.contains?(repo_id, "E4B") or String.contains?(repo_id, "e4b") do
      Gemma4MicTranscribe.Gemma4E4B.Model
    else
      Model
    end
  end

  defp param_type(nil), do: {:ok, {:bf, 16}}
  defp param_type("bf16"), do: {:ok, {:bf, 16}}
  defp param_type("f16"), do: {:ok, {:f, 16}}
  defp param_type("f32"), do: {:ok, {:f, 32}}
  defp param_type(:bf16), do: {:ok, {:bf, 16}}
  defp param_type(:f16), do: {:ok, {:f, 16}}
  defp param_type(:f32), do: {:ok, {:f, 32}}
  defp param_type({_kind, _size} = type), do: {:ok, type}
  defp param_type(other), do: {:error, "unsupported param type #{inspect(other)}"}

  # ROCm gfx1151 segfaults in the eager GPU kernels generated while unpacking
  # compressed-tensors weights and casting them to bf16, and the EXLA host
  # client has crashed in its own JIT code doing the same unpack, so
  # checkpoint loading is staged on Torchx CPU (libtorch) when available and
  # only the finished parameters are transferred to the GPU.
  defp stage_params_on_host?({EXLA.Backend, backend_opts}),
    do: backend_opts[:client] == :rocm

  defp stage_params_on_host?(_backend), do: false

  defp param_load_backend(backend) do
    cond do
      not stage_params_on_host?(backend) -> backend
      Code.ensure_loaded?(Torchx.Backend) -> {Torchx.Backend, device: :cpu}
      true -> {EXLA.Backend, client: :host}
    end
  end

  defp transfer_params(params, backend) do
    if stage_params_on_host?(backend), do: deep_transfer(params, backend), else: params
  end

  defp deep_transfer(%Axon.ModelState{} = model_state, backend) do
    %{
      model_state
      | data: deep_transfer(model_state.data, backend),
        state: deep_transfer(model_state.state, backend)
    }
  end

  defp deep_transfer(%Nx.Tensor{} = tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp deep_transfer(%{} = map, backend) when not is_struct(map),
    do: Map.new(map, fn {key, value} -> {key, deep_transfer(value, backend)} end)

  defp deep_transfer(other, _backend), do: other

  # With an EXLA backend, compile the whole forward pass into one fused XLA
  # executable per input shape instead of dispatching each Nx op eagerly.
  # Shapes are kept static (audio buckets + fixed cache lengths), so each
  # executable compiles once and is reused for every subsequent call.
  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []

  defp backend("host"), do: {:ok, Nx.BinaryBackend}
  defp backend(nil), do: backend(Config.backend())

  defp backend("exla"), do: exla_backend(:auto)
  defp backend("exla:host"), do: exla_backend(:host)
  defp backend("exla:cuda"), do: exla_backend(:cuda)
  defp backend("exla:rocm"), do: exla_backend(:rocm)

  defp backend("torchx"), do: torchx_backend(:auto)
  defp backend("torchx:cpu"), do: torchx_backend(:cpu)
  defp backend("torchx:cuda"), do: torchx_backend(:cuda)

  defp backend(other), do: {:error, "unsupported backend #{inspect(other)}"}

  defp verify_supported_model(model_name) do
    case ModelCatalog.runtime_kind(model_name) do
      :bumblebee_axon ->
        :ok

      :llama_cpp ->
        {:error,
         "#{model_name} is a GGUF checkpoint. It is runnable through llama.cpp/Ollama/LM Studio, " <>
           "but this module is the local Bumblebee/Axon runtime and does not consume GGUF files."}
    end
  end

  defp exla_backend(:rocm) do
    with :ok <- RocmPreflight.check(),
         :ok <- apply_rocm_runtime_workarounds(),
         :ok <- configure_exla_gpu_client(:rocm),
         :ok <- ensure_exla_available(),
         {:ok, backend} <- exla_backend_for_client(:rocm) do
      {:ok, backend}
    end
  end

  defp exla_backend(client) do
    with :ok <- configure_exla_gpu_client(client),
         :ok <- ensure_exla_available(),
         {:ok, backend} <- exla_backend_for_client(client) do
      {:ok, backend}
    end
  end

  defp configure_exla_gpu_client(client) when client in [:cuda, :rocm] do
    memory_fraction = exla_memory_fraction()
    clients = Application.get_env(:exla, :clients, [])

    config =
      clients
      |> Keyword.get(client, [])
      |> Keyword.merge(platform: client, preallocate: false, memory_fraction: memory_fraction)

    Application.put_env(:exla, :clients, Keyword.put(clients, client, config))
    Application.delete_env(:exla, :default_client)
    :ok
  end

  defp configure_exla_gpu_client(_client), do: :ok

  defp exla_memory_fraction do
    case System.get_env("GEMMA_EXLA_MEMORY_FRACTION") do
      nil ->
        0.55

      value ->
        case Float.parse(value) do
          {fraction, ""} when fraction > 0.0 and fraction <= 0.9 -> fraction
          _ -> 0.55
        end
    end
  end

  defp apply_rocm_runtime_workarounds do
    case RocmPreflight.apply_runtime_workarounds() do
      {:ok, true} ->
        Logger.warning(
          "runtime: applied ROCm gfx1151 XLA workarounds: XLA_FLAGS=#{System.get_env("XLA_FLAGS")}"
        )

        :ok

      {:ok, false} ->
        :ok
    end
  end

  defp ensure_exla_available do
    if Code.ensure_loaded?(EXLA.Backend) do
      case Application.ensure_all_started(:exla) do
        {:ok, _started} ->
          :ok

        {:error, {app, reason}} ->
          {:error,
           "EXLA backend requested, but failed to start #{inspect(app)}: #{inspect(reason)}"}
      end
    else
      {:error, "EXLA backend requested, but EXLA is not installed"}
    end
  end

  defp exla_backend_for_client(:auto), do: {:ok, EXLA.Backend}

  defp exla_backend_for_client(client) when client in [:host, :cuda, :rocm],
    do: {:ok, {EXLA.Backend, client: client}}

  defp torchx_backend(device) do
    with :ok <- ensure_torchx_available(),
         {:ok, device} <- torchx_device(device) do
      {:ok, {Torchx.Backend, device: device}}
    end
  end

  defp ensure_torchx_available do
    if Code.ensure_loaded?(Torchx.Backend) and Code.ensure_loaded?(Torchx) do
      :ok
    else
      {:error, "Torchx backend requested, but Torchx is not installed"}
    end
  end

  defp torchx_device(:auto), do: {:ok, Torchx.default_device()}
  defp torchx_device(:cpu), do: {:ok, :cpu}

  defp torchx_device(:cuda) do
    if Torchx.device_available?(:cuda) do
      {:ok, :cuda}
    else
      {:error,
       "Torchx CUDA device requested, but CUDA is not available. This install is likely using CPU LibTorch; recompile Torchx with LIBTORCH_TARGET=cu129, cu128, or cu126 for your CUDA stack."}
    end
  end

  defp log_backend_details(false, _backend), do: :ok

  defp log_backend_details(true, Nx.BinaryBackend),
    do: Logger.debug("runtime: using Nx.BinaryBackend")

  defp log_backend_details(true, EXLA.Backend) do
    Logger.debug(fn ->
      "runtime: exla selected_client=:auto xla_target=#{inspect(System.get_env("XLA_TARGET"))}"
    end)
  end

  defp log_backend_details(true, {EXLA.Backend, opts}) do
    Logger.debug(fn ->
      "runtime: exla selected_client=#{inspect(opts[:client])} " <>
        "device_id=#{inspect(opts[:device_id])} xla_target=#{inspect(System.get_env("XLA_TARGET"))}"
    end)
  end

  defp log_backend_details(true, {Torchx.Backend, opts}) do
    Logger.debug(fn ->
      cuda_available? = Torchx.device_available?(:cuda)
      cuda_count = if cuda_available?, do: Torchx.device_count(:cuda), else: 0

      "runtime: torchx default_device=#{inspect(Torchx.default_device())} " <>
        "selected_device=#{inspect(opts[:device])} cuda_available=#{cuda_available?} cuda_count=#{cuda_count}"
    end)
  end

  defp log_backend_details(true, backend),
    do: Logger.debug("runtime: using backend #{inspect(backend)}")

  defp verify_bumblebee_available do
    if Code.ensure_loaded?(Bumblebee) do
      :ok
    else
      {:error, {:missing_dependency, :bumblebee}}
    end
  end
end
