defmodule Gemma4MicTranscribe.StreamingSession do
  @moduledoc false

  use GenServer

  alias Gemma4MicTranscribe.Config
  alias Gemma4MicTranscribe.Gemma4Unified.AudioFeatureExtractor
  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Runtime
  alias Gemma4MicTranscribe.SpeechGate

  @default_frame_ms 20.0
  @default_speech_start_ms 120.0
  @default_speech_end_silence_ms 500.0
  @default_min_utterance_ms 350.0
  @default_max_utterance_ms 8_000.0
  @default_partial_interval_ms 1_000.0
  # Partials are throwaway UI feedback; cap their generation cost so a
  # rambling partial cannot burn the full max_response_tokens budget.
  @default_partial_max_response_tokens 16
  @default_tts_echo_window_ms 12_000.0
  @default_tts_similarity_threshold 0.78
  @default_audio_token_buckets [50, 100, 200]

  defstruct [
    :runtime_module,
    :runtime,
    :runtime_opts,
    :sample_rate,
    :frame_ms,
    :frame_samples,
    :speech_start_ms,
    :speech_end_silence_ms,
    :min_utterance_ms,
    :max_utterance_ms,
    :partial_interval_ms,
    :partial_max_response_tokens,
    :partials?,
    :speech_threshold,
    :speech_min_active_ratio,
    :speech_max_zero_crossing_rate,
    :tts_echo_window_ms,
    :tts_similarity_threshold,
    :audio_token_buckets,
    :warmup?,
    :incremental?,
    :utterance_cache,
    :utterance_cached_samples,
    :pending_samples,
    :next_frame_start_ms,
    :candidate_samples,
    :candidate_start_ms,
    :candidate_active_ms,
    :in_speech?,
    :utterance_id,
    :next_utterance_id,
    :utterance_samples,
    :utterance_start_ms,
    :utterance_last_active_end_ms,
    :trailing_silence_ms,
    :last_partial_ms,
    :tts_history
  ]

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts)
  end

  def push_audio(session, samples, timestamp_ms) when is_list(samples) do
    GenServer.call(session, {:push_audio, samples, timestamp_ms}, :infinity)
  end

  def push_tts(session, text, timestamp_ms, opts \\ []) when is_binary(text) do
    GenServer.call(session, {:push_tts, text, timestamp_ms, opts}, :infinity)
  end

  def flush(session) do
    GenServer.call(session, :flush, :infinity)
  end

  def json_event!(event), do: Jason.encode!(event)

  @impl true
  def init(opts) do
    sample_rate = Keyword.get(opts, :sample_rate, 16_000)
    frame_ms = Keyword.get(opts, :frame_ms, @default_frame_ms)
    frame_samples = max(1, round(sample_rate * frame_ms / 1000))

    state = %__MODULE__{
      runtime_module: Keyword.get(opts, :runtime_module, Runtime),
      runtime: Keyword.get(opts, :runtime),
      runtime_opts: runtime_opts(opts),
      sample_rate: sample_rate,
      frame_ms: frame_ms,
      frame_samples: frame_samples,
      speech_start_ms: Keyword.get(opts, :speech_start_ms, @default_speech_start_ms),
      speech_end_silence_ms:
        Keyword.get(opts, :speech_end_silence_ms, @default_speech_end_silence_ms),
      min_utterance_ms: Keyword.get(opts, :min_utterance_ms, @default_min_utterance_ms),
      max_utterance_ms: Keyword.get(opts, :max_utterance_ms, @default_max_utterance_ms),
      partial_interval_ms: Keyword.get(opts, :partial_interval_ms, @default_partial_interval_ms),
      partial_max_response_tokens:
        Keyword.get(opts, :partial_max_response_tokens, @default_partial_max_response_tokens),
      partials?: Keyword.get(opts, :partials, true),
      speech_threshold: Keyword.get(opts, :speech_threshold, Config.speech_threshold()),
      speech_min_active_ratio:
        Keyword.get(opts, :speech_min_active_ratio, Config.speech_min_active_ratio()),
      speech_max_zero_crossing_rate:
        Keyword.get(
          opts,
          :speech_max_zero_crossing_rate,
          Config.speech_max_zero_crossing_rate()
        ),
      tts_echo_window_ms: Keyword.get(opts, :tts_echo_window_ms, @default_tts_echo_window_ms),
      tts_similarity_threshold:
        Keyword.get(opts, :tts_similarity_threshold, @default_tts_similarity_threshold),
      audio_token_buckets: Keyword.get(opts, :audio_token_buckets, @default_audio_token_buckets),
      warmup?: Keyword.get(opts, :warmup, true),
      # Off by default: measured slower than whole-utterance prefill (final lag
      # 9.3s/18.2s vs 4.7s/11.9s) because each partial still costs more than the
      # partial interval, so the backlog grows. See README.
      incremental?: Keyword.get(opts, :incremental_prefill, false),
      utterance_cache: nil,
      utterance_cached_samples: 0,
      pending_samples: [],
      next_frame_start_ms: nil,
      candidate_samples: [],
      candidate_start_ms: nil,
      candidate_active_ms: 0.0,
      in_speech?: false,
      utterance_id: nil,
      next_utterance_id: 1,
      utterance_samples: [],
      utterance_start_ms: nil,
      utterance_last_active_end_ms: nil,
      trailing_silence_ms: 0.0,
      last_partial_ms: nil,
      tts_history: []
    }

    if Keyword.get(opts, :preload_runtime, false) do
      case ensure_runtime(state) do
        {:ok, state} -> {:ok, state}
        {:error, reason, _state} -> {:stop, reason}
      end
    else
      {:ok, state}
    end
  end

  @impl true
  def handle_call({:push_audio, samples, timestamp_ms}, _from, state) do
    state =
      state
      |> maybe_initialize_clock(timestamp_ms)
      |> Map.update!(:pending_samples, &(&1 ++ samples))

    {state, events} = process_complete_frames(state, [])
    {:reply, {:ok, Enum.reverse(events)}, state}
  end

  @impl true
  def handle_call({:push_tts, text, timestamp_ms, _opts}, _from, state) do
    event = %{
      type: "tts",
      text: text,
      timestamp_ms: round(timestamp_ms),
      send_to_llm: false
    }

    history_item = %{
      text: text,
      normalized: normalize_text(text),
      timestamp_ms: timestamp_ms
    }

    state =
      state
      |> prune_tts_history(timestamp_ms)
      |> Map.update!(:tts_history, &[history_item | &1])

    {:reply, {:ok, [event]}, state}
  end

  @impl true
  def handle_call(:flush, _from, state) do
    {state, events} =
      state
      |> process_pending_frame([])
      |> flush_speech()

    {:reply, {:ok, Enum.reverse(events)}, state}
  end

  defp runtime_opts(opts) do
    [
      model_name: Keyword.get(opts, :model_name, Config.default_model_name()),
      backend: Keyword.get(opts, :backend, Config.backend()),
      max_response_tokens: Keyword.get(opts, :max_response_tokens, Config.max_response_tokens()),
      no_repeat_ngram_size: Keyword.get(opts, :no_repeat_ngram_size, 0),
      request_timeout_seconds:
        Keyword.get(opts, :request_timeout_seconds, Config.request_timeout_seconds()),
      param_type: Keyword.get(opts, :param_type),
      packed_weights: Keyword.get(opts, :packed_weights, true),
      prompt: Keyword.get(opts, :prompt, Config.default_prompt()),
      system_message: Keyword.get(opts, :system_message),
      debug: Keyword.get(opts, :debug, false),
      debug_top_k: Keyword.get(opts, :debug_top_k, 0)
    ]
  end

  defp maybe_initialize_clock(%__MODULE__{next_frame_start_ms: nil} = state, timestamp_ms),
    do: %{state | next_frame_start_ms: timestamp_ms}

  defp maybe_initialize_clock(state, _timestamp_ms), do: state

  defp process_complete_frames(%__MODULE__{} = state, events) do
    if length(state.pending_samples) >= state.frame_samples do
      {frame, pending_samples} = Enum.split(state.pending_samples, state.frame_samples)
      duration_ms = length(frame) * 1000 / state.sample_rate
      frame_start_ms = state.next_frame_start_ms
      frame_end_ms = frame_start_ms + duration_ms

      {state, events} =
        state
        |> Map.merge(%{
          pending_samples: pending_samples,
          next_frame_start_ms: frame_end_ms
        })
        |> process_frame(frame, frame_start_ms, frame_end_ms, events)

      process_complete_frames(state, events)
    else
      {state, events}
    end
  end

  defp process_pending_frame(%__MODULE__{pending_samples: []} = state, events),
    do: {state, events}

  defp process_pending_frame(%__MODULE__{} = state, events) do
    frame = state.pending_samples
    duration_ms = length(frame) * 1000 / state.sample_rate
    frame_start_ms = state.next_frame_start_ms || 0.0
    frame_end_ms = frame_start_ms + duration_ms

    state =
      %{state | pending_samples: [], next_frame_start_ms: frame_end_ms}

    process_frame(state, frame, frame_start_ms, frame_end_ms, events)
  end

  defp process_frame(state, frame, frame_start_ms, frame_end_ms, events) do
    active? = active_frame?(state, frame)

    if state.in_speech? do
      process_speech_frame(state, frame, frame_start_ms, frame_end_ms, active?, events)
    else
      process_idle_frame(state, frame, frame_start_ms, frame_end_ms, active?, events)
    end
  end

  defp process_idle_frame(state, frame, frame_start_ms, frame_end_ms, true, events) do
    candidate_start_ms = state.candidate_start_ms || frame_start_ms
    candidate_active_ms = state.candidate_active_ms + (frame_end_ms - frame_start_ms)
    # Frames are prepended and flattened on demand to keep per-frame cost O(1).
    candidate_samples = [frame | state.candidate_samples]

    state = %{
      state
      | candidate_samples: candidate_samples,
        candidate_start_ms: candidate_start_ms,
        candidate_active_ms: candidate_active_ms
    }

    if candidate_active_ms >= state.speech_start_ms do
      utterance_id = "utt-#{state.next_utterance_id}"

      event = %{
        type: "speech_start",
        utterance_id: utterance_id,
        start_ms: round(candidate_start_ms),
        stable: false,
        send_to_llm: false
      }

      state = %{
        state
        | in_speech?: true,
          utterance_id: utterance_id,
          next_utterance_id: state.next_utterance_id + 1,
          utterance_samples: candidate_samples,
          utterance_start_ms: candidate_start_ms,
          utterance_last_active_end_ms: frame_end_ms,
          candidate_samples: [],
          candidate_start_ms: nil,
          candidate_active_ms: 0.0,
          trailing_silence_ms: 0.0,
          last_partial_ms: frame_end_ms
      }

      {state, [event | events]}
    else
      {state, events}
    end
  end

  defp process_idle_frame(state, _frame, _frame_start_ms, _frame_end_ms, false, events) do
    state = %{state | candidate_samples: [], candidate_start_ms: nil, candidate_active_ms: 0.0}
    {state, events}
  end

  defp process_speech_frame(state, frame, _frame_start_ms, frame_end_ms, active?, events) do
    trailing_silence_ms =
      if active? do
        0.0
      else
        state.trailing_silence_ms + state.frame_ms
      end

    utterance_last_active_end_ms =
      if active? do
        frame_end_ms
      else
        state.utterance_last_active_end_ms
      end

    state = %{
      state
      | utterance_samples: [frame | state.utterance_samples],
        trailing_silence_ms: trailing_silence_ms,
        utterance_last_active_end_ms: utterance_last_active_end_ms
    }

    duration_ms = frame_end_ms - state.utterance_start_ms

    cond do
      trailing_silence_ms >= state.speech_end_silence_ms ->
        finalize_utterance(state, :speech_end, events)

      duration_ms >= state.max_utterance_ms ->
        finalize_utterance(state, :max_duration, events)

      partial_due?(state, frame_end_ms) ->
        emit_partial(state, frame_end_ms, events)

      true ->
        {state, events}
    end
  end

  defp flush_speech({%__MODULE__{in_speech?: true} = state, events}) do
    finalize_utterance(state, :flush, events)
  end

  defp flush_speech({state, events}) do
    state = %{state | candidate_samples: [], candidate_start_ms: nil, candidate_active_ms: 0.0}
    {state, events}
  end

  defp partial_due?(%__MODULE__{partials?: false}, _frame_end_ms), do: false

  defp partial_due?(state, frame_end_ms) do
    frame_end_ms - (state.last_partial_ms || state.utterance_start_ms) >=
      state.partial_interval_ms
  end

  defp emit_partial(state, frame_end_ms, events) do
    case transcribe(state, utterance_samples(state),
           max_new_tokens: state.partial_max_response_tokens
         ) do
      {:ok, text, metrics, state} ->
        event = %{
          type: "partial",
          utterance_id: state.utterance_id,
          text: text,
          start_ms: round(state.utterance_start_ms),
          end_ms: round(frame_end_ms),
          stable: false,
          send_to_llm: false,
          metrics: metrics
        }

        {%{state | last_partial_ms: frame_end_ms}, [event | events]}

      {:error, reason, state} ->
        event = error_event(state, reason)
        {%{state | last_partial_ms: frame_end_ms}, [event | events]}
    end
  end

  defp finalize_utterance(state, reason, events) do
    start_ms = state.utterance_start_ms
    end_ms = state.utterance_last_active_end_ms || state.next_frame_start_ms || start_ms
    duration_ms = end_ms - start_ms

    speech_end_event = %{
      type: "speech_end",
      utterance_id: state.utterance_id,
      start_ms: round(start_ms),
      end_ms: round(end_ms),
      reason: Atom.to_string(reason),
      stable: true,
      send_to_llm: false
    }

    {event, state} =
      cond do
        duration_ms < state.min_utterance_ms ->
          {suppressed_event(state, :too_short, start_ms, end_ms), state}

        not utterance_speech?(state) ->
          {suppressed_event(state, :not_speech, start_ms, end_ms), state}

        true ->
          final_transcript_event(state, start_ms, end_ms)
      end

    state = reset_utterance(state)
    {state, [event, speech_end_event | events]}
  end

  defp final_transcript_event(state, start_ms, end_ms) do
    case transcribe(state, utterance_samples(state)) do
      {:ok, text, metrics, state} ->
        cond do
          normalize_text(text) == "" ->
            {suppressed_event(state, :empty_transcript, start_ms, end_ms), state}

          tts_echo?(state, text, end_ms) ->
            {suppressed_event(state, :tts_echo, start_ms, end_ms, text, metrics), state}

          true ->
            {%{
               type: "final",
               utterance_id: state.utterance_id,
               text: text,
               start_ms: round(start_ms),
               end_ms: round(end_ms),
               stable: true,
               send_to_llm: true,
               metrics: metrics
             }, state}
        end

      {:error, reason, state} ->
        {error_event(state, reason), state}
    end
  end

  defp transcribe(state, samples, generate_opts \\ []) do
    with {:ok, state} <- ensure_runtime(state) do
      if incremental?(state) do
        incremental_transcribe(state, samples, generate_opts)
      else
        full_transcribe(state, samples, generate_opts)
      end
    else
      {:error, reason, state} -> {:error, reason, state}
    end
  end

  defp incremental?(state) do
    state.incremental? and Code.ensure_loaded?(state.runtime_module) and
      function_exported?(state.runtime_module, :start_utterance, 2)
  end

  # Only the audio that arrived since the last call is prefilled; the cache
  # already covers everything before it.
  defp incremental_transcribe(state, samples, generate_opts) do
    {state, append_ms} =
      timed(fn ->
        state = ensure_utterance_cache(state)
        new_samples = Enum.drop(samples, state.utterance_cached_samples)

        cache =
          state.runtime_module.append_audio(state.utterance_cache, new_samples)

        %{
          state
          | utterance_cache: cache,
            utterance_cached_samples: state.utterance_cached_samples + length(new_samples)
        }
      end)

    {{status, text_or_reason}, generate_ms} =
      timed(fn ->
        case state.runtime_module.transcribe_utterance(state.utterance_cache, generate_opts) do
          {:ok, text} -> {:ok, String.trim(text)}
          {:error, reason} -> {:error, reason}
        end
      end)

    audio_tokens = Map.get(state.utterance_cache, :audio_tokens, 0)

    metrics = %{
      input_build_ms: append_ms,
      generate_ms: generate_ms,
      audio_tokens: audio_tokens,
      bucket_seconds:
        audio_tokens * AudioFeatureExtractor.samples_per_token() / state.sample_rate,
      incremental: true
    }

    case status do
      :ok -> {:ok, text_or_reason, metrics, state}
      :error -> {:error, text_or_reason, state}
    end
  end

  defp ensure_utterance_cache(%__MODULE__{utterance_cache: nil} = state) do
    {:ok, cache} =
      state.runtime_module.start_utterance(state.runtime,
        prompt: Keyword.fetch!(state.runtime_opts, :prompt),
        system_message: Keyword.get(state.runtime_opts, :system_message)
      )

    %{state | utterance_cache: cache, utterance_cached_samples: 0}
  end

  defp ensure_utterance_cache(state), do: state

  defp full_transcribe(state, samples, generate_opts) do
    with {:ok, state} <- ensure_runtime(state) do
      {input, input_build_ms} =
        timed(fn ->
          Input.build(samples,
            prompt: Keyword.fetch!(state.runtime_opts, :prompt),
            system_message: Keyword.get(state.runtime_opts, :system_message),
            audio_token_count: audio_token_bucket(state, length(samples))
          )
        end)

      {{status, text_or_reason}, generate_ms} =
        timed(fn ->
          case state.runtime_module.generate(
                 state.runtime,
                 input,
                 Keyword.merge(
                   [timeout_seconds: Keyword.get(state.runtime_opts, :request_timeout_seconds)],
                   generate_opts
                 )
               ) do
            {:ok, text} -> {:ok, String.trim(text)}
            {:error, reason} -> {:error, reason}
          end
        end)

      metrics = %{
        input_build_ms: input_build_ms,
        generate_ms: generate_ms,
        audio_tokens: input.audio.token_count,
        bucket_seconds:
          input.audio.token_count * input.audio.samples_per_token / state.sample_rate
      }

      case status do
        :ok -> {:ok, text_or_reason, metrics, state}
        :error -> {:error, text_or_reason, state}
      end
    else
      {:error, reason, state} -> {:error, reason, state}
    end
  end

  defp ensure_runtime(%__MODULE__{runtime: nil} = state) do
    case state.runtime_module.load(state.runtime_opts) do
      {:ok, runtime} ->
        state = %{state | runtime: runtime}

        case maybe_warmup(state) do
          :ok -> {:ok, state}
          {:error, reason} -> {:error, reason, state}
        end

      {:error, reason} ->
        {:error, reason, state}
    end
  rescue
    exception -> {:error, Exception.message(exception), state}
  end

  defp ensure_runtime(%__MODULE__{} = state), do: {:ok, state}

  # Compile the prefill/decode executables for every configured audio-token
  # bucket up front so no live utterance pays JIT compilation latency.
  defp maybe_warmup(state) do
    if state.warmup? and Code.ensure_loaded?(state.runtime_module) and
         function_exported?(state.runtime_module, :warmup, 2) do
      state.runtime_module.warmup(state.runtime,
        audio_token_counts: state.audio_token_buckets,
        prompt: Keyword.fetch!(state.runtime_opts, :prompt),
        system_message: Keyword.get(state.runtime_opts, :system_message)
      )
    else
      :ok
    end
  end

  defp audio_token_bucket(state, sample_count) do
    actual_tokens = ceil_div(sample_count, AudioFeatureExtractor.samples_per_token())

    state.audio_token_buckets
    |> Enum.sort()
    |> Enum.find(actual_tokens, &(&1 >= actual_tokens))
  end

  defp active_frame?(state, frame) do
    analysis =
      SpeechGate.analyze(frame,
        sample_rate: state.sample_rate,
        min_speech_seconds: state.frame_ms / 1000,
        threshold: state.speech_threshold,
        min_active_ratio: 0.0,
        max_zero_crossing_rate: state.speech_max_zero_crossing_rate,
        frame_seconds: state.frame_ms / 1000
      )

    analysis.speech?
  end

  defp utterance_samples(state) do
    state.utterance_samples |> Enum.reverse() |> Enum.concat()
  end

  defp utterance_speech?(state) do
    analysis =
      SpeechGate.analyze(utterance_samples(state),
        sample_rate: state.sample_rate,
        min_speech_seconds: state.min_utterance_ms / 1000,
        threshold: state.speech_threshold,
        min_active_ratio: state.speech_min_active_ratio,
        max_zero_crossing_rate: state.speech_max_zero_crossing_rate
      )

    analysis.speech?
  end

  defp suppressed_event(state, reason, start_ms, end_ms, text \\ nil, metrics \\ %{}) do
    %{
      type: "suppressed",
      utterance_id: state.utterance_id,
      reason: Atom.to_string(reason),
      text: text,
      start_ms: round(start_ms),
      end_ms: round(end_ms),
      stable: true,
      send_to_llm: false,
      metrics: metrics
    }
  end

  defp error_event(state, reason) do
    %{
      type: "error",
      utterance_id: state.utterance_id,
      reason: inspect(reason),
      stable: true,
      send_to_llm: false
    }
  end

  defp reset_utterance(state) do
    %{
      state
      | utterance_cache: nil,
        utterance_cached_samples: 0,
        in_speech?: false,
        utterance_id: nil,
        utterance_samples: [],
        utterance_start_ms: nil,
        utterance_last_active_end_ms: nil,
        trailing_silence_ms: 0.0,
        last_partial_ms: nil
    }
  end

  defp tts_echo?(state, text, end_ms) do
    normalized = normalize_text(text)

    Enum.any?(state.tts_history, fn %{normalized: tts_text, timestamp_ms: timestamp_ms} ->
      abs(end_ms - timestamp_ms) <= state.tts_echo_window_ms and
        similarity(normalized, tts_text) >= state.tts_similarity_threshold
    end)
  end

  defp prune_tts_history(state, timestamp_ms) do
    history =
      Enum.filter(state.tts_history, fn %{timestamp_ms: tts_timestamp_ms} ->
        timestamp_ms - tts_timestamp_ms <= state.tts_echo_window_ms
      end)

    %{state | tts_history: history}
  end

  defp normalize_text(nil), do: ""

  defp normalize_text(text) do
    text
    |> String.downcase()
    |> String.replace(~r/[^[:alnum:]\s]/u, " ")
    |> String.replace(~r/\s+/, " ")
    |> String.trim()
  end

  defp similarity("", _right), do: 0.0
  defp similarity(_left, ""), do: 0.0
  defp similarity(left, right) when left == right, do: 1.0

  defp similarity(left, right) do
    cond do
      String.contains?(left, right) or String.contains?(right, left) ->
        1.0

      true ->
        left_words = left |> String.split(" ", trim: true) |> MapSet.new()
        right_words = right |> String.split(" ", trim: true) |> MapSet.new()
        intersection = MapSet.intersection(left_words, right_words) |> MapSet.size()
        union = MapSet.union(left_words, right_words) |> MapSet.size()

        if union == 0, do: 0.0, else: intersection / union
    end
  end

  defp timed(fun) do
    started_at = System.monotonic_time(:millisecond)
    result = fun.()
    {result, System.monotonic_time(:millisecond) - started_at}
  end

  defp ceil_div(0, _denominator), do: 0
  defp ceil_div(numerator, denominator), do: div(numerator + denominator - 1, denominator)
end
