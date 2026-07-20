defmodule Gemma4MicTranscribe.CLI do
  @moduledoc false

  require Logger

  alias Gemma4MicTranscribe.Audio
  alias Gemma4MicTranscribe.Config
  alias Gemma4MicTranscribe.ModelCatalog
  alias Gemma4MicTranscribe.StreamingSession
  alias Gemma4MicTranscribe.Trace
  alias Gemma4MicTranscribe.Transcriber

  defmodule RunConfig do
    @moduledoc false
    defstruct wav: nil,
              skip_windows: 0,
              max_windows: nil,
              stream_wav: false,
              realtime: false,
              repeat: 1,
              output: "text",
              chunk_ms: 100.0,
              system_message: nil,
              system_message_source: :none,
              prompt: Config.default_prompt(),
              window_seconds: 5.0,
              stride_seconds: 2.5,
              sample_rate: 16_000,
              request_timeout_seconds: Config.request_timeout_seconds(),
              model_name: Config.default_model_name(),
              max_response_tokens: Config.max_response_tokens(),
              no_repeat_ngram: 0,
              backend: Config.backend(),
              param_type: "bf16",
              weights: "packed",
              incremental_prefill: false,
              warmup: true,
              speech_gate: Config.speech_gate?(),
              min_speech_seconds: Config.min_speech_seconds(),
              speech_threshold: Config.speech_threshold(),
              speech_min_active_ratio: Config.speech_min_active_ratio(),
              speech_max_zero_crossing_rate: Config.speech_max_zero_crossing_rate(),
              speech_start_ms: 120.0,
              speech_end_silence_ms: 500.0,
              min_utterance_ms: 350.0,
              max_utterance_ms: 8_000.0,
              partial_interval_ms: 1_000.0,
              partial_max_response_tokens: 16,
              partials: true,
              tts_text: nil,
              tts_timestamp_ms: 0.0,
              debug: false,
              debug_top_k: 0,
              trace: false
  end

  @switches [
    help: :boolean,
    list_models: :boolean,
    wav: :string,
    skip_windows: :integer,
    max_windows: :integer,
    stream_wav: :boolean,
    realtime: :boolean,
    repeat: :integer,
    output: :string,
    chunk_ms: :float,
    system_message: :string,
    system_message_file: :string,
    prompt: :string,
    window_seconds: :float,
    stride_seconds: :float,
    sample_rate: :integer,
    request_timeout_seconds: :float,
    model_name: :string,
    max_response_tokens: :integer,
    no_repeat_ngram: :integer,
    backend: :string,
    param_type: :string,
    weights: :string,
    incremental_prefill: :boolean,
    warmup: :boolean,
    speech_gate: :boolean,
    min_speech_seconds: :float,
    speech_threshold: :float,
    speech_min_active_ratio: :float,
    speech_max_zero_crossing_rate: :float,
    speech_start_ms: :float,
    speech_end_silence_ms: :float,
    min_utterance_ms: :float,
    max_utterance_ms: :float,
    partial_interval_ms: :float,
    partial_max_response_tokens: :integer,
    partials: :boolean,
    tts_text: :string,
    tts_timestamp_ms: :float,
    debug: :boolean,
    debug_top_k: :integer,
    trace: :boolean
  ]

  @aliases [h: :help]

  def main(argv, opts \\ []) do
    case parse(argv) do
      {:help, text} ->
        IO.puts(text)
        0

      {:list_models, text} ->
        IO.puts(text)
        0

      {:ok, config} ->
        run(config, opts)

      {:error, message} ->
        IO.puts(:stderr, "error: #{message}")
        1
    end
  end

  def parse(argv) do
    case OptionParser.parse(argv, strict: @switches, aliases: @aliases) do
      {opts, [], []} ->
        cond do
          Keyword.get(opts, :help, false) ->
            {:help, usage()}

          Keyword.get(opts, :list_models, false) ->
            {:list_models, ModelCatalog.format()}

          true ->
            validate(opts)
        end

      {_opts, _args, invalid} ->
        {:error, "invalid option(s): #{format_invalid(invalid)}"}
    end
  end

  def run(%RunConfig{wav: nil}, _opts) do
    IO.puts(
      :stderr,
      "error: microphone input is not supported yet; use --wav PATH"
    )

    2
  end

  def run(%RunConfig{} = config, opts) do
    configure_logger(config)

    debug(config, fn ->
      "cli: preparing WAV windows path=#{inspect(config.wav)} sample_rate=#{config.sample_rate} " <>
        "window_seconds=#{config.window_seconds} stride_seconds=#{config.stride_seconds} " <>
        "skip_windows=#{config.skip_windows} max_windows=#{inspect(config.max_windows)}"
    end)

    debug(config, fn ->
      "cli: speech_gate=#{config.speech_gate} min_speech_seconds=#{config.min_speech_seconds} " <>
        "speech_threshold=#{config.speech_threshold} speech_min_active_ratio=#{config.speech_min_active_ratio} " <>
        "speech_max_zero_crossing_rate=#{config.speech_max_zero_crossing_rate}"
    end)

    debug(config, fn ->
      "cli: prompt config model=#{inspect(config.model_name)} backend=#{inspect(config.backend)} " <>
        "prompt_bytes=#{byte_size(config.prompt)} system_message_bytes=#{byte_size_or_zero(config.system_message)} " <>
        "system_message=#{config.system_message not in [nil, ""]} " <>
        "system_message_source=#{inspect(config.system_message_source)} " <>
        "system_message_sha256=#{inspect(system_message_hash(config.system_message))}"
    end)

    if config.trace do
      Trace.with_tracing(fn -> run_mode(config, opts) end)
    else
      run_mode(config, opts)
    end
  end

  defp run_mode(config, opts) do
    if config.stream_wav do
      run_stream_wav(config, opts)
    else
      run_windowed_wav(config, opts)
    end
  end

  defp validate(opts) do
    config = %RunConfig{
      wav: Keyword.get(opts, :wav),
      skip_windows: Keyword.get(opts, :skip_windows, 0),
      max_windows: Keyword.get(opts, :max_windows),
      stream_wav: Keyword.get(opts, :stream_wav, false),
      realtime: Keyword.get(opts, :realtime, false),
      repeat: Keyword.get(opts, :repeat, 1),
      output: Keyword.get(opts, :output, "text"),
      chunk_ms: Keyword.get(opts, :chunk_ms, 100.0),
      system_message: Keyword.get(opts, :system_message),
      prompt: Keyword.get(opts, :prompt, Config.default_prompt()),
      window_seconds: Keyword.get(opts, :window_seconds, 5.0),
      stride_seconds: Keyword.get(opts, :stride_seconds, 2.5),
      sample_rate: Keyword.get(opts, :sample_rate, 16_000),
      request_timeout_seconds:
        Keyword.get(opts, :request_timeout_seconds, Config.request_timeout_seconds()),
      model_name: Keyword.get(opts, :model_name, Config.default_model_name()),
      max_response_tokens: Keyword.get(opts, :max_response_tokens, Config.max_response_tokens()),
      no_repeat_ngram: Keyword.get(opts, :no_repeat_ngram, 0),
      backend: Keyword.get(opts, :backend, Config.backend()),
      param_type: Keyword.get(opts, :param_type, "bf16"),
      weights: Keyword.get(opts, :weights, "packed"),
      incremental_prefill: Keyword.get(opts, :incremental_prefill, false),
      warmup: Keyword.get(opts, :warmup, true),
      speech_gate: Keyword.get(opts, :speech_gate, Config.speech_gate?()),
      min_speech_seconds: Keyword.get(opts, :min_speech_seconds, Config.min_speech_seconds()),
      speech_threshold: Keyword.get(opts, :speech_threshold, Config.speech_threshold()),
      speech_min_active_ratio:
        Keyword.get(opts, :speech_min_active_ratio, Config.speech_min_active_ratio()),
      speech_max_zero_crossing_rate:
        Keyword.get(
          opts,
          :speech_max_zero_crossing_rate,
          Config.speech_max_zero_crossing_rate()
        ),
      speech_start_ms: Keyword.get(opts, :speech_start_ms, 120.0),
      speech_end_silence_ms: Keyword.get(opts, :speech_end_silence_ms, 500.0),
      min_utterance_ms: Keyword.get(opts, :min_utterance_ms, 350.0),
      max_utterance_ms: Keyword.get(opts, :max_utterance_ms, 8_000.0),
      partial_interval_ms: Keyword.get(opts, :partial_interval_ms, 1_000.0),
      partial_max_response_tokens: Keyword.get(opts, :partial_max_response_tokens, 16),
      partials: Keyword.get(opts, :partials, true),
      tts_text: Keyword.get(opts, :tts_text),
      tts_timestamp_ms: Keyword.get(opts, :tts_timestamp_ms, 0.0),
      debug: Keyword.get(opts, :debug, false),
      debug_top_k: Keyword.get(opts, :debug_top_k, 0),
      trace: Keyword.get(opts, :trace, false)
    }

    with :ok <- validate_positive(config.window_seconds, "--window-seconds"),
         :ok <- validate_positive(config.stride_seconds, "--stride-seconds"),
         :ok <- validate_positive(config.chunk_ms, "--chunk-ms"),
         :ok <- validate_positive(config.repeat, "--repeat"),
         :ok <- validate_positive(config.sample_rate, "--sample-rate"),
         :ok <- validate_positive(config.request_timeout_seconds, "--request-timeout-seconds"),
         :ok <- validate_positive(config.max_response_tokens, "--max-response-tokens"),
         :ok <- validate_positive(config.min_speech_seconds, "--min-speech-seconds"),
         :ok <- validate_positive(config.speech_threshold, "--speech-threshold"),
         :ok <- validate_positive(config.speech_start_ms, "--speech-start-ms"),
         :ok <- validate_positive(config.speech_end_silence_ms, "--speech-end-silence-ms"),
         :ok <- validate_positive(config.min_utterance_ms, "--min-utterance-ms"),
         :ok <- validate_positive(config.max_utterance_ms, "--max-utterance-ms"),
         :ok <- validate_positive(config.partial_interval_ms, "--partial-interval-ms"),
         :ok <-
           validate_positive(config.partial_max_response_tokens, "--partial-max-response-tokens"),
         :ok <- validate_ratio(config.speech_min_active_ratio, "--speech-min-active-ratio"),
         :ok <-
           validate_ratio(
             config.speech_max_zero_crossing_rate,
             "--speech-max-zero-crossing-rate"
           ),
         :ok <- validate_non_negative(config.skip_windows, "--skip-windows"),
         :ok <- validate_non_negative_float(config.tts_timestamp_ms, "--tts-timestamp-ms"),
         :ok <- validate_optional_positive(config.max_windows, "--max-windows"),
         :ok <- validate_non_negative(config.debug_top_k, "--debug-top-k"),
         :ok <- validate_non_negative(config.no_repeat_ngram, "--no-repeat-ngram"),
         :ok <- validate_param_type(config.param_type),
         :ok <- validate_weights(config.weights),
         :ok <- validate_output(config.output),
         {:ok, system_message, system_message_source} <-
           read_system_message(config.system_message, Keyword.get(opts, :system_message_file)),
         :ok <- validate_wav(config.wav) do
      {:ok,
       %{config | system_message: system_message, system_message_source: system_message_source}}
    end
  end

  defp run_windowed_wav(config, opts) do
    with {:ok, runtime_module} <- runtime_module(config, opts),
         {:ok, windows} <- wav_windows(config),
         {:ok, results} <-
           Transcriber.transcribe_windows(windows,
             model_name: config.model_name,
             backend: config.backend,
             param_type: config.param_type,
             packed_weights: config.weights in ["packed", "hybrid"],
             hybrid_weights: config.weights == "hybrid",
             warmup: config.warmup,
             max_response_tokens: config.max_response_tokens,
             no_repeat_ngram_size: config.no_repeat_ngram,
             prompt: config.prompt,
             system_message: config.system_message,
             request_timeout_seconds: config.request_timeout_seconds,
             debug: config.debug,
             speech_gate: config.speech_gate,
             min_speech_seconds: config.min_speech_seconds,
             speech_threshold: config.speech_threshold,
             speech_min_active_ratio: config.speech_min_active_ratio,
             speech_max_zero_crossing_rate: config.speech_max_zero_crossing_rate,
             debug_top_k: config.debug_top_k,
             runtime_module: runtime_module,
             on_window_result: &print_window_result/1
           ) do
      if Enum.any?(results, fn {:ok, _window, text} -> text != "" end), do: 0, else: 3
    else
      {:error, reason} ->
        IO.puts(:stderr, "error: #{format_reason(reason)}")
        1
    end
  end

  defp run_stream_wav(config, opts) do
    with {:ok, runtime_module} <- runtime_module(config, opts),
         {:ok, session} <-
           StreamingSession.start_link(streaming_session_opts(config, runtime_module)) do
      base_sink = Keyword.get(opts, :event_sink, &print_stream_event(config, &1))

      # In realtime mode the audio clock starts at the first chunk; events are
      # annotated with lag_ms = wall time since clock start - event end_ms,
      # which is how far each event trails the live audio.
      clock_start_ms = System.monotonic_time(:millisecond)

      event_sink = fn event ->
        event = annotate_lag(event, config, clock_start_ms)
        base_sink.(event)
        event
      end

      try do
        # The session keeps one audio timeline across passes, so later passes
        # continue it rather than restarting; resetting only the wall clock
        # would make lag negative.
        {counts, _offset} =
          Enum.reduce(1..config.repeat, {%{finals: 0, errors: 0, lags: %{}}, 0.0}, fn pass,
                                                                                      {counts,
                                                                                       offset} ->
            counts =
              counts
              |> emit_stream_events(maybe_push_tts(session, config), event_sink)
              |> push_streaming_wav(session, config, event_sink, clock_start_ms, offset)
              |> emit_stream_events(flush_streaming_wav(session), event_sink)

            if config.repeat > 1 do
              IO.puts(:stderr, "bench: pass #{pass}/#{config.repeat} complete")
            end

            {counts, offset + wav_duration_ms(config)}
          end)

        print_lag_summary(config, counts)

        cond do
          counts.errors > 0 -> 1
          counts.finals > 0 -> 0
          true -> 3
        end
      after
        if Process.alive?(session), do: GenServer.stop(session)
      end
    else
      {:error, reason} ->
        IO.puts(:stderr, "error: #{format_reason(reason)}")
        1
    end
  end

  defp annotate_lag(event, %RunConfig{realtime: true}, clock_start_ms) do
    case event do
      %{end_ms: end_ms} ->
        lag_ms = System.monotonic_time(:millisecond) - clock_start_ms - round(end_ms)
        Map.put(event, :lag_ms, lag_ms)

      _event ->
        event
    end
  end

  defp annotate_lag(event, _config, _clock_start_ms), do: event

  defp record_lag(counts, %{lag_ms: lag_ms, type: type}) do
    update_in(counts.lags[type], &[lag_ms | &1 || []])
  end

  defp record_lag(counts, _event), do: counts

  defp print_lag_summary(%RunConfig{realtime: true} = config, %{lags: lags})
       when map_size(lags) > 0 do
    Enum.each(Enum.sort(lags), fn {type, lag_list} ->
      lag_list = Enum.sort(lag_list)
      count = length(lag_list)
      avg = round(Enum.sum(lag_list) / count)

      IO.puts(
        :stderr,
        "bench: #{type} events=#{count} lag_ms min=#{List.first(lag_list)} " <>
          "avg=#{avg} max=#{List.last(lag_list)}"
      )

      # A final is held until the endpointer has seen enough silence, so its lag
      # mixes two independent costs. Reporting them apart keeps transcription
      # speed comparable across endpointing settings, and matches how streaming
      # ASR services report latency.
      if type == "final" do
        eot = round(config.speech_end_silence_ms)
        transcript = Enum.map(lag_list, &max(&1 - eot, 0))
        transcript_avg = round(Enum.sum(transcript) / count)

        IO.puts(
          :stderr,
          "bench: #{type} eot_ms=#{eot} transcript_ms min=#{List.first(transcript)} " <>
            "avg=#{transcript_avg} max=#{List.last(transcript)}"
        )
      end
    end)
  end

  defp print_lag_summary(_config, _counts), do: :ok

  defp streaming_session_opts(config, runtime_module) do
    [
      runtime_module: runtime_module,
      model_name: config.model_name,
      backend: config.backend,
      param_type: config.param_type,
      packed_weights: config.weights in ["packed", "hybrid"],
      hybrid_weights: config.weights == "hybrid",
      incremental_prefill: config.incremental_prefill,
      warmup: config.warmup,
      # Lag numbers are only meaningful against a loaded, warmed model, so
      # realtime mode loads before the audio clock starts.
      preload_runtime: config.realtime,
      max_response_tokens: config.max_response_tokens,
      no_repeat_ngram_size: config.no_repeat_ngram,
      prompt: config.prompt,
      system_message: config.system_message,
      request_timeout_seconds: config.request_timeout_seconds,
      sample_rate: config.sample_rate,
      speech_threshold: config.speech_threshold,
      speech_min_active_ratio: config.speech_min_active_ratio,
      speech_max_zero_crossing_rate: config.speech_max_zero_crossing_rate,
      speech_start_ms: config.speech_start_ms,
      speech_end_silence_ms: config.speech_end_silence_ms,
      min_utterance_ms: config.min_utterance_ms,
      max_utterance_ms: config.max_utterance_ms,
      partial_interval_ms: config.partial_interval_ms,
      partial_max_response_tokens: config.partial_max_response_tokens,
      partials: config.partials,
      debug: config.debug,
      debug_top_k: config.debug_top_k
    ]
  end

  defp maybe_push_tts(_session, %RunConfig{tts_text: nil}), do: []

  defp maybe_push_tts(session, %RunConfig{} = config) do
    case StreamingSession.push_tts(session, config.tts_text, config.tts_timestamp_ms) do
      {:ok, events} -> events
    end
  end

  defp push_streaming_wav(counts, session, config, event_sink, clock_start_ms, offset_ms) do
    config.wav
    |> Audio.read_wav_samples!(config.sample_rate)
    |> Audio.stream_sample_chunks(config.sample_rate, config.chunk_ms)
    |> Enum.reduce(counts, fn {samples, chunk_ms}, counts ->
      timestamp_ms = chunk_ms + offset_ms
      maybe_pace(config, clock_start_ms, timestamp_ms)

      case StreamingSession.push_audio(session, samples, timestamp_ms) do
        {:ok, events} -> emit_stream_events(counts, events, event_sink)
      end
    end)
  rescue
    exception ->
      emit_stream_events(
        counts,
        [%{type: "error", reason: Exception.message(exception), send_to_llm: false}],
        event_sink
      )
  end

  defp wav_duration_ms(config) do
    length(Audio.read_wav_samples!(config.wav, config.sample_rate)) * 1000 / config.sample_rate
  end

  defp flush_streaming_wav(session) do
    case StreamingSession.flush(session) do
      {:ok, events} -> events
    end
  end

  # A chunk is not pushed before its audio timestamp has elapsed on the wall
  # clock, simulating a live microphone. When generation runs behind, pushes
  # happen late and the lag becomes visible in lag_ms.
  defp maybe_pace(%RunConfig{realtime: true}, clock_start_ms, timestamp_ms) do
    target_ms = clock_start_ms + round(timestamp_ms)
    delay_ms = target_ms - System.monotonic_time(:millisecond)
    if delay_ms > 0, do: Process.sleep(delay_ms)
  end

  defp maybe_pace(_config, _clock_start_ms, _timestamp_ms), do: :ok

  defp emit_stream_events(counts, events, event_sink) do
    Enum.reduce(events, counts, fn event, counts ->
      event = event_sink.(event)
      counts = record_lag(counts, event)

      cond do
        event[:type] == "final" and event[:send_to_llm] ->
          %{counts | finals: counts.finals + 1}

        event[:type] == "error" ->
          %{counts | errors: counts.errors + 1}

        true ->
          counts
      end
    end)
  end

  defp wav_windows(config) do
    windows =
      config.wav
      |> Audio.stream_wav_windows(
        config.sample_rate,
        config.window_seconds,
        config.stride_seconds
      )
      |> Stream.drop(config.skip_windows)
      |> maybe_take(config.max_windows)
      |> Enum.to_list()

    if windows == [] do
      {:error, :no_wav_windows}
    else
      debug(config, fn -> "cli: selected #{length(windows)} WAV window(s)" end)
      {:ok, windows}
    end
  rescue
    exception -> {:error, exception}
  end

  defp read_system_message(nil, nil), do: {:ok, nil, :none}

  defp read_system_message(system_message, nil), do: {:ok, system_message, :system_message}

  defp read_system_message(nil, path) do
    expanded_path = Path.expand(path)
    {:ok, expanded_path |> File.read!() |> String.trim(), {:system_message_file, expanded_path}}
  end

  defp read_system_message(_system_message, _path),
    do: {:error, "use either --system-message or --system-message-file, not both"}

  defp validate_wav(nil), do: :ok

  defp validate_wav(path) do
    if File.regular?(Path.expand(path)) do
      :ok
    else
      {:error, "--wav file not found: #{path}"}
    end
  end

  defp validate_positive(value, _name) when is_number(value) and value > 0, do: :ok
  defp validate_positive(_value, name), do: {:error, "#{name} must be positive"}
  defp validate_non_negative(value, _name) when is_integer(value) and value >= 0, do: :ok
  defp validate_non_negative(_value, name), do: {:error, "#{name} must be zero or positive"}
  defp validate_non_negative_float(value, _name) when is_number(value) and value >= 0.0, do: :ok

  defp validate_non_negative_float(_value, name),
    do: {:error, "#{name} must be zero or positive"}

  defp validate_ratio(value, _name) when is_number(value) and value >= 0.0 and value <= 1.0,
    do: :ok

  defp validate_ratio(_value, name), do: {:error, "#{name} must be between 0 and 1"}
  defp validate_param_type(param_type) when param_type in ["bf16", "f16", "f32"], do: :ok
  defp validate_param_type(_param_type), do: {:error, "--param-type must be bf16, f16, or f32"}
  defp validate_weights(weights) when weights in ["packed", "bf16", "hybrid"], do: :ok
  defp validate_weights(_weights), do: {:error, "--weights must be packed, bf16, or hybrid"}
  defp validate_output(output) when output in ["text", "jsonl"], do: :ok
  defp validate_output(_output), do: {:error, "--output must be text or jsonl"}
  defp validate_optional_positive(nil, _name), do: :ok
  defp validate_optional_positive(value, _name) when is_integer(value) and value > 0, do: :ok
  defp validate_optional_positive(_value, name), do: {:error, "#{name} must be positive"}
  defp byte_size_or_zero(nil), do: 0
  defp byte_size_or_zero(text) when is_binary(text), do: byte_size(text)

  defp system_message_hash(nil), do: nil
  defp system_message_hash(""), do: nil

  defp system_message_hash(text) when is_binary(text) do
    text
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
    |> binary_part(0, 12)
  end

  defp maybe_take(windows, nil), do: windows
  defp maybe_take(windows, count), do: Stream.take(windows, count)

  defp runtime_module(%RunConfig{} = config, opts) do
    case Keyword.fetch(opts, :runtime_module) do
      {:ok, runtime_module} -> {:ok, runtime_module}
      :error -> ModelCatalog.runtime_module(config.model_name)
    end
  end

  defp print_window_result({:ok, window, text}) do
    if text != "" do
      start = Audio.frames_to_timestamp(window.start_frame, window.sample_rate)
      finish = Audio.frames_to_timestamp(window.end_frame, window.sample_rate)
      IO.puts("[#{start}-#{finish}] #{text}")
    end
  end

  defp print_stream_event(%RunConfig{output: "jsonl"}, event) do
    IO.puts(StreamingSession.json_event!(event))
  end

  defp print_stream_event(%RunConfig{output: "text"}, %{type: "final"} = event) do
    start = ms_to_timestamp(event.start_ms)
    finish = ms_to_timestamp(event.end_ms)
    IO.puts("[#{start}-#{finish}] #{event.text}")
  end

  defp print_stream_event(%RunConfig{output: "text"}, _event), do: :ok

  defp ms_to_timestamp(ms) do
    total_seconds = ms / 1000
    minutes = trunc(total_seconds / 60)
    remaining = total_seconds - minutes * 60

    seconds_text =
      :io_lib.format("~.1f", [remaining])
      |> IO.iodata_to_binary()
      |> String.pad_leading(4, "0")

    String.pad_leading(Integer.to_string(minutes), 2, "0") <> ":" <> seconds_text
  end

  defp configure_logger(%RunConfig{debug: true}) do
    Logger.configure(level: :debug)
    Logger.debug("cli: debug logging enabled")
  end

  defp configure_logger(%RunConfig{}), do: :ok

  defp debug(%RunConfig{debug: true}, message_fun), do: Logger.debug(message_fun)
  defp debug(%RunConfig{}, _message_fun), do: :ok

  defp format_invalid(invalid) do
    invalid
    |> Enum.map(fn {option, value} -> "#{option}=#{inspect(value)}" end)
    |> Enum.join(", ")
  end

  defp format_reason(%{message: message}) when is_binary(message), do: message
  defp format_reason(:no_wav_windows), do: "No WAV windows selected."
  defp format_reason(reason), do: inspect(reason)

  defp usage do
    """
    Usage:
      gemma_4_mic_transcribe --wav PATH [options]
      gemma_4_mic_transcribe --list-models

    Options:
      --wav PATH                         Read PCM WAV audio from a file
      --skip-windows INT                 Skip leading audio windows
      --max-windows INT                  Stop after N selected rolling audio windows, not audio tokens
      --stream-wav                       Process WAV audio as timed streaming chunks and emit utterance events
      --realtime                         Pace stream chunks to the wall clock and report event lag_ms
      --repeat INT                       Replay the audio N times against the loaded model, default 1
      --output text|jsonl                Output format for stream events, default text
      --chunk-ms FLOAT                   Streaming WAV chunk duration, default 100.0
      --system-message TEXT              System instruction for every window
      --system-message-file PATH         Read system instruction from a file
      --prompt TEXT                      User prompt paired with every audio window
      --window-seconds FLOAT             Audio window duration, default 5.0
      --stride-seconds FLOAT             Seconds between windows, default 2.5
      --sample-rate INT                  Target sample rate, default 16000
      --request-timeout-seconds FLOAT    Maximum seconds for one generation
      --model-name NAME                  Model alias or Hugging Face repo; selects the required runtime
      --max-response-tokens INT          Maximum generated text tokens per window, default 512
      --no-repeat-ngram INT              Ban repeating generated n-grams of this size, 0 disables (default)
      --backend host|torchx|torchx:cpu|torchx:cuda|exla|exla:host|exla:cuda|exla:rocm
                                        Nx/Bumblebee backend label, default torchx
      --param-type bf16|f16|f32          Model parameter/compute precision, default bf16
      --weights packed|bf16|hybrid       packed: int4 only (least memory, fast decode, slow prefill)
                                        bf16: dequantized only (fast prefill, slower decode)
                                        hybrid: both, fast prefill and fast decode (~31GB)
                                        default packed
      --incremental-prefill              Prefill audio during speech instead of after end-of-speech
      --no-warmup                        Skip startup generation warmup (JIT compiles on first utterance instead)
      --no-speech-gate                  Disable cheap local speech gating before model generation
      --min-speech-seconds FLOAT        Minimum likely speech duration before generation, default 0.25
      --speech-threshold FLOAT          RMS threshold for active audio frames, default 0.01
      --speech-min-active-ratio FLOAT   Required active-frame ratio per window, default 0.2
      --speech-max-zero-crossing-rate FLOAT
                                        Reject very noisy windows above this zero-crossing ratio, default 0.35
      --speech-start-ms FLOAT            Active speech needed to start an utterance, default 120
      --speech-end-silence-ms FLOAT      Silence needed to commit an utterance, default 500
      --min-utterance-ms FLOAT           Suppress shorter utterances, default 350
      --max-utterance-ms FLOAT           Force-commit long utterances, default 8000
      --partial-interval-ms FLOAT        Minimum time between partial transcripts, default 1000
      --partial-max-response-tokens INT  Token cap for partial transcripts, default 16
      --no-partials                      Disable unstable partial transcript events
      --tts-text TEXT                    Recent TTS text to suppress as echo in stream mode
      --tts-timestamp-ms FLOAT           Timestamp for --tts-text, default 0
      --debug                            Emit progress logs to stderr
      --debug-top-k INT                  Log top prefill token candidates after suppression, default 0
      --trace                            Log per-call BEAM durations for pipeline modules to stderr
    """
  end
end
