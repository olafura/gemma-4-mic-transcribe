defmodule Gemma4MicTranscribe.CLI do
  @moduledoc false

  require Logger

  alias Gemma4MicTranscribe.Audio
  alias Gemma4MicTranscribe.Config
  alias Gemma4MicTranscribe.ModelCatalog
  alias Gemma4MicTranscribe.Transcriber

  defmodule RunConfig do
    @moduledoc false
    defstruct wav: nil,
              skip_windows: 0,
              max_windows: nil,
              system_message: nil,
              system_message_source: :none,
              prompt: Config.default_prompt(),
              window_seconds: 5.0,
              stride_seconds: 2.5,
              sample_rate: 16_000,
              request_timeout_seconds: Config.request_timeout_seconds(),
              model_name: Config.default_model_name(),
              max_response_tokens: Config.max_response_tokens(),
              backend: Config.backend(),
              speech_gate: Config.speech_gate?(),
              min_speech_seconds: Config.min_speech_seconds(),
              speech_threshold: Config.speech_threshold(),
              speech_min_active_ratio: Config.speech_min_active_ratio(),
              speech_max_zero_crossing_rate: Config.speech_max_zero_crossing_rate(),
              debug: false,
              debug_top_k: 0
  end

  @switches [
    help: :boolean,
    list_models: :boolean,
    wav: :string,
    skip_windows: :integer,
    max_windows: :integer,
    system_message: :string,
    system_message_file: :string,
    prompt: :string,
    window_seconds: :float,
    stride_seconds: :float,
    sample_rate: :integer,
    request_timeout_seconds: :float,
    model_name: :string,
    max_response_tokens: :integer,
    backend: :string,
    speech_gate: :boolean,
    min_speech_seconds: :float,
    speech_threshold: :float,
    speech_min_active_ratio: :float,
    speech_max_zero_crossing_rate: :float,
    debug: :boolean,
    debug_top_k: :integer
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

    with {:ok, runtime_module} <- runtime_module(config, opts),
         {:ok, windows} <- wav_windows(config),
         {:ok, results} <-
           Transcriber.transcribe_windows(windows,
             model_name: config.model_name,
             backend: config.backend,
             max_response_tokens: config.max_response_tokens,
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

  defp validate(opts) do
    config = %RunConfig{
      wav: Keyword.get(opts, :wav),
      skip_windows: Keyword.get(opts, :skip_windows, 0),
      max_windows: Keyword.get(opts, :max_windows),
      system_message: Keyword.get(opts, :system_message),
      prompt: Keyword.get(opts, :prompt, Config.default_prompt()),
      window_seconds: Keyword.get(opts, :window_seconds, 5.0),
      stride_seconds: Keyword.get(opts, :stride_seconds, 2.5),
      sample_rate: Keyword.get(opts, :sample_rate, 16_000),
      request_timeout_seconds:
        Keyword.get(opts, :request_timeout_seconds, Config.request_timeout_seconds()),
      model_name: Keyword.get(opts, :model_name, Config.default_model_name()),
      max_response_tokens: Keyword.get(opts, :max_response_tokens, Config.max_response_tokens()),
      backend: Keyword.get(opts, :backend, Config.backend()),
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
      debug: Keyword.get(opts, :debug, false),
      debug_top_k: Keyword.get(opts, :debug_top_k, 0)
    }

    with :ok <- validate_positive(config.window_seconds, "--window-seconds"),
         :ok <- validate_positive(config.stride_seconds, "--stride-seconds"),
         :ok <- validate_positive(config.sample_rate, "--sample-rate"),
         :ok <- validate_positive(config.request_timeout_seconds, "--request-timeout-seconds"),
         :ok <- validate_positive(config.max_response_tokens, "--max-response-tokens"),
         :ok <- validate_positive(config.min_speech_seconds, "--min-speech-seconds"),
         :ok <- validate_positive(config.speech_threshold, "--speech-threshold"),
         :ok <- validate_ratio(config.speech_min_active_ratio, "--speech-min-active-ratio"),
         :ok <-
           validate_ratio(
             config.speech_max_zero_crossing_rate,
             "--speech-max-zero-crossing-rate"
           ),
         :ok <- validate_non_negative(config.skip_windows, "--skip-windows"),
         :ok <- validate_optional_positive(config.max_windows, "--max-windows"),
         :ok <- validate_non_negative(config.debug_top_k, "--debug-top-k"),
         {:ok, system_message, system_message_source} <-
           read_system_message(config.system_message, Keyword.get(opts, :system_message_file)),
         :ok <- validate_wav(config.wav) do
      {:ok,
       %{config | system_message: system_message, system_message_source: system_message_source}}
    end
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

  defp validate_ratio(value, _name) when is_number(value) and value >= 0.0 and value <= 1.0,
    do: :ok

  defp validate_ratio(_value, name), do: {:error, "#{name} must be between 0 and 1"}
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
      --system-message TEXT              System instruction for every window
      --system-message-file PATH         Read system instruction from a file
      --prompt TEXT                      User prompt paired with every audio window
      --window-seconds FLOAT             Audio window duration, default 5.0
      --stride-seconds FLOAT             Seconds between windows, default 2.5
      --sample-rate INT                  Target sample rate, default 16000
      --request-timeout-seconds FLOAT    Maximum seconds for one generation
      --model-name NAME                  Model alias or Hugging Face repo; selects the required runtime
      --max-response-tokens INT          Maximum generated text tokens per window, default 512
      --backend host|torchx|torchx:cpu|torchx:cuda|exla|exla:host|exla:cuda|exla:rocm
                                        Nx/Bumblebee backend label, default torchx
      --no-speech-gate                  Disable cheap local speech gating before model generation
      --min-speech-seconds FLOAT        Minimum likely speech duration before generation, default 0.25
      --speech-threshold FLOAT          RMS threshold for active audio frames, default 0.01
      --speech-min-active-ratio FLOAT   Required active-frame ratio per window, default 0.2
      --speech-max-zero-crossing-rate FLOAT
                                        Reject very noisy windows above this zero-crossing ratio, default 0.35
      --debug                            Emit progress logs to stderr
      --debug-top-k INT                  Log top prefill token candidates after suppression, default 0
    """
  end
end
