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
              prompt: Config.default_prompt(),
              window_seconds: 5.0,
              stride_seconds: 2.5,
              sample_rate: 16_000,
              request_timeout_seconds: Config.request_timeout_seconds(),
              model_name: Config.default_model_name(),
              max_response_tokens: Config.max_response_tokens(),
              backend: Config.backend(),
              debug: false
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
    debug: :boolean
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

    runtime_module = Keyword.get(opts, :runtime_module)

    debug(config, fn ->
      "cli: preparing WAV windows path=#{inspect(config.wav)} sample_rate=#{config.sample_rate} " <>
        "window_seconds=#{config.window_seconds} stride_seconds=#{config.stride_seconds} " <>
        "skip_windows=#{config.skip_windows} max_windows=#{inspect(config.max_windows)}"
    end)

    with {:ok, windows} <- wav_windows(config),
         {:ok, results} <-
           Transcriber.transcribe_windows(windows,
             model_name: config.model_name,
             backend: config.backend,
             max_response_tokens: config.max_response_tokens,
             prompt: config.prompt,
             system_message: config.system_message,
             request_timeout_seconds: config.request_timeout_seconds,
             debug: config.debug,
             runtime_module: runtime_module || Gemma4MicTranscribe.Gemma4Unified.Runtime
           ) do
      Enum.each(results, fn {:ok, window, text} ->
        start = Audio.frames_to_timestamp(window.start_frame, window.sample_rate)
        finish = Audio.frames_to_timestamp(window.end_frame, window.sample_rate)
        transcript = if text == "", do: "<no transcript>", else: text
        IO.puts("[#{start}-#{finish}] #{transcript}")
      end)

      if Enum.any?(results, fn {:ok, _window, text} -> text != "" end), do: 0, else: 3
    else
      {:error, {:unsupported_runtime, message, _runtime}} ->
        IO.puts(:stderr, "error: #{message}")
        2

      {:error, {:unsupported_runtime, message}} ->
        IO.puts(:stderr, "error: #{message}")
        2

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
      debug: Keyword.get(opts, :debug, false)
    }

    with :ok <- validate_positive(config.window_seconds, "--window-seconds"),
         :ok <- validate_positive(config.stride_seconds, "--stride-seconds"),
         :ok <- validate_positive(config.sample_rate, "--sample-rate"),
         :ok <- validate_positive(config.request_timeout_seconds, "--request-timeout-seconds"),
         :ok <- validate_positive(config.max_response_tokens, "--max-response-tokens"),
         :ok <- validate_non_negative(config.skip_windows, "--skip-windows"),
         :ok <- validate_optional_positive(config.max_windows, "--max-windows"),
         {:ok, system_message} <-
           read_system_message(config.system_message, Keyword.get(opts, :system_message_file)),
         :ok <- validate_wav(config.wav) do
      {:ok, %{config | system_message: system_message}}
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
      |> Enum.drop(config.skip_windows)
      |> maybe_take(config.max_windows)

    if windows == [] do
      {:error, :no_wav_windows}
    else
      debug(config, fn -> "cli: selected #{length(windows)} WAV window(s)" end)
      {:ok, windows}
    end
  rescue
    exception -> {:error, exception}
  end

  defp read_system_message(system_message, nil), do: {:ok, system_message}

  defp read_system_message(nil, path),
    do: {:ok, path |> Path.expand() |> File.read!() |> String.trim()}

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
  defp validate_optional_positive(nil, _name), do: :ok
  defp validate_optional_positive(value, _name) when is_integer(value) and value > 0, do: :ok
  defp validate_optional_positive(_value, name), do: {:error, "#{name} must be positive"}

  defp maybe_take(windows, nil), do: windows
  defp maybe_take(windows, count), do: Enum.take(windows, count)

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
      --max-windows INT                  Stop after N selected windows
      --system-message TEXT              System instruction for every window
      --system-message-file PATH         Read system instruction from a file
      --prompt TEXT                      User prompt paired with every audio window
      --window-seconds FLOAT             Audio window duration, default 5.0
      --stride-seconds FLOAT             Seconds between windows, default 2.5
      --sample-rate INT                  Target sample rate, default 16000
      --request-timeout-seconds FLOAT    Maximum seconds for one generation
      --model-name NAME                  Hugging Face or local model name
      --max-response-tokens INT          Maximum generated tokens
      --backend host|exla|torchx         Nx/Bumblebee backend label
      --debug                            Emit progress logs to stderr
    """
  end
end
