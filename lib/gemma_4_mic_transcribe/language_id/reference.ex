defmodule Gemma4MicTranscribe.LanguageId.Reference do
  @moduledoc """
  Reference language detectors to measure the Gemma-based detector against.

  * `whisper/3` runs whisper.cpp's `whisper-cli` on a clip with language
    auto-detection and returns the detected language and transcript.
  * `text_detector/1` loads `papluca/xlm-roberta-base-language-detection`
    through Bumblebee, a text classifier over 20 languages; `detect_text/2`
    ranks its languages for a transcript. Chained after Whisper it gives the
    "transcribe, then classify the text" baseline.

  Both are far slower than the audio tower on short clips, which is the point
  of the comparison.
  """

  @text_repo {:hf, "papluca/xlm-roberta-base-language-detection"}

  @doc "Common Voice codes that the text detector can name, mapped to its labels."
  def text_label("zh-" <> _region), do: "zh"
  def text_label(code), do: code

  @doc """
  Transcribes `wav_path` with whisper.cpp. Options: `:binary` (default from
  `WHISPER_CLI`, else `whisper-cli`), `:model` (ggml file, required),
  `:threads`. Returns `%{language, text, ms}`.
  """
  def whisper(wav_path, opts) do
    binary = Keyword.get(opts, :binary) || System.get_env("WHISPER_CLI") || "whisper-cli"
    model = Keyword.fetch!(opts, :model)
    threads = Keyword.get(opts, :threads, 8)
    output = Path.join(System.tmp_dir!(), "whisper-#{System.unique_integer([:positive])}")

    args = [
      "-m", model,
      "-f", wav_path,
      "-l", "auto",
      "-t", Integer.to_string(threads),
      "-nt", "-np",
      "-oj", "-of", output
    ]

    started = System.monotonic_time(:millisecond)
    {_stdout, status} = System.cmd(binary, args, stderr_to_stdout: true)
    ms = System.monotonic_time(:millisecond) - started
    json_path = output <> ".json"

    result =
      case {status, File.read(json_path)} do
        {0, {:ok, json}} ->
          decoded = Jason.decode!(json)

          text =
            decoded
            |> Map.get("transcription", [])
            |> Enum.map_join(" ", &String.trim(&1["text"] || ""))
            |> String.trim()

          %{language: get_in(decoded, ["result", "language"]), text: text, ms: ms}

        _other ->
          %{language: nil, text: "", ms: ms}
      end

    File.rm(json_path)
    result
  end

  @doc """
  Loads the XLM-RoBERTa text detector as an `Nx.Serving`. Options:
  `:sequence_length` (default 64), `:defn_options` (default EXLA host).
  """
  def text_detector(opts \\ []) do
    {:ok, model_info} = Bumblebee.load_model(@text_repo)
    {:ok, tokenizer} = Bumblebee.load_tokenizer(@text_repo)

    defn_options =
      Keyword.get_lazy(opts, :defn_options, fn ->
        {:ok, _started} = Application.ensure_all_started(:exla)
        [compiler: EXLA, client: :host]
      end)

    serving =
      Bumblebee.Text.text_classification(model_info, tokenizer,
        top_k: map_size(model_info.spec.id_to_label),
        compile: [batch_size: 1, sequence_length: Keyword.get(opts, :sequence_length, 64)],
        defn_options: defn_options
      )

    %{serving: serving, languages: model_info.spec.id_to_label |> Map.values() |> Enum.sort()}
  end

  @doc "Ranked `%{language, probability}` for `text`, plus the time taken."
  def detect_text(%{serving: serving}, text) do
    started = System.monotonic_time(:millisecond)
    %{predictions: predictions} = Nx.Serving.run(serving, text)
    ms = System.monotonic_time(:millisecond) - started

    ranked =
      predictions
      |> Enum.map(fn %{label: label, score: score} -> %{language: label, probability: score} end)
      |> Enum.sort_by(& &1.probability, :desc)

    %{ranked: ranked, ms: ms}
  end
end
