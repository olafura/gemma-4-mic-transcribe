defmodule Gemma4MicTranscribe.Gemma4Unified.Runtime do
  @moduledoc false

  alias Gemma4MicTranscribe.Gemma4Unified.Model
  alias Gemma4MicTranscribe.ModelCatalog

  defstruct [
    :model_name,
    :repo_id,
    :backend,
    :max_response_tokens,
    :model_info,
    :tokenizer,
    :generation_config,
    :predict_fun
  ]

  def load(opts \\ []) do
    with :ok <- verify_bumblebee_available(),
         {:ok, backend} <- backend(Keyword.get(opts, :backend, "host")) do
      model_name = Keyword.fetch!(opts, :model_name)
      repo_id = ModelCatalog.resolve(model_name)
      repo = {:hf, repo_id}

      model_opts =
        [
          module: Model,
          architecture: :for_conditional_generation
        ] ++ if(backend, do: [backend: backend], else: [])

      with {:ok, model_info} <- Bumblebee.load_model(repo, model_opts),
           {:ok, tokenizer} <- Bumblebee.load_tokenizer(repo, type: :gemma),
           {:ok, generation_config} <- Bumblebee.load_generation_config(repo, spec_module: Model) do
        {_init_fun, predict_fun} = Axon.build(model_info.model)
        model_info = %{model_info | params: Axon.ModelState.new(model_info.params)}

        generation_config =
          Bumblebee.configure(generation_config,
            max_new_tokens: Keyword.get(opts, :max_response_tokens, 512),
            strategy: %{type: :greedy_search}
          )

        {:ok,
         %__MODULE__{
           model_name: model_name,
           repo_id: repo_id,
           backend: backend,
           max_response_tokens: generation_config.max_new_tokens,
           model_info: model_info,
           tokenizer: tokenizer,
           generation_config: generation_config,
           predict_fun: predict_fun
         }}
      end
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  def generate(%__MODULE__{} = runtime, input, _opts \\ []) do
    with {:ok, input_ids} <- tokenize(runtime.tokenizer, input.prompt),
         :ok <- validate_audio_placeholders(runtime.tokenizer, input_ids, input.audio.token_count) do
      token_ids =
        greedy_generate(
          runtime,
          input_ids,
          input.audio.input_features,
          input.audio.attention_mask,
          []
        )

      {:ok, Bumblebee.Tokenizer.decode(runtime.tokenizer, token_ids)}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  defp greedy_generate(runtime, input_ids, input_features, input_features_mask, generated) do
    cond do
      length(generated) >= runtime.max_response_tokens ->
        generated

      true ->
        logits = predict_next_logits(runtime, input_ids, input_features, input_features_mask)
        token_id = next_token_id(logits, runtime.generation_config.suppressed_token_ids)

        if eos?(token_id, runtime.generation_config.eos_token_id) do
          generated
        else
          greedy_generate(
            runtime,
            input_ids ++ [token_id],
            input_features,
            input_features_mask,
            generated ++ [token_id]
          )
        end
    end
  end

  defp predict_next_logits(runtime, input_ids, input_features, input_features_mask) do
    sequence_length = length(input_ids)
    backend = runtime.backend || Nx.BinaryBackend

    inputs =
      Nx.with_default_backend(backend, fn ->
        %{
          "input_ids" => Nx.tensor([input_ids], type: :s64),
          "attention_mask" => Nx.tensor([List.duplicate(1, sequence_length)], type: :s64),
          "position_ids" => Nx.tensor([Enum.to_list(0..(sequence_length - 1))], type: :s64),
          "input_features" => input_features |> Nx.backend_transfer(backend) |> Nx.new_axis(0),
          "input_features_mask" =>
            input_features_mask |> Nx.backend_transfer(backend) |> Nx.new_axis(0)
        }
      end)

    %{logits: logits} = runtime.predict_fun.(runtime.model_info.params, inputs)

    logits[0][sequence_length - 1]
  end

  defp next_token_id(logits, suppressed_token_ids) do
    logits =
      Enum.reduce(List.wrap(suppressed_token_ids), logits, fn token_id, logits ->
        replacement = Nx.Constants.min_finite(Nx.type(logits)) |> Nx.broadcast({1})
        Nx.put_slice(logits, [token_id], replacement)
      end)

    logits
    |> Nx.argmax(axis: -1)
    |> Nx.to_number()
  end

  defp tokenize(tokenizer, prompt) do
    inputs =
      Nx.with_default_backend(Nx.BinaryBackend, fn ->
        Bumblebee.apply_tokenizer(tokenizer, [prompt])
      end)

    {:ok, Nx.to_flat_list(inputs["input_ids"])}
  end

  defp validate_audio_placeholders(tokenizer, input_ids, expected_count) do
    audio_token_id = Bumblebee.Tokenizer.token_to_id(tokenizer, "<|audio|>")

    actual_count = Enum.count(input_ids, &(&1 == audio_token_id))

    if actual_count == expected_count do
      :ok
    else
      {:error,
       "prompt has #{actual_count} audio placeholder tokens, but audio features contain #{expected_count} tokens"}
    end
  end

  defp eos?(token_id, eos_token_id), do: token_id in List.wrap(eos_token_id)

  defp backend("host"), do: {:ok, Nx.BinaryBackend}
  defp backend(nil), do: {:ok, Nx.BinaryBackend}

  defp backend("exla") do
    if Code.ensure_loaded?(EXLA.Backend),
      do: {:ok, EXLA.Backend},
      else: {:error, "EXLA backend requested, but EXLA is not installed"}
  end

  defp backend("torchx") do
    if Code.ensure_loaded?(Torchx.Backend),
      do: {:ok, Torchx.Backend},
      else: {:error, "Torchx backend requested, but Torchx is not installed"}
  end

  defp backend(other), do: {:error, "unsupported backend #{inspect(other)}"}

  defp verify_bumblebee_available do
    if Code.ensure_loaded?(Bumblebee) do
      :ok
    else
      {:error, {:missing_dependency, :bumblebee}}
    end
  end
end
