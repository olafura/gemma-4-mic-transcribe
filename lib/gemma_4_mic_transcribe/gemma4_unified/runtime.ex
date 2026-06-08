defmodule Gemma4MicTranscribe.Gemma4Unified.Runtime do
  @moduledoc false

  require Logger

  alias Gemma4MicTranscribe.Config
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
    :predict_fun,
    :debug
  ]

  def load(opts \\ []) do
    debug? = Keyword.get(opts, :debug, false)

    with :ok <- verify_bumblebee_available(),
         {:ok, backend} <- backend(Keyword.get(opts, :backend, Config.backend())) do
      model_name = Keyword.fetch!(opts, :model_name)
      repo_id = ModelCatalog.resolve(model_name)
      repo = {:hf, repo_id}

      model_opts =
        [
          module: Model,
          architecture: :for_conditional_generation
        ] ++ if(backend, do: [backend: backend], else: [])

      log_debug(debug?, fn ->
        "runtime: resolved model #{inspect(model_name)} -> #{repo_id}; backend=#{inspect(backend)}"
      end)

      log_backend_details(debug?, backend)

      with {:ok, model_info} <-
             timed_debug(
               debug?,
               "runtime: Bumblebee.load_model #{repo_id} (checkpoint download/load)",
               fn ->
                 Bumblebee.load_model(repo, model_opts)
               end
             ),
           {:ok, tokenizer} <-
             timed_debug(debug?, "runtime: Bumblebee.load_tokenizer #{repo_id}", fn ->
               Bumblebee.load_tokenizer(repo, type: :gemma)
             end),
           {:ok, generation_config} <-
             timed_debug(debug?, "runtime: Bumblebee.load_generation_config #{repo_id}", fn ->
               Bumblebee.load_generation_config(repo, spec_module: Model)
             end) do
        {_init_fun, predict_fun} =
          timed_debug(debug?, "runtime: Axon.build predict function", fn ->
            Axon.build(model_info.model)
          end)

        model_info =
          timed_debug(debug?, "runtime: wrap model params", fn ->
            %{model_info | params: Axon.ModelState.new(model_info.params)}
          end)

        generation_config =
          Bumblebee.configure(generation_config,
            max_new_tokens: Keyword.get(opts, :max_response_tokens, 512),
            strategy: %{type: :greedy_search}
          )

        log_debug(debug?, fn ->
          "runtime: loaded spec hidden_size=#{model_info.spec.hidden_size} layers=#{model_info.spec.num_blocks} " <>
            "audio_token_id=#{model_info.spec.audio_token_id} max_new_tokens=#{generation_config.max_new_tokens}"
        end)

        {:ok,
         %__MODULE__{
           model_name: model_name,
           repo_id: repo_id,
           backend: backend,
           max_response_tokens: generation_config.max_new_tokens,
           model_info: model_info,
           tokenizer: tokenizer,
           generation_config: generation_config,
           predict_fun: predict_fun,
           debug: debug?
         }}
      end
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  def generate(%__MODULE__{} = runtime, input, _opts \\ []) do
    log_debug(runtime, fn ->
      "runtime: generate start prompt_bytes=#{byte_size(input.prompt)} audio_tokens=#{input.audio.token_count}"
    end)

    with {:ok, input_ids} <-
           timed_debug(runtime, "runtime: tokenize prompt", fn ->
             tokenize(runtime.tokenizer, input.prompt)
           end),
         :ok <- validate_audio_placeholders(runtime.tokenizer, input_ids, input.audio.token_count) do
      log_debug(runtime, fn ->
        "runtime: prompt tokenized input_tokens=#{length(input_ids)} audio_tokens=#{input.audio.token_count}"
      end)

      {input_features, input_features_mask} =
        timed_debug(runtime, "runtime: prepare audio tensors", fn ->
          prepare_audio_tensors(runtime, input.audio.input_features, input.audio.attention_mask)
        end)

      token_ids =
        greedy_generate(
          runtime,
          input_ids,
          input_features,
          input_features_mask,
          []
        )

      log_debug(runtime, fn ->
        "runtime: generation finished generated_tokens=#{length(token_ids)}"
      end)

      {:ok, Bumblebee.Tokenizer.decode(runtime.tokenizer, token_ids)}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  end

  defp greedy_generate(runtime, input_ids, input_features, input_features_mask, generated) do
    cond do
      length(generated) >= runtime.max_response_tokens ->
        log_debug(runtime, fn ->
          "runtime: generation reached max_response_tokens=#{runtime.max_response_tokens}"
        end)

        generated

      true ->
        step = length(generated) + 1

        log_debug(runtime, fn ->
          "runtime: generation step #{step}/#{runtime.max_response_tokens} start context_tokens=#{length(input_ids)}"
        end)

        started_at = System.monotonic_time(:millisecond)
        logits = predict_next_logits(runtime, input_ids, input_features, input_features_mask)
        token_id = next_token_id(logits, runtime.generation_config.suppressed_token_ids)
        eos? = eos?(token_id, runtime.generation_config.eos_token_id)
        elapsed_ms = System.monotonic_time(:millisecond) - started_at

        log_debug(runtime, fn ->
          "runtime: generation step #{step}/#{runtime.max_response_tokens} token_id=#{token_id} " <>
            "eos=#{eos?} elapsed_ms=#{elapsed_ms}"
        end)

        if eos? do
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
    backend = runtime_backend(runtime)

    inputs =
      Nx.with_default_backend(backend, fn ->
        %{
          "input_ids" => Nx.tensor([input_ids], type: :s64),
          "attention_mask" => Nx.tensor([List.duplicate(1, sequence_length)], type: :s64),
          "position_ids" => Nx.tensor([Enum.to_list(0..(sequence_length - 1))], type: :s64),
          "input_features" => Nx.new_axis(input_features, 0),
          "input_features_mask" => Nx.new_axis(input_features_mask, 0)
        }
      end)

    %{logits: logits} = runtime.predict_fun.(runtime.model_info.params, inputs)

    logits[0][sequence_length - 1]
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

  defp exla_backend(client) do
    with :ok <- ensure_exla_available(),
         {:ok, backend} <- exla_backend_for_client(client) do
      {:ok, backend}
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
