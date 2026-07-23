defmodule Gemma4MicTranscribe.Gemma4.MathExpertProfiler do
  @moduledoc """
  Profiles layer-0 router affinity using real token embeddings.

  This is an inexpensive specialization probe, not a substitute for capturing
  post-attention residual states from the complete model. It compares math and
  control token corpora and ranks experts by selection and probability lift.
  """

  alias Gemma4MicTranscribe.Gemma4.ExtractedMoeLayer
  alias Gemma4MicTranscribe.Gemma4.MoeLayerArtifact

  @embedding_tensor "model.language_model.embed_tokens.weight"
  @bf16_bytes 2
  @default_backend Application.compile_env(:nx, :default_backend, Nx.BinaryBackend)

  @math_texts [
    "algebra equation variable coefficient polynomial quadratic arithmetic fraction integer prime factor",
    "geometry triangle circle radius angle Euclidean theorem proof lemma corollary",
    "calculus derivative integral differential gradient limit continuity series",
    "linear algebra matrix vector tensor eigenvalue determinant transpose orthogonal",
    "probability statistics distribution expectation variance covariance stochastic Bayesian",
    "combinatorics permutation combination graph topology discrete recurrence",
    "sine cosine tangent logarithm exponential Fourier Laplace transform",
    "solve x squared plus y squared equals compute simplify derive prove calculate"
  ]

  @control_texts [
    "garden flower soil seed tree forest mountain river ocean weather",
    "recipe kitchen bread coffee dinner ingredient vegetable fruit cooking",
    "music guitar piano melody rhythm concert singer album orchestra",
    "painting sculpture cinema theater poetry novel story literature",
    "bicycle automobile railway airplane travel hotel village city",
    "friendship family clothing furniture house window garden holiday",
    "history politics language law culture education journalism",
    "biology medicine animal plant cell anatomy health hospital"
  ]

  @doc "Runs the default math-vs-control embedding router profile."
  def profile!(artifact_path, opts \\ []) do
    backend = Keyword.get(opts, :backend, @default_backend)
    manifest = MoeLayerArtifact.read_manifest!(artifact_path)

    unless manifest.layer_index == 0 do
      raise ArgumentError,
            "embedding proxy is only valid for layer 0, got layer #{manifest.layer_index}"
    end

    tokenizer =
      Keyword.get_lazy(opts, :tokenizer, fn ->
        {:ok, tokenizer} =
          Bumblebee.load_tokenizer({:hf, manifest.source_repo}, type: :gemma)

        Bumblebee.configure(tokenizer, add_special_tokens: false)
      end)

    math_tokens =
      tokenize(
        tokenizer,
        Keyword.get(opts, :math_texts, @math_texts)
      )

    control_tokens =
      tokenize(
        tokenizer,
        Keyword.get(opts, :control_texts, @control_texts)
      )

    all_tokens = math_tokens ++ control_tokens
    embeddings = fetch_embedding_rows!(manifest, all_tokens, opts)

    input =
      all_tokens
      |> Enum.map(fn token -> Map.fetch!(embeddings, token.id) end)
      |> Nx.stack()
      |> transfer(backend)

    {_, params} = MoeLayerArtifact.load_router!(artifact_path, backend)
    top_k = manifest.top_k_experts
    eps = manifest.rms_norm_eps
    router_scalar = manifest.hidden_size ** -0.5

    route_fun =
      Nx.Defn.jit(
        fn input, params ->
          ExtractedMoeLayer.route(input, params,
            top_k: top_k,
            eps: eps,
            router_scalar: router_scalar
          )
        end,
        build_opts(backend)
      )

    routing =
      route_fun.(input, params)
      |> Nx.backend_copy(Nx.BinaryBackend)

    math_count = length(math_tokens)

    math_routing = slice_routing(routing, 0, math_count)
    control_routing = slice_routing(routing, math_count, length(control_tokens))

    experts =
      rank(
        math_routing,
        control_routing,
        manifest.num_experts
      )
      |> Enum.map(fn expert ->
        Map.put(
          expert,
          :top_math_tokens,
          top_tokens(math_routing.router_probabilities, math_tokens, expert.expert, tokenizer)
        )
      end)

    %{
      method: "layer_0_token_embedding_router_proxy",
      caveat:
        "Token embeddings are used in place of real post-attention residuals; validate candidates with full-model activation capture.",
      source_repo: manifest.source_repo,
      layer: manifest.layer_index,
      math_tokens: length(math_tokens),
      control_tokens: length(control_tokens),
      unique_embedding_rows: all_tokens |> Enum.map(& &1.id) |> Enum.uniq() |> length(),
      experts: experts
    }
  end

  @doc "Loads real checkpoint embedding rows for one or more text strings."
  def embedding_inputs!(artifact_path, texts, opts \\ []) when is_list(texts) do
    manifest = MoeLayerArtifact.read_manifest!(artifact_path)

    tokenizer =
      Keyword.get_lazy(opts, :tokenizer, fn ->
        {:ok, tokenizer} =
          Bumblebee.load_tokenizer({:hf, manifest.source_repo}, type: :gemma)

        Bumblebee.configure(tokenizer, add_special_tokens: false)
      end)

    texts =
      if Keyword.get(opts, :prepend_bos, false) do
        Enum.map(texts, &("<bos>" <> &1))
      else
        texts
      end

    tokens = tokenize(tokenizer, texts)
    embeddings = fetch_embedding_rows!(manifest, tokens, opts)

    %{
      manifest: manifest,
      tokenizer: tokenizer,
      tokens:
        Enum.map(tokens, fn token ->
          Map.put(
            token,
            :token,
            Bumblebee.Tokenizer.id_to_token(tokenizer, token.id) ||
              Bumblebee.Tokenizer.decode(tokenizer, [token.id])
          )
        end),
      input: tokens |> Enum.map(&Map.fetch!(embeddings, &1.id)) |> Nx.stack()
    }
  end

  @doc "Ranks experts from already captured math and control routing outputs."
  def rank(math_routing, control_routing, num_experts) do
    math_rows = Nx.axis_size(math_routing.router_probabilities, 0)
    control_rows = Nx.axis_size(control_routing.router_probabilities, 0)
    math_counts = selection_counts(math_routing.top_k_indices, num_experts)
    control_counts = selection_counts(control_routing.top_k_indices, num_experts)
    math_probabilities = mean_columns(math_routing.router_probabilities)
    control_probabilities = mean_columns(control_routing.router_probabilities)

    0..(num_experts - 1)
    |> Enum.map(fn expert ->
      math_count = Enum.at(math_counts, expert)
      control_count = Enum.at(control_counts, expert)
      math_rate = math_count / math_rows
      control_rate = control_count / control_rows
      math_probability = Enum.at(math_probabilities, expert)
      control_probability = Enum.at(control_probabilities, expert)
      selection_lift = math_rate - control_rate
      probability_lift = math_probability - control_probability
      pooled_rate = (math_count + control_count) / (math_rows + control_rows)

      standard_error =
        :math.sqrt(pooled_rate * (1.0 - pooled_rate) * (1.0 / math_rows + 1.0 / control_rows))

      selection_z_score =
        if standard_error == 0.0, do: 0.0, else: selection_lift / standard_error

      %{
        expert: expert,
        score: selection_z_score,
        math_selections: math_count,
        control_selections: control_count,
        math_selection_rate: math_rate,
        control_selection_rate: control_rate,
        selection_lift: selection_lift,
        selection_z_score: selection_z_score,
        selection_ratio: (math_rate + 1.0 / math_rows) / (control_rate + 1.0 / control_rows),
        math_mean_probability: math_probability,
        control_mean_probability: control_probability,
        probability_lift: probability_lift
      }
    end)
    |> Enum.sort_by(& &1.score, :desc)
  end

  defp tokenize(tokenizer, texts) do
    Enum.flat_map(texts, fn text ->
      encoded = Bumblebee.apply_tokenizer(tokenizer, [text])
      ids = Nx.to_flat_list(encoded["input_ids"])
      mask = Nx.to_flat_list(encoded["attention_mask"])

      ids
      |> Enum.zip(mask)
      |> Enum.filter(fn {_id, present?} -> present? == 1 end)
      |> Enum.map(fn {id, _present?} -> %{id: id} end)
    end)
  end

  defp fetch_embedding_rows!(manifest, tokens, opts) do
    fetch_range = Keyword.get(opts, :fetch_range, &range_get!/3)
    {header_length, header} = fetch_header!(manifest, fetch_range)
    metadata = Map.fetch!(header, @embedding_tensor)

    unless metadata["dtype"] == "BF16" and
             metadata["shape"] == [262_144, manifest.hidden_size] do
      raise ArgumentError, "unexpected Gemma 4 embedding tensor metadata"
    end

    [data_start, _data_end] = metadata["data_offsets"]
    data_base = 8 + header_length
    row_bytes = manifest.hidden_size * @bf16_bytes
    shard_url = shard_url(manifest)

    tokens
    |> Enum.map(& &1.id)
    |> Enum.uniq()
    |> Task.async_stream(
      fn token_id ->
        first = data_base + data_start + token_id * row_bytes
        last = first + row_bytes - 1
        binary = fetch_range.(shard_url, first, last)

        unless byte_size(binary) == row_bytes do
          raise ArgumentError,
                "embedding row #{token_id} returned #{byte_size(binary)} bytes, expected #{row_bytes}"
        end

        tensor =
          binary
          |> Nx.from_binary(:bf16, backend: Nx.BinaryBackend)
          |> Nx.reshape({manifest.hidden_size})

        {token_id, tensor}
      end,
      ordered: false,
      max_concurrency: Keyword.get(opts, :max_concurrency, 8),
      timeout: Keyword.get(opts, :fetch_timeout, 120_000)
    )
    |> Map.new(fn
      {:ok, row} -> row
      {:exit, reason} -> raise "embedding row download failed: #{inspect(reason)}"
    end)
  end

  defp fetch_header!(manifest, fetch_range) do
    url = shard_url(manifest)
    <<header_length::little-unsigned-64>> = fetch_exact!(url, 0, 7, fetch_range)
    header = fetch_exact!(url, 8, 7 + header_length, fetch_range) |> Jason.decode!()
    {header_length, header}
  end

  defp fetch_exact!(url, first, last, fetch_range) do
    body = fetch_range.(url, first, last)
    expected = last - first + 1

    unless byte_size(body) == expected do
      raise ArgumentError,
            "range #{first}-#{last} returned #{byte_size(body)} bytes, expected #{expected}"
    end

    body
  end

  defp range_get!(url, first, last) do
    response =
      Req.get!(url,
        headers: [{"range", "bytes=#{first}-#{last}"}],
        redirect: true,
        redirect_log_level: false,
        max_redirects: 5,
        receive_timeout: 120_000,
        retry: :transient,
        max_retries: 3
      )

    unless response.status == 206 do
      raise ArgumentError, "#{url} returned HTTP #{response.status}, expected 206"
    end

    response.body
  end

  defp shard_url(manifest) do
    "https://huggingface.co/#{manifest.source_repo}/resolve/#{manifest.source_revision}/#{manifest.source_shard}"
  end

  defp slice_routing(routing, start, length) do
    %{
      router_probabilities:
        Nx.slice_along_axis(routing.router_probabilities, start, length, axis: 0),
      top_k_indices: Nx.slice_along_axis(routing.top_k_indices, start, length, axis: 0),
      top_k_weights: Nx.slice_along_axis(routing.top_k_weights, start, length, axis: 0)
    }
  end

  defp selection_counts(indices, num_experts) do
    counts =
      indices
      |> Nx.flatten()
      |> Nx.to_flat_list()
      |> Enum.frequencies()

    Enum.map(0..(num_experts - 1), &Map.get(counts, &1, 0))
  end

  defp mean_columns(probabilities) do
    probabilities
    |> Nx.mean(axes: [0])
    |> Nx.to_flat_list()
  end

  defp top_tokens(probabilities, tokens, expert, tokenizer) do
    probabilities
    |> Nx.slice_along_axis(expert, 1, axis: 1)
    |> Nx.flatten()
    |> Nx.to_flat_list()
    |> Enum.zip(tokens)
    |> Enum.sort_by(fn {probability, _token} -> probability end, :desc)
    |> Enum.uniq_by(fn {_probability, token} -> token.id end)
    |> Enum.take(8)
    |> Enum.map(fn {probability, token} ->
      %{
        id: token.id,
        token:
          Bumblebee.Tokenizer.id_to_token(tokenizer, token.id) ||
            Bumblebee.Tokenizer.decode(tokenizer, [token.id]),
        probability: probability
      }
    end)
  end

  defp transfer(tensor, nil), do: tensor
  defp transfer(tensor, Nx.BinaryBackend), do: Nx.backend_copy(tensor, Nx.BinaryBackend)
  defp transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp build_opts(EXLA.Backend), do: [compiler: EXLA]

  defp build_opts({EXLA.Backend, backend_opts}),
    do: [compiler: EXLA] ++ Keyword.take(backend_opts, [:client, :device_id])

  defp build_opts(_backend), do: []
end
