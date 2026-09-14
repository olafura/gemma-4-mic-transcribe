defmodule Gemma4MicTranscribe.LanguageIdTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.LanguageId.Artifact
  alias Gemma4MicTranscribe.LanguageId.Corpus
  alias Gemma4MicTranscribe.LanguageId.Finetune
  alias Gemma4MicTranscribe.LanguageId.Encoder
  alias Gemma4MicTranscribe.LanguageId.Head
  alias Gemma4MicTranscribe.LanguageId.Runtime
  alias Gemma4MicTranscribe.LanguageIdCLI

  defp tiny_encoder(opts) do
    spec =
      struct(Encoder, %{
        hidden_size: 16,
        audio_hidden_size: 8,
        audio_num_blocks: 2,
        audio_num_attention_heads: 2,
        audio_conv_kernel_size: 3,
        audio_mel_bins: 8,
        audio_subsampling_conv_channels: [4, 2],
        audio_attention_chunk_size: 2,
        audio_attention_context_left: 2,
        audio_attention_context_right: 0
      })

    Bumblebee.configure(spec, opts)
  end

  defp init(spec, batch, frames) do
    model = Encoder.model(spec)
    {init_fun, predict_fun} = Axon.build(model)
    tokens = Gemma4MicTranscribe.Gemma4E4B.Spec.audio_subsampled_length(Encoder.to_spec(spec), frames)

    inputs = %{
      "input_features" => Nx.broadcast(0.1, {batch, frames, spec.audio_mel_bins}),
      "frame_mask" => Nx.broadcast(1, {batch, tokens})
    }

    {predict_fun, init_fun.(inputs, Axon.ModelState.empty()), tokens}
  end

  test "depth limits the blocks that are built and named" do
    {_predict, params, _tokens} = init(tiny_encoder(depth: 1), 1, 16)

    names = Map.keys(params.data)
    assert Enum.any?(names, &String.starts_with?(&1, "audio_encoder.subsample"))
    assert Enum.any?(names, &String.starts_with?(&1, "audio_encoder.blocks.0."))
    refute Enum.any?(names, &String.starts_with?(&1, "audio_encoder.blocks.1."))
  end

  test "config rejects depths beyond the tower" do
    assert_raise ArgumentError, ~r/depth must be between 0 and 2/, fn ->
      tiny_encoder(depth: 3)
    end
  end

  test "pooled output ignores masked frames and captures every depth" do
    spec = tiny_encoder(capture_all_depths: true)
    {predict, params, tokens} = init(spec, 2, 16)

    key = Nx.Random.key(3)
    {features, key} = Nx.Random.normal(key, shape: {1, 16, 8})
    {noise, _key} = Nx.Random.normal(key, shape: {1, 16, 8})

    # second row shares the first half of the audio and differs after it
    half = 8
    tail = Nx.slice_along_axis(features, half, 16 - half, axis: 1)
    other = Nx.concatenate([Nx.slice_along_axis(features, 0, half, axis: 1), tail |> Nx.add(noise |> Nx.slice_along_axis(half, 16 - half, axis: 1))], axis: 1)
    batch = Nx.concatenate([features, other], axis: 0)

    real = div(tokens, 2)
    mask = Nx.less(Nx.iota({2, tokens}, axis: 1), real) |> Nx.as_type(:s64)

    outputs = predict.(params, %{"input_features" => batch, "frame_mask" => mask})

    assert Map.keys(outputs) |> Enum.sort() == ["depth_0", "depth_1", "depth_2"]
    assert Nx.shape(outputs["depth_2"]) == {2, 8}
    assert Nx.type(outputs["depth_2"]) == {:f, 32}

    # frames past the mask differ between the rows, but the causal tower with
    # bounded left context keeps the kept frames identical, so pooling agrees
    for key <- Map.keys(outputs) do
      rows = Nx.to_batched(outputs[key], 1) |> Enum.map(&Nx.squeeze(&1, axes: [0]))
      assert_all_close(Enum.at(rows, 0), Enum.at(rows, 1))
    end
  end

  test "mean_std pooling doubles the feature width" do
    spec = tiny_encoder(depth: 1, pooling: :mean_std)
    {predict, params, tokens} = init(spec, 1, 16)

    inputs = %{
      "input_features" => Nx.iota({1, 16, 8}, type: :f32) |> Nx.multiply(0.01),
      "frame_mask" => Nx.broadcast(1, {1, tokens})
    }

    assert %{"depth_1" => pooled} = predict.(params, inputs)
    assert Nx.shape(pooled) == {1, 16}
    assert pooled |> Nx.slice_along_axis(8, 8, axis: 1) |> Nx.greater(0) |> Nx.all() |> Nx.to_number() == 1
  end

  test "head separates linearly separable clusters and evaluates per language" do
    key = Nx.Random.key(7)
    {noise, _key} = Nx.Random.normal(key, shape: {60, 4}, type: :f32)
    centers = Nx.tensor([[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]])
    labels = Nx.tensor(List.duplicate(0, 20) ++ List.duplicate(1, 20) ++ List.duplicate(2, 20))
    x = Nx.add(noise, Nx.take(centers, labels))

    head = Head.train(x, labels, ["aa", "bb", "cc"], steps: 200)
    evaluation = Head.evaluate(head, x, labels)

    assert evaluation.accuracy > 0.95
    assert Map.keys(evaluation.per_language) |> Enum.sort() == ["aa", "bb", "cc"]
    assert evaluation.per_language["aa"].samples == 20
    assert Head.parameter_count(head) == 4 + 4 + 4 * 3 + 3

    log_probs = Head.log_probs(head, x)
    assert_all_close(log_probs |> Nx.exp() |> Nx.sum(axes: [1]), Nx.broadcast(1.0, {60}))

    round_trip = head |> Head.to_tensors() |> Head.from_tensors(head.languages)
    assert Head.evaluate(round_trip, x, labels).accuracy == evaluation.accuracy
  end

  test "artifact keeps only the truncated tower and detects after a round trip" do
    spec = tiny_encoder(capture_all_depths: true, pooling: :mean_std)
    {_predict, params, _tokens} = init(spec, 1, 16)

    runtime = Runtime.load(spec: spec, params: params, backend: "torchx:cpu", seconds: 1)
    assert %{parameters: total} = Runtime.size(runtime)
    sizes = Runtime.layer_sizes(runtime)
    assert length(sizes.blocks) == 2
    assert Runtime.parameters_at_depth(sizes, 2) == total

    samples = for i <- 0..3999, do: :math.sin(i / 7)
    prepared = Runtime.prepare(runtime, samples)
    features = runtime |> Runtime.encode([prepared]) |> Map.fetch!("depth_1")
    assert Nx.shape(features) == {1, 16}

    x = Nx.concatenate([features, Nx.add(features, 1.0)])
    head = Head.train(x, Nx.tensor([0, 1]), ["xx", "yy"], steps: 50)

    artifact = Artifact.build(runtime, 1, head, meta: %{note: "test"})
    refute Enum.any?(Map.keys(artifact.params.data), &String.starts_with?(&1, "audio_encoder.blocks.1."))
    assert Artifact.size(artifact).tower.parameters == Runtime.parameters_at_depth(sizes, 1)

    path = Path.join(System.tmp_dir!(), "language-id-#{System.unique_integer([:positive])}")
    on_exit(fn -> File.rm_rf(path) end)

    assert Artifact.save!(artifact, path) == path
    assert_raise ArgumentError, ~r/already exists/, fn -> Artifact.save!(artifact, path) end

    loaded = Artifact.load!(path)
    assert loaded.depth == 1
    assert loaded.languages == ["xx", "yy"]
    assert loaded.seconds == 1
    assert loaded.meta == %{note: "test"}
    assert Map.keys(loaded.params.data) |> Enum.sort() == Map.keys(artifact.params.data) |> Enum.sort()

    loaded_runtime = Artifact.runtime(loaded, backend: "torchx:cpu")
    ranked = Artifact.detect(loaded, loaded_runtime, samples)

    assert [%{language: "xx", probability: first}, %{language: "yy", probability: second}] = ranked
    assert first > second
    assert_in_delta first + second, 1.0, 1.0e-5

    expected = artifact |> Artifact.detect(runtime, samples) |> hd()
    assert_in_delta expected.probability, first, 1.0e-5
  end

  test "cli parses each subcommand and validates its options" do
    assert {:ok, :extract, extract} = LanguageIdCLI.parse(["extract", "--output", "feats", "--split", "test", "--pooling", "mean_std"])
    assert extract.backend == "exla:rocm"
    assert extract.per_language == 200
    assert extract.pooling == "mean_std"

    assert {:error, "extract requires --output"} = LanguageIdCLI.parse(["extract"])
    assert {:error, _} = LanguageIdCLI.parse(["extract", "--output", "feats", "--split", "other"])
    assert {:error, _} = LanguageIdCLI.parse(["extract", "--output", "feats", "--pooling", "max"])

    assert {:ok, :sweep, sweep} = LanguageIdCLI.parse(["sweep", "--train", "a", "--test", "b", "--depths", "1,3"])
    assert sweep.depths == [1, 3]
    assert sweep.weight_decay == 0.01
    assert {:error, _} = LanguageIdCLI.parse(["sweep", "--train", "a", "--test", "b", "--depths", "1,x"])

    assert {:ok, :export, export} = LanguageIdCLI.parse(["export", "--train", "a", "--depth", "5", "--artifact", "out"])
    assert export.depth == 5
    assert {:error, "export requires --depth"} = LanguageIdCLI.parse(["export", "--train", "a", "--artifact", "out"])

    assert {:ok, :detect, detect} = LanguageIdCLI.parse(["detect", "--artifact", "out", "--input", "clip.wav"])
    assert detect.backend == "torchx:cpu"
    assert detect.top_k == 5

    assert {:ok, :compare, compare} =
             LanguageIdCLI.parse(["compare", "--artifact", "out", "--whisper-model", "ggml-base.bin", "--split", "test"])

    assert compare.backend == "torchx:cpu"
    assert {:error, "compare requires --whisper-model or --reference"} =
             LanguageIdCLI.parse(["compare", "--artifact", "out"])

    assert {:ok, :compare, gated} =
             LanguageIdCLI.parse(["compare", "--artifact", "out", "--reference", "base.json"])

    assert gated.reference == "base.json"

    assert {:ok, :inputs, inputs} = LanguageIdCLI.parse(["inputs", "--output", "mel", "--seconds", "1"])
    assert inputs.seconds == 1
    assert {:error, "inputs requires --output"} = LanguageIdCLI.parse(["inputs"])

    assert {:ok, :finetune, finetune} =
             LanguageIdCLI.parse([
               "finetune",
               "--inputs-train",
               "mel",
               "--train",
               "feats",
               "--depth",
               "3",
               "--artifact",
               "out",
               "--freeze",
               "audio_encoder.subsample,audio_encoder.blocks.0"
             ])

    assert finetune.learning_rate == 2.0e-5
    assert finetune.epochs == 3
    assert finetune.freeze == ["audio_encoder.subsample", "audio_encoder.blocks.0"]
    assert {:error, "finetune requires --depth"} = LanguageIdCLI.parse(["finetune", "--inputs-train", "mel", "--train", "f", "--artifact", "o"])

    assert {:help, usage} = LanguageIdCLI.parse([])
    assert usage =~ "language_id detect"
    assert {:error, _} = LanguageIdCLI.parse(["unknown"])
  end

  test "onset trimming drops leading silence but keeps a short lead-in" do
    silence = fn ms -> for _ <- 1..(16 * ms), into: <<>>, do: <<0.0::little-float-32>> end
    tone = fn ms -> for i <- 1..(16 * ms), into: <<>>, do: <<0.5 * :math.sin(i / 3)::little-float-32>> end
    click = fn -> for _ <- 1..16, into: <<>>, do: <<0.9::little-float-32>> end

    audio = silence.(800) <> tone.(300) <> silence.(200)
    trimmed = Corpus.trim_onset(audio, lead: 0.1)
    # 100 ms of silence kept ahead of the 800 ms onset
    assert byte_size(trimmed) == byte_size(audio) - 700 * 16 * 4

    with_click = click.() <> silence.(799) <> tone.(300)
    assert byte_size(Corpus.trim_onset(with_click, lead: 0.1)) == byte_size(with_click) - 700 * 16 * 4

    assert Corpus.trim_onset(silence.(500)) == silence.(500)
    assert Corpus.trim_onset(tone.(50)) == tone.(50)
  end

  test "dense head parameters reproduce the standardized logistic head" do
    key = Nx.Random.key(3)
    {x, key} = Nx.Random.normal(key, shape: {40, 6}, type: :f32)
    {labels, _key} = Nx.Random.randint(key, 0, 3, shape: {40})
    head = Head.train(Nx.multiply(x, 3.0), labels, ["aa", "bb", "cc"], steps: 50)

    dense = Finetune.head_parameters(head)
    folded = x |> Nx.multiply(3.0) |> Nx.dot(dense["kernel"]) |> Nx.add(dense["bias"])
    folded_log_probs = Nx.subtract(folded, Nx.logsumexp(folded, axes: [1], keep_axes: true))
    assert_all_close(folded_log_probs, Head.log_probs(head, Nx.multiply(x, 3.0)))

    back = Finetune.head_from_parameters(dense, head.languages)
    assert_all_close(Head.log_probs(back, Nx.multiply(x, 3.0)), Head.log_probs(head, Nx.multiply(x, 3.0)))
  end

  test "fine-tuning model trains end to end on a tiny tower and exports an artifact" do
    spec = tiny_encoder(depth: 1, pooling: :mean_std)
    {_predict, params, tokens} = init(spec, 2, 16)
    frames = 16

    runtime = %Runtime{spec: spec, params: params, seconds: 1, frames: frames, tokens: tokens}
    languages = ["aa", "bb"]

    inputs = %Finetune.Inputs{
      features: Nx.concatenate([Nx.broadcast(0.5, {4, frames, 8}), Nx.broadcast(-0.5, {4, frames, 8})]),
      masks: Nx.broadcast(1, {8, tokens}),
      labels: Nx.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
      languages: languages,
      seconds: 1,
      keys: Enum.map(1..8, &"clip#{&1}")
    }

    width = 2 * spec.audio_hidden_size

    head = %Head{
      mean: Nx.broadcast(0.0, {width}),
      std: Nx.broadcast(1.0, {width}),
      kernel: Nx.broadcast(0.0, {width, 2}),
      bias: Nx.tensor([0.0, 0.0]),
      languages: languages
    }

    log = fn progress -> send(self(), {:progress, progress}) end

    result =
      Finetune.train(inputs, 1,
        runtime: runtime,
        head: head,
        test: inputs,
        epochs: 2,
        batch_size: 4,
        learning_rate: 1.0e-2,
        compiler_opts: [],
        log: log
      )

    assert_received {:progress, %{epoch: 0, accuracy: initial}}
    assert_received {:progress, %{epoch: 1, loss: loss}}
    assert is_float(loss) and loss > 0
    assert result.evaluation.accuracy >= initial
    assert Map.keys(result.model_state.data) |> Enum.sort() |> List.last() == "head"

    artifact = Finetune.to_artifact(runtime, result.model_state, 1, languages, 1)
    assert artifact.depth == 1
    assert artifact.languages == languages
    {block_layer, tensors} = Enum.find(artifact.params.data, fn {name, _} -> String.starts_with?(name, "audio_encoder.blocks.0") end)
    assert is_binary(block_layer)
    assert Enum.all?(tensors, fn {_name, tensor} -> Nx.type(tensor) == {:bf, 16} end)
    refute Map.has_key?(artifact.params.data, "head")
  end

  defp assert_all_close(left, right) do
    assert Nx.all_close(left, right, atol: 1.0e-4, rtol: 1.0e-4) |> Nx.to_number() == 1,
           "tensors differ: #{inspect(left)} vs #{inspect(right)}"
  end

  test "non-finite gradients apply no update and are counted" do
    {init, update} = Finetune.guard_non_finite(Polaris.Optimizers.sgd(learning_rate: 0.5))
    params = %{"layer" => %{"kernel" => Nx.tensor([1.0, 2.0])}}
    state = init.(params)

    good = %{"layer" => %{"kernel" => Nx.tensor([1.0, 1.0])}}
    {updates, state} = update.(good, state, params)
    assert Nx.to_flat_list(updates["layer"]["kernel"]) == [-0.5, -0.5]
    assert Nx.to_number(state.skipped) == 0

    bad = %{"layer" => %{"kernel" => Nx.tensor([:nan, 1.0])}}
    {updates, state} = update.(bad, state, params)
    assert Nx.to_flat_list(updates["layer"]["kernel"]) == [0.0, 0.0]
    assert Nx.to_number(state.skipped) == 1
  end
end
