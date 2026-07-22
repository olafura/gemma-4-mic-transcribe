defmodule Gemma4MicTranscribe.Gemma4UnifiedTest do
  use ExUnit.Case, async: true

  import Bitwise, only: [&&&: 2, <<<: 2, |||: 2]

  alias Gemma4MicTranscribe.Gemma4Unified.AudioFeatureExtractor
  alias Gemma4MicTranscribe.Gemma4Unified.ChannelState
  alias Gemma4MicTranscribe.Gemma4Unified.CompressedTensors
  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Model
  alias Gemma4MicTranscribe.Gemma4Unified.Q4DualGemv
  alias Gemma4MicTranscribe.Gemma4Unified.TokenSelection
  alias Gemma4MicTranscribe.Gemma4Unified.Transcript
  alias Gemma4MicTranscribe.Gemma4Unified.Runtime
  alias Gemma4MicTranscribe.ModelCatalog
  alias Gemma4MicTranscribe.RocmPreflight
  alias Gemma4MicTranscribe.Gemma4Unified.Prompt

  test "audio feature extractor chunks raw 16 kHz audio into 640-sample soft tokens" do
    features = AudioFeatureExtractor.extract(List.duplicate(0.25, 641))

    assert features.token_count == 2
    assert Nx.shape(features.input_features) == {2, 640}
    assert Nx.to_flat_list(features.attention_mask) == [1, 1]
    assert features.input_features[1][1] |> Nx.to_number() == 0.0
  end

  test "audio feature extractor matches LiteRT-LM skip-mel raw PCM framing" do
    samples = Enum.map(1..1281, &(&1 / 1000))
    features = AudioFeatureExtractor.extract(samples)

    assert features.samples_per_token == 640
    assert features.token_count == 3
    assert Nx.shape(features.input_features) == {3, 640}

    assert_in_delta features.input_features[0][0] |> Nx.to_number(), 0.001, 1.0e-6
    assert_in_delta features.input_features[0][639] |> Nx.to_number(), 0.64, 1.0e-6
    assert_in_delta features.input_features[1][0] |> Nx.to_number(), 0.641, 1.0e-6
    assert_in_delta features.input_features[1][639] |> Nx.to_number(), 1.28, 1.0e-6
    assert_in_delta features.input_features[2][0] |> Nx.to_number(), 1.281, 1.0e-6
    assert features.input_features[2][1] |> Nx.to_number() == 0.0
  end

  test "audio feature extractor truncates at max token count" do
    features = AudioFeatureExtractor.extract(List.duplicate(0.0, 2_000), max_tokens: 2)

    assert features.token_count == 2
    assert Nx.shape(features.input_features) == {2, 640}
  end

  test "audio feature extractor pads to a requested token bucket" do
    features = AudioFeatureExtractor.extract(List.duplicate(0.5, 641), audio_token_count: 4)

    assert features.token_count == 4
    assert Nx.shape(features.input_features) == {4, 640}
    assert Nx.to_flat_list(features.attention_mask) == [1, 1, 0, 0]
  end

  test "prompt expands Gemma4 audio marker and opens the model turn" do
    prompt = Prompt.build("System", "Transcribe.", 3)

    assert prompt ==
             "<bos><|turn>system\nSystem<turn|>\n" <>
               "<|turn>user\n" <>
               "Transcribe." <>
               "\n\n" <>
               Prompt.audio_begin() <>
               String.duplicate(Prompt.audio_token(), 3) <>
               Prompt.audio_end() <>
               "<turn|>\n" <>
               "<|turn>model\n" <>
               "<|channel>thought\n<channel|>"
  end

  test "input builder combines prompt and audio features" do
    input = Input.build(List.duplicate(0.0, 640), prompt: "Transcribe.")

    assert input.audio.token_count == 1
    assert input.prompt =~ Prompt.audio_begin() <> Prompt.audio_token() <> Prompt.audio_end()
  end

  test "token selection suppresses configured logits with a stable mask" do
    suppression_mask = TokenSelection.suppression_mask([1, 3], 5, Nx.BinaryBackend)
    logits = Nx.tensor([0.0, 99.0, 2.0, 100.0, 4.0])

    assert TokenSelection.next_token_id(logits, suppression_mask) == 4
  end

  test "token selection reports top tokens after suppression" do
    suppression_mask = TokenSelection.suppression_mask([3], 5, Nx.BinaryBackend)
    logits = Nx.tensor([0.0, 2.0, 4.0, 100.0, 3.0])

    assert TokenSelection.top_tokens(logits, suppression_mask, 2) == [{2, 4.0}, {4, 3.0}]
  end

  test "token selection skips banned candidates and falls back when all are banned" do
    suppression_mask = TokenSelection.suppression_mask([], 5, Nx.BinaryBackend)
    logits = Nx.tensor([[[0.0, 9.0, 1.0, 2.0, 8.0]]])

    assert TokenSelection.next_allowed_token_id_from_sequence(logits, suppression_mask, []) == 1
    assert TokenSelection.next_allowed_token_id_from_sequence(logits, suppression_mask, [1]) == 4

    assert TokenSelection.next_allowed_token_id_from_sequence(
             logits,
             suppression_mask,
             [0, 1, 2, 3, 4],
             3
           ) == 1
  end

  test "token selection can select from the last batched sequence position" do
    suppression_mask = TokenSelection.suppression_mask([3], 5, Nx.BinaryBackend)

    logits =
      Nx.tensor([
        [
          [0.0, 9.0, 1.0, 2.0, 3.0],
          [0.0, 2.0, 4.0, 100.0, 3.0]
        ]
      ])

    assert TokenSelection.next_token_id_from_sequence(logits, suppression_mask) == 2

    assert TokenSelection.top_tokens_from_sequence(logits, suppression_mask, 2) == [
             {2, 4.0},
             {4, 3.0}
           ]

    assert TokenSelection.next_token_with_margin_from_sequence(logits, suppression_mask) ==
             {2, 1.0}
  end

  test "transcript filtering removes tagged channel spans before decode" do
    assert Transcript.strip_tagged_span([1, 100, 45518, 101, 2], 100, 101) == [1, 2]
    assert Transcript.strip_tagged_span([1, 100, 45518, 2], 100, 101) == [1]
  end

  test "transcript filtering strips channel headers without dropping following text" do
    token_ids = [
      1,
      100,
      901,
      101,
      2,
      100,
      900,
      101,
      3,
      4
    ]

    assert Transcript.strip_tagged_span(token_ids, 100, 101) == [1, 2, 3, 4]
  end

  test "channel generation state allows one hidden channel span before content" do
    channel_token_ids = %{start: 100, end: 101}

    assert ChannelState.advance(ChannelState.initial(), 100, channel_token_ids) ==
             :inside_channel

    assert ChannelState.advance(:inside_channel, 45518, channel_token_ids) == :inside_channel
    assert ChannelState.advance(:inside_channel, 101, channel_token_ids) == :content
    assert ChannelState.advance(ChannelState.initial(), 200, channel_token_ids) == :content
    assert ChannelState.advance(:content, 100, channel_token_ids) == :content
    assert ChannelState.content() == :content
  end

  test "KV cache uses Gemma4 per-layer attention head sizes" do
    spec =
      Bumblebee.configure(Model,
        num_blocks: 2,
        num_attention_heads: 2,
        attention_head_size: 4,
        global_attention_head_size: 8,
        layer_types: [:sliding_attention, :full_attention]
      )

    cache = Model.init_cache(spec, 1, 5, %{})
    first_block = elem(cache.blocks, 0)
    second_block = elem(cache.blocks, 1)

    assert Nx.shape(first_block.self_attention.key) == {1, 5, 2, 4}
    assert Nx.shape(first_block.self_attention.value) == {1, 5, 2, 4}
    assert Nx.shape(second_block.self_attention.key) == {1, 5, 2, 8}
    assert Nx.shape(second_block.self_attention.value) == {1, 5, 2, 8}
  end

  test "local Gemma4Unified model graph runs with a tiny config" do
    spec =
      Bumblebee.configure(Model,
        vocab_size: 32,
        max_positions: 16,
        hidden_size: 8,
        intermediate_size: 16,
        num_blocks: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        num_global_key_value_heads: 1,
        attention_head_size: 4,
        global_attention_head_size: 4,
        attention_window_size: 4,
        layer_types: [:sliding_attention, :full_attention],
        boa_token_id: 5,
        audio_token_id: 7,
        eoa_token_id: 9,
        audio_embed_dim: 4,
        final_logit_softcapping: nil
      )

    model = Bumblebee.build_model(spec)
    {init_fun, predict_fun} = Axon.build(model)

    inputs = %{
      "input_ids" => Nx.tensor([[2, 7, 3]], type: :s64),
      "attention_mask" => Nx.tensor([[1, 1, 1]], type: :s64),
      "position_ids" => Nx.tensor([[0, 1, 2]], type: :s64),
      "input_features" => Nx.tensor([[[0.0, 0.1, 0.2, 0.3]]], type: {:f, 32}),
      "input_features_mask" => Nx.tensor([[1]], type: :s64)
    }

    params = init_fun.(inputs, Axon.ModelState.empty())
    outputs = predict_fun.(params, inputs)

    assert Nx.shape(outputs.logits) == {1, 3, 32}
  end

  test "logits_last_only emits logits for only the final position" do
    spec =
      Bumblebee.configure(Model,
        vocab_size: 32,
        max_positions: 16,
        hidden_size: 8,
        intermediate_size: 16,
        num_blocks: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        num_global_key_value_heads: 1,
        attention_head_size: 4,
        global_attention_head_size: 4,
        attention_window_size: 4,
        layer_types: [:sliding_attention, :full_attention],
        boa_token_id: 5,
        audio_token_id: 7,
        eoa_token_id: 9,
        audio_embed_dim: 4,
        final_logit_softcapping: nil
      )

    inputs = %{
      "input_ids" => Nx.tensor([[2, 7, 3]], type: :s64),
      "attention_mask" => Nx.tensor([[1, 1, 1]], type: :s64),
      "position_ids" => Nx.tensor([[0, 1, 2]], type: :s64),
      "input_features" => Nx.tensor([[[0.0, 0.1, 0.2, 0.3]]], type: {:f, 32}),
      "input_features_mask" => Nx.tensor([[1]], type: :s64)
    }

    full_model = Bumblebee.build_model(spec)
    {init_fun, full_predict_fun} = Axon.build(full_model)
    params = init_fun.(inputs, Axon.ModelState.empty())
    full_logits = full_predict_fun.(params, inputs).logits

    last_model = Bumblebee.build_model(Bumblebee.configure(spec, logits_last_only: true))
    {_init_fun, last_predict_fun} = Axon.build(last_model)
    last_logits = last_predict_fun.(params, inputs).logits

    assert Nx.shape(last_logits) == {1, 1, 32}

    assert full_logits
           |> Nx.slice_along_axis(2, 1, axis: 1)
           |> Nx.all_close(last_logits)
           |> Nx.to_number() == 1
  end

  test "init_cache uses the configured cache type" do
    spec =
      Bumblebee.configure(Model,
        num_blocks: 1,
        num_attention_heads: 2,
        attention_head_size: 4,
        layer_types: [:sliding_attention],
        cache_type: {:bf, 16}
      )

    cache = Model.init_cache(spec, 1, 5, %{})
    first_block = elem(cache.blocks, 0)

    assert Nx.type(first_block.self_attention.key) == {:bf, 16}
    assert Nx.type(first_block.self_attention.value) == {:bf, 16}
  end

  test "chunked prefill matches one-shot prefill" do
    # Incremental prefill appends several tokens to the cache at a non-zero
    # offset, which the single-shot path never does. This isolates that.
    spec =
      Bumblebee.configure(Model,
        vocab_size: 32,
        max_positions: 16,
        hidden_size: 8,
        intermediate_size: 16,
        num_blocks: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        num_global_key_value_heads: 1,
        attention_head_size: 4,
        global_attention_head_size: 4,
        attention_window_size: 8,
        layer_types: [:sliding_attention, :full_attention],
        audio_token_id: 7,
        audio_embed_dim: 4,
        final_logit_softcapping: nil
      )

    model = Bumblebee.build_model(spec)
    {init_fun, predict_fun} = Axon.build(model)

    token_ids = [2, 3, 4, 5, 6]
    silent = Nx.broadcast(0.0, {1, 1, 4})

    one_shot_inputs = %{
      "input_ids" => Nx.tensor([token_ids], type: :s64),
      "attention_mask" => Nx.tensor([List.duplicate(1, 5)], type: :s64),
      "position_ids" => Nx.tensor([[0, 1, 2, 3, 4]], type: :s64),
      "input_features" => silent,
      "input_features_mask" => Nx.tensor([[0]], type: :s64),
      "cache" => Model.init_cache(spec, 1, 8, %{})
    }

    params = init_fun.(one_shot_inputs, Axon.ModelState.empty())
    one_shot = predict_fun.(params, one_shot_inputs)
    one_shot_last = one_shot.logits[[0, 4]]

    # Same tokens, prefilled as [2, 3] then [4, 5, 6] at offset 2.
    first = %{
      one_shot_inputs
      | "input_ids" => Nx.tensor([[2, 3]], type: :s64),
        "attention_mask" => Nx.tensor([[1, 1]], type: :s64),
        "position_ids" => Nx.tensor([[0, 1]], type: :s64)
    }

    first_out = predict_fun.(params, first)

    second = %{
      "input_ids" => Nx.tensor([[4, 5, 6]], type: :s64),
      "attention_mask" => Nx.tensor([[1, 1, 1]], type: :s64),
      "position_ids" => Nx.tensor([[2, 3, 4]], type: :s64),
      "input_features" => silent,
      "input_features_mask" => Nx.tensor([[0]], type: :s64),
      "cache" => first_out.cache
    }

    chunked = predict_fun.(params, second)
    chunked_last = chunked.logits[[0, 2]]

    assert Nx.to_number(chunked.cache.offset) == 5

    max_diff =
      one_shot_last
      |> Nx.subtract(chunked_last)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert max_diff < 1.0e-5,
           "chunked prefill diverges from one-shot by #{max_diff}"
  end

  test "local Gemma4Unified model graph advances KV cache across prefill and decode" do
    spec =
      Bumblebee.configure(Model,
        vocab_size: 32,
        max_positions: 16,
        hidden_size: 8,
        intermediate_size: 16,
        num_blocks: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        num_global_key_value_heads: 1,
        attention_head_size: 4,
        global_attention_head_size: 4,
        attention_window_size: 4,
        layer_types: [:sliding_attention],
        boa_token_id: 5,
        audio_token_id: 7,
        eoa_token_id: 9,
        audio_embed_dim: 4,
        final_logit_softcapping: nil
      )

    model = Bumblebee.build_model(spec)
    {init_fun, predict_fun} = Axon.build(model)
    cache = Model.init_cache(spec, 1, 5, %{})

    prefill_inputs = %{
      "input_ids" => Nx.tensor([[2, 7, 3]], type: :s64),
      "attention_mask" => Nx.tensor([[1, 1, 1]], type: :s64),
      "position_ids" => Nx.tensor([[0, 1, 2]], type: :s64),
      "input_features" => Nx.tensor([[[0.0, 0.1, 0.2, 0.3]]], type: {:f, 32}),
      "input_features_mask" => Nx.tensor([[1]], type: :s64),
      "cache" => cache
    }

    params = init_fun.(prefill_inputs, Axon.ModelState.empty())
    prefill_outputs = predict_fun.(params, prefill_inputs)

    assert Nx.shape(prefill_outputs.logits) == {1, 3, 32}
    assert Nx.to_number(prefill_outputs.cache.offset) == 3

    decode_inputs = %{
      "input_ids" => Nx.tensor([[4]], type: :s64),
      "attention_mask" => Nx.tensor([[1]], type: :s64),
      "position_ids" => Nx.tensor([[3]], type: :s64),
      "input_features" => Nx.broadcast(0.0, {1, 1, 4}),
      "input_features_mask" => Nx.tensor([[0]], type: :s64),
      "cache" => prefill_outputs.cache
    }

    decode_outputs = predict_fun.(params, decode_inputs)

    assert Nx.shape(decode_outputs.logits) == {1, 1, 32}
    assert Nx.to_number(decode_outputs.cache.offset) == 4
  end

  test "audio placeholders receive projected audio vectors in order" do
    spec =
      Bumblebee.configure(Model,
        vocab_size: 8,
        max_positions: 8,
        hidden_size: 4,
        intermediate_size: 8,
        num_blocks: 0,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        num_global_key_value_heads: 1,
        attention_head_size: 4,
        global_attention_head_size: 4,
        layer_types: [],
        audio_token_id: 7,
        audio_embed_dim: 4,
        final_logit_softcapping: nil,
        tie_word_embeddings: false
      )

    model = Bumblebee.build_model(spec)
    {_init_fun, predict_fun} = Axon.build(model)

    inputs = %{
      "input_ids" => Nx.tensor([[1, 7, 3, 7]], type: :s64),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1]], type: :s64),
      "position_ids" => Nx.tensor([[0, 1, 2, 3]], type: :s64),
      "input_features" =>
        Nx.tensor([[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]], type: {:f, 32}),
      "input_features_mask" => Nx.tensor([[1, 1]], type: :s64)
    }

    params =
      Axon.ModelState.new(%{
        "audio_embedder.projection" => %{
          "kernel" => Nx.eye(4, type: {:f, 32})
        },
        "embedder.token_embedding" => %{
          "kernel" =>
            Nx.tensor(
              [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0]
              ],
              type: {:f, 32}
            )
        },
        "language_modeling_head.output" => %{
          "kernel" =>
            Nx.tensor(
              [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]
              ],
              type: {:f, 32}
            )
        },
        "output_norm" => %{
          "weight" => Nx.tensor([1.0, 1.0, 1.0, 1.0], type: {:f, 32})
        }
      })

    logits = predict_fun.(params, inputs).logits

    assert_close_list(logit_head(logits, 0), [2.0, 0.0, 0.0, 0.0])
    assert_close_list(logit_head(logits, 1), rms_vector([1.0, 2.0, 3.0, 4.0]))
    assert_close_list(logit_head(logits, 2), [0.0, 0.0, 2.0, 0.0])
    assert_close_list(logit_head(logits, 3), rms_vector([4.0, 3.0, 2.0, 1.0]))
  end

  test "full attention with k_eq_v reuses key projection without key norm leakage" do
    params =
      tiny_model_params(
        num_blocks: 1,
        layer_types: [:full_attention],
        attention_k_eq_v: true
      )

    paths = param_paths(params)

    assert "decoder.blocks.0.self_attention.key.kernel" in paths
    assert "decoder.blocks.0.self_attention.key_norm.weight" in paths
    assert "decoder.blocks.0.layer_scalar.layer_scalar" in paths
    refute "decoder.blocks.0.self_attention.value.kernel" in paths
  end

  test "full attention without k_eq_v keeps an explicit value projection" do
    params =
      tiny_model_params(
        num_blocks: 1,
        layer_types: [:full_attention],
        attention_k_eq_v: false
      )

    assert "decoder.blocks.0.self_attention.value.kernel" in param_paths(params)
  end

  test "params mapping loads Gemma4 layer scalar buffers" do
    mapping = Bumblebee.HuggingFace.Transformers.Model.params_mapping(%Model{})

    assert mapping["decoder.blocks.{n}.layer_scalar"] == "model.language_model.layers.{n}"
  end

  test "compressed-tensors params mapping uses packed linear kernels" do
    mapping =
      Bumblebee.HuggingFace.Transformers.Model.params_mapping(%Model{
        quantization_config: %{"quant_method" => "compressed-tensors"}
      })

    # Weights stay packed, so each linear loads a packed tensor and its scales
    # rather than one dequantized kernel.
    assert %{
             "packed" =>
               {[{"model.language_model.layers.{n}.self_attn.q_proj", "weight_packed"}],
                packed_builder},
             "scales" =>
               {[{"model.language_model.layers.{n}.self_attn.q_proj", "weight_scale"}],
                scales_builder}
           } = mapping["decoder.blocks.{n}.self_attention.query"]

    assert is_function(packed_builder, 1)
    assert is_function(scales_builder, 1)
    assert mapping["embedder.token_embedding"] == "model.language_model.embed_tokens"
  end

  test "hybrid params mapping loads dequantized kernels alongside packed weights" do
    mapping =
      Bumblebee.HuggingFace.Transformers.Model.params_mapping(%Model{
        quantization_config: %{"quant_method" => "compressed-tensors"},
        packed_linear: true,
        hybrid_linear: true
      })

    # Prefill reads "kernel" via rocBLAS; decode reads "packed"/"scales".
    assert %{"kernel" => {_, _}, "packed" => {_, _}, "scales" => {_, _}} =
             mapping["decoder.blocks.{n}.self_attention.query"]
  end

  test "compressed-tensors linear kernel unpacks uint4b8 group scales" do
    row_a = List.flatten(List.duplicate([-8, -1, 0, 1, 2, 3, 6, -1], 4))
    row_b = List.flatten(List.duplicate([7, 6, 5, 4, 3, 2, 1, 0], 4))

    packed =
      Nx.tensor(
        [
          Enum.map(Enum.chunk_every(row_a, 8), &pack_int4/1),
          Enum.map(Enum.chunk_every(row_b, 8), &pack_int4/1)
        ],
        type: :s32
      )

    scales = Nx.tensor([[0.5], [2.0]], type: :f32)

    kernel = CompressedTensors.linear_kernel([packed, scales])

    assert Nx.shape(kernel) == {32, 2}

    assert_close_list(
      Nx.to_flat_list(kernel),
      row_a
      |> Enum.zip(row_b)
      |> Enum.flat_map(fn {a, b} -> [a * 0.5, b * 2.0] end)
    )
  end

  test "model config loads Gemma4Unified composite config fields" do
    spec =
      Bumblebee.HuggingFace.Transformers.Config.load(%Model{}, %{
        "audio_token_id" => 12,
        "boa_token_id" => 11,
        "eoa_token_index" => 13,
        "eos_token_id" => [1, 50],
        "initializer_range" => 0.01,
        "quantization_config" => %{"quant_method" => "compressed-tensors"},
        "audio_config" => %{"audio_embed_dim" => 4, "rms_norm_eps" => 1.0e-5},
        "text_config" => %{
          "vocab_size" => 32,
          "max_position_embeddings" => 128,
          "hidden_size" => 8,
          "intermediate_size" => 16,
          "enable_moe_block" => true,
          "moe_intermediate_size" => 4,
          "num_experts" => 8,
          "top_k_experts" => 2,
          "num_hidden_layers" => 2,
          "num_attention_heads" => 2,
          "num_key_value_heads" => 1,
          "num_global_key_value_heads" => 1,
          "attention_k_eq_v" => true,
          "head_dim" => 4,
          "global_head_dim" => 4,
          "hidden_activation" => "gelu_pytorch_tanh",
          "rms_norm_eps" => 1.0e-5,
          "sliding_window" => 8,
          "layer_types" => ["sliding_attention", "full_attention"],
          "final_logit_softcapping" => nil,
          "rope_parameters" => %{
            "sliding_attention" => %{"rope_theta" => 10_000.0},
            "full_attention" => %{
              "rope_theta" => 1_000_000.0,
              "partial_rotary_factor" => 0.25
            }
          }
        }
      })

    assert spec.vocab_size == 32
    assert spec.hidden_size == 8
    assert spec.enable_moe_block
    assert spec.moe_intermediate_size == 4
    assert spec.num_experts == 8
    assert spec.top_k_experts == 2
    assert spec.boa_token_id == 11
    assert spec.audio_token_id == 12
    assert spec.eoa_token_id == 13
    assert spec.attention_k_eq_v == true
    assert spec.audio_embed_dim == 4
    assert spec.layer_types == [:sliding_attention, :full_attention]
    assert spec.quantization_config == %{"quant_method" => "compressed-tensors"}
  end

  defp tiny_model_params(opts) do
    spec =
      Bumblebee.configure(Model,
        vocab_size: 32,
        max_positions: 16,
        hidden_size: 8,
        intermediate_size: 16,
        num_blocks: Keyword.fetch!(opts, :num_blocks),
        num_attention_heads: 2,
        num_key_value_heads: 1,
        num_global_key_value_heads: 1,
        attention_head_size: 4,
        global_attention_head_size: 4,
        layer_types: Keyword.fetch!(opts, :layer_types),
        attention_k_eq_v: Keyword.fetch!(opts, :attention_k_eq_v),
        audio_embed_dim: 4,
        final_logit_softcapping: nil
      )

    model = Bumblebee.build_model(spec)
    {init_fun, _predict_fun} = Axon.build(model)

    inputs = %{
      "input_ids" => Nx.tensor([[2, 7, 3]], type: :s64),
      "attention_mask" => Nx.tensor([[1, 1, 1]], type: :s64),
      "position_ids" => Nx.tensor([[0, 1, 2]], type: :s64),
      "input_features" => Nx.tensor([[[0.0, 0.1, 0.2, 0.3]]], type: {:f, 32}),
      "input_features_mask" => Nx.tensor([[1]], type: :s64)
    }

    init_fun.(inputs, Axon.ModelState.empty())
  end

  defp param_paths(%Axon.ModelState{data: data}) do
    data
    |> param_paths([])
    |> Enum.sort()
  end

  defp param_paths(%Nx.Tensor{}, path), do: [Enum.join(path, ".")]

  defp param_paths(map, path) when is_map(map) do
    Enum.flat_map(map, fn {key, value} -> param_paths(value, path ++ [to_string(key)]) end)
  end

  defp param_paths(list, path) when is_list(list) do
    list
    |> Enum.with_index()
    |> Enum.flat_map(fn {value, index} -> param_paths(value, path ++ [to_string(index)]) end)
  end

  defp param_paths(_other, _path), do: []

  defp logit_head(logits, position) do
    logits
    |> Nx.slice([0, position, 0], [1, 1, 4])
    |> Nx.reshape({4})
    |> Nx.to_flat_list()
  end

  defp rms_vector(values) do
    rms =
      values
      |> Enum.map(&(&1 * &1))
      |> Enum.sum()
      |> Kernel./(length(values))
      |> Kernel.+(1.0e-6)
      |> :math.sqrt()

    Enum.map(values, &(&1 / rms))
  end

  defp assert_close_list(actual, expected) do
    Enum.zip(actual, expected)
    |> Enum.each(fn {actual, expected} ->
      assert_in_delta actual, expected, 1.0e-5
    end)
  end

  defp pack_int4(values) do
    values
    |> Enum.with_index()
    |> Enum.reduce(0, fn {value, index}, packed ->
      packed ||| (value + 8 &&& 0xF) <<< (index * 4)
    end)
    |> signed_i32()
  end

  defp signed_i32(value) when value >= 0x8000_0000, do: value - 0x1_0000_0000
  defp signed_i32(value), do: value

  test "model catalog resolves the friendly Gemma4 alias" do
    for model <- ModelCatalog.all() do
      assert ModelCatalog.resolve(model.name) == model.hf_repo
      assert ModelCatalog.resolve(model.hf_repo) == model.hf_repo
    end

    assert ModelCatalog.runtime_kind("gemma4-12b-unified") == :bumblebee_axon

    assert ModelCatalog.runtime_kind("gemma4-12b-qat-w4a16-ct") == :bumblebee_axon

    assert ModelCatalog.artifact_format("google/gemma-4-12B-it-qat-w4a16-ct") ==
             :compressed_tensors

    assert ModelCatalog.runtime_module("gemma4-12b-qat-w4a16-ct") ==
             {:ok, Gemma4MicTranscribe.Gemma4Unified.Runtime}

    assert ModelCatalog.runtime_kind("gemma4-12b-qat-q4_0-gguf") == :llama_cpp
    assert ModelCatalog.artifact_format("gemma4-12b-qat-q4_0-gguf") == :gguf

    assert {:error, gguf_runtime_message} =
             ModelCatalog.runtime_module("gemma4-12b-qat-q4_0-gguf")

    assert gguf_runtime_message =~ "llama.cpp"
  end

  test "Axon runtime refuses GGUF before backend setup" do
    assert {:error, gguf_message} =
             Runtime.load(model_name: "gemma4-12b-qat-q4_0-gguf", backend: "exla:rocm")

    assert gguf_message =~ "GGUF"
    assert gguf_message =~ "llama.cpp"
  end

  test "ROCm preflight parses GPU targets from rocm_agent_enumerator output" do
    assert RocmPreflight.parse_gfx_targets("gfx000\ngfx1151\n") == ["gfx1151"]
  end

  test "ROCm preflight parses XLA offload bundle targets" do
    output = """
    Extracting offload bundle: libxla_extension.so.0.hipv4-amdgcn-amd-amdhsa--gfx1100
    Extracting offload bundle: libxla_extension.so.1.hipv4-amdgcn-amd-amdhsa--gfx1200
    """

    assert RocmPreflight.parse_offload_targets(output) == ["gfx1100", "gfx1200"]
  end

  test "ROCm preflight reports inspected XLA extension path when bundles are missing" do
    tmp_dir = Path.join(System.tmp_dir!(), "gemma-rocm-preflight-#{System.unique_integer()}")
    File.mkdir_p!(tmp_dir)
    on_exit(fn -> File.rm_rf(tmp_dir) end)

    xla_extension = Path.join(tmp_dir, "libxla_extension.so")
    llvm_objdump = Path.join(tmp_dir, "llvm-objdump")
    rocm_agent = Path.join(tmp_dir, "rocm_agent_enumerator")

    File.write!(xla_extension, "")
    File.write!(llvm_objdump, "#!/bin/sh\nprintf '%s\\n' 'file format elf64-x86-64'\n")
    File.write!(rocm_agent, "#!/bin/sh\nprintf '%s\\n' 'gfx1151'\n")
    File.chmod!(llvm_objdump, 0o755)
    File.chmod!(rocm_agent, 0o755)

    assert {:error, message} =
             RocmPreflight.check(
               xla_extension_path: xla_extension,
               llvm_objdump: llvm_objdump,
               rocm_agent_enumerator: rocm_agent
             )

    assert message =~ "no ROCm offload bundles"
    assert message =~ xla_extension
  end

  test "ROCm compatibility-only preflight ignores allocator memory reservations" do
    tmp_dir = Path.join(System.tmp_dir!(), "gemma-rocm-preflight-#{System.unique_integer()}")
    File.mkdir_p!(tmp_dir)
    on_exit(fn -> File.rm_rf(tmp_dir) end)

    xla_extension = Path.join(tmp_dir, "libxla_extension.so")
    llvm_objdump = Path.join(tmp_dir, "llvm-objdump")
    rocm_agent = Path.join(tmp_dir, "rocm_agent_enumerator")

    File.write!(xla_extension, "")

    File.write!(
      llvm_objdump,
      "#!/bin/sh\nprintf '%s\\n' 'hipv4-amdgcn-amd-amdhsa--gfx1151'\n"
    )

    File.write!(rocm_agent, "#!/bin/sh\nprintf '%s\\n' 'gfx1151'\n")
    File.chmod!(llvm_objdump, 0o755)
    File.chmod!(rocm_agent, 0o755)

    low_memory = """
    {"card0": {"VRAM Total Memory (B)": "68719476736", "VRAM Total Used Memory (B)": "60129542144"}}
    """

    assert :ok =
             RocmPreflight.check(
               xla_extension_path: xla_extension,
               llvm_objdump: llvm_objdump,
               rocm_agent_enumerator: rocm_agent,
               rocm_smi_output: low_memory,
               min_free_bytes: 24 * 1024 * 1024 * 1024,
               skip_memory_budget: true
             )
  end

  test "repacked int4 weights dequantize to the same matrix as the dequant path" do
    out_features = 4
    # in_features must cover whole 32-element quant groups
    packed_cols = 8
    in_features = packed_cols * 8
    group_size = CompressedTensors.quant_group_size()
    scale_cols = div(in_features, group_size)

    # Deterministic biased nibbles 0..15 packed 8-per-word along in_features.
    nibbles =
      Nx.iota({out_features, packed_cols, 8}, type: :u32)
      |> Nx.multiply(7)
      |> Nx.remainder(16)

    multipliers =
      Enum.map(0..7, &:erlang.bsl(1, &1 * 4))
      |> Nx.tensor(type: :u32)
      |> Nx.reshape({1, 1, 8})

    packed =
      nibbles
      |> Nx.multiply(multipliers)
      |> Nx.sum(axes: [2])
      |> Nx.bitcast(:s32)

    # bf16 throughout: repack_scales stores bf16 because that is the type the
    # kernel's FFI signature requires.
    scales =
      Nx.iota({out_features, scale_cols}, type: {:f, 32})
      |> Nx.add(1)
      |> Nx.divide(100)
      |> Nx.as_type({:bf, 16})

    # Reference: existing dequant path, {in_features, out_features}
    reference = CompressedTensors.linear_kernel([packed, scales])
    assert Nx.shape(reference) == {in_features, out_features}

    repacked = CompressedTensors.repack_kernel([packed])
    repacked_scales = CompressedTensors.repack_scales([scales])

    assert Nx.shape(repacked) == {div(in_features, 8), out_features}
    assert Nx.shape(repacked_scales) == {scale_cols, out_features}

    # Dequantize the repacked layout the way the kernel does.
    shifts = Nx.iota({1, 8, 1}, axis: 1, type: :s32) |> Nx.multiply(4)

    dequantized =
      repacked
      |> Nx.new_axis(1)
      |> Nx.right_shift(shifts)
      |> Nx.bitwise_and(0xF)
      |> Nx.subtract(8)
      |> Nx.reshape({in_features, out_features})
      |> Nx.as_type({:f, 32})
      |> Nx.multiply(
        repacked_scales
        |> Nx.new_axis(1)
        |> Nx.broadcast({scale_cols, group_size, out_features})
        |> Nx.reshape({in_features, out_features})
      )

    # Tolerance is bf16 rounding of the product (~2^-8 relative), not repack error.
    max_diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(dequantized, reference))))
    max_value = Nx.to_number(Nx.reduce_max(Nx.abs(reference)))

    assert max_diff / max_value < 0.01
  end

  test "dual packed projection fallback concatenates both results" do
    x = Nx.broadcast(Nx.tensor(1, type: :bf16), {32})
    packed = Nx.broadcast(Nx.tensor(0, type: :s32), {4, 2})
    first_scales = Nx.broadcast(Nx.tensor(0.5, type: :bf16), {1, 2})
    second_scales = Nx.broadcast(Nx.tensor(0.25, type: :bf16), {1, 2})

    project =
      Nx.Defn.jit(fn x, packed, first_scales, second_scales ->
        Q4DualGemv.dot(x, packed, first_scales, packed, second_scales)
      end)

    assert Nx.to_flat_list(project.(x, packed, first_scales, second_scales)) ==
             [-128.0, -128.0, -64.0, -64.0]
  end

  test "banned ngram tokens block completing an already-generated trigram" do
    # sequence in order: [5, 6, 7, 9, 5, 6] -> reversed [6, 5, 9, 7, 6, 5]
    # last bigram [5, 6] previously continued with 7, so 7 is banned
    assert Runtime.banned_ngram_token_ids([6, 5, 9, 7, 6, 5], 3) == [7]

    # a fresh bigram bans nothing
    assert Runtime.banned_ngram_token_ids([9, 7], 3) == []

    # too little history or disabled sizes ban nothing
    assert Runtime.banned_ngram_token_ids([7], 3) == []
    assert Runtime.banned_ngram_token_ids([6, 5, 9, 7, 6, 5], 0) == []
    assert Runtime.banned_ngram_token_ids([6, 5, 9, 7, 6, 5], 1) == []
  end

  test "leftover audio decomposes into whole chunk sizes without padding" do
    # 37 tokens should become 25 + 10 + 2, not 37 single-token prefills and not
    # one padded 50-token chunk.
    assert Runtime.decompose_tokens(37) == [25, 10, 2]
    assert Runtime.decompose_tokens(50) == [50]
    assert Runtime.decompose_tokens(1) == [1]
    assert Runtime.decompose_tokens(0) == []
    assert Enum.sum(Runtime.decompose_tokens(123)) == 123
  end

  test "prefill masks exclude padded audio tokens and keep positions contiguous" do
    # prompt: [text, text, audio, audio, PAD, PAD, text] -> 7 tokens,
    # audio span starts at 2 with 2 real of 4 bucketed tokens
    masks = Runtime.prefill_masks(7, 2, 2, 4)

    assert masks.attention_mask == [1, 1, 1, 1, 0, 0, 1]
    assert masks.position_ids == [0, 1, 2, 3, 0, 0, 4]
    assert masks.content_length == 5
  end

  test "prefill masks are all-ones without padding" do
    masks = Runtime.prefill_masks(4, 1, 2, 2)

    assert masks.attention_mask == [1, 1, 1, 1]
    assert masks.position_ids == [0, 1, 2, 3]
    assert masks.content_length == 4
  end

  test "ROCm preflight adds gfx1151 XLA workarounds" do
    all_flags =
      "--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer= " <>
        "--xla_gpu_enable_triton_gemm=false"

    assert RocmPreflight.runtime_workaround_flags(["gfx1151"], nil) == {all_flags, true}

    assert RocmPreflight.runtime_workaround_flags(["gfx1151"], "--xla_dump_to=/tmp/xla") ==
             {"--xla_dump_to=/tmp/xla " <> all_flags, true}
  end

  test "ROCm preflight preserves explicit XLA autotune settings" do
    assert RocmPreflight.runtime_workaround_flags(
             ["gfx1151"],
             "--xla_gpu_autotune_level=2 --xla_gpu_enable_command_buffer=FUSION " <>
               "--xla_gpu_enable_triton_gemm=true"
           ) ==
             {"--xla_gpu_autotune_level=2 --xla_gpu_enable_command_buffer=FUSION " <>
                "--xla_gpu_enable_triton_gemm=true", false}

    assert RocmPreflight.runtime_workaround_flags(["gfx1100"], nil) == {nil, false}
  end

  test "ROCm preflight parses rocm-smi JSON memory output" do
    output = """
    WARNING: AMD GPU device(s) is/are in a low-power state.
    {"card0": {"VRAM Total Memory (B)": "68719476736", "VRAM Total Used Memory (B)": "1523736576"}}
    """

    assert {:ok, info} = RocmPreflight.parse_memory_info(output)
    assert info.total == 68_719_476_736
    assert info.used == 1_523_736_576
    assert info.free == 67_195_740_160
  end

  test "ROCm preflight rejects low VRAM headroom" do
    output = """
    {"card0": {"VRAM Total Memory (B)": "68719476736", "VRAM Total Used Memory (B)": "60129542144"}}
    """

    assert {:error, message} =
             RocmPreflight.memory_budget(
               rocm_smi: System.find_executable("printf"),
               min_free_bytes: 24 * 1024 * 1024 * 1024,
               rocm_smi_output: output
             )

    assert message =~ "GPU VRAM headroom is too low"
  end
end
