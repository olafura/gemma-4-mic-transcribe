defmodule Gemma4MicTranscribe.Gemma4UnifiedTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4Unified.AudioFeatureExtractor
  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Model
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

  test "prompt expands Gemma4 audio marker like LiteRT-LM input data" do
    prompt = Prompt.build("System", "Transcribe.", 3)

    assert prompt ==
             "<bos><|turn>system\nSystem<turn|>\n" <>
               "<|turn>user\n" <>
               Prompt.audio_begin() <>
               String.duplicate(Prompt.audio_token(), 3) <>
               Prompt.audio_end() <>
               "Transcribe.<turn|>\n" <>
               "<|turn>model\n"
  end

  test "input builder combines prompt and audio features" do
    input = Input.build(List.duplicate(0.0, 640), prompt: "Transcribe.")

    assert input.audio.token_count == 1
    assert input.prompt =~ Prompt.audio_begin() <> Prompt.audio_token() <> Prompt.audio_end()
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

  test "model config loads Gemma4Unified composite config fields" do
    spec =
      Bumblebee.HuggingFace.Transformers.Config.load(%Model{}, %{
        "audio_token_id" => 12,
        "boa_token_id" => 11,
        "eoa_token_index" => 13,
        "eos_token_id" => [1, 50],
        "initializer_range" => 0.01,
        "audio_config" => %{"audio_embed_dim" => 4, "rms_norm_eps" => 1.0e-5},
        "text_config" => %{
          "vocab_size" => 32,
          "max_position_embeddings" => 128,
          "hidden_size" => 8,
          "intermediate_size" => 16,
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
    assert spec.boa_token_id == 11
    assert spec.audio_token_id == 12
    assert spec.eoa_token_id == 13
    assert spec.attention_k_eq_v == true
    assert spec.audio_embed_dim == 4
    assert spec.layer_types == [:sliding_attention, :full_attention]
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

  test "model catalog resolves the friendly Gemma4 alias" do
    assert ModelCatalog.resolve("gemma4-12b-unified") == "google/gemma-4-12B-it"
    assert ModelCatalog.resolve("google/gemma-4-12B-it") == "google/gemma-4-12B-it"
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

  test "ROCm preflight adds gfx1151 XLA autotune workaround" do
    assert RocmPreflight.runtime_workaround_flags(["gfx1151"], nil) ==
             {"--xla_gpu_autotune_level=0", true}

    assert RocmPreflight.runtime_workaround_flags(["gfx1151"], "--xla_dump_to=/tmp/xla") ==
             {"--xla_dump_to=/tmp/xla --xla_gpu_autotune_level=0", true}
  end

  test "ROCm preflight preserves explicit XLA autotune settings" do
    assert RocmPreflight.runtime_workaround_flags(["gfx1151"], "--xla_gpu_autotune_level=2") ==
             {"--xla_gpu_autotune_level=2", false}

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
