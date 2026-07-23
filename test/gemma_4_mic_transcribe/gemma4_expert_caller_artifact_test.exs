defmodule Gemma4MicTranscribe.Gemma4.ExpertCallerArtifactTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4.ExpertCallerArtifact

  @hidden 4
  @heads 2
  @kv_heads 1
  @head_dim 2

  @tag :tmp_dir
  test "range-extracts a later decoder layer's attention caller", %{tmp_dir: tmp_dir} do
    tensors = synthetic_tensors(1)
    source = tensors |> Safetensors.dump() |> IO.iodata_to_binary()

    config = %{
      "text_config" => %{
        "num_hidden_layers" => 2,
        "layer_types" => ["full_attention", "sliding_attention"],
        "hidden_size" => @hidden,
        "num_attention_heads" => @heads,
        "num_key_value_heads" => @kv_heads,
        "head_dim" => @head_dim,
        "rms_norm_eps" => 1.0e-6,
        "sliding_window" => 8,
        "rope_parameters" => %{
          "full_attention" => %{"rope_theta" => 1_000_000.0},
          "sliding_attention" => %{"rope_theta" => 10_000.0}
        }
      }
    }

    index = %{
      "metadata" => %{"total_size" => byte_size(source)},
      "weight_map" => Map.new(tensors, fn {name, _} -> {name, "model.safetensors"} end)
    }

    fetch_json = fn url ->
      if String.ends_with?(url, "config.json"), do: config, else: index
    end

    fetch_range = fn _url, first, last ->
      binary_part(source, first, last - first + 1)
    end

    artifact = Path.join(tmp_dir, "caller")

    manifest =
      ExpertCallerArtifact.extract!(artifact,
        layer: 1,
        fetch_json: fetch_json,
        fetch_range: fetch_range
      )

    assert manifest.hidden_size == @hidden
    assert manifest.layer_index == 1
    assert manifest.attention_type == "sliding_attention"
    assert manifest.sliding_window == 8
    assert manifest.parameter_count == 60

    {loaded_manifest, params} = ExpertCallerArtifact.load!(artifact, Nx.BinaryBackend)

    assert loaded_manifest.parameter_sha256 == manifest.parameter_sha256

    assert params |> Map.keys() |> Enum.sort() ==
             [
               :input_norm,
               :key,
               :key_norm,
               :output,
               :post_attention_norm,
               :query,
               :query_norm,
               :value
             ]

    assert Nx.shape(params.query) == {@heads * @head_dim, @hidden}
    assert Nx.shape(params.key) == {@kv_heads * @head_dim, @hidden}
  end

  @tag :tmp_dir
  test "extracts full attention with global dimensions and shared K/V projection", %{
    tmp_dir: tmp_dir
  } do
    tensors = full_attention_tensors()

    {norm_tensors, attention_tensors} =
      Map.split(tensors, [
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.post_attention_layernorm.weight"
      ])

    sources = %{
      "attention.safetensors" => attention_tensors |> Safetensors.dump() |> IO.iodata_to_binary(),
      "norms.safetensors" => norm_tensors |> Safetensors.dump() |> IO.iodata_to_binary()
    }

    config = %{
      "text_config" => %{
        "num_hidden_layers" => 1,
        "layer_types" => ["full_attention"],
        "hidden_size" => @hidden,
        "num_attention_heads" => @heads,
        "num_key_value_heads" => @kv_heads,
        "head_dim" => @head_dim,
        "global_head_dim" => 4,
        "num_global_key_value_heads" => 1,
        "attention_k_eq_v" => true,
        "rms_norm_eps" => 1.0e-6,
        "sliding_window" => 8,
        "rope_parameters" => %{
          "full_attention" => %{
            "rope_theta" => 1_000_000.0,
            "rope_type" => "proportional",
            "partial_rotary_factor" => 0.25
          }
        }
      }
    }

    index = %{
      "metadata" => %{
        "total_size" => sources |> Map.values() |> Enum.map(&byte_size/1) |> Enum.sum()
      },
      "weight_map" =>
        Map.new(tensors, fn {name, _} ->
          shard =
            if String.contains?(name, "layernorm"),
              do: "norms.safetensors",
              else: "attention.safetensors"

          {name, shard}
        end)
    }

    fetch_json = fn url ->
      if String.ends_with?(url, "config.json"), do: config, else: index
    end

    fetch_range = fn url, first, last ->
      {shard, source} = Enum.find(sources, fn {shard, _} -> String.ends_with?(url, shard) end)
      assert is_binary(shard)
      binary_part(source, first, last - first + 1)
    end

    artifact = Path.join(tmp_dir, "full-caller")

    manifest =
      ExpertCallerArtifact.extract!(artifact,
        layer: 0,
        fetch_json: fetch_json,
        fetch_range: fetch_range
      )

    assert manifest.attention_type == "full_attention"
    assert manifest.head_dim == 4
    assert manifest.num_key_value_heads == 1
    assert manifest.alternative_attention
    assert manifest.partial_rotary_factor == 0.25

    assert Enum.sort(manifest.source_shards) ==
             ["attention.safetensors", "norms.safetensors"]

    assert manifest.parameter_count == 96

    {_, params} = ExpertCallerArtifact.load!(artifact, Nx.BinaryBackend)

    refute Map.has_key?(params, :value)
    assert Nx.shape(params.query) == {8, @hidden}
    assert Nx.shape(params.key) == {4, @hidden}
    assert Nx.shape(params.output) == {@hidden, 8}
  end

  defp synthetic_tensors(layer) do
    prefix = "model.language_model.layers.#{layer}"

    %{
      "#{prefix}.input_layernorm.weight" => ramp({@hidden}, 7),
      "#{prefix}.post_attention_layernorm.weight" => ramp({@hidden}, 11),
      "#{prefix}.self_attn.q_proj.weight" => ramp({@heads * @head_dim, @hidden}, 13),
      "#{prefix}.self_attn.k_proj.weight" => ramp({@kv_heads * @head_dim, @hidden}, 17),
      "#{prefix}.self_attn.v_proj.weight" => ramp({@kv_heads * @head_dim, @hidden}, 19),
      "#{prefix}.self_attn.o_proj.weight" => ramp({@hidden, @heads * @head_dim}, 23),
      "#{prefix}.self_attn.q_norm.weight" => ramp({@head_dim}, 29),
      "#{prefix}.self_attn.k_norm.weight" => ramp({@head_dim}, 31)
    }
  end

  defp full_attention_tensors do
    prefix = "model.language_model.layers.0"

    %{
      "#{prefix}.input_layernorm.weight" => ramp({@hidden}, 7),
      "#{prefix}.post_attention_layernorm.weight" => ramp({@hidden}, 11),
      "#{prefix}.self_attn.q_proj.weight" => ramp({8, @hidden}, 13),
      "#{prefix}.self_attn.k_proj.weight" => ramp({4, @hidden}, 17),
      "#{prefix}.self_attn.o_proj.weight" => ramp({@hidden, 8}, 23),
      "#{prefix}.self_attn.q_norm.weight" => ramp({4}, 29),
      "#{prefix}.self_attn.k_norm.weight" => ramp({4}, 31)
    }
  end

  defp ramp(shape, divisor) do
    shape
    |> Nx.iota(type: :f32)
    |> Nx.add(1)
    |> Nx.divide(divisor)
    |> Nx.as_type(:bf16)
  end
end
