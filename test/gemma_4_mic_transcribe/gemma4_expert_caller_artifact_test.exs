defmodule Gemma4MicTranscribe.Gemma4.ExpertCallerArtifactTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4.ExpertCallerArtifact

  @hidden 4
  @heads 2
  @kv_heads 1
  @head_dim 2

  @tag :tmp_dir
  test "range-extracts only the layer-0 attention caller", %{tmp_dir: tmp_dir} do
    tensors = synthetic_tensors()
    source = tensors |> Safetensors.dump() |> IO.iodata_to_binary()

    config = %{
      "text_config" => %{
        "hidden_size" => @hidden,
        "num_attention_heads" => @heads,
        "num_key_value_heads" => @kv_heads,
        "head_dim" => @head_dim,
        "rms_norm_eps" => 1.0e-6,
        "sliding_window" => 8,
        "rope_parameters" => %{
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
        fetch_json: fetch_json,
        fetch_range: fetch_range
      )

    assert manifest.hidden_size == @hidden
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

  defp synthetic_tensors do
    prefix = "model.language_model.layers.0"

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

  defp ramp(shape, divisor) do
    shape
    |> Nx.iota(type: :f32)
    |> Nx.add(1)
    |> Nx.divide(divisor)
    |> Nx.as_type(:bf16)
  end
end
