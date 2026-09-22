defmodule Gemma4MicTranscribe.SystemOneTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4.SystemOne
  alias Gemma4MicTranscribe.Gemma4.SystemOne.Prompt, as: SystemOnePrompt
  alias Gemma4MicTranscribe.Gemma4.SystemOne.Trainer
  alias Gemma4MicTranscribe.Gemma4.SystemOneArtifact
  alias Gemma4MicTranscribe.Gemma4Unified.Input
  alias Gemma4MicTranscribe.Gemma4Unified.Model
  alias Gemma4MicTranscribe.Gemma4Unified.Q4Gemv
  alias Gemma4MicTranscribe.SystemOneCLI

  describe "prompt rendering" do
    test "renders state, question and option names" do
      text =
        SystemOnePrompt.render(%{
          "id" => "d1-01",
          "state" => %{"battery" => 12, "charger" => "unplugged"},
          "question" => "  Should I leave now?  ",
          "options" => %{"leave" => "go anyway", "charge" => "wait for 80%"}
        })

      assert text ==
               """
               State: {"battery":12,"charger":"unplugged"}

               Should I leave now?

               Options: charge, leave\
               """
    end

    test "omits the options block and keeps a pre-rendered state" do
      text = SystemOnePrompt.render(%{"state" => " already json ", "question" => "Why?"})

      assert text == "State: already json\n\nWhy?"
    end

    test "takes options given as a bare list in order" do
      text =
        SystemOnePrompt.render(%{
          "state" => %{},
          "question" => "Which?",
          "options" => ["wait", "act"]
        })

      assert String.ends_with?(text, "Options: wait, act")
    end

    test "raises on a missing field" do
      assert_raise ArgumentError, ~r/"question"/, fn ->
        SystemOnePrompt.render(%{"state" => %{}})
      end
    end

    test "a spoken question keeps the state and options as text and points at the audio" do
      text =
        SystemOnePrompt.render_audio(%{
          "state" => %{"job" => "contract.pdf"},
          "question" => "ignored, the WAV carries it",
          "audio" => "q1.wav",
          "options" => ["hp_2200", "brother_l"]
        })

      assert text ==
               """
               State: {"job":"contract.pdf"}

               The question is spoken in the audio that follows.

               Options: hp_2200, brother_l\
               """

      # The audio slot follows the text inside the same user turn, so the
      # model reads the state first and then hears the question.
      input =
        Input.build(List.duplicate(0.1, 640 * 3),
          prompt: text,
          audio_token_count: 5,
          thought_channel: true
        )

      assert input.audio.token_count == 5
      assert Nx.to_flat_list(input.audio.attention_mask) == [1, 1, 1, 0, 0]

      assert input.prompt ==
               "<bos><|turn>user\n" <>
                 text <>
                 "\n\n<|audio>" <>
                 String.duplicate("<|audio|>", 5) <>
                 "<audio|><turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    end
  end

  describe "build_text/2" do
    test "matches the official chat template for a plain user turn" do
      input = Input.build_text("Should I leave now?")

      assert input.prompt ==
               "<bos><|turn>user\nShould I leave now?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"

      assert input.prompt_without_response == input.prompt
      assert input.response == nil
      assert input.thought_channel == true
      refute input.prompt =~ "<|audio"
    end

    test "inserts a system turn and can end at the model turn" do
      input =
        Input.build_text("Should I leave now?",
          system_message: "Be helpful.",
          thought_channel: false
        )

      assert input.prompt ==
               "<bos><|turn>system\nBe helpful.<turn|>\n<|turn>user\nShould I leave now?<turn|>\n<|turn>model\n"

      assert input.thought_channel == false
    end

    test "teacher forces the response after the model turn header" do
      input = Input.build_text("Should I leave now?", response: "Yes.")

      assert input.prompt == input.prompt_without_response <> "Yes.<turn|>"
      assert input.response == "Yes."
    end

    test "carries one masked silent audio frame and no audio tokens" do
      input = Input.build_text("Should I leave now?")

      assert Nx.shape(input.audio.input_features) == {1, 640}
      assert Nx.to_flat_list(input.audio.attention_mask) == [0]
      assert input.audio.token_count == 0
    end
  end

  describe "response_range/3" do
    test "is nil without a teacher-forced response" do
      input = Input.build_text("Should I leave now?")

      assert Input.response_range(input, [1, 2, 3], [1, 2, 3]) == nil
    end

    test "is the span the response tokens occupy" do
      input = Input.build_text("Should I leave now?", response: "Yes.")

      assert Input.response_range(input, [1, 2, 3, 4, 5], [1, 2, 3]) == %{start: 3, length: 2}
    end
  end

  describe "bucket/2" do
    test "picks the smallest bucket that fits" do
      buckets = [64, 128, 256, 384]

      assert SystemOneCLI.bucket(1, buckets) == {:ok, 64}
      assert SystemOneCLI.bucket(64, buckets) == {:ok, 64}
      assert SystemOneCLI.bucket(65, buckets) == {:ok, 128}
      assert SystemOneCLI.bucket(384, buckets) == {:ok, 384}
    end

    test "refuses a prompt longer than the largest bucket" do
      assert {:error, message} = SystemOneCLI.bucket(385, [64, 128, 256, 384])
      assert message =~ "385 tokens"
      assert message =~ "384"
    end
  end

  describe "cache argument parsing" do
    test "fills in the defaults" do
      assert {:ok, :cache, opts} =
               SystemOneCLI.parse(["cache", "--input", "in.jsonl", "--output", "out"])

      assert opts.backend == "exla:rocm"
      assert opts.buckets == [64, 128, 256, 384]
      assert opts.prefix_artifact == "artifacts/gemma4-12b-packed-prefix-0-44"
      assert opts.thought_channel == true
      assert opts.last_prompt_token_only == false
      assert opts.verify_padding == nil
      assert opts.limit == nil
    end

    test "parses the overrides" do
      assert {:ok, :cache, opts} =
               SystemOneCLI.parse([
                 "cache",
                 "--input",
                 "in.jsonl",
                 "--output",
                 "out",
                 "--backend",
                 "binary",
                 "--buckets",
                 "256, 64,64",
                 "--no-thought-channel",
                 "--last-prompt-token-only",
                 "--verify-padding",
                 "384",
                 "--limit",
                 "3"
               ])

      assert opts.backend == "binary"
      assert opts.buckets == [64, 256]
      assert opts.thought_channel == false
      assert opts.last_prompt_token_only == true
      assert opts.verify_padding == 384
      assert opts.limit == 3
    end

    test "requires the input and output paths and well-formed buckets" do
      assert {:error, message} = SystemOneCLI.parse(["cache", "--output", "out"])
      assert message =~ "--input"

      assert {:error, message} = SystemOneCLI.parse(["cache", "--input", "in.jsonl"])
      assert message =~ "--output"

      assert {:error, message} =
               SystemOneCLI.parse([
                 "cache",
                 "--input",
                 "in.jsonl",
                 "--output",
                 "out",
                 "--buckets",
                 "64,many"
               ])

      assert message =~ "--buckets"
    end

    test "reports an unknown subcommand and prints usage without arguments" do
      assert {:error, message} = SystemOneCLI.parse(["explain"])
      assert message =~ "unknown subcommand explain"

      assert {:help, usage} = SystemOneCLI.parse([])
      assert usage =~ "mix gemma.system_one cache"
    end
  end

  describe "train, generate and regress argument parsing" do
    test "train fills in the trainer's defaults" do
      assert {:ok, :train, opts} =
               SystemOneCLI.parse(["train", "--cache", "cache", "--output", "expert"])

      defaults = Trainer.defaults()

      assert opts.layers == defaults.layers
      assert opts.batch_size == defaults.batch_size
      assert opts.expert_learning_rate == defaults.expert_learning_rate
      assert opts.router_learning_rate == defaults.router_learning_rate
      assert opts.checkpoint_every == defaults.checkpoint_every
      assert opts.stage_backend == "torchx:cpu"
      assert opts.resume == true
      assert opts.dry_run == false
      assert opts.max_steps == nil
    end

    test "train parses the overrides" do
      assert {:ok, :train, opts} =
               SystemOneCLI.parse([
                 "train",
                 "--cache",
                 "cache",
                 "--output",
                 "expert",
                 "--layers",
                 "46,47",
                 "--expert-lr",
                 "0.0005",
                 "--max-steps",
                 "5",
                 "--no-resume",
                 "--dry-run"
               ])

      assert opts.layers == [46, 47]
      assert opts.expert_learning_rate == 0.0005
      assert opts.max_steps == 5
      assert opts.resume == false
      assert opts.dry_run == true
    end

    test "train requires a cache and an output and well-formed layers" do
      assert {:error, message} = SystemOneCLI.parse(["train", "--output", "expert"])
      assert message =~ "--cache"

      assert {:error, message} = SystemOneCLI.parse(["train", "--cache", "cache"])
      assert message =~ "--output"

      assert {:error, message} =
               SystemOneCLI.parse([
                 "train",
                 "--cache",
                 "cache",
                 "--output",
                 "expert",
                 "--layers",
                 "45,tail"
               ])

      assert message =~ "--layers"
    end

    test "generate defaults to base Gemma with the gate probe on" do
      assert {:ok, :generate, opts} =
               SystemOneCLI.parse(["generate", "--input", "in.jsonl", "--output", "out.jsonl"])

      assert opts.expert == nil
      assert opts.force_router_closed == false
      assert opts.gate_probe == true
      assert opts.thought_channel == true
      assert opts.max_new_tokens == 64
      assert opts.audio_seconds == 8.0
      assert opts.buckets == [64, 128, 256, 384]
      assert opts.tail_artifact == "artifacts/gemma4-12b-packed-tail-45-47"
    end

    test "generate parses the expert options" do
      assert {:ok, :generate, opts} =
               SystemOneCLI.parse([
                 "generate",
                 "--input",
                 "in.jsonl",
                 "--output",
                 "out.jsonl",
                 "--expert",
                 "artifacts/system-one-expert",
                 "--force-router-closed",
                 "--no-gate-probe",
                 "--max-new-tokens",
                 "16",
                 "--audio-seconds",
                 "6"
               ])

      assert opts.expert == "artifacts/system-one-expert"
      assert opts.force_router_closed == true
      assert opts.gate_probe == false
      assert opts.max_new_tokens == 16
      assert opts.audio_seconds == 6.0
      assert opts.gate_floor == nil
    end

    test "generate and regress take a gate floor that overrides the artifact's" do
      assert {:ok, :generate, opts} =
               SystemOneCLI.parse([
                 "generate",
                 "--input",
                 "in.jsonl",
                 "--output",
                 "out.jsonl",
                 "--gate-floor",
                 "0.5"
               ])

      assert opts.gate_floor == 0.5

      assert {:ok, :regress, opts} = SystemOneCLI.parse(["regress", "--gate-floor", "0.5"])
      assert opts.gate_floor == 0.5
      assert {:ok, :regress, opts} = SystemOneCLI.parse(["regress"])
      assert opts.gate_floor == nil
    end

    test "train takes the gate floor, the lead weighting and an artifact to start from" do
      defaults = Trainer.defaults()

      assert {:ok, :train, opts} =
               SystemOneCLI.parse(["train", "--cache", "cache", "--output", "expert"])

      assert opts.gate_floor == defaults.gate_floor
      assert opts.lead_tokens == defaults.lead_tokens
      assert opts.lead_weight == defaults.lead_weight
      assert opts.gate_mode == :response
      assert opts.init_from == nil

      # A newly trained artifact records the floor its router was supervised
      # against, not the 0.05 round 1 used.
      assert defaults.gate_floor == 0.5

      assert {:ok, :train, opts} =
               SystemOneCLI.parse([
                 "train",
                 "--cache",
                 "cache",
                 "--output",
                 "expert",
                 "--gate-floor",
                 "0.25",
                 "--lead-tokens",
                 "2",
                 "--lead-weight",
                 "5.0",
                 "--gate-mode",
                 "classifier",
                 "--init-from",
                 "artifacts/system-one/round1"
               ])

      assert opts.gate_floor == 0.25
      assert opts.lead_tokens == 2
      assert opts.lead_weight == 5.0
      assert opts.gate_mode == :classifier
      assert opts.init_from == "artifacts/system-one/round1"

      assert {:error, message} =
               SystemOneCLI.parse(["train", "--cache", "c", "--output", "e", "--gate-mode", "x"])

      assert message =~ "response or classifier"
    end

    test "regress defaults to the plan's reference ids and the base comparison" do
      assert {:ok, :regress, opts} = SystemOneCLI.parse(["regress"])

      assert opts.expected == [712, 81686, 3124, 8178, 586, 5756, 506, 5597, 2214]
      assert opts.compare_base == true
      assert opts.strict_reference == false
      assert opts.wav == "journal1.wav"
      assert opts.seconds == 5.0
      assert opts.max_new_tokens == 32
    end

    test "regress takes its own reference ids" do
      assert {:ok, :regress, opts} =
               SystemOneCLI.parse(["regress", "--expected", "1, 2,3", "--strict-reference"])

      assert opts.expected == [1, 2, 3]
      assert opts.strict_reference == true

      assert {:error, message} = SystemOneCLI.parse(["regress", "--expected", "1,two"])
      assert message =~ "--expected"
    end
  end

  describe "expert graph" do
    test "adds no nodes at all when no layer carries an expert" do
      spec = tiny_spec()

      # An artifact written before the field existed deserializes without it,
      # which has to build the same graph as an empty list.
      old_spec = without_field(spec, :system_one_layers)

      base = Model.decoder_block_chain_model(spec, [0, 1])
      old = Model.decoder_block_chain_model(old_spec, [0, 1])

      assert map_size(base.nodes) == map_size(old.nodes)
      assert node_names(base) == node_names(old)
      refute Enum.any?(node_names(base), &(&1 =~ "system_one"))
    end

    test "adds exactly the expert's nodes for a named layer" do
      base = Model.decoder_block_chain_model(tiny_spec(), [0, 1])
      expert = Model.decoder_block_chain_model(tiny_spec(system_one_layers: [1]), [0, 1])

      added = node_names(expert) -- node_names(base)

      assert Enum.filter(added, &(&1 =~ "system_one")) == [
               "decoder.blocks.1.system_one.expert.activation",
               "decoder.blocks.1.system_one.expert.gate",
               "decoder.blocks.1.system_one.expert.intermediate",
               "decoder.blocks.1.system_one.expert.output",
               "decoder.blocks.1.system_one.expert.product",
               "decoder.blocks.1.system_one.gate",
               "decoder.blocks.1.system_one.gated",
               "decoder.blocks.1.system_one.router",
               "decoder.blocks.1.system_one.sum"
             ]

      # The nine named nodes plus the containers `multiply` and `add` wrap
      # their operands in, and nothing of the block itself removed.
      assert map_size(expert.nodes) == map_size(base.nodes) + length(added)
      assert node_names(base) -- node_names(expert) == []
    end

    test "an untrained expert leaves the block output unchanged" do
      spec = tiny_spec()
      expert_spec = tiny_spec(system_one_layers: [0, 1])
      inputs = chain_inputs(spec)

      base = Model.decoder_block_chain_model(spec, [0, 1])
      {base_init, base_predict} = Axon.build(base)
      base_params = base_init.(inputs, Axon.ModelState.empty())

      expert = Model.decoder_block_chain_model(expert_spec, [0, 1])
      {expert_init, expert_predict} = Axon.build(expert)
      expert_params = expert_init.(inputs, base_params)

      # Zero `output` and a router clamped shut: exactly 0.0 is added, not
      # something that rounds to it.
      assert Nx.to_number(
               Nx.all(
                 Nx.equal(
                   base_predict.(base_params, inputs),
                   expert_predict.(expert_params, inputs)
                 )
               )
             ) == 1

      assert Nx.to_flat_list(
               expert_params.data["decoder.blocks.0.system_one.expert.output"]["kernel"]
             )
             |> Enum.all?(&(&1 == 0.0))

      assert Nx.to_flat_list(expert_params.data["decoder.blocks.0.system_one.router"]["bias"]) ==
               [SystemOne.closed_router_bias()]
    end

    test "clamps a gate under the floor to exactly zero" do
      gate = Nx.tensor([0.0, 0.02, 0.05, 0.4, 1.0])
      kept = Nx.to_flat_list(gate)

      assert Nx.to_flat_list(SystemOne.clamp_gate(gate, 0.05)) ==
               [0.0, 0.0] ++ Enum.drop(kept, 2)

      assert Nx.to_flat_list(SystemOne.clamp_gate(gate, 1.0)) == [0.0, 0.0, 0.0, 0.0, 1.0]
      assert Nx.to_flat_list(SystemOne.clamp_gate(gate, 0.0)) == kept
    end

    test "the router's gate is floor-clamped at inference and raw in training" do
      model =
        Axon.input("hidden_state", shape: {nil, nil, 4})
        |> SystemOne.router(name: "router", gate_floor: 0.05)

      inputs = %{"hidden_state" => Nx.broadcast(0.0, {1, 2, 4})}
      {init_fun, inference} = Axon.build(model)
      params = init_fun.(inputs, Axon.ModelState.empty())
      {_init_fun, training} = Axon.build(model, mode: :train)

      # sigmoid(-4) is 0.018, under the floor.
      assert Nx.to_flat_list(inference.(params, inputs)) == [0.0, 0.0]

      [first, _second] = Nx.to_flat_list(training.(params, inputs).prediction)
      assert_in_delta first, 0.01799, 1.0e-4
    end

    test "names the artifact's own parameter nodes" do
      assert SystemOne.parameter_node?("decoder.blocks.45.system_one.router")
      assert SystemOne.parameter_node?("decoder.blocks.45.system_one.expert.output")
      refute SystemOne.parameter_node?("decoder.blocks.45.self_attention.query")
      refute SystemOne.parameter_node?("output_norm")
    end
  end

  describe "expert artifact" do
    @tag :tmp_dir
    test "round trips through safetensors and a manifest", %{tmp_dir: tmp_dir} do
      spec = artifact_spec()
      artifact = %{SystemOneArtifact.new(spec, [45, 47]) | step: 400, meta: %{cache: "c"}}
      path = Path.join(tmp_dir, "expert")

      assert ^path = SystemOneArtifact.save!(artifact, path)
      assert File.exists?(Path.join(path, "parameters.safetensors"))

      manifest = SystemOneArtifact.manifest!(path)
      assert manifest.kind == :system_one_expert
      assert manifest.layers == [45, 47]
      assert manifest.expert_size == 4
      assert manifest.step == 400
      assert manifest.size.parameters == SystemOneArtifact.size(artifact).parameters

      loaded = SystemOneArtifact.load!(path)

      assert loaded.layers == artifact.layers
      assert loaded.hidden_size == artifact.hidden_size
      assert loaded.gate_floor == artifact.gate_floor
      assert loaded.step == 400
      assert loaded.meta == %{cache: "c"}
      assert Map.keys(loaded.params) == Map.keys(artifact.params)

      for {node_name, parameters} <- artifact.params, {name, tensor} <- parameters do
        assert Nx.to_flat_list(loaded.params[node_name][name]) == Nx.to_flat_list(tensor)
      end

      assert_raise ArgumentError, ~r/already exists/, fn ->
        SystemOneArtifact.save!(artifact, path)
      end

      assert SystemOneArtifact.save!(artifact, path, overwrite: true, type: {:bf, 16})

      assert Nx.type(
               SystemOneArtifact.load!(path).params["decoder.blocks.45.system_one.router"]["bias"]
             ) ==
               {:bf, 16}
    end

    @tag :tmp_dir
    test "overwriting keeps the checkpoints written under the artifact", %{tmp_dir: tmp_dir} do
      artifact = SystemOneArtifact.new(artifact_spec(), [45])
      path = Path.join(tmp_dir, "expert")

      SystemOneArtifact.save!(artifact, path)
      checkpoint = Trainer.checkpoint_path(path, 25)
      SystemOneArtifact.save!(artifact, checkpoint)

      SystemOneArtifact.save!(artifact, path, overwrite: true, keep: ["checkpoints"])

      assert Trainer.latest_checkpoint(path) == {25, checkpoint}
      assert SystemOneArtifact.manifest!(checkpoint).layers == [45]

      SystemOneArtifact.save!(artifact, path, overwrite: true)

      assert Trainer.latest_checkpoint(path) == nil
    end

    test "installs itself on a spec and forces the router closed" do
      base = tiny_spec()
      artifact = SystemOneArtifact.new(artifact_spec(), [1])

      installed = SystemOneArtifact.spec(base, artifact)
      assert SystemOne.layers(installed) == [1]
      assert SystemOne.expert_size(installed) == 4
      assert SystemOne.gate_floor(installed) == 0.05

      closed = SystemOneArtifact.spec(base, artifact, force_router_closed: true)
      assert SystemOne.gate_floor(closed) == 1.0

      # An artifact trained before the floor was raised is evaluated at the
      # new one without being retrained or rewritten.
      raised = SystemOneArtifact.spec(base, artifact, gate_floor: 0.5)
      assert SystemOne.gate_floor(raised) == 0.5

      # `nil` is "no override", not "no floor": the CLI passes the option
      # through whether or not it was given.
      assert SystemOne.gate_floor(SystemOneArtifact.spec(base, artifact, gate_floor: nil)) == 0.05
    end

    test "merges the expert into a parameter map without touching the model" do
      params = Axon.ModelState.new(%{"output_norm" => %{"scale" => Nx.tensor([1.0])}})

      merged =
        SystemOneArtifact.merge_parameters(params, %{
          "decoder.blocks.45.system_one.router" => %{"bias" => Nx.tensor([-4.0])}
        })

      assert Map.keys(merged.data) == ["decoder.blocks.45.system_one.router", "output_norm"]
      assert Nx.to_flat_list(merged.data["output_norm"]["scale"]) == [1.0]
    end
  end

  describe "trainer" do
    test "gathers logits at the positions it is given" do
      spec = tiny_spec()
      inputs = chain_inputs(spec)

      tail = Model.decoder_tail_model(spec, [0, 1])
      {tail_init, tail_predict} = Axon.build(tail)
      params = tail_init.(inputs, Axon.ModelState.empty())

      gathered =
        spec
        |> Model.decoder_block_chain_model([0, 1])
        |> Model.gathered_output_logits(Axon.input("label_positions", shape: {nil, nil}), spec)

      {gathered_init, gathered_predict} = Axon.build(gathered)
      gathered_inputs = Map.put(inputs, "label_positions", Nx.tensor([[3, 1]], type: :s64))
      gathered_params = gathered_init.(gathered_inputs, params)

      logits = gathered_predict.(gathered_params, gathered_inputs)
      assert Nx.shape(logits) == {1, 2, spec.vocab_size}

      # The tail model reads the last position, which is the first gathered one.
      # The head runs over 2 positions instead of 4, so the sums differ in the
      # last bits.
      assert Nx.to_number(
               Nx.all_close(logits[0][0], tail_predict.(params, inputs)[0],
                 atol: 1.0e-6,
                 rtol: 1.0e-6
               )
             ) == 1
    end

    test "builds fixed-shape batches with the response shifted one position back" do
      rows = [
        cache_row("a", "system_one", 6, 4, 2),
        cache_row("b", "replay", 3, 0, 0)
      ]

      {inputs, targets} =
        Trainer.batch(rows,
          sequence_length: 8,
          max_response_tokens: 3,
          hidden_size: 2
        )

      assert Nx.shape(inputs["hidden_state"]) == {2, 8, 2}
      assert Nx.shape(inputs["label_positions"]) == {2, 3}
      assert Nx.shape(targets.labels) == {2, 3}

      assert Nx.to_flat_list(inputs["attention_mask"][0]) == [1, 1, 1, 1, 1, 1, 0, 0]
      assert Nx.to_flat_list(inputs["position_ids"][0]) == [0, 1, 2, 3, 4, 5, 6, 7]

      assert Nx.to_flat_list(inputs["label_positions"][0]) == [3, 4, 5]
      assert Nx.to_flat_list(targets.labels[0]) == [104, 105, 0]
      assert Nx.to_flat_list(targets.label_mask[0]) == [1.0, 1.0, 0.0]

      # Rows padded to the bucket contribute nothing past their own tokens.
      assert Nx.to_flat_list(inputs["hidden_state"][0][6]) == [0.0, 0.0]

      # The router is pushed open from the position that predicts the first
      # response token (3, one before the response at 4) to the end of the
      # row, and shut on the prompt before it: those positions are the chat
      # template and the question, which an ordinary request has too.
      assert Nx.to_flat_list(targets.gate_open_mask[0]) == [0, 0, 0, 1, 1, 1, 0, 0]
      assert Nx.to_flat_list(targets.gate_closed_mask[0]) == [1, 1, 1, 0, 0, 0, 0, 0]

      # A replay row is shut on every real position, prompt and response
      # alike, and padding is in neither mask.
      assert Nx.to_flat_list(targets.gate_open_mask[1]) == [0, 0, 0, 0, 0, 0, 0, 0]
      assert Nx.to_flat_list(targets.gate_closed_mask[1]) == [1, 1, 1, 0, 0, 0, 0, 0]

      # The two reported means stay per kind, so `gate_mean_replay` is the
      # same number as before the masks were split.
      assert Nx.to_flat_list(targets.system_one_mask[0]) == [1, 1, 1, 1, 1, 1, 0, 0]
      assert Nx.to_flat_list(targets.replay_mask[0]) == [0, 0, 0, 0, 0, 0, 0, 0]
      assert Nx.to_flat_list(targets.system_one_mask[1]) == [0, 0, 0, 0, 0, 0, 0, 0]
      assert Nx.to_flat_list(targets.replay_mask[1]) == [1, 1, 1, 0, 0, 0, 0, 0]

      # No base logits cached, so the replay KL is masked out entirely.
      assert Nx.to_flat_list(targets.base_mask[1]) == [0.0, 0.0, 0.0]
    end

    test "opens the gate from the first position when a row is all response" do
      {_inputs, targets} =
        Trainer.batch([cache_row("a", "system_one", 3, 0, 3)],
          sequence_length: 4,
          max_response_tokens: 3,
          hidden_size: 2
        )

      # There is no prompt position before the response, so nothing is pushed
      # shut and the clamped `response_start - 1` does not wrap to the end.
      assert Nx.to_flat_list(targets.gate_open_mask[0]) == [1, 1, 1, 0]
      assert Nx.to_flat_list(targets.gate_closed_mask[0]) == [0, 0, 0, 0]
    end

    test "weights the lead response tokens of a System One row only" do
      rows = [
        cache_row("a", "system_one", 6, 4, 2),
        cache_row("b", "replay", 6, 4, 2)
      ]

      {_inputs, targets} =
        Trainer.batch(rows,
          sequence_length: 8,
          max_response_tokens: 3,
          hidden_size: 2,
          lead_tokens: 1,
          lead_weight: 3.0
        )

      assert Nx.to_flat_list(targets.label_mask[0]) == [1.0, 1.0, 0.0]
      assert Nx.to_flat_list(targets.label_weight[0]) == [3.0, 1.0, 0.0]

      # A replay row is reproducing the base model, not making a decision, so
      # its first token is worth no more than the rest.
      assert Nx.to_flat_list(targets.label_weight[1]) == [1.0, 1.0, 0.0]
    end

    test "under the classifier gate mode only an underspecified row opens the gate" do
      rows = [
        Map.put(cache_row("a-d", "system_one", 6, 4, 2), :decidable, true),
        Map.put(cache_row("a-u", "system_one", 6, 4, 2), :decidable, false),
        # An older cache has no flag, so the twin suffix stands in for it.
        cache_row("b-d", "system_one", 6, 4, 2),
        cache_row("c", "replay", 6, 4, 2)
      ]

      {_inputs, targets} =
        Trainer.batch(rows,
          sequence_length: 8,
          max_response_tokens: 3,
          hidden_size: 2,
          lead_tokens: 1,
          lead_weight: 3.0,
          gate_mode: :classifier
        )

      # A decidable item is base Gemma's to answer: shut everywhere, like a
      # replay row, and its gold reply carries no cross-entropy.
      for row <- [0, 2] do
        assert Nx.to_flat_list(targets.gate_open_mask[row]) == [0, 0, 0, 0, 0, 0, 0, 0]
        assert Nx.to_flat_list(targets.gate_closed_mask[row]) == [1, 1, 1, 1, 1, 1, 0, 0]
        assert Nx.to_flat_list(targets.label_weight[row]) == [0.0, 0.0, 0.0]
        assert Nx.to_flat_list(targets.system_one_mask[row]) == [1, 1, 1, 1, 1, 1, 0, 0]
      end

      # The underspecified twin is trained exactly as under `:response`.
      assert Nx.to_flat_list(targets.gate_open_mask[1]) == [0, 0, 0, 1, 1, 1, 0, 0]
      assert Nx.to_flat_list(targets.gate_closed_mask[1]) == [1, 1, 1, 0, 0, 0, 0, 0]
      assert Nx.to_flat_list(targets.label_weight[1]) == [3.0, 1.0, 0.0]

      assert Nx.to_flat_list(targets.gate_closed_mask[3]) == [1, 1, 1, 1, 1, 1, 0, 0]
      assert Nx.to_flat_list(targets.label_weight[3]) == [1.0, 1.0, 0.0]

      # Without the mode nothing changes for the same rows.
      {_inputs, targets} =
        Trainer.batch(rows, sequence_length: 8, max_response_tokens: 3, hidden_size: 2)

      assert Nx.to_flat_list(targets.gate_open_mask[0]) == [0, 0, 0, 1, 1, 1, 0, 0]

      assert_raise ArgumentError, ~r/needs a decidable flag/, fn ->
        Trainer.batch([cache_row("x", "system_one", 6, 4, 2)],
          sequence_length: 8,
          max_response_tokens: 3,
          hidden_size: 2,
          gate_mode: :classifier
        )
      end
    end

    test "refuses a row longer than the cache bucket" do
      assert_raise ArgumentError, ~r/longer than the cache bucket/, fn ->
        Trainer.batch([cache_row("a", "system_one", 6, 4, 2)],
          sequence_length: 4,
          max_response_tokens: 3,
          hidden_size: 2
        )
      end
    end

    test "masks the padded response slots out of every loss term" do
      targets = %{
        labels: Nx.tensor([[1, 2]], type: :s64),
        label_mask: Nx.tensor([[1.0, 0.0]]),
        label_weight: Nx.tensor([[1.0, 0.0]]),
        gate_open_mask: Nx.tensor([[1.0, 0.0]]),
        gate_closed_mask: Nx.tensor([[0.0, 1.0]]),
        system_one_mask: Nx.tensor([[1.0, 0.0]]),
        replay_mask: Nx.tensor([[0.0, 1.0]]),
        base_values: Nx.broadcast(0.0, {1, 2, 1}),
        base_ids: Nx.broadcast(0, {1, 2, 1}) |> Nx.as_type(:s64),
        base_mask: Nx.tensor([[0.0, 0.0]])
      }

      outputs = %{
        logits: Nx.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]]),
        gate_logits: %{"45" => Nx.tensor([[[-0.5], [1.0]]])}
      }

      components = Trainer.components(targets, outputs)

      assert_in_delta Nx.to_number(components.ce), 0.55144, 1.0e-4
      assert Nx.to_number(components.kl) == 0.0

      # The BCE comes off the logit: `-log sigmoid(z)` where the target is
      # open, `-log (1 - sigmoid(z))` where it is shut.
      assert_in_delta Nx.to_number(components.gate_open), -:math.log(sigmoid(-0.5)), 1.0e-5
      assert_in_delta Nx.to_number(components.gate_closed), -:math.log(1 - sigmoid(1.0)), 1.0e-5

      assert_in_delta Nx.to_number(components.gate_mean_system_one), sigmoid(-0.5), 1.0e-6
      assert_in_delta Nx.to_number(components.gate_mean_replay), sigmoid(1.0), 1.0e-6

      # Both positions are on the wrong side of a gate of 0.5 here.
      assert_in_delta Nx.to_number(components.gate_false_open), 1.0, 1.0e-6
      assert_in_delta Nx.to_number(components.gate_false_closed), 1.0, 1.0e-6

      # The masked slot's label is never read.
      masked = Trainer.components(%{targets | labels: Nx.tensor([[1, 0]], type: :s64)}, outputs)
      assert Nx.to_number(masked.ce) == Nx.to_number(components.ce)

      total = Trainer.total(components, %{kl: 1.0, gate_open: 0.5, gate_closed: 2.0})

      assert_in_delta Nx.to_number(total),
                      Nx.to_number(components.ce) + 0.5 * Nx.to_number(components.gate_open) +
                        2.0 * Nx.to_number(components.gate_closed),
                      1.0e-5
    end

    test "averages the two halves of the router BCE separately" do
      # One open-target position against three closed-target ones, which is
      # roughly the ratio a real batch has. Pooled, the open term would be
      # worth a quarter of what it is here.
      targets = %{
        labels: Nx.broadcast(0, {1, 1}) |> Nx.as_type(:s64),
        label_mask: Nx.tensor([[0.0]]),
        label_weight: Nx.tensor([[0.0]]),
        gate_open_mask: Nx.tensor([[0.0, 0.0, 0.0, 1.0]]),
        gate_closed_mask: Nx.tensor([[1.0, 1.0, 1.0, 0.0]]),
        system_one_mask: Nx.tensor([[1.0, 1.0, 1.0, 1.0]]),
        replay_mask: Nx.broadcast(0.0, {1, 4}),
        base_values: Nx.broadcast(0.0, {1, 1, 1}),
        base_ids: Nx.broadcast(0, {1, 1, 1}) |> Nx.as_type(:s64),
        base_mask: Nx.tensor([[0.0]])
      }

      logits = [-2.0, -3.0, 4.0, 2.0]

      outputs = %{
        logits: Nx.broadcast(0.0, {1, 1, 2}),
        gate_logits: %{"45" => logits |> Nx.tensor() |> Nx.reshape({1, 4, 1})}
      }

      components = Trainer.components(targets, outputs)

      assert_in_delta Nx.to_number(components.gate_open), -:math.log(sigmoid(2.0)), 1.0e-5

      closed =
        [-2.0, -3.0, 4.0]
        |> Enum.map(&(-:math.log(1 - sigmoid(&1))))
        |> Enum.sum()
        |> Kernel./(3)

      assert_in_delta Nx.to_number(components.gate_closed), closed, 1.0e-5

      # One of the three closed positions sits at a logit of 4, so a third of
      # them would fire at inference; the open one is on the right side.
      assert_in_delta Nx.to_number(components.gate_false_open), 1 / 3, 1.0e-6
      assert Nx.to_number(components.gate_false_closed) == 0.0
    end

    test "counts false opens on replay rows separately from the whole closed side" do
      # Row 0 is a System One row whose prompt prefix has one position on the
      # wrong side; row 1 is a replay row with none. The aggregate mixes the
      # two, so the replay regression needs its own number.
      targets = %{
        labels: Nx.broadcast(0, {2, 1}) |> Nx.as_type(:s64),
        label_mask: Nx.broadcast(0.0, {2, 1}),
        label_weight: Nx.broadcast(0.0, {2, 1}),
        gate_open_mask: Nx.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]),
        gate_closed_mask: Nx.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
        system_one_mask: Nx.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]),
        replay_mask: Nx.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
        base_values: Nx.broadcast(0.0, {2, 1, 1}),
        base_ids: Nx.broadcast(0, {2, 1, 1}) |> Nx.as_type(:s64),
        base_mask: Nx.broadcast(0.0, {2, 1})
      }

      logits = [[4.0, -3.0, -2.0, 2.0], [-1.0, -1.0, -1.0, -1.0]]

      outputs = %{
        logits: Nx.broadcast(0.0, {2, 1, 2}),
        gate_logits: %{"45" => logits |> Nx.tensor() |> Nx.reshape({2, 4, 1})}
      }

      components = Trainer.components(targets, outputs)

      # One wrong position out of the seven closed-target ones overall, and
      # none of the four that belong to the replay row.
      assert_in_delta Nx.to_number(components.gate_false_open), 1 / 7, 1.0e-6
      assert Nx.to_number(components.gate_false_open_replay) == 0.0
    end

    test "keeps the router BCE finite at a saturated gate" do
      # `log sigmoid(-90)` is `log 0.0` read off the sigmoid; through
      # `softplus` it is 90.
      targets = %{
        labels: Nx.broadcast(0, {1, 1}) |> Nx.as_type(:s64),
        label_mask: Nx.tensor([[0.0]]),
        label_weight: Nx.tensor([[0.0]]),
        gate_open_mask: Nx.tensor([[1.0, 0.0]]),
        gate_closed_mask: Nx.tensor([[0.0, 1.0]]),
        system_one_mask: Nx.tensor([[1.0, 0.0]]),
        replay_mask: Nx.tensor([[0.0, 1.0]]),
        base_values: Nx.broadcast(0.0, {1, 1, 1}),
        base_ids: Nx.broadcast(0, {1, 1, 1}) |> Nx.as_type(:s64),
        base_mask: Nx.tensor([[0.0]])
      }

      outputs = %{
        logits: Nx.broadcast(0.0, {1, 1, 2}),
        gate_logits: %{"45" => Nx.tensor([[[-90.0], [90.0]]])}
      }

      components = Trainer.components(targets, outputs)

      assert_in_delta Nx.to_number(components.gate_open), 90.0, 1.0e-3
      assert_in_delta Nx.to_number(components.gate_closed), 90.0, 1.0e-3
    end

    test "trains the router at its own learning rate" do
      assert Trainer.router_node?("decoder.blocks.45.system_one.router")
      refute Trainer.router_node?("decoder.blocks.45.system_one.expert.output")
    end

    test "dequantizes packed nodes into dense kernels" do
      packed = Nx.tensor([[0x01234567, 0x89ABCDEF]], type: :s32)
      scales = Nx.tensor([[2.0, 0.5]])

      params = %{
        "decoder.blocks.45.ffn.gate" => %{"packed" => packed, "scales" => scales},
        "decoder.blocks.45.self_attention.norm" => %{"scale" => Nx.tensor([1.0, 2.0])}
      }

      dequantized = Trainer.dequantize(params, group_size: 8)

      kernel = dequantized["decoder.blocks.45.ffn.gate"]["kernel"]
      assert Map.keys(dequantized["decoder.blocks.45.ffn.gate"]) == ["kernel"]

      # `{k, n}`, which is what `Axon.dense` wants of a kernel.
      assert Nx.shape(kernel) == {8, 2}
      assert Nx.to_flat_list(kernel) == Nx.to_flat_list(Q4Gemv.dequantize(packed, scales, 8))

      assert Nx.to_flat_list(dequantized["decoder.blocks.45.self_attention.norm"]["scale"]) ==
               [1.0, 2.0]
    end

    test "casts only the vocabulary projection to the head type" do
      params = %{
        "decoder.blocks.45.self_attention.norm" => %{"scale" => Nx.tensor([1.0, 2.0])},
        "language_modeling_head.output" => %{"kernel" => Nx.tensor([[1.0, 2.0]])}
      }

      dequantized = Trainer.dequantize(params, head_type: {:bf, 16})

      assert Nx.type(dequantized["language_modeling_head.output"]["kernel"]) == {:bf, 16}
      assert Nx.type(dequantized["decoder.blocks.45.self_attention.norm"]["scale"]) == {:f, 32}
    end

    test "initialises the vocabulary projection at the head type" do
      spec = Trainer.training_spec(tiny_spec(), [1])
      inputs = Map.put(chain_inputs(spec), "label_positions", Nx.tensor([[3, 1]], type: :s64))

      {init, _predict} = Axon.build(Trainer.model(spec, [1], head_type: {:bf, 16}))
      params = init.(inputs, Axon.ModelState.empty()).data

      # A layer casts its parameters to its own policy, so the head keeps a
      # bf16 kernel only because its node carries that policy too.
      assert Nx.type(params["language_modeling_head.output"]["kernel"]) == {:bf, 16}
      assert Nx.type(params["decoder.blocks.1.self_attention.query"]["kernel"]) == {:f, 32}
    end

    test "the memory estimate follows the head type" do
      spec = tiny_spec()
      bf16 = Trainer.memory_estimate(spec, [1], 8, 256, 48, head_type: {:bf, 16})
      f32 = Trainer.memory_estimate(spec, [1], 8, 256, 48, head_type: {:f, 32})

      assert bf16.head_bytes < f32.head_bytes
      assert bf16.frozen_bytes == f32.frozen_bytes
      assert bf16.total_bytes < f32.total_bytes

      # Two buffers, because the step hands the head both of the layouts it
      # uses and differentiates nothing through it.
      assert f32.head_bytes == 2 * 4 * spec.vocab_size * spec.hidden_size
    end

    test "supplies the dropout keys a train-mode build needs" do
      spec = Trainer.training_spec(tiny_spec(), [1])

      inputs =
        spec |> chain_inputs() |> Map.put("label_positions", Nx.tensor([[1, 2]], type: :s64))

      model = Trainer.hidden_model(spec, [1])
      {init, predict} = Axon.build(model, mode: :train)

      keys = Trainer.dropout_state(model, 42)
      assert map_size(keys) > 0

      # The tail artifact holds weights only, so the trainer starts from a
      # parameter map without any dropout state in it.
      weights_only =
        init.(inputs, Axon.ModelState.empty()).data
        |> Map.drop(Map.keys(keys))

      state = %Axon.ModelState{
        data: Map.merge(weights_only, keys),
        parameters: %{},
        state: %{},
        frozen_parameters: %{}
      }

      assert %{prediction: %{hidden: hidden, gate_logits: gate_logits}} = predict.(state, inputs)
      assert Nx.shape(hidden) == {1, 2, spec.hidden_size}

      # The second output is the router's dense node, before the sigmoid: a
      # freshly initialised router is at the closed bias everywhere.
      logit = Map.fetch!(gate_logits, "1")
      assert Nx.shape(logit) == {1, 4, 1}

      assert Nx.to_flat_list(logit) ==
               List.duplicate(SystemOne.closed_router_bias(), 4)
    end

    test "a newly trained artifact records the run's gate floor" do
      defaults = Trainer.defaults()

      # The tail spec carries round 1's 0.05; a new run overrides it, so the
      # artifact it writes is loaded at the floor its router was trained for.
      assert SystemOne.gate_floor(artifact_spec()) == 0.05

      spec = Trainer.training_spec(artifact_spec(), [1])
      assert SystemOne.gate_floor(spec) == defaults.gate_floor
      assert SystemOneArtifact.new(spec, [1]).gate_floor == defaults.gate_floor

      assert SystemOne.gate_floor(Trainer.training_spec(artifact_spec(), [1], gate_floor: 0.2)) ==
               0.2
    end

    test "splits the step without changing the gradient" do
      spec = Trainer.training_spec(tiny_spec(final_logit_softcapping: 30.0), [1])
      weights = %{kl: 1.0, gate_open: 0.05, gate_closed: 0.5}

      inputs =
        spec |> chain_inputs() |> Map.put("label_positions", Nx.tensor([[1, 2]], type: :s64))

      targets = %{
        labels: Nx.tensor([[3, 5]], type: :s64),
        label_mask: Nx.tensor([[1.0, 1.0]]),
        label_weight: Nx.tensor([[3.0, 1.0]]),
        gate_open_mask: Nx.tensor([[0.0, 1.0, 1.0, 0.0]]),
        gate_closed_mask: Nx.tensor([[1.0, 0.0, 0.0, 0.0]]),
        system_one_mask: Nx.tensor([[1.0, 1.0, 1.0, 0.0]]),
        replay_mask: Nx.tensor([[0.0, 0.0, 0.0, 0.0]]),
        base_values: Nx.tensor([[[0.5, -0.25], [1.0, 0.75]]]),
        base_ids: Nx.tensor([[[3, 7], [5, 2]]], type: :s64),
        base_mask: Nx.tensor([[1.0, 1.0]])
      }

      {init, predict} = Axon.build(Trainer.model(spec, [1]), mode: :train)
      model_state = init.(inputs, Axon.ModelState.empty())
      data = opened_expert(model_state.data)

      {expert, frozen} =
        Map.split_with(data, fn {name, _parameters} -> SystemOne.parameter_node?(name) end)

      # Frozen tensors are arguments of the jitted function, never closures,
      # which is also what the trainer does with them.
      monolithic =
        Nx.Defn.jit(fn expert, frozen, inputs, targets ->
          Nx.Defn.value_and_grad(expert, fn expert ->
            %{prediction: outputs} =
              predict.(%{model_state | data: Map.merge(frozen, expert)}, inputs)

            targets |> Trainer.components(outputs) |> Trainer.total(weights)
          end)
        end)

      {value, reference} = monolithic.(expert, frozen, inputs, targets)

      kernel = frozen["language_modeling_head.output"]["kernel"]

      head = %{
        weight: frozen["output_norm"]["weight"],
        kernel: kernel,
        kernel_t: Nx.transpose(kernel)
      }

      tail = Map.drop(frozen, ["output_norm", "language_modeling_head.output"])
      {_init, hidden_predict} = Axon.build(Trainer.hidden_model(spec, [1]), mode: :train)

      forward =
        Nx.Defn.jit(&Trainer.tail_forward(hidden_predict, &1, &2, &3, &4))

      head_step =
        Nx.Defn.jit(
          &Trainer.head_terms(&1, &2, &3,
            epsilon: spec.layer_norm_epsilon,
            softcapping: spec.final_logit_softcapping,
            kl_weight: weights.kl
          )
        )

      tail_step =
        Nx.Defn.jit(&Trainer.tail_gradients(hidden_predict, &1, &2, &3, &4, &5, weights))

      {hidden, gates} = forward.(tail, expert, inputs, targets)
      {terms, cotangent} = head_step.(head, hidden, targets)
      {_surrogate, gradients} = tail_step.(tail, expert, inputs, targets, cotangent)

      assert_close(Trainer.total(Map.merge(terms, gates), weights), value, "loss")

      # Every expert and router tensor, so a wrong cotangent cannot hide in one
      # of them; the zero-initialised output kernel is opened first, otherwise
      # the gradients above it are all zero and match trivially.
      assert map_size(gradients) == map_size(reference)

      largest =
        for {_name, parameters} <- reference, {_key, tensor} <- parameters do
          tensor |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
        end

      assert Enum.max(largest) > 1.0e-3, "the reference gradients are all zero"

      for {name, parameters} <- reference, {key, expected} <- parameters do
        assert_close(gradients[name][key], expected, name <> "." <> key)
      end
    end

    @tag :tmp_dir
    test "resumes from the highest numbered checkpoint", %{tmp_dir: tmp_dir} do
      assert Trainer.latest_checkpoint(tmp_dir) == nil

      for step <- [200, 1400, 600] do
        File.mkdir_p!(Trainer.checkpoint_path(tmp_dir, step))
      end

      File.mkdir_p!(Path.join([tmp_dir, "checkpoints", "scratch"]))

      assert {1400, path} = Trainer.latest_checkpoint(tmp_dir)
      assert Path.basename(path) == "step-001400"
      assert path == Trainer.checkpoint_path(tmp_dir, 1400)
    end

    test "plans the run and its memory without touching a device" do
      spec = Trainer.training_spec(tiny_spec(), [0, 1])
      cache = %{rows: Enum.map(1..10, &%{id: "r#{&1}"}), sequence_length: 8}

      plan = Trainer.plan(cache, spec, [0, 1], 4, 3, epochs: 2)

      assert plan.batches_per_epoch == 2
      assert plan.dropped_rows == 2
      assert plan.steps == 4
      assert plan.sequence_length == 8

      # Two layers of 3 * hidden * expert_size plus a router of hidden + 1.
      assert plan.parameters.expert == 2 * (3 * 8 * 4 + 9)
      assert plan.memory.expert_bytes == plan.parameters.expert * 20
      assert plan.memory.total_bytes > plan.memory.expert_bytes

      assert Trainer.plan(cache, spec, [0, 1], 4, 3, epochs: 2, max_steps: 3).steps == 3
    end

    test "shuffles deterministically and differently per epoch" do
      assert Trainer.shuffle(8, 42, 0) == Trainer.shuffle(8, 42, 0)
      assert Enum.sort(Trainer.shuffle(8, 42, 0)) == Enum.to_list(0..7)
      refute Trainer.shuffle(8, 42, 0) == Trainer.shuffle(8, 42, 1)
      refute Trainer.shuffle(8, 42, 0) == Trainer.shuffle(8, 7, 0)
    end
  end

  defp tiny_spec(overrides \\ []) do
    Model
    |> Bumblebee.configure(
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
      layer_types: [:sliding_attention, :full_attention],
      attention_window_size: 4,
      audio_embed_dim: 4,
      audio_token_id: 7,
      boa_token_id: 6,
      eoa_token_id: 8,
      final_logit_softcapping: nil
    )
    |> Bumblebee.configure(Keyword.merge([system_one_expert_size: 4], overrides))
  end

  defp artifact_spec do
    %{
      hidden_size: 8,
      system_one_expert_size: 4,
      system_one_gate_floor: 0.05,
      activation: :gelu_approx_tanh
    }
  end

  defp chain_inputs(spec) do
    %{
      "hidden_state" => Nx.iota({1, 4, spec.hidden_size}, type: :f32) |> Nx.divide(32),
      "position_ids" => Nx.tensor([[0, 1, 2, 3]], type: :s64),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1]], type: :s64)
    }
  end

  defp sigmoid(x), do: 1 / (1 + :math.exp(-x))

  defp cache_row(id, kind, length, response_start, response_length) do
    %{
      id: id,
      kind: kind,
      length: length,
      response_start: response_start,
      response_length: response_length,
      hidden_state: Nx.broadcast(1.0, {length, 2}),
      input_ids: Nx.tensor(Enum.map(0..(length - 1), &(100 + &1)), type: :s64),
      base_values: nil,
      base_ids: nil
    }
  end

  # A freshly initialised expert adds exactly nothing, and its zero output
  # kernel leaves every gradient above it at zero, which any split would
  # match. This opens it up so the comparison means something.
  defp opened_expert(data) do
    Map.new(data, fn {name, parameters} ->
      if SystemOne.parameter_node?(name) do
        {name, Map.new(parameters, fn {key, tensor} -> {key, open(tensor)} end)}
      else
        {name, parameters}
      end
    end)
  end

  defp open(tensor) do
    tensor
    |> Nx.shape()
    |> Nx.iota(type: Nx.type(tensor))
    |> Nx.divide(37)
    |> Nx.add(0.03)
    |> Nx.add(tensor)
  end

  defp assert_close(actual, expected, label) do
    difference =
      actual |> Nx.subtract(expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    assert difference < 1.0e-5, "#{label} differs by #{difference}"
  end

  # A spec deserialized from a manifest written before the field existed.
  defp without_field(spec, key), do: Map.delete(spec, key)

  defp node_names(%Axon{nodes: nodes}) do
    nodes |> Map.values() |> Enum.map(&node_name/1) |> Enum.sort()
  end

  defp node_name(%Axon.Node{name: name, op_name: op_name}) when is_function(name, 2),
    do: name.(op_name, %{})

  defp node_name(%Axon.Node{name: name}), do: name
end
