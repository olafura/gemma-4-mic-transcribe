defmodule Gemma4MicTranscribe.Gemma4.SystemOne.ServerTest do
  use ExUnit.Case, async: true

  alias Gemma4MicTranscribe.Gemma4.SystemOne.Server
  alias Gemma4MicTranscribe.SystemOneCLI

  setup do
    parent = self()

    # Stands in for the router: reports what it was given and streams a
    # canned reply, or fails on request.
    route = fn request, emit ->
      audio = request["audio"]
      send(parent, {:routed, request, audio && File.read!(audio)})

      if request["prompt"] == "fail", do: raise("no bucket fits")

      emit.(%{event: "route", route: "ask", ask_score: 0.93})
      emit.(%{event: "text", text: "Which"})
      emit.(%{event: "text", text: " one?"})
      Map.merge(request, %{"route" => "ask", "reply" => "Which one?"})
    end

    socket = Server.listen!(route, 0)
    {:ok, port} = :inet.port(socket)
    on_exit(fn -> :gen_tcp.close(socket) end)
    %{url: "http://127.0.0.1:#{port}"}
  end

  defp post(url, body) do
    {:ok, {{_, status, _}, headers, body}} =
      :httpc.request(:post, {~c"#{url}/route", [], ~c"application/json", body}, [], [])

    {status, headers, body |> to_string() |> String.split("\n", trim: true)}
  end

  test "streams the route, the reply's pieces and the finished row", %{url: url} do
    {200, headers, lines} = post(url, Jason.encode!(%{"id" => "q1", "prompt" => "Pick one"}))

    assert {~c"content-type", ~c"application/x-ndjson"} in headers

    assert [
             %{"event" => "route", "route" => "ask", "ask_score" => 0.93},
             %{"event" => "text", "text" => "Which"},
             %{"event" => "text", "text" => " one?"},
             %{"event" => "done", "id" => "q1", "reply" => "Which one?"}
           ] = Enum.map(lines, &Jason.decode!/1)

    assert_received {:routed, %{"id" => "q1", "prompt" => "Pick one"}, nil}
  end

  test "hands a spoken request's WAV over as a file and removes it after", %{url: url} do
    wav = "RIFF....WAVEfmt fake"
    body = Jason.encode!(%{"prompt" => "Say it", "audio_wav" => Base.encode64(wav)})
    {200, _headers, lines} = post(url, body)

    assert_received {:routed, request, ^wav}
    refute Map.has_key?(request, "audio_wav")
    refute File.exists?(request["audio"])
    refute lines |> List.last() |> Jason.decode!() |> Map.has_key?("audio")
  end

  test "drops an audio path sent by the client", %{url: url} do
    {200, _headers, _lines} =
      post(url, Jason.encode!(%{"prompt" => "x", "audio" => "/etc/passwd"}))

    assert_received {:routed, request, nil}
    refute Map.has_key?(request, "audio")
  end

  test "ends the stream with an error and keeps serving", %{url: url} do
    {200, _headers, lines} = post(url, Jason.encode!(%{"prompt" => "fail"}))

    assert [%{"event" => "error", "error" => "no bucket fits"}] =
             Enum.map(lines, &Jason.decode!/1)

    {200, _headers, lines} = post(url, Jason.encode!(%{"prompt" => "again"}))
    assert %{"event" => "done"} = lines |> List.last() |> Jason.decode!()
  end

  test "rejects a body that is not a JSON object", %{url: url} do
    {400, _headers, [line]} = post(url, "[1, 2]")
    assert %{"error" => "the body must be one JSON object"} = Jason.decode!(line)
    {400, _headers, _lines} = post(url, "not json")
  end

  test "answers health checks and serves the page", %{url: url} do
    {:ok, {{_, 200, _}, _, body}} = :httpc.request(~c"#{url}/health")
    assert Jason.decode!(to_string(body)) == %{"status" => "ok"}

    {:ok, {{_, 200, _}, _, body}} = :httpc.request(~c"#{url}/")
    assert to_string(body) =~ "System One router"
  end

  test "serve takes route's options and a port" do
    assert {:ok, :serve, opts} =
             SystemOneCLI.parse(["serve", "--port", "8080", "--confidence", "0.95"])

    assert opts.port == 8080
    assert opts.confidence == 0.95
    assert opts.stop_early == true
    assert opts.decide_only == false
    assert opts.buckets == [256, 384, 512]

    assert {:ok, :serve, %{port: 7860}} = SystemOneCLI.parse(["serve"])
    assert {:error, _message} = SystemOneCLI.parse(["serve", "--input", "in.jsonl"])
  end
end
