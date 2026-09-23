defmodule Gemma4MicTranscribe.Gemma4.SystemOne.Server do
  @moduledoc """
  A dependency-free HTTP front for the router (`mix gemma.system_one serve`)
  that streams each reply as Gemma writes it.

  Routes:

    * `GET /` - a page to try requests from the browser
    * `GET /health` - `{"status":"ok"}` once the model is loaded
    * `POST /route` - the body is one request as `route --input` takes it (a
      `prompt`, or a System One item), with a spoken request's WAV as base64
      in `audio_wav` instead of a path in `audio`. The response is chunked
      NDJSON: a `route` event once the route is chosen (`answer`, `ask` or
      `reason`, with `ask_score` and `confidence`), `text` events that each
      carry the reply's next piece, and a `done` event with the finished row
      as `route` writes it, whose `reply` is authoritative. A request that
      fails after the stream started ends with an `error` event instead.

  Requests run one at a time on a single worker, since they share the GPU;
  the others wait in its queue. The server is plain `:gen_tcp` in `:http_bin`
  packet mode, as `LanguageId.Server` is: one process per connection, the
  body bounded by `:max_body` (default 16 MB).
  """

  @max_body 16 * 1024 * 1024

  @doc """
  Listens on `port` (0 picks a free one) and answers `POST /route` with
  `route.(request, emit)`, which calls `emit` with each event map as the
  reply is made and returns the finished row. Returns the listening socket.
  """
  def listen!(route, port, opts \\ []) when is_function(route, 2) do
    {:ok, socket} =
      :gen_tcp.listen(port, [
        :binary,
        packet: :http_bin,
        active: false,
        reuseaddr: true,
        backlog: 64
      ])

    worker = spawn_link(fn -> work(route) end)
    state = %{worker: worker, max_body: Keyword.get(opts, :max_body, @max_body)}
    spawn_link(fn -> accept_loop(socket, state) end)
    socket
  end

  @doc "Serves forever; used by the `serve` command."
  def run!(route, port, opts \\ []) do
    listen!(route, port, opts)
    Process.sleep(:infinity)
  end

  # The GPU's single queue. EXLA frees device buffers when the process that
  # holds them collects them, so the worker collects after every request.
  defp work(route) do
    receive do
      {:route, request, from, ref} ->
        emit = fn event -> send(from, {ref, :event, event}) end

        try do
          send(from, {ref, :done, route.(request, emit)})
        rescue
          error -> send(from, {ref, :error, Exception.message(error)})
        end

        :erlang.garbage_collect()
        work(route)
    end
  end

  defp accept_loop(socket, state) do
    case :gen_tcp.accept(socket) do
      {:ok, client} ->
        pid = spawn(fn -> serve(client, state) end)
        :gen_tcp.controlling_process(client, pid)
        send(pid, :go)
        accept_loop(socket, state)

      {:error, :closed} ->
        :ok
    end
  end

  defp serve(client, state) do
    receive do
      :go -> :ok
    end

    case read_request(client, state.max_body) do
      {:ok, method, path, body} -> handle(client, method, path, body, state)
      {:error, reason} -> reply(client, 400, "text/plain", "bad request: #{inspect(reason)}\n")
    end

    :gen_tcp.close(client)
  end

  defp handle(client, :GET, "/health", _body, _state),
    do: reply(client, 200, "application/json", Jason.encode!(%{status: "ok"}))

  defp handle(client, :GET, "/", _body, _state),
    do: reply(client, 200, "text/html; charset=utf-8", page())

  defp handle(client, :POST, "/route", body, state) do
    case request(body) do
      {:ok, request, wav} ->
        try do
          stream(client, request, state.worker)
        after
          if wav, do: File.rm(wav)
        end

      {:error, message} ->
        reply(client, 400, "application/json", Jason.encode!(%{error: message}))
    end
  end

  defp handle(client, _method, _path, _body, _state),
    do: reply(client, 404, "text/plain", "not found\n")

  # A spoken request's WAV goes to a temporary file that stands in for the
  # `audio` path `route` reads. A path sent by the client is dropped, so a
  # request cannot read files on the server.
  defp request(body) do
    case Jason.decode(body) do
      {:ok, %{"audio_wav" => wav} = request} when is_binary(wav) ->
        case Base.decode64(wav, ignore: :whitespace) do
          {:ok, bytes} ->
            path =
              Path.join(System.tmp_dir!(), "system-one-#{System.unique_integer([:positive])}.wav")

            File.write!(path, bytes)
            {:ok, request |> Map.delete("audio_wav") |> Map.put("audio", path), path}

          :error ->
            {:error, "audio_wav must be base64"}
        end

      {:ok, %{} = request} ->
        {:ok, Map.delete(request, "audio"), nil}

      {:ok, _other} ->
        {:error, "the body must be one JSON object"}

      {:error, error} ->
        {:error, "the body must be JSON: #{Exception.message(error)}"}
    end
  end

  defp stream(client, request, worker) do
    ref = Process.monitor(worker)
    send(worker, {:route, request, self(), ref})

    :gen_tcp.send(client, [
      "HTTP/1.1 200 OK\r\n",
      "content-type: application/x-ndjson\r\n",
      "cache-control: no-cache\r\n",
      "transfer-encoding: chunked\r\n",
      "connection: close\r\n\r\n"
    ])

    relay(client, ref)
    :gen_tcp.send(client, "0\r\n\r\n")
  end

  # A client that hangs up only stops the relay; the worker still finishes
  # the request.
  defp relay(client, ref) do
    receive do
      {^ref, :event, event} ->
        if chunk(client, event) == :ok, do: relay(client, ref)

      {^ref, :done, row} ->
        Process.demonitor(ref, [:flush])
        chunk(client, Map.merge(Map.delete(row, "audio"), %{"event" => "done"}))

      {^ref, :error, message} ->
        Process.demonitor(ref, [:flush])
        chunk(client, %{event: "error", error: message})

      {:DOWN, ^ref, :process, _pid, reason} ->
        chunk(client, %{event: "error", error: "the worker stopped: #{inspect(reason)}"})
    end
  end

  defp chunk(client, event) do
    line = Jason.encode!(event) <> "\n"
    :gen_tcp.send(client, [Integer.to_string(byte_size(line), 16), "\r\n", line, "\r\n"])
  end

  defp reply(client, status, type, body) do
    reason =
      case status do
        200 -> "OK"
        400 -> "Bad Request"
        404 -> "Not Found"
        _other -> "Error"
      end

    :gen_tcp.send(client, [
      "HTTP/1.1 #{status} #{reason}\r\n",
      "content-type: #{type}\r\n",
      "content-length: #{byte_size(body)}\r\n",
      "connection: close\r\n\r\n",
      body
    ])
  end

  defp read_request(client, max_body) do
    with {:ok, {:http_request, method, {:abs_path, path}, _version}} <-
           :gen_tcp.recv(client, 0, 10_000),
         {:ok, headers} <- read_headers(client, %{}),
         length = Map.get(headers, "content-length", 0),
         :ok <- if(length <= max_body, do: :ok, else: {:error, :body_too_large}),
         {:ok, body} <- read_body(client, length) do
      {:ok, method, path |> String.split("?") |> hd(), body}
    else
      {:ok, other} -> {:error, other}
      {:error, reason} -> {:error, reason}
    end
  end

  defp read_headers(client, headers) do
    case :gen_tcp.recv(client, 0, 10_000) do
      {:ok, {:http_header, _, name, _, value}} ->
        key = name |> to_string() |> String.downcase()
        value = if key == "content-length", do: String.to_integer(value), else: value
        read_headers(client, Map.put(headers, key, value))

      {:ok, :http_eoh} ->
        {:ok, headers}

      {:ok, other} ->
        {:error, other}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp read_body(_client, 0), do: {:ok, ""}

  defp read_body(client, length) do
    :inet.setopts(client, packet: :raw)
    :gen_tcp.recv(client, length, 60_000)
  end

  defp page do
    """
    <!doctype html>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Gemma 4 System One router</title>
    <style>
      body { font: 15px/1.5 system-ui, sans-serif; max-width: 42rem; margin: 3rem auto; padding: 0 1rem; }
      textarea { width: 100%; box-sizing: border-box; font: inherit; }
      pre { background: #f4f4f4; padding: 1rem; white-space: pre-wrap; }
      #route { font-weight: 600; }
    </style>
    <h1>Gemma 4 System One router</h1>
    <p>Unmodified Gemma 4 12B with a router in front that answers now, reasons first or
    asks back. The reply streams as it is written.</p>
    <p><textarea id="prompt" rows="4">What is 17 * 23?</textarea></p>
    <p><button id="go">Route</button> <span id="route"></span></p>
    <pre id="reply"></pre>
    <p id="timing"></p>
    <p>From anywhere else:</p>
    <pre>curl -N -d '{"prompt": "What is 17 * 23?", "answer": "number"}' $URL/route</pre>
    <script>
      document.getElementById("go").onclick = async () => {
        const [route, reply, timing] = ["route", "reply", "timing"].map(id => document.getElementById(id));
        route.textContent = "routing...";
        reply.textContent = timing.textContent = "";
        const started = performance.now();
        let first = null;
        const response = await fetch("route", { method: "POST",
          body: JSON.stringify({ prompt: document.getElementById("prompt").value }) });
        const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
        let buffer = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += value;
          const lines = buffer.split("\\n");
          buffer = lines.pop();
          for (const line of lines.filter(Boolean)) {
            const event = JSON.parse(line);
            if (event.event === "route") route.textContent = event.route;
            if (event.event === "text") {
              first ??= performance.now() - started;
              reply.textContent += event.text;
            }
            if (event.event === "done") {
              reply.textContent = event.reply ?? "";
              timing.textContent = `first text ${Math.round(first ?? 0)} ms, done ${Math.round(performance.now() - started)} ms`;
            }
            if (event.event === "error") route.textContent = "error: " + event.error;
          }
        }
      };
    </script>
    """
  end
end
