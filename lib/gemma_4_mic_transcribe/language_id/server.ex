defmodule Gemma4MicTranscribe.LanguageId.Server do
  @moduledoc """
  A dependency-free HTTP front for a saved detector, so the Docker image can
  run as a Hugging Face Space (port 7860) or anywhere else.

  Routes:

    * `GET /` - a page that uploads a clip from the browser
    * `GET /health` - `{"status":"ok"}` once the detector is loaded
    * `POST /detect` - the request body is an audio file in any format ffmpeg
      reads; the response lists every language with its probability, best
      first, plus the detector's window and the time spent. `?languages=de,en`
      restricts the answer to those languages (renormalised), as does the
      `:candidates` option given to the server for every request.

  The server is plain `:gen_tcp` in `:http_bin` packet mode: one process per
  connection, the request body bounded by `:max_body` (default 16 MB).
  """

  alias Gemma4MicTranscribe.LanguageId.Artifact
  alias Gemma4MicTranscribe.LanguageId.Corpus

  @max_body 16 * 1024 * 1024

  @doc """
  Listens on `port` and serves `artifact` with `runtime` until the calling
  process exits. Returns the listening socket.
  """
  def listen!(artifact, runtime, port, opts \\ []) do
    max_body = Keyword.get(opts, :max_body, @max_body)

    {:ok, socket} =
      :gen_tcp.listen(port, [:binary, packet: :http_bin, active: false, reuseaddr: true, backlog: 64])

    state = %{artifact: artifact, runtime: runtime, max_body: max_body, candidates: Keyword.get(opts, :candidates)}
    parent = self()
    spawn_link(fn -> accept_loop(socket, state, parent) end)
    socket
  end

  @doc "Serves forever; used by the `serve` command."
  def run!(artifact, runtime, port, opts \\ []) do
    listen!(artifact, runtime, port, opts)
    Process.sleep(:infinity)
  end

  defp accept_loop(socket, state, parent) do
    case :gen_tcp.accept(socket) do
      {:ok, client} ->
        pid = spawn(fn -> serve(client, state) end)
        :gen_tcp.controlling_process(client, pid)
        send(pid, :go)
        accept_loop(socket, state, parent)

      {:error, :closed} ->
        :ok
    end
  end

  defp serve(client, state) do
    receive do
      :go -> :ok
    end

    response =
      case read_request(client, state.max_body) do
        {:ok, method, path, body} -> handle(method, path, body, state)
        {:error, reason} -> {400, "text/plain", "bad request: #{inspect(reason)}\n"}
      end

    :gen_tcp.send(client, encode(response))
    :gen_tcp.close(client)
  end

  defp read_request(client, max_body) do
    with {:ok, {:http_request, method, {:abs_path, path}, _version}} <- :gen_tcp.recv(client, 0, 10_000),
         {:ok, headers} <- read_headers(client, %{}),
         length = Map.get(headers, "content-length", 0),
         :ok <- if(length <= max_body, do: :ok, else: {:error, :body_too_large}),
         {:ok, body} <- read_body(client, length) do
      {:ok, method, path, body}
    else
      {:ok, other} -> {:error, other}
      {:error, reason} -> {:error, reason}
    end
  end

  defp read_headers(client, headers) do
    case :gen_tcp.recv(client, 0, 10_000) do
      {:ok, {:http_header, _, name, _, value}} ->
        key = name |> to_string() |> String.downcase()

        value =
          case key do
            "content-length" -> String.to_integer(value)
            _other -> value
          end

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

  @doc false
  def handle(method, path, body, state) do
    {route, query} =
      case String.split(path, "?", parts: 2) do
        [route] -> {route, %{}}
        [route, query] -> {route, URI.decode_query(query)}
      end

    route(method, route, query, body, state)
  end

  defp route(:GET, "/health", _query, _body, _state), do: {200, "application/json", Jason.encode!(%{status: "ok"})}

  defp route(:GET, "/", _query, _body, state), do: {200, "text/html; charset=utf-8", page(state.artifact)}

  defp route(:POST, "/detect", _query, "", _state),
    do: {400, "application/json", Jason.encode!(%{error: "send the audio file as the request body"})}

  defp route(:POST, "/detect", query, body, %{artifact: artifact, runtime: runtime} = state) do
    file = Path.join(System.tmp_dir!(), "language-id-serve-#{System.unique_integer([:positive])}")
    File.write!(file, body)

    candidates =
      case query do
        %{"languages" => list} -> String.split(list, ",", trim: true)
        _none -> Map.get(state, :candidates)
      end

    try do
      samples = Corpus.decode!(file, artifact.seconds)
      started = System.monotonic_time(:millisecond)
      ranked = Artifact.detect(artifact, runtime, samples, candidates: candidates)

      {200, "application/json",
       Jason.encode!(%{
         languages: ranked,
         seconds: artifact.seconds,
         ms: System.monotonic_time(:millisecond) - started
       })}
    rescue
      error -> {422, "application/json", Jason.encode!(%{error: Exception.message(error)})}
    after
      File.rm(file)
    end
  end

  defp route(_method, _path, _query, _body, _state), do: {404, "text/plain", "not found\n"}

  @doc false
  def encode({status, type, body}) do
    reason =
      case status do
        200 -> "OK"
        400 -> "Bad Request"
        404 -> "Not Found"
        422 -> "Unprocessable Content"
        _other -> "Error"
      end

    [
      "HTTP/1.1 #{status} #{reason}\r\n",
      "content-type: #{type}\r\n",
      "content-length: #{byte_size(body)}\r\n",
      "connection: close\r\n\r\n",
      body
    ]
  end

  defp page(artifact) do
    """
    <!doctype html>
    <meta charset="utf-8">
    <title>Gemma 4 spoken language detector</title>
    <style>
      body { font: 15px/1.5 system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem; }
      pre { background: #f4f4f4; padding: 1rem; overflow-x: auto; }
      table { border-collapse: collapse; } td { padding: 0.15rem 1rem 0.15rem 0; }
    </style>
    <h1>Gemma 4 spoken language detector</h1>
    <p>The first #{artifact.depth} conformer blocks of the Gemma 4 E2B audio tower with a
    #{length(artifact.languages)}-way head, listening to a #{artifact.seconds} s window that starts at the
    first sound in the clip.</p>
    <p><input id="file" type="file" accept="audio/*"> <button id="go">Detect</button></p>
    <div id="out"></div>
    <p>From anywhere else:</p>
    <pre>curl --data-binary @clip.mp3 $URL/detect</pre>
    <pre>curl --data-binary @clip.mp3 "$URL/detect?languages=de,en,fr"</pre>
    <p>Languages: #{Enum.join(artifact.languages, ", ")}</p>
    <script>
      document.getElementById("go").onclick = async () => {
        const file = document.getElementById("file").files[0];
        const out = document.getElementById("out");
        if (!file) { out.textContent = "pick a clip first"; return; }
        out.textContent = "detecting...";
        const response = await fetch("detect", { method: "POST", body: file });
        const result = await response.json();
        if (!response.ok) { out.textContent = result.error; return; }
        out.innerHTML = "<table>" + result.languages.slice(0, 5).map(r =>
          `<tr><td>${r.language}</td><td>${(r.probability * 100).toFixed(1)}%</td></tr>`).join("") +
          `</table><p>${result.ms} ms</p>`;
      };
    </script>
    """
  end
end
