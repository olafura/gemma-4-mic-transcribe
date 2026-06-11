defmodule Gemma4MicTranscribe.Gemma4Unified.ChannelState do
  @moduledoc false

  def initial, do: :before_content
  def content, do: :content

  def advance(:before_content, token_id, %{start: token_id}) when is_integer(token_id),
    do: :inside_channel

  def advance(:inside_channel, token_id, %{end: token_id}) when is_integer(token_id),
    do: :content

  def advance(:inside_channel, _token_id, _channel_token_ids), do: :inside_channel

  def advance(_state, _token_id, _channel_token_ids), do: :content
end
