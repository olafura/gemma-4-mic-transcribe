# Spoken language detector from the Gemma 4 E2B audio tower, packaged the way
# a Hugging Face Docker Space expects (uid 1000, port 7860) so the same image
# runs as a Space, as a `hf jobs run` validation job with the Common Voice
# bucket mounted at /data, or on any Docker host.
#
#   docker build -t gemma-language-id .
#   docker run --rm -p 7860:7860 gemma-language-id
#   docker run --rm -v "$PWD/common_voice:/data/common_voice:ro" gemma-language-id \
#     language_id validate --artifact artifacts/language-id/ft-depth5-1s-e2 --per-language 30
#
# Only the language-ID slice of the project is compiled (MIX_TARGET=language_id
# in mix.exs): the vendored EXLA/XLA build, Boombox and WebRTC stay out, and
# inference runs on Torchx CPU.

ARG ELIXIR_IMAGE=hexpm/elixir:1.20.2-erlang-29.0.3-debian-bookworm-20260713-slim

FROM ${ELIXIR_IMAGE} AS build

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake git curl ca-certificates unzip && \
    rm -rf /var/lib/apt/lists/*

ENV MIX_ENV=prod \
    MIX_TARGET=language_id \
    GEMMA4_ESCRIPT=language_id \
    LIBTORCH_TARGET=cpu

WORKDIR /app

RUN mix local.hex --force && mix local.rebar --force

# Dependencies first so source edits do not repeat the libtorch download and
# the NIF builds (torchx, explorer and tokenizers are all fetched or compiled
# here).
COPY mix.exs mix.lock ./
COPY vendor/exla/mix.exs vendor/exla/mix.exs
COPY vendor/xla/mix.exs vendor/xla/mix.exs
COPY vendor/ex_libsrt/mix.exs vendor/ex_libsrt/mix.exs
RUN mix deps.get && mix deps.compile

COPY config config
COPY lib lib
RUN mix compile --warnings-as-errors && mix escript.build

# The torchx NIF finds libtorch through $ORIGIN/libtorch, which the build
# leaves as a symlink into the download cache; replace it with the shared
# libraries themselves so the runtime stage only copies the build tree.
RUN cd _build/language_id_prod/lib/torchx/priv && rm libtorch && mkdir libtorch && \
    cp /app/deps/torchx/cache/libtorch-*-cpu/lib/libtorch.so \
       /app/deps/torchx/cache/libtorch-*-cpu/lib/libtorch_cpu.so \
       /app/deps/torchx/cache/libtorch-*-cpu/lib/libc10.so \
       /app/deps/torchx/cache/libtorch-*-cpu/lib/libtorch_global_deps.so \
       /app/deps/torchx/cache/libtorch-*-cpu/lib/libgomp-*.so.1 libtorch/

FROM ${ELIXIR_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app:$PATH \
    MIX_ENV=prod
WORKDIR $HOME/app

COPY --chown=user --from=build /app/language_id ./language_id
COPY --chown=user --from=build /app/_build/language_id_prod/lib ./_build/language_id_prod/lib
COPY --chown=user artifacts/language-id/ft-depth5-1s-e2 ./artifacts/language-id/ft-depth5-1s-e2

EXPOSE 7860

CMD ["language_id", "serve", "--artifact", "artifacts/language-id/ft-depth5-1s-e2", "--port", "7860"]
