# Builds the System One tool (MIX_TARGET=system_one_cuda in mix.exs) as a
# release against the precompiled cuda12 XLA archive, for timing `route` on
# the Nvidia GPUs of `hf jobs run`. Only the build output is used: the job
# runs the release from the plain Elixir image with the CUDA libraries
# installed from the nvidia-*-cu12 wheels (see job.sh).
#
#   docker build -f hf-space/system-one/Dockerfile -t gemma-system-one:cuda .
#   docker run --rm gemma-system-one:cuda tar -C /app/_build/system_one_cuda_prod/rel -cz system_one \
#     > system_one-cuda.tar.gz

ARG ELIXIR_IMAGE=hexpm/elixir:1.20.2-erlang-29.0.3-debian-bookworm-20260713-slim

FROM ${ELIXIR_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake git curl ca-certificates unzip && \
    rm -rf /var/lib/apt/lists/*

# The vendored EXLA compiles its packed-int4 CUDA kernels (q4_gemv_cuda.cu)
# only when nvcc is on the PATH; without them `route` on packed weights fails
# with "No FFI handler registered for exla_q4_gemv on a platform CUDA".
RUN curl -fsSLo /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-nvcc-12-9 cuda-cudart-dev-12-9 && \
    rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda-12.9/bin:$PATH

ENV MIX_ENV=prod \
    MIX_TARGET=system_one_cuda \
    LIBTORCH_TARGET=cpu \
    XLA_TARGET=cuda12

WORKDIR /app

RUN mix local.hex --force && mix local.rebar --force

COPY mix.exs mix.lock ./
COPY vendor vendor
RUN mix deps.get && mix deps.compile

COPY config config
COPY lib lib
RUN mix compile

# As in the Dockerfile at the root: the torchx NIF reaches libtorch through a
# symlink into the download cache and EXLA's priv links into vendor/exla/cache,
# so both become real files before the release copies them.
RUN cd _build/system_one_cuda_prod/lib/torchx/priv && rm libtorch && mkdir libtorch && \
    cp /app/deps/torchx/cache/libtorch-*-cpu/lib/libtorch.so \
       /app/deps/torchx/cache/libtorch-*-cpu/lib/libtorch_cpu.so \
       /app/deps/torchx/cache/libtorch-*-cpu/lib/libc10.so \
       /app/deps/torchx/cache/libtorch-*-cpu/lib/libtorch_global_deps.so \
       /app/deps/torchx/cache/libtorch-*-cpu/lib/libgomp-*.so.1 libtorch/ && \
    cd /app/_build/system_one_cuda_prod/lib/exla && cp -rL priv priv.copy && rm -rf priv && mv priv.copy priv

RUN mix release system_one
