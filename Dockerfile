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
#
# MIX_TARGET=language_id_cuda adds EXLA on the precompiled cuda12 XLA archive
# (no CUDA toolkit is needed to compile it; the vendored EXLA skips its nvcc
# helpers when nvcc is absent). The runtime stage then installs the CUDA
# libraries XLA loads at run time (cudart, cuBLAS, cuDNN, cuFFT, cuSPARSE,
# NVRTC 12.9, NCCL, NVSHMEM 3.3 and ptxas) from the nvidia-*-cu12 wheels, so
# the image still starts from the plain Elixir image and only libcuda.so.1
# comes from the host driver (`--gpus all`). Pass --backend exla:cuda:
#
#   docker build --build-arg MIX_TARGET=language_id_cuda -t gemma-language-id:cuda .
#   docker run --rm --gpus all gemma-language-id:cuda \
#     language_id serve --backend exla:cuda --artifact artifacts/language-id/ft-depth5-1s-e2
#
# The same wheel list is what jobs/hf_job_cuda.sh in the Common Voice bucket
# installs at job time on a t4-small `hf jobs run`.

ARG ELIXIR_IMAGE=hexpm/elixir:1.20.2-erlang-29.0.3-debian-bookworm-20260713-slim
ARG MIX_TARGET=language_id

FROM ${ELIXIR_IMAGE} AS build
ARG MIX_TARGET
ARG XLA_TARGET=cuda12

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake git curl ca-certificates unzip && \
    rm -rf /var/lib/apt/lists/*

ENV MIX_ENV=prod \
    MIX_TARGET=${MIX_TARGET} \
    GEMMA4_ESCRIPT=language_id \
    LIBTORCH_TARGET=cpu \
    XLA_TARGET=${XLA_TARGET}

WORKDIR /app

RUN mix local.hex --force && mix local.rebar --force

# Dependencies first so source edits do not repeat the libtorch download and
# the NIF builds (torchx, explorer, tokenizers and, for the CUDA target, the
# vendored EXLA are all fetched or compiled here).
COPY mix.exs mix.lock ./
COPY vendor vendor
RUN mix deps.get && mix deps.compile

COPY config config
COPY lib lib
RUN mix compile --warnings-as-errors && mix escript.build

# The torchx NIF finds libtorch through $ORIGIN/libtorch, which the build
# leaves as a symlink into the download cache; replace it with the shared
# libraries themselves so the runtime stage only copies the build tree. EXLA
# likewise leaves its NIF and the XLA extension as links into vendor/exla/cache.
RUN cd _build/${MIX_TARGET}_prod/lib/torchx/priv && rm libtorch && mkdir libtorch && \
    cp /app/deps/torchx/cache/libtorch-*-cpu/lib/libtorch.so \
       /app/deps/torchx/cache/libtorch-*-cpu/lib/libtorch_cpu.so \
       /app/deps/torchx/cache/libtorch-*-cpu/lib/libc10.so \
       /app/deps/torchx/cache/libtorch-*-cpu/lib/libtorch_global_deps.so \
       /app/deps/torchx/cache/libtorch-*-cpu/lib/libgomp-*.so.1 libtorch/ && \
    if [ -d /app/_build/${MIX_TARGET}_prod/lib/exla/priv ]; then \
      cd /app/_build/${MIX_TARGET}_prod/lib/exla && cp -rL priv priv.copy && rm -rf priv && mv priv.copy priv; \
    fi

FROM ${ELIXIR_IMAGE}
ARG MIX_TARGET

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# CUDA user-space libraries for the cuda target. XLA dlopens the NVSHMEM
# bootstrap/transport plugins and libnvrtc-builtins by SONAME, so every wheel
# lib dir is flattened into one directory on LD_LIBRARY_PATH; NVSHMEM must
# stay on 3.3 (3.4+ renames the transports to .so.6 and the NIF fails to
# load), and NVRTC on 12.9 to match libnvrtc-builtins.so.12.9. The nvcc
# wheel becomes /usr/local/cuda, where XLA looks for ptxas and libdevice.
RUN if [ "${MIX_TARGET}" = "language_id_cuda" ]; then \
      apt-get update && apt-get install -y --no-install-recommends python3-pip && \
      pip install --no-cache-dir --break-system-packages --no-deps --target /opt/nvidia \
        nvidia-cublas-cu12 'nvidia-cuda-nvrtc-cu12==12.9.*' nvidia-cuda-runtime-cu12 \
        nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 \
        nvidia-nvjitlink-cu12 'nvidia-nvshmem-cu12==3.3.*' 'nvidia-cuda-nvcc-cu12==12.9.*' && \
      mkdir -p /opt/nvidia/lib && \
      for f in /opt/nvidia/nvidia/*/lib/*.so*; do ln -s "$f" /opt/nvidia/lib/; done && \
      ln -s /opt/nvidia/nvidia/cuda_nvcc /usr/local/cuda && \
      apt-get purge -y python3-pip && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*; \
    fi
ENV LD_LIBRARY_PATH=/opt/nvidia/lib

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app:/usr/local/cuda/bin:$PATH \
    MIX_ENV=prod
WORKDIR $HOME/app

COPY --chown=user --from=build /app/language_id ./language_id
COPY --chown=user --from=build /app/_build/${MIX_TARGET}_prod/lib ./_build/${MIX_TARGET}_prod/lib
COPY --chown=user artifacts/language-id/ft-depth5-1s-e2 ./artifacts/language-id/ft-depth5-1s-e2

EXPOSE 7860

CMD ["language_id", "serve", "--artifact", "artifacts/language-id/ft-depth5-1s-e2", "--port", "7860"]
