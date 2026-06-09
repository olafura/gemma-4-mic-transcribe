import Config

config :nx, :default_backend, Torchx.Backend

config :exla, :clients,
  host: [platform: :host],
  cuda: [platform: :cuda, preallocate: false, memory_fraction: 0.55],
  rocm: [platform: :rocm, preallocate: false, memory_fraction: 0.55],
  tpu: [platform: :tpu]
