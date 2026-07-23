// int4 weight-only GEMV/GEMM for CUDA.
//
// The packed layout matches compressed-tensors w4a16 after our loader:
//   packed : {k/8, n} int32, row-major, 8 biased nibbles per int32
//   scales : {k/group_size, n} bf16
//   x      : {k} or {seq, k} bf16
//   out    : {n} or {seq, n} f32
//
// Adjacent threads own adjacent output columns, so packed weights and scales
// are read coalescently. Decode splits k across threadIdx.y and reduces the
// partials in shared memory. Prefill reuses each unpacked word across a tile
// of tokens instead of materialising a dequantized weight matrix.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kCols = 32;
constexpr int kSplitK = 8;
constexpr int kNibblesPerWord = 8;

__device__ __forceinline__ float2 load_bf16_pair(const __nv_bfloat16 *values) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(values));
}

__global__ void q4_gemv_kernel(const int32_t *__restrict__ packed,
                               const __nv_bfloat16 *__restrict__ scales,
                               const __nv_bfloat16 *__restrict__ x,
                               float *__restrict__ out, int k, int n,
                               int group_size) {
  __shared__ float partials[kSplitK][kCols];

  const int col = blockIdx.x * kCols + threadIdx.x;
  const int split = threadIdx.y;
  const int words_per_col = k / kNibblesPerWord;
  const int words_per_group = group_size / kNibblesPerWord;
  float acc = 0.0f;

  if (col < n) {
    for (int word = split; word < words_per_col; word += kSplitK) {
      const int32_t packed_word = packed[word * n + col];
      const int k_base = word * kNibblesPerWord;
      const float scale =
          __bfloat162float(scales[(word / words_per_group) * n + col]);
      float partial = 0.0f;

#pragma unroll
      for (int pair = 0; pair < kNibblesPerWord / 2; ++pair) {
        const int low = ((packed_word >> (pair * 8)) & 0xF) - 8;
        const int high = ((packed_word >> (pair * 8 + 4)) & 0xF) - 8;
        const float2 activations = load_bf16_pair(x + k_base + pair * 2);
        partial += static_cast<float>(low) * activations.x +
                   static_cast<float>(high) * activations.y;
      }

      acc += partial * scale;
    }
  }

  partials[split][threadIdx.x] = acc;
  __syncthreads();

  if (split == 0 && col < n) {
    float sum = 0.0f;

#pragma unroll
    for (int s = 0; s < kSplitK; ++s) {
      sum += partials[s][threadIdx.x];
    }

    out[col] = sum;
  }
}

__global__ void q4_dual_gemv_kernel(const int32_t *__restrict__ packed_a,
                                    const __nv_bfloat16 *__restrict__ scales_a,
                                    const int32_t *__restrict__ packed_b,
                                    const __nv_bfloat16 *__restrict__ scales_b,
                                    const __nv_bfloat16 *__restrict__ x,
                                    float *__restrict__ out, int k, int n,
                                    int group_size) {
  __shared__ float partials[kSplitK][kCols];

  const int output_col = blockIdx.x * kCols + threadIdx.x;
  const int split = threadIdx.y;
  const bool second = output_col >= n;
  const int col = second ? output_col - n : output_col;
  const int32_t *packed = second ? packed_b : packed_a;
  const __nv_bfloat16 *scales = second ? scales_b : scales_a;
  const int words_per_col = k / kNibblesPerWord;
  const int words_per_group = group_size / kNibblesPerWord;
  float acc = 0.0f;

  if (col < n) {
    for (int word = split; word < words_per_col; word += kSplitK) {
      const int32_t packed_word = packed[word * n + col];
      const int k_base = word * kNibblesPerWord;
      const float scale =
          __bfloat162float(scales[(word / words_per_group) * n + col]);
      float partial = 0.0f;

#pragma unroll
      for (int pair = 0; pair < kNibblesPerWord / 2; ++pair) {
        const int low = ((packed_word >> (pair * 8)) & 0xF) - 8;
        const int high = ((packed_word >> (pair * 8 + 4)) & 0xF) - 8;
        const float2 activations = load_bf16_pair(x + k_base + pair * 2);
        partial += static_cast<float>(low) * activations.x +
                   static_cast<float>(high) * activations.y;
      }

      acc += partial * scale;
    }
  }

  partials[split][threadIdx.x] = acc;
  __syncthreads();

  if (split == 0 && col < n) {
    float sum = 0.0f;

#pragma unroll
    for (int s = 0; s < kSplitK; ++s) {
      sum += partials[s][threadIdx.x];
    }

    out[output_col] = sum;
  }
}

constexpr int kGemmCols = 64;
constexpr int kSeqTile = 16;

__global__ void q4_gemm_kernel(const int32_t *__restrict__ packed,
                               const __nv_bfloat16 *__restrict__ scales,
                               const __nv_bfloat16 *__restrict__ x,
                               float *__restrict__ out, int k, int n, int seq,
                               int group_size) {
  const int col = blockIdx.x * kGemmCols + threadIdx.x;
  const int seq_base = blockIdx.y * kSeqTile;
  if (col >= n)
    return;

  const int words_per_col = k / kNibblesPerWord;
  const int words_per_group = group_size / kNibblesPerWord;
  float acc[kSeqTile];

#pragma unroll
  for (int token_offset = 0; token_offset < kSeqTile; ++token_offset) {
    acc[token_offset] = 0.0f;
  }

  for (int word = 0; word < words_per_col; ++word) {
    const int32_t packed_word = packed[word * n + col];
    const float scale =
        __bfloat162float(scales[(word / words_per_group) * n + col]);
    const int k_base = word * kNibblesPerWord;
    float weights[kNibblesPerWord];

#pragma unroll
    for (int nibble = 0; nibble < kNibblesPerWord; ++nibble) {
      weights[nibble] =
          static_cast<float>(((packed_word >> (nibble * 4)) & 0xF) - 8) * scale;
    }

#pragma unroll
    for (int token_offset = 0; token_offset < kSeqTile; ++token_offset) {
      const int token = seq_base + token_offset;

      if (token < seq) {
        const __nv_bfloat16 *row = x + static_cast<size_t>(token) * k + k_base;
        float sum = 0.0f;

#pragma unroll
        for (int pair = 0; pair < kNibblesPerWord / 2; ++pair) {
          const float2 activations = load_bf16_pair(row + pair * 2);
          sum += weights[pair * 2] * activations.x +
                 weights[pair * 2 + 1] * activations.y;
        }

        acc[token_offset] += sum;
      }
    }
  }

#pragma unroll
  for (int token_offset = 0; token_offset < kSeqTile; ++token_offset) {
    const int token = seq_base + token_offset;
    if (token < seq) {
      out[static_cast<size_t>(token) * n + col] = acc[token_offset];
    }
  }
}

ffi::Error q4_gemm_impl(cudaStream_t stream, ffi::AnyBuffer x,
                        ffi::AnyBuffer packed, ffi::AnyBuffer scales,
                        int64_t group_size, ffi::Result<ffi::AnyBuffer> out) {
  auto packed_dims = packed.dimensions();
  auto x_dims = x.dimensions();

  if (packed_dims.size() != 2 || x_dims.size() != 2) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "q4_gemm expects packed {k/8, n} and x {seq, k}");
  }

  const int64_t n = packed_dims[1];
  const int64_t seq = x_dims[0];
  const int64_t k = x_dims[1];

  if (packed_dims[0] * kNibblesPerWord != k) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "q4_gemm packed rows must equal k/8");
  }

  if (group_size <= 0 || k % group_size != 0 ||
      group_size % kNibblesPerWord != 0) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "q4_gemm group_size must divide k and be a multiple of 8");
  }

  const dim3 blocks(static_cast<uint32_t>((n + kGemmCols - 1) / kGemmCols),
                    static_cast<uint32_t>((seq + kSeqTile - 1) / kSeqTile));

  q4_gemm_kernel<<<blocks, kGemmCols, 0, stream>>>(
      static_cast<const int32_t *>(packed.untyped_data()),
      static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
      static_cast<const __nv_bfloat16 *>(x.untyped_data()),
      static_cast<float *>(out->untyped_data()), static_cast<int>(k),
      static_cast<int>(n), static_cast<int>(seq), static_cast<int>(group_size));

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(error));
  }

  return ffi::Error::Success();
}

ffi::Error q4_gemv_impl(cudaStream_t stream, ffi::AnyBuffer x,
                        ffi::AnyBuffer packed, ffi::AnyBuffer scales,
                        int64_t group_size, ffi::Result<ffi::AnyBuffer> out) {
  auto packed_dims = packed.dimensions();
  auto x_dims = x.dimensions();

  if (packed_dims.size() != 2 || x_dims.size() != 1) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "q4_gemv expects packed {k/8, n} and x {k}");
  }

  const int64_t n = packed_dims[1];
  const int64_t k = x_dims[0];

  if (packed_dims[0] * kNibblesPerWord != k) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "q4_gemv packed rows must equal k/8");
  }

  if (group_size <= 0 || k % group_size != 0 ||
      group_size % kNibblesPerWord != 0) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "q4_gemv group_size must divide k and be a multiple of 8");
  }

  const uint32_t blocks = static_cast<uint32_t>((n + kCols - 1) / kCols);

  q4_gemv_kernel<<<blocks, dim3(kCols, kSplitK), 0, stream>>>(
      static_cast<const int32_t *>(packed.untyped_data()),
      static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
      static_cast<const __nv_bfloat16 *>(x.untyped_data()),
      static_cast<float *>(out->untyped_data()), static_cast<int>(k),
      static_cast<int>(n), static_cast<int>(group_size));

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(error));
  }

  return ffi::Error::Success();
}

ffi::Error q4_dual_gemv_impl(cudaStream_t stream, ffi::AnyBuffer x,
                             ffi::AnyBuffer packed_a, ffi::AnyBuffer scales_a,
                             ffi::AnyBuffer packed_b, ffi::AnyBuffer scales_b,
                             int64_t group_size,
                             ffi::Result<ffi::AnyBuffer> out) {
  auto packed_a_dims = packed_a.dimensions();
  auto packed_b_dims = packed_b.dimensions();
  auto x_dims = x.dimensions();

  if (packed_a_dims.size() != 2 || packed_b_dims.size() != 2 ||
      packed_b_dims[0] != packed_a_dims[0] ||
      packed_b_dims[1] != packed_a_dims[1] || x_dims.size() != 1) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "q4_dual_gemv expects matching packed matrices and x {k}");
  }

  const int64_t n = packed_a_dims[1];
  const int64_t k = x_dims[0];

  if (packed_a_dims[0] * kNibblesPerWord != k || group_size <= 0 ||
      k % group_size != 0 || group_size % kNibblesPerWord != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "q4_dual_gemv has incompatible k or group_size");
  }

  const uint32_t blocks = static_cast<uint32_t>((2 * n + kCols - 1) / kCols);

  q4_dual_gemv_kernel<<<blocks, dim3(kCols, kSplitK), 0, stream>>>(
      static_cast<const int32_t *>(packed_a.untyped_data()),
      static_cast<const __nv_bfloat16 *>(scales_a.untyped_data()),
      static_cast<const int32_t *>(packed_b.untyped_data()),
      static_cast<const __nv_bfloat16 *>(scales_b.untyped_data()),
      static_cast<const __nv_bfloat16 *>(x.untyped_data()),
      static_cast<float *>(out->untyped_data()), static_cast<int>(k),
      static_cast<int>(n), static_cast<int>(group_size));

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(error));
  }

  return ffi::Error::Success();
}

} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(exla_q4_gemv_cuda, q4_gemv_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Attr<int64_t>("group_size")
                                  .Ret<ffi::AnyBuffer>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_q4_gemv", "CUDA",
                         exla_q4_gemv_cuda);

XLA_FFI_DEFINE_HANDLER_SYMBOL(exla_q4_dual_gemv_cuda, q4_dual_gemv_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Attr<int64_t>("group_size")
                                  .Ret<ffi::AnyBuffer>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_q4_dual_gemv", "CUDA",
                         exla_q4_dual_gemv_cuda);

XLA_FFI_DEFINE_HANDLER_SYMBOL(exla_q4_gemm_cuda, q4_gemm_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Attr<int64_t>("group_size")
                                  .Ret<ffi::AnyBuffer>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_q4_gemm", "CUDA",
                         exla_q4_gemm_cuda);
