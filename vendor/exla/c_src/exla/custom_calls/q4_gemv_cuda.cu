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
#include <mma.h>

#include <cstdint>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kCols = 32;
// Decode is bound by how many weight loads are in flight. 32 k-splits of 4
// independent words each keep enough of them outstanding to reach most of an
// A100's bandwidth even when n/32 blocks barely cover the SMs.
constexpr int kSplitK = 32;
constexpr int kUnroll = 4;
constexpr int kNibblesPerWord = 8;

__device__ __forceinline__ float2 load_bf16_pair(const __nv_bfloat16 *values) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(values));
}

// Sum over one packed word of (nibble - 8) * x, for the 8 activations it
// covers (16 bytes, loaded at once and broadcast across the warp).
__device__ __forceinline__ float word_dot(int32_t packed_word,
                                          const __nv_bfloat16 *x) {
  const uint4 raw = *reinterpret_cast<const uint4 *>(x);
  const uint32_t pairs[4] = {raw.x, raw.y, raw.z, raw.w};
  float partial = 0.0f;

#pragma unroll
  for (int pair = 0; pair < kNibblesPerWord / 2; ++pair) {
    const float2 activations = __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162 *>(&pairs[pair]));
    const int low = ((packed_word >> (pair * 8)) & 0xF) - 8;
    const int high = ((packed_word >> (pair * 8 + 4)) & 0xF) - 8;
    partial += static_cast<float>(low) * activations.x +
               static_cast<float>(high) * activations.y;
  }

  return partial;
}

// One column's share of x . dequantize(packed, scales): the words
// split, split + kSplitK, ... of column col.
__device__ __forceinline__ float column_partial(
    const int32_t *__restrict__ packed, const __nv_bfloat16 *__restrict__ scales,
    const __nv_bfloat16 *__restrict__ x, int col, int split, int words_per_col,
    int n, int words_per_group) {
  float acc = 0.0f;

  for (int base = split; base < words_per_col; base += kSplitK * kUnroll) {
    int32_t words[kUnroll];
    float word_scales[kUnroll];

#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int word = base + u * kSplitK;
      words[u] = word < words_per_col ? packed[word * n + col] : 0;
      word_scales[u] =
          word < words_per_col
              ? __bfloat162float(scales[(word / words_per_group) * n + col])
              : 0.0f;
    }

#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int word = base + u * kSplitK;
      if (word < words_per_col) {
        acc += word_dot(words[u], x + word * kNibblesPerWord) * word_scales[u];
      }
    }
  }

  return acc;
}

__device__ __forceinline__ float reduce_splits(float partials[kSplitK][kCols],
                                               float acc) {
  partials[threadIdx.y][threadIdx.x] = acc;
  __syncthreads();

  float sum = 0.0f;
  if (threadIdx.y == 0) {
#pragma unroll
    for (int s = 0; s < kSplitK; ++s) {
      sum += partials[s][threadIdx.x];
    }
  }

  return sum;
}

__global__ void q4_gemv_kernel(const int32_t *__restrict__ packed,
                               const __nv_bfloat16 *__restrict__ scales,
                               const __nv_bfloat16 *__restrict__ x,
                               float *__restrict__ out, int k, int n,
                               int group_size) {
  __shared__ float partials[kSplitK][kCols];

  const int col = blockIdx.x * kCols + threadIdx.x;
  const float acc =
      col < n ? column_partial(packed, scales, x, col, threadIdx.y,
                               k / kNibblesPerWord, n,
                               group_size / kNibblesPerWord)
              : 0.0f;
  const float sum = reduce_splits(partials, acc);

  if (threadIdx.y == 0 && col < n) {
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
  const bool second = output_col >= n;
  const int col = second ? output_col - n : output_col;
  const float acc =
      col < n ? column_partial(second ? packed_b : packed_a,
                               second ? scales_b : scales_a, x, col,
                               threadIdx.y, k / kNibblesPerWord, n,
                               group_size / kNibblesPerWord)
              : 0.0f;
  const float sum = reduce_splits(partials, acc);

  if (threadIdx.y == 0 && col < n) {
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

// Prefill on tensor cores (sm_80+, bf16 WMMA). Each block computes a 64x64
// output tile with 4 warps of 32x32, stepping k by one 32-wide quant group:
// the activations are copied to shared memory and the packed weights are
// dequantized into it as bf16, then multiplied with f32 accumulation. The
// scalar kernel above stays for older GPUs and other group sizes.
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 32;
constexpr int kTileThreads = 128;
constexpr int kXStride = kTileK + 8;
constexpr int kWStride = kTileN + 8;
constexpr int kOutStride = kTileN + 4;

__global__ void __launch_bounds__(kTileThreads)
    q4_gemm_tc_kernel(const int32_t *__restrict__ packed,
                      const __nv_bfloat16 *__restrict__ scales,
                      const __nv_bfloat16 *__restrict__ x,
                      float *__restrict__ out, int k, int n, int seq) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using namespace nvcuda;

  constexpr int kInputBytes =
      (kTileM * kXStride + kTileK * kWStride) * sizeof(__nv_bfloat16);
  constexpr int kOutputBytes = kTileM * kOutStride * sizeof(float);
  __shared__ __align__(32) unsigned char
      shared[kInputBytes > kOutputBytes ? kInputBytes : kOutputBytes];
  __nv_bfloat16 *xs = reinterpret_cast<__nv_bfloat16 *>(shared);
  __nv_bfloat16 *ws = xs + kTileM * kXStride;
  float *outs = reinterpret_cast<float *>(shared);

  const int m0 = blockIdx.y * kTileM;
  const int n0 = blockIdx.x * kTileN;
  const int thread = threadIdx.x;
  const int warp = thread / 32;
  const int warp_m = (warp / 2) * 32;
  const int warp_n = (warp % 2) * 32;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::fill_fragment(acc[i][j], 0.0f);

  for (int k0 = 0; k0 < k; k0 += kTileK) {
    // 64 rows x 32 activations = 256 16-byte chunks, 2 per thread.
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int chunk = thread + i * kTileThreads;
      const int row = chunk / 4;
      const int part = (chunk % 4) * 8;
      uint4 values = make_uint4(0, 0, 0, 0);
      if (m0 + row < seq) {
        values = *reinterpret_cast<const uint4 *>(
            x + static_cast<size_t>(m0 + row) * k + k0 + part);
      }
      *reinterpret_cast<uint4 *>(xs + row * kXStride + part) = values;
    }

    // 4 packed words x 64 columns, 2 per thread, dequantized to 8 rows each.
    const int group = k0 / kTileK;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int index = thread + i * kTileThreads;
      const int word = index / kTileN;
      const int col = index % kTileN;
      int32_t packed_word = 0;
      float scale = 0.0f;
      if (n0 + col < n) {
        packed_word = packed[static_cast<size_t>(k0 / kNibblesPerWord + word) * n +
                             n0 + col];
        scale = __bfloat162float(scales[static_cast<size_t>(group) * n + n0 + col]);
      }
#pragma unroll
      for (int nibble = 0; nibble < kNibblesPerWord; ++nibble) {
        const float weight =
            static_cast<float>(((packed_word >> (nibble * 4)) & 0xF) - 8) * scale;
        ws[(word * kNibblesPerWord + nibble) * kWStride + col] =
            __float2bfloat16(weight);
      }
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kTileK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
          a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major>
          b[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        wmma::load_matrix_sync(a[i], xs + (warp_m + i * 16) * kXStride + kk,
                               kXStride);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma::load_matrix_sync(b[j], ws + kk * kWStride + warp_n + j * 16,
                               kWStride);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
          wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::store_matrix_sync(
          outs + (warp_m + i * 16) * kOutStride + warp_n + j * 16, acc[i][j],
          kOutStride, wmma::mem_row_major);

  __syncthreads();

  for (int index = thread; index < kTileM * kTileN; index += kTileThreads) {
    const int row = index / kTileN;
    const int col = index % kTileN;
    if (m0 + row < seq && n0 + col < n) {
      out[static_cast<size_t>(m0 + row) * n + n0 + col] =
          outs[row * kOutStride + col];
    }
  }
#endif
}

bool tensor_cores_available() {
  int device = 0;
  int major = 0;
  return cudaGetDevice(&device) == cudaSuccess &&
         cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                device) == cudaSuccess &&
         major >= 8;
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

  if (group_size == kTileK && tensor_cores_available()) {
    const dim3 tiles(static_cast<uint32_t>((n + kTileN - 1) / kTileN),
                     static_cast<uint32_t>((seq + kTileM - 1) / kTileM));

    q4_gemm_tc_kernel<<<tiles, kTileThreads, 0, stream>>>(
        static_cast<const int32_t *>(packed.untyped_data()),
        static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
        static_cast<const __nv_bfloat16 *>(x.untyped_data()),
        static_cast<float *>(out->untyped_data()), static_cast<int>(k),
        static_cast<int>(n), static_cast<int>(seq));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(error));
    }

    return ffi::Error::Success();
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
