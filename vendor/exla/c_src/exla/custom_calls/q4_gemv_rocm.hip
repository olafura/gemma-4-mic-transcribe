// int4 weight-only GEMV for ROCm.
//
// XLA's generic codegen dequantizes packed int4 weights into bf16 before the
// GEMM, so a 4-bit model reads 16-bit traffic and decode is bandwidth-bound at
// the dequantized size. This kernel reads the packed nibbles directly: each
// workgroup owns one output column and streams that column's packed weights,
// unpacking in registers.
//
// Layout (matches compressed-tensors w4a16 after our loader):
//   packed : {k/8, n} int32, row-major, 8 nibbles per int32 along k
//   scales : {k/group_size, n} bf16, one scale per (group, column)
//   x      : {k} bf16 activations
//   out    : {n} f32
//
// Nibbles are stored low-to-high within each int32 and biased by 8.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

namespace {

// Columns per workgroup, along threadIdx.x so adjacent lanes read adjacent
// memory. Parallelising over columns alone starves the GPU on the narrow
// layers (n = 2048 is only 8 workgroups), so threadIdx.y splits the k loop
// and the partial sums are reduced in LDS.
constexpr int kCols = 32;
constexpr int kSplitK = 8;
constexpr int kNibblesPerWord = 8;
#if defined(__gfx1151__)
using bf16x2_bits = short __attribute__((ext_vector_type(2)));
#endif

// Reduction happens in LDS rather than across lanes, so the kernel does not
// depend on the wavefront size (RDNA defaults to wave32, CDNA to wave64).
__global__ void q4_gemv_kernel(const int32_t *__restrict__ packed,
                               const __hip_bfloat16 *__restrict__ scales,
                               const __hip_bfloat16 *__restrict__ x,
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

      // group_size is a multiple of 8, so all nibbles in a word share a scale.
      const float scale =
          __bfloat162float(scales[(word / words_per_group) * n + col]);

      float partial = 0.0f;

#pragma unroll
#if defined(__gfx1151__)
      // Accumulate the integer weights before applying their shared scale, as
      // in the scalar path, but consume two bf16 activations per instruction.
      for (int pair = 0; pair < kNibblesPerWord / 2; ++pair) {
        const int low = ((packed_word >> (pair * 8)) & 0xF) - 8;
        const int high = ((packed_word >> (pair * 8 + 4)) & 0xF) - 8;
        const __hip_bfloat162 quantized(__float2bfloat16(low),
                                        __float2bfloat16(high));
        const __hip_bfloat162_raw quantized_raw = quantized;
        const bf16x2_bits weights = {
            static_cast<short>(quantized_raw.x),
            static_cast<short>(quantized_raw.y)};
        const bf16x2_bits activations =
            *reinterpret_cast<const bf16x2_bits *>(x + k_base + pair * 2);
        partial = __builtin_amdgcn_fdot2_f32_bf16(
            weights, activations, partial, false);
      }
#else
      for (int nib = 0; nib < kNibblesPerWord; ++nib) {
        const int weight = ((packed_word >> (nib * 4)) & 0xF) - 8;
        partial +=
            static_cast<float>(weight) * __bfloat162float(x[k_base + nib]);
      }
#endif

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

// Two projections of the same activation vector share one launch. The output
// is {a, b} concatenated along columns, so the normal gated FFN can slice it
// without materialising or repacking either weight matrix.
__global__ void q4_dual_gemv_kernel(
    const int32_t *__restrict__ packed_a,
    const __hip_bfloat16 *__restrict__ scales_a,
    const int32_t *__restrict__ packed_b,
    const __hip_bfloat16 *__restrict__ scales_b,
    const __hip_bfloat16 *__restrict__ x, float *__restrict__ out, int k, int n,
    int group_size) {
  __shared__ float partials[kSplitK][kCols];

  const int output_col = blockIdx.x * kCols + threadIdx.x;
  const int split = threadIdx.y;
  const bool second = output_col >= n;
  const int col = second ? output_col - n : output_col;
  const int32_t *packed = second ? packed_b : packed_a;
  const __hip_bfloat16 *scales = second ? scales_b : scales_a;

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
#if defined(__gfx1151__)
      for (int pair = 0; pair < kNibblesPerWord / 2; ++pair) {
        const int low = ((packed_word >> (pair * 8)) & 0xF) - 8;
        const int high = ((packed_word >> (pair * 8 + 4)) & 0xF) - 8;
        const __hip_bfloat162 quantized(__float2bfloat16(low),
                                        __float2bfloat16(high));
        const __hip_bfloat162_raw quantized_raw = quantized;
        const bf16x2_bits weights = {
            static_cast<short>(quantized_raw.x),
            static_cast<short>(quantized_raw.y)};
        const bf16x2_bits activations =
            *reinterpret_cast<const bf16x2_bits *>(x + k_base + pair * 2);
        partial = __builtin_amdgcn_fdot2_f32_bf16(
            weights, activations, partial, false);
      }
#else
      for (int nib = 0; nib < kNibblesPerWord; ++nib) {
        const int weight = ((packed_word >> (nib * 4)) & 0xF) - 8;
        partial +=
            static_cast<float>(weight) * __bfloat162float(x[k_base + nib]);
      }
#endif

      acc += partial * scale;
    }
  }

  partials[split][threadIdx.x] = acc;
  __syncthreads();

  if (split == 0 && col < n) {
    float sum = 0.0f;

#pragma unroll
    for (int s = 0; s < kSplitK; ++s) sum += partials[s][threadIdx.x];
    out[output_col] = sum;
  }
}

// Prefill multiplies a whole token sequence. Dequantizing first materialises
// the bf16 matrix (measured 1.46s for one Gemma prefill), so this kernel
// unpacks each weight once into registers and reuses it across a tile of
// tokens, keeping weight traffic proportional to seq / kSeqTile.
constexpr int kGemmCols = 64;
constexpr int kSeqTile = 16;

__global__ void q4_gemm_kernel(const int32_t *__restrict__ packed,
                               const __hip_bfloat16 *__restrict__ scales,
                               const __hip_bfloat16 *__restrict__ x,
                               float *__restrict__ out, int k, int n, int seq,
                               int group_size) {
  const int col = blockIdx.x * kGemmCols + threadIdx.x;
  const int seq_base = blockIdx.y * kSeqTile;
  if (col >= n) return;

  const int words_per_col = k / kNibblesPerWord;
  const int words_per_group = group_size / kNibblesPerWord;

  float acc[kSeqTile];
#pragma unroll
  for (int t = 0; t < kSeqTile; ++t) acc[t] = 0.0f;

  for (int word = 0; word < words_per_col; ++word) {
    const int32_t packed_word = packed[word * n + col];
    const float scale =
        __bfloat162float(scales[(word / words_per_group) * n + col]);
    const int k_base = word * kNibblesPerWord;

#if defined(__gfx1151__)
    // gfx1151 can accumulate two bf16 products per instruction. Keeping the
    // dequantized weights in bf16 also matches the model's weight dtype.
    bf16x2_bits weights[kNibblesPerWord / 2];
#pragma unroll
    for (int pair = 0; pair < kNibblesPerWord / 2; ++pair) {
      const int low = ((packed_word >> (pair * 8)) & 0xF) - 8;
      const int high = ((packed_word >> (pair * 8 + 4)) & 0xF) - 8;
      const __hip_bfloat162 values(__float2bfloat16(low * scale),
                                   __float2bfloat16(high * scale));
      const __hip_bfloat162_raw raw = values;
      weights[pair] = {static_cast<short>(raw.x),
                       static_cast<short>(raw.y)};
    }
#else
    float weights[kNibblesPerWord];
#pragma unroll
    for (int nib = 0; nib < kNibblesPerWord; ++nib) {
      weights[nib] =
          static_cast<float>(((packed_word >> (nib * 4)) & 0xF) - 8) * scale;
    }
#endif

#pragma unroll
    for (int t = 0; t < kSeqTile; ++t) {
      const int token = seq_base + t;
      if (token < seq) {
        const __hip_bfloat16 *row = x + static_cast<size_t>(token) * k + k_base;
        float sum = 0.0f;
#pragma unroll
#if defined(__gfx1151__)
        for (int pair = 0; pair < kNibblesPerWord / 2; ++pair) {
          const bf16x2_bits values =
              *reinterpret_cast<const bf16x2_bits *>(row + pair * 2);
          sum = __builtin_amdgcn_fdot2_f32_bf16(weights[pair], values, sum,
                                                 false);
        }
#else
        for (int nib = 0; nib < kNibblesPerWord; ++nib) {
          sum += weights[nib] * __bfloat162float(row[nib]);
        }
#endif
        acc[t] += sum;
      }
    }
  }

  for (int t = 0; t < kSeqTile; ++t) {
    const int token = seq_base + t;
    if (token < seq) {
      out[static_cast<size_t>(token) * n + col] = acc[t];
    }
  }
}

ffi::Error q4_gemm_impl(hipStream_t stream, ffi::AnyBuffer x,
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

  hipLaunchKernelGGL(q4_gemm_kernel, blocks, dim3(kGemmCols), 0, stream,
                     static_cast<const int32_t *>(packed.untyped_data()),
                     static_cast<const __hip_bfloat16 *>(scales.untyped_data()),
                     static_cast<const __hip_bfloat16 *>(x.untyped_data()),
                     static_cast<float *>(out->untyped_data()),
                     static_cast<int>(k), static_cast<int>(n),
                     static_cast<int>(seq), static_cast<int>(group_size));

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(err));
  }

  return ffi::Error::Success();
}

ffi::Error q4_gemv_impl(hipStream_t stream, ffi::AnyBuffer x,
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

  hipLaunchKernelGGL(q4_gemv_kernel, dim3(blocks), dim3(kCols, kSplitK), 0,
                     stream,
                     static_cast<const int32_t *>(packed.untyped_data()),
                     static_cast<const __hip_bfloat16 *>(scales.untyped_data()),
                     static_cast<const __hip_bfloat16 *>(x.untyped_data()),
                     static_cast<float *>(out->untyped_data()),
                     static_cast<int>(k), static_cast<int>(n),
                     static_cast<int>(group_size));

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(err));
  }

  return ffi::Error::Success();
}

ffi::Error q4_dual_gemv_impl(
    hipStream_t stream, ffi::AnyBuffer x, ffi::AnyBuffer packed_a,
    ffi::AnyBuffer scales_a, ffi::AnyBuffer packed_b, ffi::AnyBuffer scales_b,
    int64_t group_size, ffi::Result<ffi::AnyBuffer> out) {
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

  const uint32_t blocks =
      static_cast<uint32_t>((2 * n + kCols - 1) / kCols);

  hipLaunchKernelGGL(
      q4_dual_gemv_kernel, dim3(blocks), dim3(kCols, kSplitK), 0, stream,
      static_cast<const int32_t *>(packed_a.untyped_data()),
      static_cast<const __hip_bfloat16 *>(scales_a.untyped_data()),
      static_cast<const int32_t *>(packed_b.untyped_data()),
      static_cast<const __hip_bfloat16 *>(scales_b.untyped_data()),
      static_cast<const __hip_bfloat16 *>(x.untyped_data()),
      static_cast<float *>(out->untyped_data()), static_cast<int>(k),
      static_cast<int>(n), static_cast<int>(group_size));

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(err));
  }

  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(exla_q4_gemv_rocm, q4_gemv_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Attr<int64_t>("group_size")
                                  .Ret<ffi::AnyBuffer>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_q4_gemv", "ROCM",
                         exla_q4_gemv_rocm);

XLA_FFI_DEFINE_HANDLER_SYMBOL(exla_q4_dual_gemv_rocm, q4_dual_gemv_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Attr<int64_t>("group_size")
                                  .Ret<ffi::AnyBuffer>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_q4_dual_gemv", "ROCM",
                         exla_q4_dual_gemv_rocm);

XLA_FFI_DEFINE_HANDLER_SYMBOL(exla_q4_gemm_rocm, q4_gemm_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Attr<int64_t>("group_size")
                                  .Ret<ffi::AnyBuffer>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "exla_q4_gemm", "ROCM",
                         exla_q4_gemm_rocm);
