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
      for (int nib = 0; nib < kNibblesPerWord; ++nib) {
        const int weight = ((packed_word >> (nib * 4)) & 0xF) - 8;
        partial +=
            static_cast<float>(weight) * __bfloat162float(x[k_base + nib]);
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
