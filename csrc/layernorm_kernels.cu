#include "type_convert.cuh"
#include "dispatch_utils.h"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef USE_ROCM
#include <cub/cub.cuh>
#else
#include <hipcub/hipcub.hpp>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cmath>
#include "cuda_compat.h"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && CUDART_VERSION >= 12010
#define NVCC_SUPPORTS_MULTICAST 1
#endif

#include <ATen/ATen.h>
#if !defined(USE_ROCM)
#include <cuda_bf16.h>
#endif

namespace vllm
{

/* 
* ********************************************************** *
* Code copied from Pytorch codebase                          *
* Multimem All Reduce Meta kernel                            *
* ********************************************************** *
*/
template <typename T>
__inline__ size_t get_alignment(T ptr_or_size)
{
  auto val = reinterpret_cast<uintptr_t>(ptr_or_size);
  if (val % 16 == 0)
  {
    return 16;
  }
  else if (val % 8 == 0)
  {
    return 8;
  }
  else if (val % 4 == 0)
  {
    return 4;
  }
  else if (val % 2 == 0)
  {
    return 2;
  }
  else
  {
    return 1;
  }
}

template <>
__inline__ size_t get_alignment<size_t>(size_t size)
{
  return get_alignment(reinterpret_cast<void *>(size));
}

template <bool Value, class... Args>
inline constexpr bool dependent_bool_value = Value;

template <class... Args>
inline constexpr bool dependent_false = dependent_bool_value<false, Args...>;

template <auto... Args>
inline constexpr bool dependent_false_nt =
    dependent_bool_value<false, decltype(Args)...>;

enum class MemOpSem
{
  Relaxed,
  Acquire,
  Release,
  AcqRel,
};

#define CAS_ASM(addr, compare, val, old_val, sem)                 \
asm volatile("atom.global" sem ".sys.cas.b32 %0, [%1], %2, %3;" \
              : "=r"(old_val)                                    \
              : "l"(addr), "r"(compare), "r"(val)                \
              : "memory");

template <MemOpSem Sem>
__device__ __forceinline__ uint32_t
cas(uint32_t *addr, uint32_t compare, uint32_t val)
{
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  CUDA_KERNEL_ASSERT(false);
  return 0;
#else
  uint32_t old_val;
  if constexpr (Sem == MemOpSem::Relaxed)
  {
    CAS_ASM(addr, compare, val, old_val, ".relaxed");
  }
  else if constexpr (Sem == MemOpSem::Acquire)
  {
    CAS_ASM(addr, compare, val, old_val, ".acquire");
  }
  else if constexpr (Sem == MemOpSem::Release)
  {
    CAS_ASM(addr, compare, val, old_val, ".release");
  }
  else
  {
    static_assert(dependent_false_nt<Sem>);
  }
  return old_val;
#endif
}

__device__ __forceinline__ void trap()
{
#if defined(USE_ROCM)
  assert(0);
#else
  __trap();
#endif
}

__device__ __forceinline__ size_t global_timer_ns()
{
#if defined(USE_ROCM)
  CUDA_KERNEL_ASSERT(false);
  return 0;
#else
  size_t val;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(val) : : "memory");
  return val;
#endif
}

constexpr size_t ns_per_ms = 1e6;

template <MemOpSem Sem>
__device__ __forceinline__ bool try_put_signal(
    uint32_t *addr,
    size_t timeout_ms)
{
  size_t deadline = global_timer_ns() + timeout_ms * ns_per_ms;
  while (cas<Sem>(addr, 0, 1) != 0)
  {
    if (timeout_ms != 0 && global_timer_ns() > deadline)
    {
      return false;
    }
  }
  return true;
}

template <MemOpSem Sem>
__device__ __forceinline__ bool try_wait_signal(
    uint32_t *addr,
    size_t timeout_ms)
{
  size_t deadline = global_timer_ns() + timeout_ms * ns_per_ms;
  while (cas<Sem>(addr, 1, 0) != 1)
  {
    if (timeout_ms != 0 && global_timer_ns() > deadline)
    {
      return false;
    }
  }
  return true;
}

template <MemOpSem Sem>
__device__ __forceinline__ void put_signal(uint32_t *addr)
{
  while (cas<Sem>(addr, 0, 1) != 0)
    ;
}

template <MemOpSem Sem>
__device__ __forceinline__ void wait_signal(uint32_t *addr)
{
  while (cas<Sem>(addr, 1, 0) != 1)
    ;
}

// Synchronizes blocks with matching blockIdx across participating devices.
// Note: sync_remote_block itself is not a system level barrier/fence. It is a
// building block for expressing different synchronization patterns.
//
// Pattern 0: Ensures that all writes to symm_mem buffers from previous
// kernels across all devices are visible to the current kernel:
//
//   sync_remote_blocks<MemOpSem::Relaxed>(...);
//   __syncthreads();
//
// Pattern 1: Ensures that all writes to symm_mem buffers from the current
// block are visible to all remote blocks with matching blockIdx:
//
//   __syncthreads();
//   sync_remote_blocks<MemOpSem::AcqRel>(...);
//   __syncthreads();
//
// Pattern 2: Ensures that symm_mem buffers read by the current kernel are safe
// for writing by subsequent kernels across all devices.
//
//   __syncthreads();
//   sync_remote_blocks<MemOpSem::Relaxed>(...);
template <MemOpSem Sem>
__device__ __forceinline__ void sync_remote_blocks(
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size);

template <>
__device__ __forceinline__ void sync_remote_blocks<MemOpSem::Relaxed>(
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size)
{
  if (threadIdx.x < world_size)
  {
    auto target_rank = threadIdx.x;
    put_signal<MemOpSem::Relaxed>(
        signal_pads[target_rank] + blockIdx.x * world_size + rank);
    wait_signal<MemOpSem::Relaxed>(
        signal_pads[rank] + blockIdx.x * world_size + target_rank);
  }
}

template <>
__device__ __forceinline__ void sync_remote_blocks<MemOpSem::AcqRel>(
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size)
{
  if (threadIdx.x < world_size)
  {
    auto target_rank = threadIdx.x;
    put_signal<MemOpSem::Release>(
        signal_pads[target_rank] + blockIdx.x * world_size + rank);
    wait_signal<MemOpSem::Acquire>(
        signal_pads[rank] + blockIdx.x * world_size + target_rank);
  }
}

template <int Size>
union Vec;

template <>
union Vec<4>
{
  uint16_t u16[2];
  uint32_t u32, as_scalar;
  float f32;
};

template <>
union Vec<8>
{
  uint16_t u16[4];
  uint32_t u32[2];
  uint64_t u64, as_scalar;
  float f32[2];
};

template <>
union alignas(16) Vec<16>
{
  uint16_t u16[8];
  uint32_t u32[4];
  uint64_t u64[2];
  uint4 u128, as_scalar;
  float f32[4];
};

template <typename T>
struct MultimemLdReduce
{
  template <int Alignment>
  __device__ __inline__ Vec<Alignment> operator()(T *mc_ptr)
  {
    static_assert(dependent_false<T>);
  }
};

template <int Alignment, typename T>
__device__ __inline__ Vec<Alignment> multimem_ld_reduce_add(T *mc_ptr)
{
  MultimemLdReduce<T> functor;
  return functor.template operator()<Alignment>(mc_ptr);
}

#if defined(USE_ROCM) || !defined(NVCC_SUPPORTS_MULTICAST)
#define SPECIALIZE_MULTIMEM_LD_REDUCE_VEC_32(type, asm_type, acc_prec) \
  template <>                                                          \
  struct MultimemLdReduce<type>                                        \
  {                                                                    \
    template <int Alignment>                                           \
    __device__ __inline__ Vec<Alignment> operator()(type *mc_ptr)      \
    {                                                                  \
      CUDA_KERNEL_ASSERT(false);                                       \
    }                                                                  \
  };
#else
#define SPECIALIZE_MULTIMEM_LD_REDUCE_VEC_32(type, asm_type, acc_prec)    \
  template <>                                                             \
  struct MultimemLdReduce<type>                                           \
  {                                                                       \
    template <int Alignment>                                              \
    __device__ __inline__ Vec<Alignment> operator()(type *mc_ptr)         \
    {                                                                     \
      Vec<Alignment> vec;                                                 \
      if constexpr (Alignment == 16)                                      \
      {                                                                   \
        asm("multimem.ld_reduce.relaxed.sys.global.add" acc_prec          \
            ".v4" asm_type " {%0,%1,%2,%3}, [%4];"                        \
            : "=r"(vec.u32[0]),                                           \
              "=r"(vec.u32[1]),                                           \
              "=r"(vec.u32[2]),                                           \
              "=r"(vec.u32[3])                                            \
            : "l"(mc_ptr)                                                 \
            : "memory");                                                  \
      }                                                                   \
      else if constexpr (Alignment == 8)                                  \
      {                                                                   \
        asm("multimem.ld_reduce.relaxed.sys.global.add" acc_prec          \
            ".v2" asm_type " {%0,%1}, [%2];"                              \
            : "=r"(vec.u32[0]), "=r"(vec.u32[1])                          \
            : "l"(mc_ptr)                                                 \
            : "memory");                                                  \
      }                                                                   \
      else if constexpr (Alignment == 4)                                  \
      {                                                                   \
        asm("multimem.ld_reduce.relaxed.sys.global.add" acc_prec asm_type \
            " %0, [%1];"                                                  \
            : "=r"(vec.u32)                                               \
            : "l"(mc_ptr)                                                 \
            : "memory");                                                  \
      }                                                                   \
      return vec;                                                         \
    }                                                                     \
  };
#endif

  SPECIALIZE_MULTIMEM_LD_REDUCE_VEC_32(at::BFloat16, ".bf16x2", ".acc::f32");
  SPECIALIZE_MULTIMEM_LD_REDUCE_VEC_32(float, ".f32", "");

  template <int Alignment, typename T>
  __device__ __inline__ void multimem_st(T *mc_ptr, Vec<Alignment> &vec)
  {
#if defined(USE_ROCM) || !defined(NVCC_SUPPORTS_MULTICAST)
    CUDA_KERNEL_ASSERT(false);
#else
    if constexpr (Alignment == 16)
    {
      asm("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
          :
          : "l"(mc_ptr),
            "r"(vec.u32[0]),
            "r"(vec.u32[1]),
            "r"(vec.u32[2]),
            "r"(vec.u32[3])
          : "memory");
    }
    else if constexpr (Alignment == 8)
    {
      asm("multimem.st.relaxed.sys.global.v2.f32 [%0], {%1,%2};"
          :
          : "l"(mc_ptr), "r"(vec.u32[0]), "r"(vec.u32[1])
          : "memory");
    }
    else if constexpr (Alignment == 4)
    {
      asm("multimem.st.relaxed.sys.global.f32 [%0], %1;"
          :
          : "l"(mc_ptr), "r"(vec.u32)
          : "memory");
    }
    else
    {
      static_assert(dependent_false<T>);
    }
#endif
  }
/* 
* ********************************************************** *
* Code copied from Pytorch codebase                          *
* End:: Multimem All Reduce Meta kernel                      *
* ********************************************************** *
*/


/*
* ******************************************************** *
* RMS NORM KERNEL                                          *
* ******************************************************** * 
*/ 

// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t *__restrict__ out,          // [..., hidden_size]
    const scalar_t *__restrict__ input,  // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size)
{
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x)
  {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0)
  {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x)
  {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

/*
* ****************************************************** *
* RMS NORM IN-PLACE KERNEL                               *
* ****************************************************** * 
*/ 
template <typename scalar_t>
__global__ void rms_norm_inplace_kernel(
    scalar_t *__restrict__ out,          // [..., hidden_size]
    scalar_t *__restrict__ input,        // [..., hidden_size] — will be updated in-place
    const scalar_t *__restrict__ weight, // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size)
{

  __shared__ float s_variance;
  float variance = 0.0f;

  // First pass: compute variance
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x)
  {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0)
  {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // Second pass: copy to out, normalize input in-place
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x)
  {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] = x;                                            // copy original input to `out`
    input[blockIdx.x * hidden_size + idx] = ((scalar_t)(x * s_variance)) * weight[idx]; // normalize in-place
  }
}

/* 
* ********************************************************* *
* FUSED RESIDUAL ADD + RMS NORM KERNEL                      *
* Function specialization in the case of FP16/BF16 tensors. *
* ********************************************************* *
* */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size)
{
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
      not aliased in practice. Argument pointers should not be dereferenced
      in this kernel as that would be undefined behavior */
  auto *__restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width> *>(input);
  auto *__restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width> *>(residual);
  auto *__restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width> *>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x)
  {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = input_v[id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0)
  {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x)
  {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    input_v[id] = temp;
  }
}

/* 
* ********************************************************* *
* FUSED RESIDUAL ADD + RMS NORM KERNEL                      *
* GENERIC                                                   *
* ********************************************************* *
*/
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size)
{
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x)
  {
    scalar_t z = input[blockIdx.x * hidden_size + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0)
  {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x)
  {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

/* 
* ********************************************************* *
* BLOCK REDUCE SUM                                          *
* ********************************************************* *
*/
template <typename T, int NUM>
__inline__ __device__ T warpReduceSum(T *val)
{
#pragma unroll
  for (int i = 0; i < NUM; i++)
  {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(0xffffffff, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSum(T *val)
{
  __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSum<T, NUM>(val);

  if (lane == 0)
  {
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++)
  {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSum<T, NUM>(val);
  return (T)0.0f;
}

/* 
* ********************************************************* *
* FUSED RESIDUAL ADD + RMS NORM CTA-BASED KERNEL            *
* Function specialization in the case of      BF16 tensors. *
* ********************************************************* *
*/
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_cta_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    const float epsilon,
    const int num_tokens,
    const int hidden_size)
{

  // Check vectorization assumptions
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  int tokens_per_iter = (num_tokens + gridDim.x - 1) / gridDim.x;
  using vec_t = _f16Vec<scalar_t, width>;

  // Type-punned vector pointers
  auto *__restrict__ input_v = reinterpret_cast<vec_t *>(input);
  auto *__restrict__ residual_v = reinterpret_cast<vec_t *>(residual);
  auto *__restrict__ weight_v = reinterpret_cast<const vec_t *>(weight);

#pragma unroll
  for (int iter = 0; iter < tokens_per_iter; iter++)
  {
    float variance[1] = {0.0f};
    int token_id = blockIdx.x + iter * gridDim.x;
    if (token_id >= num_tokens)
    {
      return;
    }
    const int tid = threadIdx.x;
    const int bdimx = blockDim.x;

    __shared__ float s_variance;
    int offset = token_id * vec_hidden_size;
    auto input_o = input_v + offset;
    auto residual_o = residual_v + offset;
    // First pass: add + accumulate sum of squares
    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      vec_t temp = input_o[idx];
      temp += residual_o[idx];
      variance[0] += temp.sum_squares(); // FP32 accumulation
      residual_o[idx] = temp;
    }

    // Block-wide variance reduction using CUB
    // using BlockReduce = cub::BlockReduce<float, 256>;
    // __shared__ typename BlockReduce::TempStorage reduce_storage;
    // float var_out = BlockReduce(reduce_storage).Reduce(variance, cub::Sum{});
    blockReduceSum<float, 1>(variance);
    if (threadIdx.x == 0)
    {
      s_variance = rsqrtf(variance[0] / hidden_size + epsilon);
    }
    __syncthreads();

    // Second pass: normalize and apply weight
    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      vec_t shared_weight = weight_v[idx];
      vec_t temp = residual_o[idx];
      temp *= s_variance;
      temp *= shared_weight;
      input_o[idx] = temp;
    }
  }
}
/* 
* ********************************************************* *
* FUSED RESIDUAL ADD + RMS NORM CTA-BASED KERNEL            *
* GENERIC: NOT SUPPORTED                                    *
* ********************************************************* *
*/
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_cta_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size)
{
  /* Not supported */
}

/* 
* ********************************************************* *
* FUSED RS + RESIDUAL ADD + RMS NORM + AG CTA-BASED KERNEL  *
* Function specialization in the case of      BF16 tensors. *
* ********************************************************* *
*/
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_rs_ln_ag_cta_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ mcptr,        // [..., hidden_size] multimem_ptr
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const float epsilon,
    const int num_tokens,
    const int hidden_size)
{

  // Check vectorization assumptions
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  using vec_t = _f16Vec<scalar_t, width>;

  // Type-punned vector pointers
  auto *__restrict__ input_v = reinterpret_cast<vec_t *>(input);
  auto *__restrict__ residual_v = reinterpret_cast<vec_t *>(residual);
  auto *__restrict__ weight_v = reinterpret_cast<const vec_t *>(weight);
  int tokens_per_iter = (num_tokens + gridDim.x - 1) / gridDim.x;

  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
  __syncthreads();

  #pragma unroll
  for (int iter = 0; iter < tokens_per_iter; iter++)
  {
    int token_id = blockIdx.x + iter * gridDim.x;
    if (token_id >= num_tokens)
      continue;
    float variance[1] = {0.0f};
    const int tid = threadIdx.x;
    const int bdimx = blockDim.x;

    __shared__ float s_variance;
    int offset = token_id * vec_hidden_size;
    int offset_scalar = token_id * hidden_size;
    auto input_o = input_v + offset;
    auto residual_o = residual_v + offset;

    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      auto mtemp = multimem_ld_reduce_add<16>(mcptr + offset_scalar + idx * width);
      vec_t temp = *(reinterpret_cast<vec_t *>(&mtemp));
      // vec_t temp = input_o[idx];
      temp += residual_o[idx];
      variance[0] += temp.sum_squares(); // FP32 accumulation
      residual_o[idx] = temp;
    }

    blockReduceSum<float, 1>(variance);
    if (threadIdx.x == 0)
    {
      s_variance = rsqrtf(variance[0] / hidden_size + epsilon);
    }
    __syncthreads();

    // Second pass: normalize and apply weight
    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      vec_t shared_weight = weight_v[idx];
      vec_t temp = residual_o[idx];
      temp *= s_variance;
      temp *= shared_weight;
      // input_o[idx] = temp;
      multimem_st<16>(mcptr + offset_scalar + idx * width, *(reinterpret_cast<Vec<16> *>(&temp)));
    }
  }
  __syncthreads();
  sync_remote_blocks<MemOpSem::AcqRel>(signal_pads, rank, world_size);
}

/* 
* ********************************************************* *
* FUSED RS + RESIDUAL ADD + RMS NORM + AG CTA-BASED KERNEL  *
* GENERIC NOT SUPPORT                                       *
* ********************************************************* *
*/
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_rs_ln_ag_cta_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ mcptr,        // [..., hidden_size] multimem_ptr
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const float epsilon,
    const int num_tokens,
    const int hidden_size)
{
  /* Not supported */
}

/* 
* ********************************************************* *
* FUSED RS + RESIDUAL ADD + RMS NORM      CTA-BASED KERNEL  *
* Function specialization in the case of      BF16 tensors. *
* ********************************************************* *
*/
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_rs_ln_cta_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ mcptr,        // [..., hidden_size] multimem_ptr
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const float epsilon,
    const int num_tokens,
    const int hidden_size)
{
  // Check vectorization assumptions
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  using vec_t = _f16Vec<scalar_t, width>;

  // Type-punned vector pointers
  auto *__restrict__ input_v = reinterpret_cast<vec_t *>(input);
  auto *__restrict__ residual_v = reinterpret_cast<vec_t *>(residual);
  auto *__restrict__ weight_v = reinterpret_cast<const vec_t *>(weight);

  int tokens_per_iter = (num_tokens + gridDim.x - 1) / gridDim.x;

  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
  __syncthreads();

  #pragma unroll
  for (int iter = 0; iter < tokens_per_iter; iter++)
  {
    int token_id = blockIdx.x + iter * gridDim.x;
    if (token_id >= num_tokens)
      continue;
    float variance[1] = {0.0f};
    const int tid = threadIdx.x;
    const int bdimx = blockDim.x;

    __shared__ float s_variance;
    int offset = token_id * vec_hidden_size;
    int offset_scalar = token_id * hidden_size;
    auto input_o = input_v + offset;
    auto residual_o = residual_v + offset;

    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      auto mtemp = multimem_ld_reduce_add<16>(mcptr + offset_scalar + idx * width);
      vec_t temp = *(reinterpret_cast<vec_t *>(&mtemp));
      // vec_t temp = input_o[idx];
      temp += residual_o[idx];
      variance[0] += temp.sum_squares(); // FP32 accumulation
      residual_o[idx] = temp;
    }

    blockReduceSum<float, 1>(variance);
    if (threadIdx.x == 0)
    {
      s_variance = rsqrtf(variance[0] / hidden_size + epsilon);
    }
    __syncthreads();

    // Second pass: normalize and apply weight
    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      vec_t shared_weight = weight_v[idx];
      vec_t temp = residual_o[idx];
      temp *= s_variance;
      temp *= shared_weight;
      input_o[idx] = temp;
    }
  }
  __syncthreads();
  sync_remote_blocks<MemOpSem::AcqRel>(signal_pads, rank, world_size);
}

/* 
* ********************************************************* *
* FUSED RS + RESIDUAL ADD + RMS NORM      CTA-BASED KERNEL  *
* GENERIC NOT SUPPORTED                                     *
* ********************************************************* *
*/
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_rs_ln_cta_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ mcptr,        // [..., hidden_size] multimem_ptr
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const float epsilon,
    const int num_tokens,
    const int hidden_size)
{
  /* Not supported */
}

/* 
* ********************************************************* *
* ALL REDUCE (Multimem)                   CTA-BASED KERNEL  *
* Function specialization in the case of      BF16 tensors. *
* ********************************************************* *
*/
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
ar_cta_kernel(
    scalar_t *__restrict__ mcptr, // [..., hidden_size] multimem_ptr
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const int num_tokens,
    const int hidden_size)
{

  // Check vectorization assumptions
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  int tokens_per_iter = (num_tokens + gridDim.x - 1) / gridDim.x;

  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
  __syncthreads();

  #pragma unroll
  for (int iter = 0; iter < tokens_per_iter; iter++)
  {
    int token_id = blockIdx.x + iter * gridDim.x;
    if (token_id >= num_tokens)
      continue;
    const int tid = threadIdx.x;
    const int bdimx = blockDim.x;

    int offset_scalar = token_id * hidden_size;
    // First pass: add + accumulate sum of squares
    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    { // all reduce
      auto mtemp = multimem_ld_reduce_add<16>(mcptr + offset_scalar + idx * width);
      multimem_st<16>(mcptr + offset_scalar + idx * width, mtemp);
    }
  }
  __syncthreads();
  sync_remote_blocks<MemOpSem::AcqRel>(signal_pads, rank, world_size);
}

/* 
* ********************************************************* *
* ALL REDUCE (Multimem)                   CTA-BASED KERNEL  *
* GENERIC NOT SUPPORTED                                     *
* ********************************************************* *
*/
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
ar_cta_kernel(
    scalar_t *__restrict__ mcptr, // [..., hidden_size] multimem_ptr
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const int num_tokens,
    const int hidden_size)
{
  /* Not supported */
}

/* 
* ********************************************************* *
* REDUCE SCATTER (Multimem)               CTA-BASED KERNEL  *
* Function specialization in the case of      BF16 tensors. *
* ********************************************************* *
*/
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
rs_cta_kernel(
    scalar_t *__restrict__ input, // [..., hidden_size]
    scalar_t *__restrict__ mcptr, // [..., hidden_size] multimem_ptr
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const int num_tokens,
    const int hidden_size)
{

  // Check vectorization assumptions
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  using vec_t = _f16Vec<scalar_t, width>;

  // Type-punned vector pointers
  auto *__restrict__ input_v = reinterpret_cast<vec_t *>(input);
  int tokens_per_iter = (num_tokens + gridDim.x - 1) / gridDim.x;

  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
  __syncthreads();

  #pragma unroll
  for (int iter = 0; iter < tokens_per_iter; iter++)
  {
    int token_id = blockIdx.x + iter * gridDim.x;
    if (token_id >= num_tokens)
      continue;
    const int tid = threadIdx.x;
    const int bdimx = blockDim.x;
    int offset = token_id * vec_hidden_size;
    int offset_scalar = token_id * hidden_size;
    auto input_o = input_v + offset;

    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      auto mtemp = multimem_ld_reduce_add<16>(mcptr + offset_scalar + idx * width);
      input_o[idx] = *(reinterpret_cast<vec_t *>(&mtemp));
    }
  }
  __syncthreads();
  sync_remote_blocks<MemOpSem::AcqRel>(signal_pads, rank, world_size);
}

/* 
* ********************************************************* *
* REDUCE SCATTER (Multimem)               CTA-BASED KERNEL  *
* GENERIC NOT SUPPORTED                                     *
* ********************************************************* *
*/
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
rs_cta_kernel(
    scalar_t *__restrict__ input, // [..., hidden_size]
    scalar_t *__restrict__ mcptr, // [..., hidden_size] multimem_ptr
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const int num_tokens,
    const int hidden_size)
{
  /* Not supported */
}

/* 
* ********************************************************* *
* ALL GATHER (Multimem)                   CTA-BASED KERNEL  *
* Function specialization in the case of      BF16 tensors. *
* ********************************************************* *
*/
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
ag_cta_kernel(
    scalar_t *__restrict__ input, // [..., hidden_size]
    scalar_t *__restrict__ mcptr, // [..., hidden_size] multimem_ptr
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const int num_tokens,
    const int hidden_size)
{

  // Check vectorization assumptions
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  using vec_t = _f16Vec<scalar_t, width>;

  // Type-punned vector pointers
  auto *__restrict__ input_v = reinterpret_cast<vec_t *>(input);

  int tokens_per_iter = (num_tokens + gridDim.x - 1) / gridDim.x;

  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
  __syncthreads();

  #pragma unroll
  for (int iter = 0; iter < tokens_per_iter; iter++)
  {
    int token_id = blockIdx.x + iter * gridDim.x;
    if (token_id >= num_tokens)
      continue;
    const int tid = threadIdx.x;
    const int bdimx = blockDim.x;
    int offset = token_id * vec_hidden_size;
    int offset_scalar = token_id * hidden_size;
    auto input_o = input_v + offset;

    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      vec_t temp = input_o[idx];
      multimem_st<16>(mcptr + offset_scalar + idx * width, *(reinterpret_cast<Vec<16> *>(&temp)));
    }
  }
  __syncthreads();
  sync_remote_blocks<MemOpSem::AcqRel>(signal_pads, rank, world_size);
}

/* 
* ********************************************************* *
* ALL GATHER (Multimem)                   CTA-BASED KERNEL  *
* GENERIC NOT SUPPORTED                                     *
* ********************************************************* *
*/
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
ag_cta_kernel(
    scalar_t *__restrict__ input, // [..., hidden_size]
    scalar_t *__restrict__ mcptr, // [..., hidden_size] multimem_ptr
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const int num_tokens,
    const int hidden_size)
{
  /* Not supported */
}

/* 
* ***************************************************************** *
* SIMPLE: FUSED RS + RESIDUAL ADD + RMS NORM + AG CTA-BASED KERNEL  *
* Function specialization in the case o  BF16 tensors.              *
* ***************************************************************** *
*/
template <typename scalar_t, int width>
__device__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
simple_fusion_add_rms_norm_cta_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    const float epsilon,
    const int num_tokens,
    const int hidden_size)
{

  // Check vectorization assumptions
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  int tokens_per_iter = (num_tokens + gridDim.x - 1) / gridDim.x;
  using vec_t = _f16Vec<scalar_t, width>;

  // Type-punned vector pointers
  auto *__restrict__ input_v = reinterpret_cast<vec_t *>(input);
  auto *__restrict__ residual_v = reinterpret_cast<vec_t *>(residual);
  auto *__restrict__ weight_v = reinterpret_cast<const vec_t *>(weight);

  #pragma unroll
  for (int iter = 0; iter < tokens_per_iter; iter++)
  {
    float variance[1] = {0.0f};
    int token_id = blockIdx.x + iter * gridDim.x;
    if (token_id >= num_tokens)
    {
      return;
    }
    const int tid = threadIdx.x;
    const int bdimx = blockDim.x;

    __shared__ float s_variance;
    int offset = token_id * vec_hidden_size;
    auto input_o = input_v + offset;
    auto residual_o = residual_v + offset;
    // First pass: add + accumulate sum of squares
    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      vec_t temp = input_o[idx];
      temp += residual_o[idx];
      variance[0] += temp.sum_squares(); // FP32 accumulation
      residual_o[idx] = temp;
    }

    blockReduceSum<float, 1>(variance);
    if (threadIdx.x == 0)
    {
      s_variance = rsqrtf(variance[0] / hidden_size + epsilon);
    }
    __syncthreads();

    // Second pass: normalize and apply weight
    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      vec_t shared_weight = weight_v[idx];
      vec_t temp = residual_o[idx];
      temp *= s_variance;
      temp *= shared_weight;
      input_o[idx] = temp;
    }
  }
}
template <typename scalar_t, int width>
__device__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
simple_fusion_add_rms_norm_cta_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size)
{
  /* Not supported */
}

template <typename scalar_t, int width>
__device__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
simple_fusion_rs_cta_kernel(
    scalar_t *__restrict__ input, // [..., hidden_size]
    scalar_t *__restrict__ mcptr, // [..., hidden_size] multimem_ptr
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const int num_tokens,
    const int hidden_size)
{
  // Check vectorization assumptions
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  using vec_t = _f16Vec<scalar_t, width>;

  // Type-punned vector pointers
  auto *__restrict__ input_v = reinterpret_cast<vec_t *>(input);
  int tokens_per_iter = (num_tokens + gridDim.x - 1) / gridDim.x;

  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
  __syncthreads();

  #pragma unroll
  for (int iter = 0; iter < tokens_per_iter; iter++)
  {
    int token_id = blockIdx.x + iter * gridDim.x;
    if (token_id >= num_tokens)
      continue;
    const int tid = threadIdx.x;
    const int bdimx = blockDim.x;
    int offset = token_id * vec_hidden_size;
    int offset_scalar = token_id * hidden_size;
    auto input_o = input_v + offset;

    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      auto mtemp = multimem_ld_reduce_add<16>(mcptr + offset_scalar + idx * width);
      input_o[idx] = *(reinterpret_cast<vec_t *>(&mtemp));
    }
  }
  __syncthreads();
  sync_remote_blocks<MemOpSem::AcqRel>(signal_pads, rank, world_size);
}

template <typename scalar_t, int width>
__device__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
simple_fusion_rs_cta_kernel(
    scalar_t *__restrict__ input, // [..., hidden_size]
    scalar_t *__restrict__ mcptr, // [..., hidden_size] multimem_ptr
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const int num_tokens,
    const int hidden_size)
{
  /* Not supported */
}

template <typename scalar_t, int width>
__device__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
simple_fusion_ag_cta_kernel(
    scalar_t *__restrict__ input, // [..., hidden_size]
    scalar_t *__restrict__ mcptr, // [..., hidden_size] multimem_ptr
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const int num_tokens,
    const int hidden_size)
{
  // Check vectorization assumptions
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  using vec_t = _f16Vec<scalar_t, width>;

  // Type-punned vector pointers
  auto *__restrict__ input_v = reinterpret_cast<vec_t *>(input);

  int tokens_per_iter = (num_tokens + gridDim.x - 1) / gridDim.x;

  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
  __syncthreads();

  #pragma unroll
  for (int iter = 0; iter < tokens_per_iter; iter++)
  {
    int token_id = blockIdx.x + iter * gridDim.x;
    if (token_id >= num_tokens)
      continue;
    const int tid = threadIdx.x;
    const int bdimx = blockDim.x;
    int offset = token_id * vec_hidden_size;
    int offset_scalar = token_id * hidden_size;
    auto input_o = input_v + offset;

    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      vec_t temp = input_o[idx];
      multimem_st<16>(mcptr + offset_scalar + idx * width, *(reinterpret_cast<Vec<16> *>(&temp)));
    }
  }
  __syncthreads();
  sync_remote_blocks<MemOpSem::AcqRel>(signal_pads, rank, world_size);
}

template <typename scalar_t, int width>
__device__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
simple_fusion_ag_cta_kernel(
    scalar_t *__restrict__ input, // [..., hidden_size]
    scalar_t *__restrict__ mcptr, // [..., hidden_size] multimem_ptr
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const int num_tokens,
    const int hidden_size)
{
  /* Not supported */
}

template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
simple_fusion_rs_ln_ag_cta_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ mcptr,        // [..., hidden_size] multimem_ptr
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const float epsilon,
    const int num_tokens,
    const int hidden_size)
{
  // Check vectorization assumptions
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  simple_fusion_rs_cta_kernel<scalar_t, width>(
      input, mcptr, signal_pads, rank, world_size, num_tokens, hidden_size);
  __syncthreads();
  simple_fusion_add_rms_norm_cta_kernel<scalar_t, width>(
      input, residual, weight, epsilon, num_tokens, hidden_size);
  __syncthreads();
  simple_fusion_ag_cta_kernel<scalar_t, width>(
      input, mcptr, signal_pads, rank, world_size, num_tokens, hidden_size);
}
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
simple_fusion_rs_ln_ag_cta_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ mcptr,        // [..., hidden_size] multimem_ptr
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const float epsilon,
    const int num_tokens,
    const int hidden_size)
{
  /* Not supported */
}

} // namespace vllm


/* 
* RMS NORM 
*/
void rms_norm(torch::Tensor &out,    // [..., hidden_size]
              torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &weight, // [hidden_size]
              double epsilon)
{
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&]
                               { vllm::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
                                     out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                     weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size); });
}

/* 
* RMS NORM IN-PLACE
*/
void rms_norm_inplace(torch::Tensor &out,    // [..., hidden_size]
                      torch::Tensor &input,  // [..., hidden_size]
                      torch::Tensor &weight, // [hidden_size]
                      double epsilon)
{
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_inplace_kernel", [&]
                               { vllm::rms_norm_inplace_kernel<scalar_t><<<grid, block, 0, stream>>>(
                                     out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                     weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size); });
}

/* 
* FUSED RESIDUAL ADD + RMS NORM 
*/
#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                                                                             \
  VLLM_DISPATCH_FLOATING_TYPES(                                                                                                      \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] { vllm::fused_add_rms_norm_kernel<scalar_t, width>                       \
                                                                  <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),           \
                                                                                               residual.data_ptr<scalar_t>(),        \
                                                                                               weight.data_ptr<scalar_t>(), epsilon, \
                                                                                               num_tokens, hidden_size); });

void fused_add_rms_norm(torch::Tensor &input,    // [..., hidden_size]
                        torch::Tensor &residual, // [..., hidden_size]
                        torch::Tensor &weight,   // [hidden_size]
                        double epsilon)
{
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0)
  {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  }
  else
  {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}

/* 
* FUSED RESIDUAL ADD + RMS NORM (CTA-BASED)
*/
#define LAUNCH_FUSED_ADD_RMS_NORM_CTA(width)                                                                                             \
  VLLM_DISPATCH_FLOATING_TYPES(                                                                                                          \
      input.scalar_type(), "fused_add_rms_norm_cta_kernel", [&] { vllm::fused_add_rms_norm_cta_kernel<scalar_t, width>                   \
                                                                      <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),           \
                                                                                                   residual.data_ptr<scalar_t>(),        \
                                                                                                   weight.data_ptr<scalar_t>(), epsilon, \
                                                                                                   num_tokens, hidden_size); });

void fused_add_rms_norm_cta(torch::Tensor &input,    // [..., hidden_size]
                            torch::Tensor &residual, // [..., hidden_size]
                            torch::Tensor &weight,   // [hidden_size]
                            int64_t MAX_CTAs,
                            double epsilon)
{
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(MAX_CTAs);                                          // full coverage
  dim3 block(std::min(1024, (hidden_size / 8 + 31) / 32 * 32)); // match kernel assumptions
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0)
  {
    LAUNCH_FUSED_ADD_RMS_NORM_CTA(8);
  }
  else
  {
    // LAUNCH_FUSED_ADD_RMS_NORM_CTA(0);
    TORCH_CHECK(false, "Input, residual, and weight tensors must be 16-byte aligned and hidden_size must be divisible by 8 for optimized kernel.");
  }
}

/* 
* ******************************************************************* *
* Fused ReduceScatter plus Fused(Residual, RMSNorm) plus AllGather    *
* ******************************************************************* *
*/
#define LAUNCH_FUSED_RS_LN_AG_CTA(width)                                                                                                   \
  VLLM_BF16_DISPATCH_FLOATING_TYPES(                                                                                                       \
      input.scalar_type(), "fused_rs_ln_ag_cta_kernel", [&] { vllm::fused_rs_ln_ag_cta_kernel<scalar_t, width>                             \
                                                                  <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),                 \
                                                                                               reinterpret_cast<scalar_t *>(mcptr),        \
                                                                                               residual.data_ptr<scalar_t>(),              \
                                                                                               weight.data_ptr<scalar_t>(),                \
                                                                                               reinterpret_cast<uint32_t **>(signal_pads), \
                                                                                               static_cast<size_t>(rank),                  \
                                                                                               static_cast<size_t>(world_size),            \
                                                                                               epsilon, num_tokens, hidden_size); });
void fused_rs_ln_ag_cta(torch::Tensor &input,    // [..., hidden_size]
                        torch::Tensor &residual, // [..., hidden_size]
                        torch::Tensor &weight,   // [hidden_size]
                        int64_t mcptr,           // [..., hidden_size] multimem_ptr
                        int64_t signal_pads,     // [..., hidden_size] signal pads
                        int64_t rank,
                        int64_t world_size,
                        int64_t MAX_CTAs,
                        double epsilon)
{
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(MAX_CTAs);                                          // full coverage
  dim3 block(std::min(1024, (hidden_size / 8 + 31) / 32 * 32)); // match kernel assumptions
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0)
  {
    LAUNCH_FUSED_RS_LN_AG_CTA(8);
  }
  else
  {
    // LAUNCH_FUSED_ADD_RMS_NORM_CTA(0);
    TORCH_CHECK(false, "Input, residual, and weight tensors must be 16-byte aligned and hidden_size must be divisible by 8 for optimized kernel.");
  }
}

/* 
* ******************************************************************* *
* Fused ReduceScatter plus Fused(Residual, RMSNorm)                   *
* ******************************************************************* *
*/
#define LAUNCH_FUSED_RS_LN_CTA(width)                                                                                                   \
  VLLM_BF16_DISPATCH_FLOATING_TYPES(                                                                                                    \
      input.scalar_type(), "fused_rs_ln_cta_kernel", [&] { vllm::fused_rs_ln_cta_kernel<scalar_t, width>                                \
                                                               <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),                 \
                                                                                            reinterpret_cast<scalar_t *>(mcptr),        \
                                                                                            residual.data_ptr<scalar_t>(),              \
                                                                                            weight.data_ptr<scalar_t>(),                \
                                                                                            reinterpret_cast<uint32_t **>(signal_pads), \
                                                                                            static_cast<size_t>(rank),                  \
                                                                                            static_cast<size_t>(world_size),            \
                                                                                            epsilon, num_tokens, hidden_size); });
void fused_rs_ln_cta(torch::Tensor &input,    // [..., hidden_size]
                     torch::Tensor &residual, // [..., hidden_size]
                     torch::Tensor &weight,   // [hidden_size]
                     int64_t mcptr,           // [..., hidden_size] multimem_ptr
                     int64_t signal_pads,     // [..., hidden_size] signal pads
                     int64_t rank,
                     int64_t world_size,
                     int64_t MAX_CTAs,
                     double epsilon)
{
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(MAX_CTAs);                                          // full coverage
  dim3 block(std::min(1024, (hidden_size / 8 + 31) / 32 * 32)); // match kernel assumptions
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0)
  {
    LAUNCH_FUSED_RS_LN_CTA(8);
  }
  else
  {
    // LAUNCH_FUSED_ADD_RMS_NORM_CTA(0);
    TORCH_CHECK(false, "Input, residual, and weight tensors must be 16-byte aligned and hidden_size must be divisible by 8 for optimized kernel.");
  }
}

/* 
* ******************************************************************* *
* Multimem AllReduce (CTA-Based)                                      *
* ******************************************************************* *
*/
#define LAUNCH_AR_CTA(width)                                                                                                   \
  VLLM_BF16_DISPATCH_FLOATING_TYPES(                                                                                           \
      input.scalar_type(), "ar_cta_kernel", [&] { vllm::ar_cta_kernel<scalar_t, width>                                         \
                                                      <<<grid, block, 0, stream>>>(reinterpret_cast<scalar_t *>(mcptr),        \
                                                                                   reinterpret_cast<uint32_t **>(signal_pads), \
                                                                                   static_cast<size_t>(rank),                  \
                                                                                   static_cast<size_t>(world_size),            \
                                                                                   num_tokens, hidden_size); });
void multimem_ar_cta(torch::Tensor &input, // [..., hidden_size]
                     int64_t mcptr,        // [..., hidden_size] multimem_ptr
                     int64_t signal_pads,  // [..., hidden_size] signal pads
                     int64_t rank,
                     int64_t world_size,
                     int64_t MAX_CTAs)
{
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(MAX_CTAs);                                          // full coverage
  dim3 block(std::min(1024, (hidden_size / 8 + 31) / 32 * 32)); // match kernel assumptions
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  if (hidden_size % 8 == 0)
  {
    LAUNCH_AR_CTA(8);
  }
  else
  {
    TORCH_CHECK(false, "hidden_size must be divisible by 8 for optimized kernel.");
  }
}

/* 
* ******************************************************************* *
* Multimem ReduceScatter (CTA-Based)                                  *
* ******************************************************************* *
*/
#define LAUNCH_RS_CTA(width)                                                                                                   \
  VLLM_BF16_DISPATCH_FLOATING_TYPES(                                                                                           \
      input.scalar_type(), "rs_cta_kernel", [&] { vllm::rs_cta_kernel<scalar_t, width>                                         \
                                                      <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),                 \
                                                                                   reinterpret_cast<scalar_t *>(mcptr),        \
                                                                                   reinterpret_cast<uint32_t **>(signal_pads), \
                                                                                   static_cast<size_t>(rank),                  \
                                                                                   static_cast<size_t>(world_size),            \
                                                                                   num_tokens, hidden_size); });
void multimem_rs_cta(torch::Tensor &input, // [..., hidden_size]
                     int64_t mcptr,        // [..., hidden_size] multimem_ptr
                     int64_t signal_pads,  // [..., hidden_size] signal pads
                     int64_t rank,
                     int64_t world_size,
                     int64_t MAX_CTAs)
{
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(MAX_CTAs);                                          // full coverage
  dim3 block(std::min(1024, (hidden_size / 8 + 31) / 32 * 32)); // match kernel assumptions
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  bool ptrs_are_aligned = inp_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0)
  {
    LAUNCH_RS_CTA(8);
  }
  else
  {
    // LAUNCH_FUSED_ADD_RMS_NORM_CTA(0);
    TORCH_CHECK(false, "Input tensors must be 16-byte aligned and hidden_size must be divisible by 8 for optimized kernel.");
  }
}

/* 
* ******************************************************************* *
* Multimem AllGather (CTA-Based)                                      *
* ******************************************************************* *
*/
#define LAUNCH_AG_CTA(width)                                                                                                   \
  VLLM_BF16_DISPATCH_FLOATING_TYPES(                                                                                           \
      input.scalar_type(), "ag_cta_kernel", [&] { vllm::ag_cta_kernel<scalar_t, width>                                         \
                                                      <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),                 \
                                                                                   reinterpret_cast<scalar_t *>(mcptr),        \
                                                                                   reinterpret_cast<uint32_t **>(signal_pads), \
                                                                                   static_cast<size_t>(rank),                  \
                                                                                   static_cast<size_t>(world_size),            \
                                                                                   num_tokens, hidden_size); });
void multimem_ag_cta(torch::Tensor &input, // [..., hidden_size]
                     int64_t mcptr,        // [..., hidden_size] multimem_ptr
                     int64_t signal_pads,  // [..., hidden_size] signal pads
                     int64_t rank,
                     int64_t world_size,
                     int64_t MAX_CTAs)
{
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(MAX_CTAs);                                          // full coverage
  dim3 block(std::min(1024, (hidden_size / 8 + 31) / 32 * 32)); // match kernel assumptions
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  bool ptrs_are_aligned = inp_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0)
  {
    LAUNCH_AG_CTA(8);
  }
  else
  {
    // LAUNCH_FUSED_ADD_RMS_NORM_CTA(0);
    TORCH_CHECK(false, "Input tensors must be 16-byte aligned and hidden_size must be divisible by 8 for optimized kernel.");
  }
}

/* 
* ************************************************************************** *
* SimpleFusion ReduceScatter plus Fused(Residual, RMSNorm) plus AllGather    *
* ************************************************************************** *
*/
#define LAUNCH_SIMPLE_FUSION_RS_LN_AG_CTA(width)                                                                                                   \
  VLLM_BF16_DISPATCH_FLOATING_TYPES(                                                                                                          \
      input.scalar_type(), "simple_fusion_rs_ln_ag_cta_kernel", [&] { vllm::simple_fusion_rs_ln_ag_cta_kernel<scalar_t, width>                          \
                                                                     <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),                 \
                                                                                                  reinterpret_cast<scalar_t *>(mcptr),        \
                                                                                                  residual.data_ptr<scalar_t>(),              \
                                                                                                  weight.data_ptr<scalar_t>(),                \
                                                                                                  reinterpret_cast<uint32_t **>(signal_pads), \
                                                                                                  static_cast<size_t>(rank),                  \
                                                                                                  static_cast<size_t>(world_size),            \
                                                                                                  epsilon, num_tokens, hidden_size); });
void simple_fusion_rs_ln_ag_cta(torch::Tensor &input,    // [..., hidden_size]
                           torch::Tensor &residual, // [..., hidden_size]
                           torch::Tensor &weight,   // [hidden_size]
                           int64_t mcptr,           // [..., hidden_size] multimem_ptr
                           int64_t signal_pads,     // [..., hidden_size] signal pads
                           int64_t rank,
                           int64_t world_size,
                           int64_t MAX_CTAs,
                           double epsilon)
{
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(MAX_CTAs);                                          // full coverage
  dim3 block(std::min(1024, (hidden_size / 8 + 31) / 32 * 32)); // match kernel assumptions
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0)
  {
    LAUNCH_SIMPLE_FUSION_RS_LN_AG_CTA(8);
  }
  else
  {
    // LAUNCH_FUSED_ADD_RMS_NORM_CTA(0);
    TORCH_CHECK(false, "Input, residual, and weight tensors must be 16-byte aligned and hidden_size must be divisible by 8 for optimized kernel.");
  }
}