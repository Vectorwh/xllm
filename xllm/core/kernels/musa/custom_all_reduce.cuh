// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-FileCopyrightText: Copyright (c) 2026, Moore Threads Technology Co.,
// Ltd.

#pragma once

#include <glog/logging.h>

#include "core/kernels/musa/custom_collective_common.cuh"

namespace xllm::kernel::musa {
// Maximal number of blocks in allreduce kernel.
constexpr int32_t kMaxThreadsPerBlock = 1024;

template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
    cross_device_reduce_1stage(RankData* _dp,
                               RankSignals sg,
                               Signal* self_sg,
                               T* __restrict__ result,
                               int32_t rank,
                               int32_t size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  // do the actual reduction
  for (int32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    reinterpret_cast<P*>(result)[idx] = packed_reduce<P, ngpus, A>(
        reinterpret_cast<const P**>(&dp.ptrs[0]), idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return reinterpret_cast<P*>(sg + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
    cross_device_reduce_2stage(RankData* _dp,
                               RankSignals sg,
                               Signal* self_sg,
                               T* __restrict__ result,
                               int32_t rank,
                               int32_t size) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int32_t part = size / ngpus;
  int32_t start = rank * part;
  int32_t end = rank == ngpus - 1 ? size : start + part;
  int32_t largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int32_t i = 0; i < ngpus; i++) {
    int32_t target = (rank + i) % ngpus;
    ptrs[i] = reinterpret_cast<const P*>(_dp->ptrs[target]);
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  barrier_at_start<ngpus>(sg, self_sg, rank);

  // stage 1: reduce scatter
  for (int32_t idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  barrier_at_end<ngpus>(sg, self_sg, rank);

  // stage 2: allgather. Note: it's important to match the tid between
  // the two stages, because visibility across devices is only guaranteed
  // between threads that have the same tid. If thread i computes the sum of
  // start + i in the first stage, then thread i also gathers start + i from
  // all ranks.

  for (int32_t idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int32_t i = 0; i < ngpus; i++) {
      int32_t gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int32_t dst_idx = gather_from_rank * part + idx;
        reinterpret_cast<P*>(result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

class CustomAllreduce final {
 public:
  int32_t rank_;
  int32_t world_size_;
  // 0 selects the size heuristic, 1 forces one-stage, 2 forces two-stage.
  int32_t algorithm_;

  RankSignals sg_;
  // Stores a map from a pointer to its peer pointers from all ranks.
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  // Stores rank data from all ranks (peer pointers registered via
  // register_buffer). Eager only; graph capture is routed to MCCL upstream.
  RankData *d_rank_data_base_, *d_rank_data_end_;

  /**
   * Signals are an array of ipc-enabled buffers from all ranks.
   * For each of the buffer, the layout is as follows:
   * | -- sizeof(Signal) -- | ------ a few MB ----- |
   * The first section is for allreduce synchronization, and the second
   * section is for storing the intermediate results required by some
   * allreduce algos.
   *
   * Note: this class does not own any device memory. Any required buffers
   * are passed in from the constructor.
   */
  CustomAllreduce(Signal** signals,
                  void* rank_data,
                  size_t rank_data_sz,
                  int32_t rank,
                  int32_t world_size)
      : rank_(rank),
        world_size_(world_size),
        algorithm_(0),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int32_t i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
    }
  }

  void set_algorithm(int32_t algorithm) { algorithm_ = algorithm; }

  void check_rank_data_capacity(size_t num = 1) {
    CHECK(d_rank_data_base_ + num <= d_rank_data_end_)
        << "Rank data buffer is overflowed by "
        << (d_rank_data_base_ + num - d_rank_data_end_);
  }

  /**
   * Register already-shared IPC pointers.
   */
  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) {
      data.ptrs[i] = ptrs[i];
    }
    auto d_data = d_rank_data_base_++;
    CUDACHECK(
        cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  /**
   * Performs allreduce, assuming input has already been registered.
   *
   * Block and grid default configs are results after careful grid search.
   * Using 36 blocks give the best or close to the best runtime on the devices
   * I tried: A100, A10, A30, T4, V100. You'll notice that NCCL kernels also
   * only take a small amount of SMs. Not quite sure the underlying reason,
   * but my guess is that too many SMs will cause contention on NVLink bus.
   */
  template <typename T>
  void allreduce(cudaStream_t stream,
                 T* input,
                 T* output,
                 int32_t size,
                 int32_t threads = 512,
                 int32_t block_limit = defaultBlockLimit) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0) {
      LOG(FATAL) << "custom allreduce currently requires input length to be "
                    "multiple of "
                 << d;
    }
    CHECK_LE(block_limit, kMaxBlocks) << "max supported block limit is "
                                      << kMaxBlocks << ". Got " << block_limit;

    auto it = buffers_.find(input);
    CHECK(it != buffers_.end())
        << "buffer address " << reinterpret_cast<uint64_t>(input)
        << " is not registered!";
    RankData* ptrs = it->second;

    size /= d;
    auto bytes = size * sizeof(typename packed_t<T>::P);
    int32_t blocks = std::min(block_limit, (size + threads - 1) / threads);

    // Parse the algorithm selector once. Only XLLM_CUSTOM_ALLREDUCE_ALGO is
    // read; when it is unset the size-based dispatch below decides.
    static const int32_t kEnvAlgo = [] {
      const char* env_algo = std::getenv("XLLM_CUSTOM_ALLREDUCE_ALGO");
      if (env_algo == nullptr) {
        return 0;
      }
      if (std::strcmp(env_algo, "1stage") == 0 ||
          std::strcmp(env_algo, "oneshot") == 0) {
        return 1;
      }
      if (std::strcmp(env_algo, "2stage") == 0 ||
          std::strcmp(env_algo, "twoshot") == 0) {
        return 2;
      }
      LOG(FATAL) << "Invalid XLLM_CUSTOM_ALLREDUCE_ALGO: " << env_algo
                 << ". Valid values: 1stage, oneshot, 2stage, twoshot";
    }();
    bool force_1stage = (kEnvAlgo == 1);
    bool force_2stage = (kEnvAlgo == 2);
    // The validated MUSA implementation uses size-based dispatch for FP32 and
    // its selector applies to reduced-precision kernels only: at world>2 a
    // two-stage (rotated) FP32 reduction order differs from the size-based one
    // and would change residual/HC values. At world==2 the reduction is a
    // single commutative 2-term add, so it is order-independent -- there we do
    // honor the selector, so the wrapper's two-stage policy for world==2 (TP2 /
    // cp_group / world_group) is not silently ignored (the two-layer contract).
    const bool fp32_exempt = std::is_same<T, float>::value && world_size_ != 2;
    if (!force_1stage && !force_2stage && !fp32_exempt) {
      if (algorithm_ == 1) {
        force_1stage = true;
      } else if (algorithm_ == 2) {
        force_2stage = true;
      }
    }
    // A selector set by the wrapper is intentionally ignored for FP32 at
    // world>2 (size-based dispatch above); surface it once instead of letting
    // the configured algorithm be silently dropped. algorithm_ == 0 means the
    // wrapper never set one, so only warn on a real (ignored) request.
    if (fp32_exempt && algorithm_ != 0) {
      LOG_FIRST_N(WARNING, 1)
          << "Custom all-reduce: algorithm selector " << algorithm_
          << " is ignored for FP32 at world_size=" << world_size_
          << "; size-based dispatch is used instead";
    }

#define KL(ngpus, name)                           \
  name<T, ngpus><<<blocks, threads, 0, stream>>>( \
      ptrs, sg_, self_sg_, output, rank_, size);
#define REDUCE_CASE(ngpus)                              \
  case ngpus: {                                         \
    if (force_1stage) {                                 \
      KL(ngpus, cross_device_reduce_1stage);            \
    } else if (force_2stage) {                          \
      KL(ngpus, cross_device_reduce_2stage);            \
    } else {                                            \
      if (world_size_ == 2) {                           \
        KL(ngpus, cross_device_reduce_1stage);          \
      } else {                                          \
        if ((world_size_ <= 4 && bytes < 512 * 1024) || \
            (world_size_ <= 8 && bytes < 256 * 1024)) { \
          KL(ngpus, cross_device_reduce_1stage);        \
        } else {                                        \
          KL(ngpus, cross_device_reduce_2stage);        \
        }                                               \
      }                                                 \
    }                                                   \
    break;                                              \
  }

    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        LOG(FATAL) << "custom allreduce only supports num gpus in "
                      "(2,4,6,8). Actual num gpus = "
                   << world_size_;
    }
#undef REDUCE_CASE
#undef KL
  }

  ~CustomAllreduce() = default;
};

/**
 * To inspect PTX/SASS, copy paste this header file to compiler explorer and
 * add a template instantiation:
 * template void
 * xllm::kernel::musa::CustomAllreduce::allreduce<half>(cudaStream_t, half *,
 *                                                       half *, int, int, int);
 */
}  // namespace xllm::kernel::musa
