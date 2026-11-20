/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>
#include <musa_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <tuple>

#include "core/framework/parallel_state/musa_process_group.h"
#include "core/util/env_var.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace xllm {

void register_musa_custom_all_reduce_kernel();

namespace {

// Bounded store wait for cp_group custom-AR init. The default rendezvous store
// timeout is 15 minutes; a peer that dies mid-way through
// create_shared_buffer's collective would otherwise block its partner for that
// whole window. 120s is far above a normal 2-rank IPC init. On timeout the wait
// throws, init()'s catch marks the rank failed, and Phase C degrades both ranks
// to MCCL consistently instead of hanging.
constexpr std::chrono::seconds kCpInitStoreTimeout{120};

// Publishes this rank's `selected` byte under `prefix`, waits for every
// rank, then verifies all ranks agree -- the store rendezvous shared by the
// TP generic policy, the CP intent exchange (Phase A) and the CP outcome
// exchange (Phase C). `timeout` bounds the wait -- every caller passes
// kCpInitStoreTimeout so a dead peer fails fast rather than blocking for the
// store's default 15-minute rendezvous. Throws (from wait) on timeout; the
// CP callers catch it to emit their phase-specific diagnostics.
void exchange_store_byte(c10::intrusive_ptr<c10d::Store> store,
                         int32_t rank,
                         int32_t world_size,
                         const std::string& prefix,
                         uint8_t selected,
                         std::chrono::seconds timeout,
                         const std::string& phase_desc) {
  std::vector<std::string> keys;
  keys.reserve(static_cast<size_t>(world_size));
  for (int32_t peer = 0; peer < world_size; ++peer) {
    keys.emplace_back(prefix + std::to_string(peer));
  }
  store->set(keys[rank], std::vector<uint8_t>{selected});
  store->wait(keys, timeout);
  for (int32_t peer = 0; peer < world_size; ++peer) {
    CHECK(store->get(keys[peer]) == std::vector<uint8_t>{selected})
        << "MUSA " << phase_desc
        << " must match on every rank; a split would deadlock the "
           "collective. rank="
        << rank << " selected=" << static_cast<int>(selected)
        << " peer=" << peer;
  }
}

int64_t op_meta_size() {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("_C_custom_ar::meta_size", "")
                       .typed<int64_t()>();
  return op.call();
}

std::tuple<int64_t, torch::Tensor> op_allocate_shared_buffer_and_handle(
    int64_t size) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("_C_custom_ar::allocate_shared_buffer_and_handle",
                             "")
          .typed<std::tuple<int64_t, torch::Tensor>(int64_t)>();
  return op.call(size);
}

int64_t op_open_mem_handle(torch::Tensor& handle) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("_C_custom_ar::open_mem_handle", "")
                       .typed<int64_t(torch::Tensor&)>();
  return op.call(handle);
}

int64_t op_init_custom_ar(const std::vector<int64_t>& ipc_ptrs,
                          torch::Tensor& rank_data,
                          int64_t rank) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("_C_custom_ar::init_custom_ar", "")
          .typed<int64_t(
              const std::vector<int64_t>&, torch::Tensor&, int64_t)>();
  return op.call(ipc_ptrs, rank_data, rank);
}

void op_set_algorithm(int64_t custom_ar_handle, int64_t algorithm) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("_C_custom_ar::set_algorithm", "")
                       .typed<void(int64_t, int64_t)>();
  op.call(custom_ar_handle, algorithm);
}

void op_register_buffer(int64_t custom_ar_handle,
                        const std::vector<int64_t>& ipc_ptrs) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("_C_custom_ar::register_buffer", "")
                       .typed<void(int64_t, const std::vector<int64_t>&)>();
  op.call(custom_ar_handle, ipc_ptrs);
}

void op_all_reduce(int64_t custom_ar_handle,
                   const torch::Tensor& input,
                   torch::Tensor& output,
                   int64_t registered_buffer,
                   int64_t registered_buffer_size) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("_C_custom_ar::all_reduce", "")
                       .typed<void(int64_t,
                                   const torch::Tensor&,
                                   torch::Tensor&,
                                   int64_t,
                                   int64_t)>();
  op.call(custom_ar_handle,
          input,
          output,
          registered_buffer,
          registered_buffer_size);
}

void op_dispose(int64_t custom_ar_handle) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("_C_custom_ar::dispose", "")
                       .typed<void(int64_t)>();
  op.call(custom_ar_handle);
}

void op_free_shared_buffer(int64_t ptr) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("_C_custom_ar::free_shared_buffer", "")
          .typed<void(int64_t)>();
  op.call(ptr);
}

// Release every entry of an IPC pointer table: rank `rank`'s own allocation is
// cudaFree'd (op_free_shared_buffer), and every peer entry -- opened with
// cudaIpcOpenMemHandle inside op_open_mem_handle -- is closed with the matching
// IPC close. Used by the destructor and the init()/create_shared_buffer failure
// paths so a partially initialized instance never leaks peer IPC mappings.
void release_shared_buffers(std::vector<int64_t>& ptrs, int32_t rank) {
  for (int32_t i = 0; i < static_cast<int32_t>(ptrs.size()); ++i) {
    if (ptrs[i] == 0) {
      continue;
    }
    if (i == rank) {
      op_free_shared_buffer(ptrs[i]);
    } else {
      musaIpcCloseMemHandle(reinterpret_cast<void*>(ptrs[i]));
    }
    ptrs[i] = 0;
  }
  ptrs.clear();
}

bool supported_world_size(int32_t world_size) {
  return world_size == 2 || world_size == 4 || world_size == 6 ||
         world_size == 8;
}

bool supported_dtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 ||
         dtype == torch::kBFloat16;
}

bool is_weak_contiguous(const torch::Tensor& tensor) {
  if (tensor.is_contiguous()) {
    return true;
  }
  const int64_t storage_nbytes =
      static_cast<int64_t>(tensor.storage().nbytes());
  return storage_nbytes - tensor.storage_offset() * tensor.element_size() ==
         tensor.numel() * tensor.element_size();
}

bool is_16_byte_aligned(const torch::Tensor& tensor) {
  return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0;
}

}  // namespace

bool musa_custom_allreduce_enabled(ProcessGroup* process_group) {
  auto* musa_process_group = dynamic_cast<ProcessGroupImpl*>(process_group);
  if (musa_process_group == nullptr) {
    return false;
  }
  return musa_process_group->custom_allreduce_enabled();
}

std::optional<torch::Tensor> musa_allreduce_custom(
    ProcessGroup* process_group,
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& output) {
  ProcessGroupImpl* musa_process_group =
      dynamic_cast<ProcessGroupImpl*>(process_group);
  if (musa_process_group == nullptr) {
    return std::nullopt;
  }
  return musa_process_group->allreduce_custom(input, output);
}

std::optional<torch::Tensor> musa_allreduce_tp_custom(
    ProcessGroup* process_group,
    const torch::Tensor& input) {
  ProcessGroupImpl* musa_process_group =
      dynamic_cast<ProcessGroupImpl*>(process_group);
  if (musa_process_group == nullptr) {
    return std::nullopt;
  }
  return musa_process_group->allreduce_tp_custom(input);
}

MusaCustomAllReduce::MusaCustomAllReduce(int32_t rank,
                                         int32_t world_size,
                                         std::string group_name,
                                         const torch::Device& device,
                                         c10::intrusive_ptr<c10d::Store> store)
    : rank_(rank),
      world_size_(world_size),
      group_name_(std::move(group_name)),
      device_(device),
      store_(std::move(store)),
      enable_tp_allreduce_(
          util::get_bool_env("XLLM_MUSA_ENABLE_TP_CUSTOM_AR", true)) {
  // Production creates only tp_group and cp_group instances.
  // TP enters via allreduce_tp; CP via allreduce (musa_allreduce_custom).
  // Both share allreduce and IPC; group_name selects the startup policy.
  if (group_name_ == "tp_group" && world_size_ > 1) {
    CHECK(store_ != nullptr);
    const uint8_t selected = enable_tp_allreduce_ ? 1 : 0;
    exchange_store_byte(store_,
                        rank_,
                        world_size_,
                        "xllm_custom_ar/tp_group/generic_policy/",
                        selected,
                        kCpInitStoreTimeout,
                        "TP custom-allreduce policy");
    LOG(INFO) << "[MUSA_TP_REDUCE_POLICY] group=" << group_name_
              << ", rank=" << rank_ << ", world_size=" << world_size_
              << ", enabled=" << enable_tp_allreduce_
              << ", max_bytes=" << max_size_bytes_;
  }
  if (group_name_ != "tp_group" && group_name_ != "cp_group") {
    return;
  }
  if (!supported_world_size(world_size_)) {
    // Unsupported group size is a uniform, pre-init property: every rank
    // of the group sees the same world_size, so disabling the custom path
    // here cannot split the backend across ranks. Flip the TP selector too
    // so allreduce_tp() returns nullopt and every rank degrades to MCCL
    // instead of tripping CHECK(enabled_) farther down.
    enable_tp_allreduce_ = false;
    LOG(WARNING) << "Custom all-reduce disabled for " << group_name_
                 << ": unsupported world size " << world_size_
                 << ", falling back to MCCL";
    return;
  }
  if (group_name_ == "tp_group") {
    // An explicitly selected collective must not silently change backend on
    // only one rank after an IPC/device initialization failure.
    if (enable_tp_allreduce_) {
      init();
    }
    return;
  }
  // cp_group is the only group left. A per-rank soft failure or env drift
  // would put one rank on custom AR and the other on MCCL, deadlocking both
  // with no diagnostics, so enforce one agreed backend across all CP ranks.
  init_cp_group_consistently();
}

void MusaCustomAllReduce::init_cp_group_consistently() {
  // A split backend on cp_group is an undiagnosable deadlock: the custom kernel
  // spins forever in barrier_at_start while the peer waits in the MCCL
  // broadcast. We agree on the backend across every CP rank at startup. Policy:
  //   * store null, or a consensus-exchange failure/timeout (a CP peer never
  //     reached init, or intents/outcomes mismatch) -> fail fast (abort): the
  //     backend cannot be reconciled, and a one-sided MCCL degrade would split.
  //   * an init() IPC failure AFTER intent consensus -> degrade: both ranks
  //     reach Phase C and disable custom AR consistently (MCCL on both).
  CHECK(store_ != nullptr) << "CP custom all-reduce requires a c10d store.";
  // Different costs produce incompatible row counts for CP collectives.
  const std::array<double, 2>& costs = util::get_musa_pcp_costs();
  std::vector<uint8_t> cost_bytes(costs.size() * sizeof(double));
  std::memcpy(cost_bytes.data(), costs.data(), cost_bytes.size());
  std::vector<std::string> cost_keys;
  cost_keys.reserve(static_cast<size_t>(world_size_));
  for (int32_t peer = 0; peer < world_size_; ++peer) {
    cost_keys.emplace_back("xllm_pcp/costs/" + std::to_string(peer));
  }
  store_->set(cost_keys[rank_], cost_bytes);
  store_->wait(cost_keys, kCpInitStoreTimeout);
  for (int32_t peer = 0; peer < world_size_; ++peer) {
    CHECK(store_->get(cost_keys[peer]) == cost_bytes)
        << "MUSA PCP costs must match on every CP rank: "
           "XLLM_MUSA_PCP_COST_PAIR and XLLM_MUSA_PCP_COST_OTHER_KV_ROW; rank="
        << rank_ << " peer=" << peer;
  }
  const bool intent = util::get_bool_env("XLLM_MUSA_PCP_CUSTOM_AR", false);
  // Phase A: agree on intent BEFORE any IPC handle exchange. init()'s
  // create_shared_buffer() runs collective store.set/wait + peer handle opens;
  // a rank that splits off after peers entered that collective leaves them
  // blocked in store.wait instead of failing fast. Publishing intent first
  // catches any env/config split with no IPC in flight.
  {
    const uint8_t intent_byte = intent ? 1 : 0;
    try {
      exchange_store_byte(
          store_,
          rank_,
          world_size_,
          "xllm_custom_ar/cp_group/intent/",
          intent_byte,
          kCpInitStoreTimeout,
          "CP custom all-reduce intent (XLLM_MUSA_PCP_CUSTOM_AR)");
    } catch (const std::exception& e) {
      LOG(FATAL)
          << "cp_group custom all-reduce intent consensus failed on rank "
          << rank_ << ": " << e.what()
          << ". A CP peer did not reach init within "
          << kCpInitStoreTimeout.count()
          << "s; cannot agree on a consistent CP backend.";
    }
  }
  if (!intent) {
    enabled_ = false;
    LOG(INFO) << "[MUSA_CP_REDUCE_POLICY] group=" << group_name_
              << ", rank=" << rank_ << ", world_size=" << world_size_
              << ", intent=0, enabled=0";
    return;
  }
  // Phase B: all ranks intend custom -> run the IPC init.
  bool init_ok = false;
  try {
    init();
    init_ok = enabled_;
  } catch (const std::exception& e) {
    LOG(WARNING) << "CP custom all-reduce init failed on rank " << rank_ << ": "
                 << e.what();
    enabled_ = false;
  }
  // Phase C: require the effective outcome (init success) to agree too, so a
  // rank that failed *after* the collective setup is caught at startup rather
  // than splitting at runtime. A failure *inside* create_shared_buffer's
  // collective is now bounded by kCpInitStoreTimeout, so the partner trips its
  // wait within that window and both ranks degrade to MCCL consistently here.
  const uint8_t selected = init_ok ? 1 : 0;
  try {
    exchange_store_byte(store_,
                        rank_,
                        world_size_,
                        "xllm_custom_ar/cp_group/policy/",
                        selected,
                        kCpInitStoreTimeout,
                        "CP custom all-reduce backend");
  } catch (const std::exception& e) {
    LOG(FATAL)
        << "cp_group custom all-reduce backend consensus failed on rank "
        << rank_ << ": " << e.what()
        << ". A CP peer did not reach the backend exchange; cannot agree "
           "on a consistent CP backend.";
  }
  enabled_ = (selected != 0);
  LOG(INFO) << "[MUSA_CP_REDUCE_POLICY] group=" << group_name_
            << ", rank=" << rank_ << ", world_size=" << world_size_
            << ", intent=" << intent << ", init_ok=" << init_ok
            << ", enabled=" << enabled_;
}

MusaCustomAllReduce::~MusaCustomAllReduce() {
  const c10::DeviceGuard device_guard(device_);
  if (custom_ar_handle_ != 0) {
    op_dispose(custom_ar_handle_);
    custom_ar_handle_ = 0;
  }
  // Free the local allocation and close every peer IPC mapping for both tables
  // (symmetric meta/buffer release). Previously only [rank_] was freed, leaking
  // all peer IPC mappings.
  release_shared_buffers(meta_ptrs_, rank_);
  release_shared_buffers(buffer_ptrs_, rank_);
}

void MusaCustomAllReduce::init() {
  // Keep the native dispatcher registrations when linking static archives.
  register_musa_custom_all_reduce_kernel();
  const c10::DeviceGuard device_guard(device_);
  CHECK(store_ != nullptr) << "Custom all-reduce requires a c10d store.";

  const int64_t meta_bytes = op_meta_size();
  // ONLY world_size == 2 (TP2) uses two-stage communication; TP>2 keeps
  // one-stage, which is verified correct in production. TP2 is pinned to
  // two-stage because the one-stage MUSA path produced malformed outputs in a
  // world_size == 2 mixed-request Qwen3.8 MTP evaluation (fix confirmed in
  // production), matching the existing world-group two-stage policy. The
  // kernel honors this selector for reduced precision unconditionally and for
  // FP32 only at world==2 (single commutative 2-term add, order-independent;
  // world>2 FP32 stays size-based), so world==2 always runs two-stage
  const bool tp2_generic =
      enable_tp_allreduce_ && group_name_ == "tp_group" && world_size_ == 2;
  const int64_t algorithm = group_name_ == "tp_group" && !tp2_generic ? 1 : 2;
  // Allocate the meta/buffer IPC tables and initialize the handle. On any
  // failure, dispose a half-initialized handle (op_init_custom_ar may have
  // succeeded before op_set_algorithm/op_register_buffer threw), reset
  // enabled_, and release both tables (own allocation + peer IPC mappings).
  // This MUST happen here: an explicitly selected tp_group propagates the
  // exception out of the constructor, so the destructor never runs to clean up;
  // cp_group's caller catches it and degrades to MCCL.
  auto abort_init = [this]() {
    if (custom_ar_handle_ != 0) {
      op_dispose(custom_ar_handle_);
      custom_ar_handle_ = 0;
    }
    enabled_ = false;
    release_shared_buffers(meta_ptrs_, rank_);
    release_shared_buffers(buffer_ptrs_, rank_);
    rank_data_ = torch::Tensor();
  };
  try {
    meta_ptrs_ = create_shared_buffer(meta_bytes + max_size_bytes_, "meta");
    buffer_ptrs_ = create_shared_buffer(max_size_bytes_, "buffer");
    rank_data_ = torch::empty(
        {kMusaCustomArMaxBytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    custom_ar_handle_ = op_init_custom_ar(meta_ptrs_, rank_data_, rank_);
    op_set_algorithm(custom_ar_handle_, algorithm);
    op_register_buffer(custom_ar_handle_, buffer_ptrs_);
    enabled_ = true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Custom all-reduce init failed for group=" << group_name_
               << ", rank=" << rank_ << ": " << e.what();
    abort_init();
    throw;
  } catch (...) {
    LOG(ERROR) << "Custom all-reduce init failed for group=" << group_name_
               << ", rank=" << rank_ << ": unknown error";
    abort_init();
    throw;
  }

  LOG(INFO) << "Custom all-reduce enabled for group " << group_name_
            << ", rank=" << rank_ << ", world_size=" << world_size_
            << ", algorithm=" << algorithm << ", max_bytes=" << max_size_bytes_;
}

std::vector<int64_t> MusaCustomAllReduce::create_shared_buffer(
    int64_t size_in_bytes,
    const std::string& name) {
  const c10::DeviceGuard device_guard(device_);
  auto [local_ptr, local_handle] =
      op_allocate_shared_buffer_and_handle(size_in_bytes);

  const std::string key_prefix =
      "xllm_custom_ar/" + group_name_ + "/" + name + "/";
  const std::string local_key = key_prefix + std::to_string(rank_);
  std::vector<std::string> keys;
  keys.reserve(world_size_);
  for (int32_t i = 0; i < world_size_; ++i) {
    keys.push_back(key_prefix + std::to_string(i));
  }
  // Track this rank's local allocation from the start so any throw below -- a
  // store set error, or a bounded wait timeout when a peer fails mid-way -- is
  // cleaned up in the catch instead of leaking the local device buffer.
  std::vector<int64_t> ptrs(world_size_, 0);
  ptrs[rank_] = local_ptr;
  try {
    store_->set(local_key, tensor_to_bytes(local_handle));
    // Bound the init handle exchange: a peer failing mid-way must trip this
    // rank's wait within kCpInitStoreTimeout instead of blocking for the
    // default (15-minute) rendezvous timeout. cp_group then degrades via
    // Phase C; tp_group propagates the failure (no silent backend switch).
    // Only tp_group/cp_group build a custom all-reduce (the constructor
    // returns early for every other group), so the bounded wait is
    // unconditional here.
    store_->wait(keys, kCpInitStoreTimeout);
    for (int32_t i = 0; i < world_size_; ++i) {
      if (i == rank_) {
        continue;
      }
      std::vector<uint8_t> bytes = store_->get(keys[i]);
      torch::Tensor handle_tensor = bytes_to_tensor(bytes);
      ptrs[i] = op_open_mem_handle(handle_tensor);
    }
  } catch (...) {
    // Free this buffer's local allocation and close the peer mappings opened so
    // far before propagating, so a mid-way failure leaves nothing leaked.
    release_shared_buffers(ptrs, rank_);
    throw;
  }
  return ptrs;
}

std::vector<uint8_t> MusaCustomAllReduce::tensor_to_bytes(
    const torch::Tensor& tensor) const {
  torch::Tensor cpu_tensor = tensor.to(torch::kCPU).contiguous();
  std::vector<uint8_t> bytes(cpu_tensor.numel() * cpu_tensor.element_size());
  std::memcpy(bytes.data(), cpu_tensor.data_ptr<uint8_t>(), bytes.size());
  return bytes;
}

torch::Tensor MusaCustomAllReduce::bytes_to_tensor(
    const std::vector<uint8_t>& bytes) const {
  torch::Tensor tensor = torch::empty(
      {static_cast<int64_t>(bytes.size())},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  std::memcpy(tensor.data_ptr<uint8_t>(), bytes.data(), bytes.size());
  return tensor;
}

bool MusaCustomAllReduce::should_use(const torch::Tensor& input) const {
  if (!enabled_ || !input.defined() || input.numel() == 0) {
    return false;
  }
  if (input.device() != device_) {
    return false;
  }
  if (!supported_dtype(input.scalar_type())) {
    return false;
  }
  if (!is_weak_contiguous(input) || !is_16_byte_aligned(input)) {
    return false;
  }
  const int64_t input_bytes = input.numel() * input.element_size();
  if (input_bytes % 16 != 0) {
    return false;
  }
  return input_bytes <= max_size_bytes_;
}

std::optional<torch::Tensor> MusaCustomAllReduce::allreduce(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& output_workspace) {
  if (!should_use(input)) {
    return std::nullopt;
  }
  // MUSA runs kernels launched during stream capture. Custom all-reduce
  // peer pointers are published only after capture ends, so the barrier
  // spins and capture_stream.synchronize() never returns. Capture MCCL
  // instead. Eager calls still use this kernel.
  {
    musaStreamCaptureStatus capture_status = musaStreamCaptureStatusNone;
    const musaError_t capture_err = musaStreamIsCapturing(
        c10::musa::getCurrentMUSAStream(device_.index()), &capture_status);
    // A failed query leaves capture_status at its initialized "None", so
    // falling through would run the barrier while a capture is in progress and
    // deadlock. Fail fast: only a confirmed non-capture may use this kernel.
    CHECK_EQ(static_cast<int>(capture_err), static_cast<int>(musaSuccess))
        << "musaStreamIsCapturing failed; cannot confirm the stream is not "
           "capturing, refusing the custom all-reduce";
    if (capture_status != musaStreamCaptureStatusNone) {
      return std::nullopt;
    }
  }
  torch::Tensor output = output_workspace.has_value()
                             ? *output_workspace
                             : torch::empty_like(input);
  if (!output.defined() || output.sizes() != input.sizes() ||
      output.scalar_type() != input.scalar_type() ||
      output.device() != device_ || !is_weak_contiguous(output) ||
      !is_16_byte_aligned(output) || output.is_alias_of(input)) {
    return std::nullopt;
  }
  op_all_reduce(
      custom_ar_handle_, input, output, buffer_ptrs_[rank_], max_size_bytes_);
  return output;
}

std::optional<torch::Tensor> MusaCustomAllReduce::allreduce_in_slices(
    const torch::Tensor& input) {
  const int64_t bytes = static_cast<int64_t>(input.nbytes());
  if (!enabled_ || !input.defined() || input.device() != device_ ||
      !supported_dtype(input.scalar_type()) || !input.is_contiguous() ||
      !is_16_byte_aligned(input) || bytes <= max_size_bytes_ ||
      bytes % 16 != 0) {
    return std::nullopt;
  }
  const int64_t align_elems = 16 / static_cast<int64_t>(input.element_size());
  const int64_t num_slices = (bytes + max_size_bytes_ - 1) / max_size_bytes_;
  const int64_t slice_elems =
      ((input.numel() + num_slices - 1) / num_slices + align_elems - 1) /
      align_elems * align_elems;
  torch::Tensor flat_input = input.view({-1});
  torch::Tensor output = torch::empty_like(input);
  torch::Tensor flat_output = output.view({-1});
  for (int64_t offset = 0; offset < flat_input.numel(); offset += slice_elems) {
    const int64_t count = std::min(slice_elems, flat_input.numel() - offset);
    torch::Tensor in_slice = flat_input.narrow(/*dim=*/0, offset, count);
    const std::optional<torch::Tensor> reduced =
        allreduce(in_slice, flat_output.narrow(/*dim=*/0, offset, count));
    if (!reduced.has_value()) {
      CHECK_EQ(offset, 0) << "TP custom all-reduce failed after a slice";
      return std::nullopt;
    }
  }
  return output;
}

std::optional<torch::Tensor> MusaCustomAllReduce::allreduce_tp(
    const torch::Tensor& input) {
  if (!enable_tp_allreduce_ || group_name_ != "tp_group" || world_size_ <= 1) {
    return std::nullopt;
  }
  CHECK(enabled_) << "Explicit MUSA TP custom all-reduce was not initialized";
  std::optional<torch::Tensor> output = allreduce(input);
  if (!output.has_value()) {
    output = allreduce_in_slices(input);
    if (output.has_value()) {
      LOG_FIRST_N(INFO, 1) << "[MUSA_TP_REDUCE] backend=custom_slices, group="
                           << group_name_ << ", rank=" << rank_
                           << ", world_size=" << world_size_
                           << ", dtype=" << input.scalar_type()
                           << ", shape=" << input.sizes()
                           << ", bytes=" << input.nbytes();
      return output;
    }
  }
  if (output.has_value()) {
    LOG_FIRST_N(INFO, 1) << "[MUSA_TP_REDUCE] backend=custom, group="
                         << group_name_ << ", rank=" << rank_
                         << ", world_size=" << world_size_
                         << ", dtype=" << input.scalar_type()
                         << ", shape=" << input.sizes()
                         << ", bytes=" << input.nbytes()
                         << ", out_of_place=true";
  } else {
    LOG_FIRST_N(INFO, 1)
        << "[MUSA_TP_REDUCE] backend=mccl, reason=unsupported_tensor, group="
        << group_name_ << ", rank=" << rank_ << ", world_size=" << world_size_
        << ", dtype=" << input.scalar_type() << ", shape=" << input.sizes()
        << ", bytes=" << input.nbytes();
  }
  return output;
}

}  // namespace xllm
