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

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "acl/acl_rt.h"
#include "aclnn/acl_meta.h"
#include "aclnn_dsa_kv_offloading_serve_rope_planned.h"
#include "aclnn_dsa_kv_offloading_update_nope.h"
#include "aclnn_dsa_kv_offloading_update_rope_with_plan.h"
#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
extern "C" aclError aclrtHostGetDevicePointer(void*, void**, uint32_t);
namespace xllm::kernel::npu {
namespace {
class AclTensorDeleter final {
 public:
  void operator()(aclTensor* tensor) const noexcept {
    if (tensor == nullptr) {
      return;
    }
    // Teardown can run while another exception unwinds this call; aborting
    // here would mask the error that triggered the unwind.
    const aclError status = aclDestroyTensor(tensor);
    if (status != ACL_SUCCESS) {
      LOG(ERROR) << "Failed to release a stable resident ACL view; status "
                 << status;
    }
  }
};
using AclTensorHandle = std::unique_ptr<aclTensor, AclTensorDeleter>;
std::string recent_acl_error_detail() {
  // Copy before ACL cleanup can replace the thread-local error message.
  const char* detail = aclGetRecentErrMsg();
  return detail == nullptr ? std::string{} : std::string(detail);
}
// These calls prepare call-local views/executors before any KV write is
// enqueued. Propagate the original operator error; the caller must terminate
// the failed request instead of ignoring the failure or retrying execution.
template <typename Status>
void check_setup_status(Status status, const char* operation) {
  if (status == ACL_SUCCESS) {
    return;
  }
  const std::string detail = recent_acl_error_detail();
  LOG(ERROR) << operation << " failed with status " << status
             << ", detail: " << detail;
  TORCH_CHECK(
      false, operation, " failed with status ", status, ", detail: ", detail);
}
void close_acl_view(AclTensorHandle& view) {
  aclTensor* tensor = view.release();
  if (tensor == nullptr) {
    return;
  }
  check_setup_status(aclDestroyTensor(tensor), "aclDestroyTensor");
}
void close_acl_views(std::vector<AclTensorHandle>& views,
                     std::initializer_list<AclTensorHandle*> host_views) {
  for (AclTensorHandle& view : views) {
    close_acl_view(view);
  }
  for (AclTensorHandle* view : host_views) {
    close_acl_view(*view);
  }
}
AclTensorHandle make_acl_view(const torch::Tensor& tensor, void* pointer) {
  const aclDataType dtype =
      tensor.scalar_type() == torch::kBFloat16 ? ACL_BF16 : ACL_INT32;
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16 ||
                  tensor.scalar_type() == torch::kInt32,
              "Stable resident ACL views require BF16 or INT32 tensors.");
  TORCH_CHECK(tensor.is_contiguous(),
              "Stable resident ACL views require contiguous tensors.");
  // pointer already refers to the first view element; do not add storage_offset
  // again. Mapped Host uses the device alias, never the CPU storage address.
  aclTensor* handle = aclCreateTensor(tensor.sizes().data(),
                                      tensor.dim(),
                                      dtype,
                                      tensor.strides().data(),
                                      0,
                                      ACL_FORMAT_ND,
                                      tensor.sizes().data(),
                                      tensor.dim(),
                                      pointer);
  TORCH_CHECK(handle != nullptr,
              "aclCreateTensor returned a null stable resident view.");
  return AclTensorHandle(handle);
}
// The host KV is a CPU allocation whose registered device alias must reach
// the op as an ACL_FORMAT_ND aclTensor. EXEC_NPU_CMD's generic tensor
// converter infers NCHW for 4-D inputs, so this adapter keeps explicit ND
// view construction rather than using the macro.
void* mapped_host_pointer(const torch::Tensor& host) {
  TORCH_CHECK(host.device().type() == torch::kCPU,
              "Mapped host KV must be a CPU tensor.");
  TORCH_CHECK(host.scalar_type() == torch::kBFloat16,
              "Mapped host KV must use BF16.");
  TORCH_CHECK(host.is_contiguous(), "Mapped host KV must be contiguous.");
  void* pointer = nullptr;
  check_setup_status(
      aclrtHostGetDevicePointer(host.data_ptr(), &pointer, /*flag=*/0),
      "aclrtHostGetDevicePointer");
  TORCH_CHECK(pointer != nullptr,
              "Mapped host KV has no registered device alias.");
  return pointer;
}
std::vector<AclTensorHandle> make_device_views(
    const std::vector<const torch::Tensor*>& tensors,
    const torch::Device& device) {
  std::vector<AclTensorHandle> views;
  views.reserve(tensors.size());
  for (const torch::Tensor* tensor : tensors) {
    TORCH_CHECK(tensor->device() == device,
                "Stable resident device tensors must share one NPU device.");
    views.emplace_back(make_acl_view(*tensor, tensor->data_ptr()));
  }
  return views;
}
template <typename Run>
void run_resident_op(Run run,
                     uint64_t workspace_size,
                     aclOpExecutor* executor,
                     const torch::Tensor& current) {
  torch::Tensor workspace;
  void* address = nullptr;
  if (workspace_size > 0) {
    workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                             current.options().dtype(torch::kUInt8));
    address = workspace.data_ptr();
  }
  // The workspace is allocated on the current stream and consumed by a kernel
  // on that same stream, so allocator reuse stays stream-ordered; no host-side
  // synchronisation is needed to keep it alive.
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(current.device().index()).stream();
  // Execution writes shared Host/resident/selection buffers in place. A failed
  // enqueue cannot prove that none of those writes ran, so do not keep serving.
  CHECK_EQ(run(address, workspace_size, executor, stream), ACL_SUCCESS)
      << ", detail: " << recent_acl_error_detail();
}
}  // namespace
int64_t dsa_kv_offloading_selection_capacity() {
  // A missing capacity symbol reports a negative capacity. The execution
  // entries are linked directly, so this is not a load-time feature gate for
  // a package missing the entire operator family.
  using CapacityFn = int64_t (*)();
  static const auto capacity_fn =
      reinterpret_cast<CapacityFn>(aclnn::detail::get_op_api_func_addr(
          "aclnnDsaKvOffloadingSelectionCapacity"));
  if (capacity_fn == nullptr) {
    LOG(WARNING) << "aclnnDsaKvOffloadingSelectionCapacity is unavailable in "
                    "the installed vendor package.";
    return -1;
  }
  return capacity_fn();
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dsa_kv_offloading_update_nope_out(torch::Tensor& host,
                                  const torch::Tensor& block_table,
                                  const torch::Tensor& descriptor,
                                  const torch::Tensor& current_map,
                                  const torch::Tensor& topk,
                                  const torch::Tensor& append_slots,
                                  const torch::Tensor& current,
                                  torch::Tensor& selection,
                                  torch::Tensor& indices,
                                  torch::Tensor& lengths,
                                  torch::Tensor& selection_table,
                                  torch::Tensor& resident,
                                  torch::Tensor& slots,
                                  torch::Tensor& count,
                                  torch::Tensor& compact_indices) {
  TORCH_CHECK(current.device().type() == c10::DeviceType::PrivateUse1,
              "Resident current KV must be an NPU tensor.");
  const c10::DeviceGuard device_guard(current.device());
  auto host_view = make_acl_view(host, mapped_host_pointer(host));
  auto views = make_device_views({&block_table,
                                  &descriptor,
                                  &current_map,
                                  &topk,
                                  &append_slots,
                                  &current,
                                  &selection,
                                  &indices,
                                  &lengths,
                                  &selection_table,
                                  &resident,
                                  &slots,
                                  &count,
                                  &compact_indices},
                                 current.device());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_setup_status(
      aclnnDsaKvOffloadingUpdateNopeGetWorkspaceSize(host_view.get(),
                                                     views[0].get(),
                                                     views[1].get(),
                                                     views[2].get(),
                                                     views[3].get(),
                                                     views[4].get(),
                                                     views[5].get(),
                                                     views[6].get(),
                                                     views[7].get(),
                                                     views[8].get(),
                                                     views[9].get(),
                                                     views[10].get(),
                                                     views[11].get(),
                                                     views[12].get(),
                                                     views[13].get(),
                                                     &workspace_size,
                                                     &executor),
      "aclnnDsaKvOffloadingUpdateNopeGetWorkspaceSize");
  run_resident_op(
      aclnnDsaKvOffloadingUpdateNope, workspace_size, executor, current);
  close_acl_views(views, {&host_view});
  return std::make_tuple(selection, compact_indices, lengths, selection_table);
}
void dsa_kv_offloading_update_rope_with_plan_out(
    torch::Tensor& host,
    const torch::Tensor& block_table,
    const torch::Tensor& descriptor,
    const torch::Tensor& current_map,
    const torch::Tensor& topk,
    const torch::Tensor& append_slots,
    const torch::Tensor& current,
    torch::Tensor& selection,
    torch::Tensor& indices,
    torch::Tensor& lengths,
    torch::Tensor& selection_table,
    torch::Tensor& resident,
    torch::Tensor& slots,
    torch::Tensor& count,
    torch::Tensor& host_rope,
    const torch::Tensor& current_rope,
    torch::Tensor& selection_rope,
    torch::Tensor& plan_tokens,
    torch::Tensor& plan_slots) {
  TORCH_CHECK(current.device().type() == c10::DeviceType::PrivateUse1,
              "Resident current KV must be an NPU tensor.");
  const c10::DeviceGuard device_guard(current.device());
  auto host_view = make_acl_view(host, mapped_host_pointer(host));
  auto rope_view = make_acl_view(host_rope, mapped_host_pointer(host_rope));
  auto views = make_device_views({&block_table,
                                  &descriptor,
                                  &current_map,
                                  &topk,
                                  &append_slots,
                                  &current,
                                  &selection,
                                  &indices,
                                  &lengths,
                                  &selection_table,
                                  &resident,
                                  &slots,
                                  &count,
                                  &current_rope,
                                  &selection_rope,
                                  &plan_tokens,
                                  &plan_slots},
                                 current.device());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_setup_status(
      aclnnDsaKvOffloadingUpdateRopeWithPlanGetWorkspaceSize(host_view.get(),
                                                             views[0].get(),
                                                             views[1].get(),
                                                             views[2].get(),
                                                             views[3].get(),
                                                             views[4].get(),
                                                             views[5].get(),
                                                             views[6].get(),
                                                             views[7].get(),
                                                             views[8].get(),
                                                             views[9].get(),
                                                             views[10].get(),
                                                             views[11].get(),
                                                             views[12].get(),
                                                             rope_view.get(),
                                                             views[13].get(),
                                                             views[14].get(),
                                                             views[15].get(),
                                                             views[16].get(),
                                                             &workspace_size,
                                                             &executor),
      "aclnnDsaKvOffloadingUpdateRopeWithPlanGetWorkspaceSize");
  run_resident_op(aclnnDsaKvOffloadingUpdateRopeWithPlan,
                  workspace_size,
                  executor,
                  current);
  close_acl_views(views, {&host_view, &rope_view});
}
void dsa_kv_offloading_serve_rope_planned_out(torch::Tensor& host,
                                              torch::Tensor& host_rope,
                                              const torch::Tensor& block_table,
                                              const torch::Tensor& descriptor,
                                              const torch::Tensor& current_map,
                                              const torch::Tensor& plan_tokens,
                                              const torch::Tensor& plan_slots,
                                              const torch::Tensor& append_slots,
                                              const torch::Tensor& current,
                                              const torch::Tensor& current_rope,
                                              torch::Tensor& selection,
                                              torch::Tensor& selection_rope) {
  TORCH_CHECK(current.device().type() == c10::DeviceType::PrivateUse1,
              "Resident current KV must be an NPU tensor.");
  const c10::DeviceGuard device_guard(current.device());
  auto host_view = make_acl_view(host, mapped_host_pointer(host));
  auto rope_view = make_acl_view(host_rope, mapped_host_pointer(host_rope));
  auto views = make_device_views({&block_table,
                                  &descriptor,
                                  &current_map,
                                  &plan_tokens,
                                  &plan_slots,
                                  &append_slots,
                                  &current,
                                  &current_rope,
                                  &selection,
                                  &selection_rope},
                                 current.device());
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_setup_status(
      aclnnDsaKvOffloadingServeRopePlannedGetWorkspaceSize(host_view.get(),
                                                           rope_view.get(),
                                                           views[0].get(),
                                                           views[1].get(),
                                                           views[2].get(),
                                                           views[3].get(),
                                                           views[4].get(),
                                                           views[5].get(),
                                                           views[6].get(),
                                                           views[7].get(),
                                                           views[8].get(),
                                                           views[9].get(),
                                                           &workspace_size,
                                                           &executor),
      "aclnnDsaKvOffloadingServeRopePlannedGetWorkspaceSize");
  run_resident_op(
      aclnnDsaKvOffloadingServeRopePlanned, workspace_size, executor, current);
  close_acl_views(views, {&host_view, &rope_view});
}
}  // namespace xllm::kernel::npu
