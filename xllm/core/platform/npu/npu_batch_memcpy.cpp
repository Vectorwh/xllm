/* Copyright 2025-2026 The xLLM Authors.

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

#include "platform/npu/npu_batch_memcpy.h"

#include <glog/logging.h>

#include <cstring>

namespace xllm {
namespace npu {

void NPUBatchMemcpy::init(int32_t device_id) {
  if (initialized_) {
    return;
  }
  device_id_ = device_id;

  memset(&h2d_attr_, 0, sizeof(h2d_attr_));
  h2d_attr_.dstLoc.id = device_id;
  h2d_attr_.dstLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE;
  h2d_attr_.srcLoc.id = device_id;
  h2d_attr_.srcLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_HOST;

  memset(&d2h_attr_, 0, sizeof(d2h_attr_));
  d2h_attr_.dstLoc.id = device_id;
  d2h_attr_.dstLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_HOST;
  d2h_attr_.srcLoc.id = device_id;
  d2h_attr_.srcLoc.type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE;

  initialized_ = true;
}

bool NPUBatchMemcpy::submit_h2d(const std::vector<torch::Tensor>& src_tensors,
                                const std::vector<torch::Tensor>& dst_tensors,
                                Stream* stream) {
  return copy(
      src_tensors, dst_tensors, h2d_attr_, stream, submit_asynchronously_);
}

bool NPUBatchMemcpy::copy_d2h(const std::vector<torch::Tensor>& src_tensors,
                              const std::vector<torch::Tensor>& dst_tensors,
                              Stream* stream) {
  return copy(
      src_tensors, dst_tensors, d2h_attr_, stream, /*asynchronous=*/false);
}

bool NPUBatchMemcpy::submit_d2h(const std::vector<torch::Tensor>& src_tensors,
                                const std::vector<torch::Tensor>& dst_tensors,
                                Stream* stream) {
  return copy(
      src_tensors, dst_tensors, d2h_attr_, stream, submit_asynchronously_);
}

bool NPUBatchMemcpy::copy(const std::vector<torch::Tensor>& src_tensors,
                          const std::vector<torch::Tensor>& dst_tensors,
                          const aclrtMemcpyBatchAttr& attr,
                          Stream* stream,
                          bool asynchronous) {
  CHECK(initialized_) << "NPUBatchMemcpy not initialized.";
  CHECK(stream != nullptr) << "Stream must not be null.";
  CHECK_EQ(src_tensors.size(), dst_tensors.size())
      << "src and dst tensor count mismatch.";

  if (src_tensors.empty()) {
    return true;
  }

  const size_t count = src_tensors.size();

  c10::StreamGuard guard = stream->set_stream_guard();

  aclrtMemcpyBatchAttr attrs[1] = {attr};
  size_t attrs_indexes[1] = {0};

  // aclrtMemcpyBatch caps a single call at kMaxBatchCopyCount descriptors, so a
  // large transfer (e.g. a many-block prefix reload) is split into fixed-size
  // chunks issued onto the same stream in order. Synchronous copies drain once
  // at the end; submit-only callers own the completion/lifetime fence.
  std::vector<void*> srcs;
  std::vector<void*> dsts;
  std::vector<size_t> sizes;
  srcs.reserve(std::min(count, kMaxBatchCopyCount));
  dsts.reserve(std::min(count, kMaxBatchCopyCount));
  sizes.reserve(std::min(count, kMaxBatchCopyCount));

  for (size_t offset = 0; offset < count; offset += kMaxBatchCopyCount) {
    const size_t chunk = std::min(kMaxBatchCopyCount, count - offset);
    srcs.clear();
    dsts.clear();
    sizes.clear();
    for (size_t i = 0; i < chunk; ++i) {
      srcs.emplace_back(src_tensors[offset + i].data_ptr());
      dsts.emplace_back(dst_tensors[offset + i].data_ptr());
      sizes.emplace_back(static_cast<size_t>(src_tensors[offset + i].nbytes()));
    }

    size_t fail_index = SIZE_MAX;
    const auto submit = [&](bool async) {
      if (async) {
        return aclrtMemcpyBatchAsync(dsts.data(),
                                     sizes.data(),
                                     srcs.data(),
                                     sizes.data(),
                                     chunk,
                                     attrs,
                                     attrs_indexes,
                                     /*attrsCount=*/1,
                                     &fail_index,
                                     stream->get_stream()->stream());
      }
      return aclrtMemcpyBatch(dsts.data(),
                              sizes.data(),
                              srcs.data(),
                              sizes.data(),
                              chunk,
                              attrs,
                              attrs_indexes,
                              /*attrsCount=*/1,
                              &fail_index);
    };
    const aclError ret = submit(asynchronous);
    if (ret != ACL_SUCCESS || fail_index != SIZE_MAX) {
      // fail_index is chunk-local; report the position within the full batch.
      LOG(ERROR) << "aclrtMemcpyBatch failed: ret=" << ret << ", fail_index="
                 << (fail_index == SIZE_MAX ? fail_index : offset + fail_index);
      // An earlier chunk may already be in flight. Returning false without
      // draining it would let callers free or reuse live DMA buffers.
      CHECK_EQ(stream->synchronize(), 0)
          << "Cannot drain a failed NPU batch copy; DMA lifetime is unknown.";
      return false;
    }
  }

  if (!asynchronous) {
    // A synchronous caller returns its source/destination buffers to normal
    // ownership immediately. If completion cannot be established, an in-flight
    // DMA may still touch them, so this cannot be reported as a request error.
    CHECK_EQ(stream->synchronize(), 0)
        << "Cannot establish synchronous NPU copy completion; DMA lifetime is "
           "unknown.";
  }

  return true;
}

}  // namespace npu
}  // namespace xllm
