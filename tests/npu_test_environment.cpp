/* Copyright 2026 The xLLM Authors.

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

#include "tests/npu_test_environment.h"

#include <Python.h>
#include <acl/acl.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "tests/scoped_port_range.h"

namespace xllm::testing {

namespace {

// Main-thread state saved by init_npu_test_runtime(). Test tear-down code
// reacquires the GIL here so pybind11 objects held by torch_npu / torch can
// dec_ref safely during process exit (otherwise pybind11 aborts with
// PyGILState_Check failure).
PyThreadState* g_saved_thread_state = nullptr;

}  // namespace

void init_npu_test_runtime() {
  if (Py_IsInitialized()) {
    return;
  }

  // Each CTest process gets a separate HCCL listener block; forked ranks
  // inherit the same base. Preserve explicit caller configuration.
  if (std::getenv("HCCL_IF_BASE_PORT") == nullptr) {
    xllm::tests::ScopedPortRange hccl_ports(/*count=*/64);
    const std::string base = std::to_string(hccl_ports.release());
    if (setenv("HCCL_IF_BASE_PORT", base.c_str(), /*overwrite=*/0) != 0) {
      std::fprintf(stderr, "Failed to configure HCCL test ports\n");
      std::abort();
    }
  }

  // HCCL reads the NPU NIC socket range while the process initializes its
  // runtime, so the variable must be present before aclInit runs; setting it
  // later only affects processes that re-exec afterwards. HCCL_IF_BASE_PORT
  // does not cover these sockets. Reserve one block; a forked peer that forms
  // an independent HCCL group derives the block that follows it (see the
  // Mooncake round-trip test). A busy host may not have a large contiguous
  // hole left, so step down to smaller blocks instead of failing the whole
  // binary; if no block can be reserved, leave the variable unset and let the
  // cases that require it report the missing range.
  if (std::getenv("HCCL_NPU_SOCKET_PORT_RANGE") == nullptr) {
    constexpr int32_t kBlockSizes[] = {64, 16, 4, 2};
    std::string range;
    std::string last_error = "no contiguous block found";
    for (const int32_t block_size : kBlockSizes) {
      try {
        xllm::tests::ScopedPortRange npu_ports(block_size);
        const int32_t base_port = npu_ports.release();
        range = std::to_string(base_port) + "-" +
                std::to_string(base_port + block_size - 1);
        break;
      } catch (const std::runtime_error& error) {
        last_error = error.what();
      }
    }
    if (range.empty()) {
      std::fprintf(stderr,
                   "Failed to reserve HCCL NPU socket ports: %s\n",
                   last_error.c_str());
    } else if (setenv("HCCL_NPU_SOCKET_PORT_RANGE",
                      range.c_str(),
                      /*overwrite=*/0) != 0) {
      std::fprintf(stderr, "Failed to configure HCCL NPU socket ports\n");
      std::abort();
    }
  }

  // 1. Tolerate ACL_ERROR_INTERNAL_ERROR — dump server may fail to start but
  //    the runtime is still usable, matching the production launch path.
  const auto acl_ret = aclInit(nullptr);
  if (acl_ret != ACL_SUCCESS && acl_ret != ACL_ERROR_INTERNAL_ERROR) {
    std::fprintf(stderr, "aclInit failed with error %d\n", acl_ret);
    std::abort();
  }

  // 2. Boot the Python interpreter before importing torch_npu.
  setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);
  Py_InitializeEx(0);

  // 3. Import torch_npu and run its Python-side runtime init. Suppressing
  //    torch._C._get_accelerator during import matches production and avoids
  //    a spurious CUDA-vs-NPU accelerator check on import.
  const int import_ret = PyRun_SimpleString(
      "import os\n"
      "os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'\n"
      "import torch\n"
      "_orig_get_accelerator = torch._C._get_accelerator\n"
      "try:\n"
      "    torch._C._get_accelerator = lambda: torch.device('cpu')\n"
      "    import torch_npu\n"
      "finally:\n"
      "    torch._C._get_accelerator = _orig_get_accelerator\n"
      "import torch_npu.npu as _npu_mod\n"
      "try:\n"
      "    torch_npu._C._npu_init()\n"
      "except RuntimeError as e:\n"
      "    if 'already initialized' not in str(e).lower():\n"
      "        raise\n"
      "_npu_mod._initialized = True\n"
      "_npu_mod._original_pid = os.getpid()\n");
  if (import_ret != 0) {
    std::fprintf(stderr, "torch_npu Python-side init failed\n");
    std::abort();
  }

  // 4. Release the GIL so torch_npu's C++ entry points can PyGILState_Ensure
  //    without deadlocking against the main thread.
  g_saved_thread_state = PyEval_SaveThread();
}

void finalize_npu_test_runtime() {
  if (g_saved_thread_state != nullptr) {
    PyEval_RestoreThread(g_saved_thread_state);
    g_saved_thread_state = nullptr;
  }
}

}  // namespace xllm::testing

namespace {

// Mirror xllm.cpp::main()::init_npu_python_runtime() for every NPU test binary
// via a global gtest Environment. Tests that register their own custom
// Environment (e.g. AclGraphExecutorTestEnvironment) must call
// xllm::testing::init_npu_test_runtime() from their own SetUp before touching
// NPU APIs — gtest Environment ordering depends on TU link order and cannot be
// relied on, and this file's Environment cannot deterministically run first.
class NpuPythonEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { xllm::testing::init_npu_test_runtime(); }
  void TearDown() override { xllm::testing::finalize_npu_test_runtime(); }
};

::testing::Environment* const kNpuPythonEnvironment =
    ::testing::AddGlobalTestEnvironment(new NpuPythonEnvironment);

}  // namespace
