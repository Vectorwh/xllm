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

#include <unistd.h>

#include "core/distributed_runtime/spawn_worker_server/spawn_worker_server.h"

#if defined(USE_NPU)
#include "core/util/python_runtime.h"
#endif

namespace xllm {

#if defined(USE_NPU)
std::atomic<SpawnWorkerServer::ShutdownState>
    SpawnWorkerServer::shutdown_state_{ShutdownState::RUNNING};

bool SpawnWorkerServer::shutdown_requested() {
  return shutdown_state_.load(std::memory_order_acquire) ==
         ShutdownState::READY_TO_STOP;
}

void SpawnWorkerServer::handle_signal(int /*signum*/) {
  if (shutdown_state_.exchange(ShutdownState::ARMING_DEADLINE,
                               std::memory_order_relaxed) !=
      ShutdownState::RUNNING) {
    // A repeated request forces exit without claiming cleanup.
    _exit(1);
  }
  // Signals may arrive on another thread. Only allow run() to stop after the
  // deadline covers owner teardown and terminal Python cleanup.
  arm_shutdown_deadline();
  shutdown_state_.store(ShutdownState::READY_TO_STOP,
                        std::memory_order_release);
}
#else
void SpawnWorkerServer::handle_signal(int signum) {
  (void)signum;
  _exit(0);
}
#endif

}  // namespace xllm
