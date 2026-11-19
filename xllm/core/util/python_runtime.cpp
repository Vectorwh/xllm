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

#include "core/util/python_runtime.h"

#include <glog/logging.h>
#include <pybind11/embed.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace py = pybind11;

namespace xllm {
namespace {

constexpr int32_t kShutdownTimeoutSeconds = 30;
constexpr auto kTrackerShutdownTimeout = std::chrono::seconds(5);
constexpr auto kTrackerPollInterval = std::chrono::milliseconds(20);

// One process-wide deadline: the first caller starts it, later callers must
// not extend it. The flag is read and written from signal handlers.
std::atomic<bool> g_deadline_armed{false};
static_assert(std::atomic<bool>::is_always_lock_free,
              "Terminal deadline arming must be async-signal-safe");

void handle_shutdown_timeout(int /*signum*/) { std::_Exit(1); }

}  // namespace

void arm_shutdown_deadline() {
  if (g_deadline_armed.exchange(true, std::memory_order_relaxed)) {
    return;
  }
  struct sigaction action = {};
  action.sa_handler = handle_shutdown_timeout;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGALRM, &action, nullptr) != 0) {
    std::_Exit(1);
  }
  alarm(static_cast<unsigned int>(kShutdownTimeoutSeconds));
}

namespace {

// One internal cleanup attempt. Its result is consumed only by the terminal
// exit below; released resource handles do not represent historical success.
bool cleanup_python_subprocesses() {
  if (!Py_IsInitialized()) {
    return true;
  }
  py::gil_scoped_acquire gil;
  LOG(INFO) << "Stopping embedded Python subprocesses...";
  try {
    // Manager shutdown is an exit-priority finalizer. _exit_function would
    // additionally join unrelated children without a timeout.
    py::module_::import("multiprocessing.util").attr("_run_finalizers")(0);
    // Finalizers report their exceptions internally, so validate the result
    // instead of treating a normal Python return as successful cleanup.
    const py::object children =
        py::module_::import("multiprocessing").attr("active_children")();
    if (py::len(children) != 0) {
      LOG(ERROR)
          << "Embedded Python subprocesses still running after finalizers: "
          << py::str(children).cast<std::string>();
      return false;
    }
    // The tracker is not a Process child. All interpreter users have stopped,
    // so close our pipe as CPython's _stop does, but bound the wait: another
    // forked process can still hold a copy of the pipe open.
    const py::object tracker =
        py::module_::import("multiprocessing.resource_tracker")
            .attr("_resource_tracker");
    const py::object tracker_fd = tracker.attr("_fd");
    const py::object tracker_pid = tracker.attr("_pid");
    if (!tracker_fd.is_none()) {
      py::module_::import("os").attr("close")(tracker_fd);
      tracker.attr("_fd") = py::none();
    }
    if (!tracker_pid.is_none()) {
      const pid_t child_pid = tracker_pid.cast<pid_t>();
      int status = 0;
      bool tracker_reaped = false;
      const auto deadline =
          std::chrono::steady_clock::now() + kTrackerShutdownTimeout;
      {
        py::gil_scoped_release release;
        while (true) {
          const pid_t result = ::waitpid(child_pid, &status, WNOHANG);
          if (result == child_pid) {
            tracker_reaped = true;
            break;
          }
          if (result < 0 && errno == ECHILD) {
            // A forked interpreter can inherit its parent's tracker PID.
            LOG(INFO) << "Python resource tracker " << child_pid
                      << " is not a waitable child; releasing local state";
            break;
          }
          if (result < 0 && errno != EINTR) {
            const int wait_errno = errno;
            LOG(ERROR) << "Failed to reap Python resource tracker " << child_pid
                       << ", errno: " << wait_errno;
            return false;
          }
          if (std::chrono::steady_clock::now() >= deadline) {
            LOG(ERROR) << "Timed out waiting for Python resource tracker "
                       << child_pid;
            return false;
          }
          std::this_thread::sleep_for(kTrackerPollInterval);
        }
      }
      tracker.attr("_pid") = py::none();
      if (tracker_reaped && (!WIFEXITED(status) || WEXITSTATUS(status) != 0)) {
        LOG(ERROR) << "Python resource tracker " << child_pid
                   << " exited unsuccessfully, wait status: " << status;
        return false;
      }
    }
  } catch (const py::error_already_set& error) {
    LOG(ERROR) << "Failed to stop embedded Python subprocesses: "
               << error.what();
    return false;
  }
  LOG(INFO) << "Embedded multiprocessing children stopped and local resource "
               "tracker state released.";
  return true;
}

}  // namespace

[[noreturn]] void exit_without_python_teardown(int32_t exit_code) {
  if (!cleanup_python_subprocesses()) {
    exit_code = 1;
  }
  google::FlushLogFiles(google::GLOG_INFO);
  std::fflush(nullptr);
  std::_Exit(exit_code);
}

}  // namespace xllm
