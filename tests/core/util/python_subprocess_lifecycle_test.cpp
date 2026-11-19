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

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "core/distributed_runtime/spawn_worker_server/spawn_worker_server.h"
#include "core/util/python_runtime.h"

namespace {

static_assert(std::atomic<bool>::is_always_lock_free,
              "The alarm pause must be async-signal-safe");
std::atomic<bool> g_hold_next_alarm{false};
std::atomic<bool> g_alarm_entered{false};
std::atomic<bool> g_release_alarm{false};

timespec read_alarm_pause_clock() {
  timespec now{};
  if (::clock_gettime(/*clock_id=*/CLOCK_MONOTONIC, &now) != 0) {
    constexpr char kClockFailed[] = "Failed to read alarm wrapper clock\n";
    ::write(STDERR_FILENO, kClockFailed, sizeof(kClockFailed) - 1);
    ::_exit(2);
  }
  return now;
}

}  // namespace

extern "C" unsigned int __real_alarm(unsigned int seconds);

// Test-only linker wrapping pauses the real arming call without adding a hook
// to the production signal handler. The pause uses only lock-free atomics and
// async-signal-safe clock/error operations, before the production timer starts.
extern "C" unsigned int __wrap_alarm(unsigned int seconds) {
  if (g_hold_next_alarm.exchange(false, std::memory_order_relaxed)) {
    timespec deadline = read_alarm_pause_clock();
    deadline.tv_sec += 5;
    g_alarm_entered.store(true, std::memory_order_release);
    while (!g_release_alarm.load(std::memory_order_acquire)) {
      const timespec now = read_alarm_pause_clock();
      if (now.tv_sec > deadline.tv_sec ||
          (now.tv_sec == deadline.tv_sec && now.tv_nsec >= deadline.tv_nsec)) {
        constexpr char kPauseTimedOut[] =
            "Timed out waiting to release alarm wrapper\n";
        ::write(STDERR_FILENO, kPauseTimedOut, sizeof(kPauseTimedOut) - 1);
        ::_exit(2);
      }
    }
  }
  return __real_alarm(seconds);
}

namespace xllm {

#if defined(USE_NPU)
class SpawnWorkerServerTestPeer final {
 public:
  static bool shutdown_requested() {
    return SpawnWorkerServer::shutdown_requested();
  }
};
#endif

namespace {

namespace py = pybind11;

// The isolated death-test process observes the terminal owner's exit code and
// children. It adopts leftovers only to verify and reclaim its own fixtures.
py::dict terminal_fixture_scope() {
  if (::prctl(PR_SET_CHILD_SUBREAPER, 1) != 0) {
    std::_Exit(2);
  }
  std::signal(SIGINT, SpawnWorkerServer::handle_signal);
  py::initialize_interpreter(/*init_signal_handlers=*/false);
  struct sigaction action;
  if (::sigaction(SIGINT, nullptr, &action) != 0 ||
      action.sa_handler != SpawnWorkerServer::handle_signal) {
    std::_Exit(2);
  }
  py::dict scope;
  scope["terminal_exit"] = py::cpp_function([](int32_t exit_code) {
    arm_shutdown_deadline();
    py::gil_scoped_release release;
    exit_without_python_teardown(exit_code);
  });
  py::exec(R"(
import multiprocessing
import os
import signal
import time
from multiprocessing.connection import Connection
from typing import Callable

context = multiprocessing.get_context('fork')

def assert_reaped_by_owner(pid: int) -> None:
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    else:
        raise AssertionError(f'owner did not reap child {pid}')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    raise AssertionError(f'child {pid} remains after terminal exit')

def observe_terminal_exit(
    fixture: Callable[[Connection], None],
    verify: Callable[[int, tuple[int, ...]], None],
) -> None:
    receiver, sender = context.Pipe(duplex=False)
    owner = context.Process(target=fixture, args=(sender,))
    owner.start()
    sender.close()
    pids = ()
    try:
        assert receiver.poll(10), 'fixture did not publish its children'
        pids = receiver.recv()
        # Forked helpers can inherit the join sentinel. Observe the owner's
        # actual wait status without waiting for those helpers to close it.
        deadline = time.monotonic() + 10
        while owner.exitcode is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert owner.exitcode is not None, 'terminal owner did not exit'
        verify(owner.exitcode, pids)
    finally:
        receiver.close()
        if owner.is_alive():
            owner.kill()
        owner.join()
        # A successful owner already reaped its children. On a failing test,
        # only adopted, still-waitable fixture children can be killed here.
        for pid in pids:
            try:
                child, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                continue
            if child == 0:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
)",
           scope);
  return scope;
}

TEST(PythonSubprocessLifecycleDeathTest, ReapsManagerAndResourceTracker) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        if (Py_IsInitialized()) {
          std::_Exit(2);
        }
        arm_shutdown_deadline();
        exit_without_python_teardown(/*exit_code=*/7);
      },
      ::testing::ExitedWithCode(7),
      "");
  EXPECT_EXIT(
      {
        py::dict scope = terminal_fixture_scope();
        py::exec(R"(
from multiprocessing import resource_tracker, shared_memory

def fixture(connection: Connection) -> None:
    segment = shared_memory.SharedMemory(create=True, size=1)
    segment.unlink()
    segment.close()
    tracker = resource_tracker._resource_tracker
    tracker_pid = tracker._pid
    # The fork inherits a tracker it cannot wait for. Its terminal exit must
    # preserve the owner's tracker and still report successful local cleanup.
    fork_pid = os.fork()
    if fork_pid == 0:
        try:
            os.waitpid(tracker_pid, os.WNOHANG)
        except ChildProcessError:
            terminal_exit(0)
        os._exit(2)
    _, status = os.waitpid(fork_pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert tracker._pid == tracker_pid and tracker._fd is not None
    os.kill(tracker_pid, 0)
    manager = context.Manager()
    assert manager._process.is_alive()
    connection.send((manager._process.pid, tracker_pid))
    connection.close()
    terminal_exit(0)

def verify(exit_code: int, pids: tuple[int, ...]) -> None:
    assert exit_code == 0, exit_code
    for pid in pids:
        assert_reaped_by_owner(pid)
    os.write(2, b'manager and tracker reaped\n')

observe_terminal_exit(fixture, verify)
)",
                 scope);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0),
      "manager and tracker reaped");
}

TEST(PythonSubprocessLifecycleDeathTest,
     ReportsCleanupFailuresThroughExitCode) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        py::dict scope = terminal_fixture_scope();
        py::exec(R"(
import multiprocessing.util
from multiprocessing import resource_tracker, shared_memory

def remaining_child(connection: Connection) -> None:
    manager = context.Manager()
    blocker = context.Process(target=context.Event().wait)
    blocker.start()

    def fail_finalizer() -> None:
        raise RuntimeError('injected finalizer failure')

    multiprocessing.util.Finalize(None, fail_finalizer, exitpriority=0)
    connection.send((manager._process.pid, blocker.pid))
    connection.close()
    terminal_exit(0)

def verify_remaining(exit_code: int, pids: tuple[int, ...]) -> None:
    assert exit_code == 1, exit_code
    manager_pid, blocker_pid = pids
    assert_reaped_by_owner(manager_pid)
    assert os.waitpid(blocker_pid, os.WNOHANG) == (0, 0)

observe_terminal_exit(remaining_child, verify_remaining)

def boundary_failure(connection: Connection) -> None:
    def fail_terminal_cleanup(_: int) -> None:
        raise RuntimeError('injected terminal cleanup failure')

    multiprocessing.util._run_finalizers = fail_terminal_cleanup
    connection.send(())
    connection.close()
    terminal_exit(0)

def verify_failure(exit_code: int, _: tuple[int, ...]) -> None:
    assert exit_code == 1, exit_code

observe_terminal_exit(boundary_failure, verify_failure)

def abnormal_tracker(connection: Connection) -> None:
    segment = shared_memory.SharedMemory(create=True, size=1)
    segment.unlink()
    segment.close()
    tracker_pid = resource_tracker._resource_tracker._pid
    os.kill(tracker_pid, signal.SIGKILL)
    connection.send((tracker_pid,))
    connection.close()
    terminal_exit(0)

def verify_abnormal(exit_code: int, pids: tuple[int, ...]) -> None:
    assert exit_code == 1, exit_code
    assert_reaped_by_owner(pids[0])

observe_terminal_exit(abnormal_tracker, verify_abnormal)
os.write(2, b'cleanup failures preserved in exit code\n')
)",
                 scope);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0),
      "cleanup failures preserved in exit code");
}

TEST(PythonSubprocessLifecycleDeathTest,
     ReportsTrackerTimeoutWithInheritedPipe) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        py::dict scope = terminal_fixture_scope();
        py::exec(R"(
from multiprocessing import resource_tracker, shared_memory

def fixture(connection: Connection) -> None:
    ready_event = context.Event()
    release_event = context.Event()
    segment = shared_memory.SharedMemory(create=True, size=1)
    segment.unlink()
    segment.close()
    tracker_pid = resource_tracker._resource_tracker._pid
    # An unregistered fork holds the pipe, preventing the tracker from exiting.
    holder_pid = os.fork()
    if holder_pid == 0:
        ready_event.set()
        release_event.wait()
        os._exit(2)
    assert ready_event.wait(5)
    connection.send((holder_pid, tracker_pid))
    connection.close()
    terminal_exit(0)

def verify(exit_code: int, pids: tuple[int, ...]) -> None:
    assert exit_code == 1, exit_code
    assert time.monotonic() - started >= 5
    for pid in pids:
        assert os.waitpid(pid, os.WNOHANG) == (0, 0)
    os.write(2, b'tracker timeout reported in exit code\n')

started = time.monotonic()
observe_terminal_exit(fixture, verify)
)",
                 scope);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0),
      "tracker timeout reported in exit code");
}

TEST(PythonSubprocessLifecycleDeathTest, SpawnDeadlineBoundsBlockedCleanup) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
#if defined(USE_NPU)
  EXPECT_EXIT(
      {
        std::signal(SIGTERM, SpawnWorkerServer::handle_signal);
        // Exercise delivery on a worker thread, as in a real spawned process.
        g_hold_next_alarm.store(true, std::memory_order_relaxed);
        std::thread sender([]() { std::raise(SIGTERM); });
        const auto arming_deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!g_alarm_entered.load(std::memory_order_acquire)) {
          if (std::chrono::steady_clock::now() >= arming_deadline) {
            std::_Exit(2);
          }
          std::this_thread::yield();
        }
        const bool published_before_arming =
            SpawnWorkerServerTestPeer::shutdown_requested();
        g_release_alarm.store(true, std::memory_order_release);
        sender.join();
        if (published_before_arming ||
            !SpawnWorkerServerTestPeer::shutdown_requested()) {
          std::fprintf(stderr, "Shutdown published before deadline arming\n");
          std::_Exit(2);
        }
        // Require a production timer, then shorten it for this CPU regression.
        if (::alarm(/*seconds=*/1) == 0) {
          std::_Exit(2);
        }
        constexpr char kDeadlineArmed[] = "shutdown deadline armed\n";
        if (::write(
                STDERR_FILENO, kDeadlineArmed, sizeof(kDeadlineArmed) - 1) !=
            static_cast<ssize_t>(sizeof(kDeadlineArmed) - 1)) {
          std::_Exit(2);
        }
        while (true) {
          ::pause();
        }
      },
      ::testing::ExitedWithCode(1),
      "shutdown deadline armed");
#else
  EXPECT_EXIT(
      {
        std::signal(SIGTERM, SpawnWorkerServer::handle_signal);
        std::raise(SIGTERM);
        std::_Exit(2);
      },
      ::testing::ExitedWithCode(0),
      "");
#endif
}

TEST(PythonSubprocessLifecycleDeathTest, DeadlineArmingIsIdempotent) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        arm_shutdown_deadline();
        // Shorten the production deadline, then prove a later arm keeps it.
        if (::alarm(/*seconds=*/7) == 0) {
          std::_Exit(2);
        }
        arm_shutdown_deadline();
        if (::alarm(/*seconds=*/1) > 7) {
          std::_Exit(2);
        }
        while (true) {
          ::pause();
        }
      },
      ::testing::ExitedWithCode(1),
      "");
}

TEST(PythonSubprocessLifecycleDeathTest,
     TerminalDeadlineBoundsBlockedFinalizers) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        py::initialize_interpreter(/*init_signal_handlers=*/false);
        py::exec(R"(
import multiprocessing.util
import os
import threading

def _blocked_finalizer() -> None:
    # The death test captures this marker directly from stderr.
    os.write(2, b"finalizer entered\n")
    threading.Event().wait()

multiprocessing.util.Finalize(None, _blocked_finalizer, exitpriority=0)
)");
        arm_shutdown_deadline();
        // Require the production deadline, then shorten it for this regression.
        if (::alarm(/*seconds=*/1) == 0) {
          std::_Exit(2);
        }
        exit_without_python_teardown(/*exit_code=*/0);
      },
      ::testing::ExitedWithCode(1),
      "finalizer entered");
}

}  // namespace
}  // namespace xllm
