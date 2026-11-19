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

#pragma once

#include <cstdint>

namespace xllm {

// Starts the 30-second process-wide terminal teardown deadline. Idempotent and
// async-signal-safe: the first call starts the deadline and later calls do not
// extend it. Call it as soon as the quit request is observed, before stopping
// owners; do not reset it in later cleanup stages.
// The deadline owns SIGALRM for the rest of the process: arming installs the
// handler and alarm(), replacing any previous SIGALRM disposition or pending
// timer. A host that calls this must not use SIGALRM/ITIMER_REAL afterwards.
// Handler installation failure hard-exits 1.
void arm_shutdown_deadline();

// Runs multiprocessing's exit-priority finalizers, closes this process's
// tracker pipe, flushes logs/stdio and hard-exits. Remaining Process children,
// tracker wait errors, a five-second tracker timeout or an abnormal reaped
// tracker exit set the process exit code to 1. ECHILD releases local tracker
// state without inferring an exit status; another process's tracker can remain
// alive. An absent interpreter requires no Python cleanup; unregistered
// processes are not stopped.
// This is the only Python cleanup entry point: it never returns a result that
// can be retried or replaced. All interpreter users must already have stopped;
// call only at terminal process exit, not between owner lifetimes. Callers arm
// the deadline before owner teardown; this function does not restart it.
// Keeps CPython alive and skips global/static destructors: global pybind11
// objects cannot be destroyed safely without the GIL. Not signal-safe.
[[noreturn]] void exit_without_python_teardown(int32_t exit_code);

}  // namespace xllm
