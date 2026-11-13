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

#pragma once

#include <brpc/errno.pb.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "macros.h"

namespace xllm {

// Singleton class to manage cluster health status.
// A lost worker connection ends the master process. An execution timeout only
// marks the cluster unhealthy so the process can keep serving.
class HealthCheckManager {
 public:
  using HealthCheckFunc = std::function<bool()>;

  static HealthCheckManager& instance() {
    static HealthCheckManager manager;
    return manager;
  }

  // Check if the cluster is healthy
  bool is_healthy() const {
    return is_healthy_.load(std::memory_order_acquire);
  }

  // Set cluster health status to unhealthy
  void set_unhealthy(const std::string& reason = "") {
    {
      std::lock_guard<std::mutex> lock(reason_mutex_);
      unhealthy_reason_ = reason;
    }
    is_healthy_.store(false, std::memory_order_release);
  }

  // A worker process or its connection is gone. An execution timeout is not
  // fatal: the worker may still be computing and the process stays up.
  void report_worker_failure(int32_t error_code, const std::string& reason) {
    if (!worker_connection_lost(error_code)) {
      set_unhealthy(reason);
      return;
    }
    bool expected = false;
    if (!worker_lost_.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel)) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(reason_mutex_);
      unhealthy_reason_ = reason;
    }
    is_healthy_.store(false, std::memory_order_release);
    LOG(ERROR) << "Worker connection lost: " << reason;
    if (fatal_handler_) {
      fatal_handler_(reason);
    }
  }

  void set_worker_loss_handler(
      std::function<void(const std::string&)> handler) {
    fatal_handler_ = std::move(handler);
  }

  // brpc uses these values for a peer that closed or never accepted the
  // connection. EEOF is the clean peer close ("Got EOF"). A plain execution
  // timeout stays non-fatal.
  static bool worker_connection_lost(int32_t error_code) {
    return error_code == ECONNREFUSED || error_code == ECONNRESET ||
           error_code == EHOSTDOWN || error_code == ETIMEDOUT ||
           error_code == brpc::EFAILEDSOCKET || error_code == brpc::EEOF ||
           error_code == brpc::ELOGOFF || error_code == brpc::ECLOSE;
  }

  // Set cluster health status to healthy
  void set_healthy() {
    is_healthy_.store(true, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(reason_mutex_);
      unhealthy_reason_.clear();
    }
  }

  // Get the reason for unhealthy status
  std::string unhealthy_reason() const {
    std::lock_guard<std::mutex> lock(reason_mutex_);
    return unhealthy_reason_;
  }

  // Register a health check function for a worker
  void register_health_check(int worker_rank, HealthCheckFunc func) {
    health_checks_[worker_rank] = std::move(func);
  }

  // Start background health check thread
  void start_health_check_thread(int interval_ms = 3000) {
    if (health_check_running_.load()) {
      return;
    }

    health_check_running_.store(true);
    health_check_thread_ = std::thread([this, interval_ms]() {
      while (health_check_running_.load()) {
        check_all_workers();
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
      }
    });
  }

  // Stop background health check thread
  void stop_health_check_thread() {
    health_check_running_.store(false);
    if (health_check_thread_.joinable()) {
      health_check_thread_.join();
    }
  }

 private:
  HealthCheckManager()
      : is_healthy_(true), worker_lost_(false), health_check_running_(false) {}
  ~HealthCheckManager() { stop_health_check_thread(); }

  DISALLOW_COPY_AND_ASSIGN(HealthCheckManager);

  // A single Hello failure is not a lost worker. Startup can fail the probe
  // while the worker is still initializing, and one timeout must not end the
  // instance. Only this many consecutive failures for the same rank are fatal.
  static constexpr int kConsecutiveFailureLimit = 3;

  void check_all_workers() {
    for (const auto& [rank, check_func] : health_checks_) {
      if (check_func()) {
        consecutive_failures_[rank] = 0;
        continue;
      }
      const int failures = ++consecutive_failures_[rank];
      if (failures < kConsecutiveFailureLimit) {
        LOG(WARNING) << "Worker " << rank << " health check failed ("
                     << failures << "/" << kConsecutiveFailureLimit << ")";
        continue;
      }
      report_worker_failure(
          EHOSTDOWN,
          "Worker " + std::to_string(rank) + " disconnected after " +
              std::to_string(failures) + " consecutive health check failures");
      return;
    }
  }

  std::atomic<bool> is_healthy_;
  std::atomic<bool> worker_lost_;
  std::function<void(const std::string&)> fatal_handler_;
  std::atomic<bool> health_check_running_;
  mutable std::mutex reason_mutex_;
  std::string unhealthy_reason_;
  std::unordered_map<int, HealthCheckFunc> health_checks_;
  // Touched only by the health check thread.
  std::unordered_map<int, int> consecutive_failures_;
  std::thread health_check_thread_;
};

}  // namespace xllm
