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

#include "core/runtime/xservice_client.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include "core/common/etcd_client.h"

namespace xllm {
namespace {

// Replace only registry I/O in this isolated executable. Client state, locks,
// init/register/shutdown and its reader threads come from the runtime library.
std::atomic<bool> master_available{true};
std::atomic<bool> registration_succeeds{true};
std::atomic<int32_t> attempts{0};
std::promise<void> first_attempt;
std::string registered_incarnation;

void note_attempt() {
  if (attempts.fetch_add(1) == 0) {
    first_attempt.set_value();
  }
}

InstanceInfo instance() {
  InstanceInfo info;
  info.name = "127.0.0.1:9";
  info.type = "PREFILL";
  return info;
}

}  // namespace

// These definitions satisfy EtcdClient's public boundary without pulling its
// production .o from the common static archive. SyncClient is constructed, but
// no ping, watch or registration is sent to an external etcd server.
EtcdClient::EtcdClient(const std::string& address, const std::string& space)
    : client_(address), etcd_addr_(address), etcd_namespace_prefix_(space) {}
EtcdClient::EtcdClient(const std::string& address,
                       const std::string&,
                       const std::string&,
                       const std::string& space)
    : EtcdClient(address, space) {}
EtcdClient::~EtcdClient() = default;
void EtcdClient::add_watch(const std::string&, Callback, bool) {}
void EtcdClient::remove_watch(const std::string&) {}
void EtcdClient::stop_watch() {}
bool EtcdClient::get_master_service(const std::string&, std::string* address) {
  if (!master_available.load()) {
    note_attempt();
    return false;
  }
  *address = "127.0.0.1:9";
  return true;
}
bool EtcdClient::get_all_xservices(const std::string&,
                                   std::vector<std::string>*) {
  return false;
}
bool EtcdClient::register_instance(const std::string&,
                                   const std::string& value,
                                   int) {
  registered_incarnation =
      nlohmann::json::parse(value).at("incarnation_id").get<std::string>();
  note_attempt();
  return registration_succeeds.load();
}
// Temporary until MR823: the linker pulls the real etcd_client without this.
bool EtcdClient::withdraw_registration(const std::string&) { return true; }
RegistrationLookupState EtcdClient::lookup_registration(const std::string&) {
  return RegistrationLookupState::kPresent;
}

namespace {

TEST(XServiceClientLifecycleDeathTest, ReinitializesAfterCompleteShutdown) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        alarm(5);
        auto* client = XServiceClient::get_instance();
        if (!client->init("http://127.0.0.1:9", "first")) {
          _exit(1);
        }
        client->register_instance(instance());
        const std::string first = registered_incarnation;
        client->shutdown();
        if (client->initialize_done()) {
          _exit(2);
        }
        if (!client->init("http://127.0.0.1:9", "second")) {
          _exit(3);
        }
        client->register_instance(instance());
        if (first.empty() || first == registered_incarnation) {
          _exit(4);
        }
        client->shutdown();
        client->shutdown();
        _exit(client->initialize_done() ? 5 : 0);
      },
      ::testing::ExitedWithCode(0),
      ".*");
}

TEST(XServiceClientLifecycleDeathTest, CancelsRegistrationThroughPublicEntry) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        alarm(5);
        registration_succeeds.store(false);
        auto* client = XServiceClient::get_instance();
        if (!client->init("http://127.0.0.1:9", "cancel")) {
          _exit(1);
        }
        auto started = first_attempt.get_future();
        std::thread registration(
            [&] { client->register_instance(instance()); });
        if (started.wait_for(std::chrono::seconds(1)) !=
            std::future_status::ready) {
          _exit(2);
        }
        const auto start = std::chrono::steady_clock::now();
        client->shutdown();
        registration.join();
        if (std::chrono::steady_clock::now() - start >
            std::chrono::milliseconds(500)) {
          _exit(3);
        }
        const int32_t count = attempts.load();
        client->register_instance(instance());
        _exit(!client->initialize_done() && attempts.load() == count ? 0 : 4);
      },
      ::testing::ExitedWithCode(0),
      ".*");
}

TEST(XServiceClientLifecycleDeathTest, CancelsInitializationWaitingForMaster) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        alarm(5);
        master_available.store(false);
        auto* client = XServiceClient::get_instance();
        auto started = first_attempt.get_future();
        bool initialized = true;
        std::thread initialization([&] {
          initialized = client->init("http://127.0.0.1:9", "waiting");
        });
        if (started.wait_for(std::chrono::seconds(1)) !=
            std::future_status::ready) {
          _exit(1);
        }
        client->shutdown();
        initialization.join();
        _exit(!initialized && !client->initialize_done() ? 0 : 2);
      },
      ::testing::ExitedWithCode(0),
      ".*");
}

}  // namespace
}  // namespace xllm
