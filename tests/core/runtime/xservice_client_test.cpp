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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <etcd/SyncClient.hpp>
#include <etcd/v3/action_constants.hpp>
#include <future>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#include "core/common/etcd_schema.h"
#include "tests/core/common/etcd_registry_fixture.h"

namespace xllm {
namespace {

using test::FakeRegistry;

InstanceInfo instance() {
  InstanceInfo info;
  info.name = "127.0.0.1:9";
  info.type = "PREFILL";
  return info;
}

std::string registered_incarnation(const FakeRegistry& registry) {
  const std::string key =
      std::string(etcd_schema::kPrefillInstancePrefix) + instance().name;
  etcd::SyncClient reader(registry.address());
  const auto response = reader.get(key);
  if (response.error_code() == etcdv3::ERROR_KEY_NOT_FOUND) {
    return {};
  }
  CHECK(response.is_ok()) << response.error_message();
  CHECK_GT(response.value().lease(), 0);
  const auto lease = reader.leasetimetolive(response.value().lease());
  CHECK(lease.is_ok()) << lease.error_message();
  CHECK_GT(lease.value().ttl(), 0);
  return nlohmann::json::parse(response.value().as_string())
      .at("incarnation_id")
      .get<std::string>();
}

TEST(XServiceClientLifecycleDeathTest, ReinitializesAfterCompleteShutdown) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        alarm(5);
        // Exercise proxy isolation even when the runner has no proxy settings.
        setenv("grpc_proxy", "http://127.0.0.1:0", /*overwrite=*/1);
        unsetenv("no_grpc_proxy");
        unsetenv("no_proxy");
        FakeRegistry registry;
        auto* client = XServiceClient::get_instance();
        if (!client->init(registry.address(), "first")) {
          _exit(1);
        }
        client->register_instance(instance());
        const std::string first = registered_incarnation(registry);
        client->withdraw_instance_registration();
        if (!registered_incarnation(registry).empty()) {
          _exit(6);
        }
        client->shutdown();
        if (client->initialize_done()) {
          _exit(2);
        }
        if (!client->init(registry.address(), "second")) {
          _exit(3);
        }
        client->register_instance(instance());
        if (first.empty() || first == registered_incarnation(registry)) {
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
        FakeRegistry registry(
            /*mode=*/FakeRegistry::Mode::REGISTRATION_FAILURE);
        auto* client = XServiceClient::get_instance();
        if (!client->init(registry.address(), "cancel")) {
          _exit(1);
        }
        auto started = registry.first_attempt();
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
        const int32_t count = registry.attempts();
        client->register_instance(instance());
        _exit(!client->initialize_done() && registry.attempts() == count ? 0
                                                                         : 4);
      },
      ::testing::ExitedWithCode(0),
      ".*");
}

TEST(XServiceClientLifecycleDeathTest, CancelsInitializationWaitingForMaster) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        alarm(5);
        FakeRegistry registry(/*mode=*/FakeRegistry::Mode::MASTER_UNAVAILABLE);
        auto* client = XServiceClient::get_instance();
        auto started = registry.first_attempt();
        bool initialized = true;
        std::thread initialization(
            [&] { initialized = client->init(registry.address(), "waiting"); });
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
