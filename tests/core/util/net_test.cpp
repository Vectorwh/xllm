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

#include "core/util/net.h"

#include <gtest/gtest.h>

#include <array>
#include <exception>
#include <memory>
#include <thread>

#include "tests/scoped_port_range.h"

namespace xllm {
namespace net {
namespace {

TEST(TestPortRangeTest, HoldsEveryPortUntilReleased) {
  xllm::tests::ScopedPortRange ports(/*count=*/4);
  const int32_t base = ports.base_port();
  for (int32_t offset = 0; offset < 4; ++offset) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(fd, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(static_cast<uint16_t>(base + offset));
    const int result =
        bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address));
    const int error = errno;
    close(fd);
    EXPECT_EQ(result, -1);
    EXPECT_EQ(error, EADDRINUSE);
  }
  EXPECT_EQ(ports.release(), base);
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(fd, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(static_cast<uint16_t>(base));
  EXPECT_EQ(bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)),
            0);
  EXPECT_EQ(listen(fd, /*backlog=*/1), 0);
  close(fd);
}

TEST(TestPortRangeTest, ConcurrentReservationsDoNotOverlap) {
  constexpr int32_t kWorkers = 8;
  constexpr int32_t kPortsPerWorker = 16;
  std::array<std::unique_ptr<xllm::tests::ScopedPortRange>, kWorkers> ports;
  std::array<std::exception_ptr, kWorkers> errors;
  std::array<std::thread, kWorkers> workers;
  for (int32_t worker = 0; worker < kWorkers; ++worker) {
    workers[worker] = std::thread([&, worker] {
      try {
        ports[worker] =
            std::make_unique<xllm::tests::ScopedPortRange>(kPortsPerWorker);
      } catch (...) {
        errors[worker] = std::current_exception();
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (int32_t worker = 0; worker < kWorkers; ++worker) {
    ASSERT_EQ(errors[worker], nullptr);
    ASSERT_NE(ports[worker], nullptr);
    for (int32_t previous = 0; previous < worker; ++previous) {
      const int32_t first = ports[worker]->base_port();
      const int32_t second = ports[previous]->base_port();
      EXPECT_TRUE(first + kPortsPerWorker <= second ||
                  second + kPortsPerWorker <= first);
    }
  }
}

TEST(TestPortRangeTest, RejectsInvalidRangeWithoutRetrying) {
  EXPECT_THROW(xllm::tests::ScopedPortRange(/*count=*/0),
               std::invalid_argument);
  EXPECT_THROW(xllm::tests::ScopedPortRange(/*count=*/65536),
               std::invalid_argument);
}

TEST(NetTest, SelectsLoopbackForLoopbackRemote) {
  EXPECT_EQ(get_route_ip("127.0.0.1:19888"), "127.0.0.1");
}

TEST(NetTest, ResolvesRemoteHostname) {
  EXPECT_EQ(get_route_ip("localhost:19888"), "127.0.0.1");
}

TEST(NetTest, NormalizesLocalBindHosts) {
  const std::string local_ip = get_local_ip_addr();
  ASSERT_FALSE(local_ip.empty());

  EXPECT_EQ(extract_ip("0.0.0.0"), local_ip);
  EXPECT_EQ(extract_ip("127.0.0.1"), local_ip);
  EXPECT_EQ(extract_ip("localhost"), local_ip);
}

TEST(NetTest, ReturnsEmptyWhenRemoteHasNoUsableRoute) {
  EXPECT_TRUE(get_route_ip("255.255.255.255:19888").empty());
}

TEST(NetTest, ReturnsEmptyWhenRemoteCannotBeResolved) {
  EXPECT_TRUE(get_route_ip("worker.invalid:19888").empty());
}

}  // namespace
}  // namespace net
}  // namespace xllm
