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

#pragma once

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace xllm::tests {

// Reserve a contiguous block with real bind calls, not a shared fixed base.
// Keep it alive while preparing child processes. Frameworks that cannot adopt
// listening FDs need release() immediately before starting their listeners.
class ScopedPortRange final {
 public:
  explicit ScopedPortRange(int32_t count = 1) {
    if (count <= 0 || count > UINT16_MAX) {
      throw std::invalid_argument("Invalid test port range size");
    }
    sockets_.reserve(count);
    constexpr int32_t kMaxAttempts = 32;
    for (int32_t attempt = 0; attempt < kMaxAttempts; ++attempt) {
      bool available = true;
      for (int32_t offset = 0; offset < count; ++offset) {
        const int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (fd < 0) {
          const int error = errno;
          clear();
          throw std::system_error(error, std::generic_category(), "socket");
        }
        sockets_.emplace_back(fd);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_ANY);
        address.sin_port =
            htons(static_cast<uint16_t>(offset == 0 ? 0 : base_port_ + offset));
        if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) !=
            0) {
          const int error = errno;
          clear();
          if (error != EADDRINUSE) {
            throw std::system_error(error, std::generic_category(), "bind");
          }
          available = false;
          break;
        }
        if (offset == 0) {
          socklen_t size = sizeof(address);
          if (getsockname(fd, reinterpret_cast<sockaddr*>(&address), &size) !=
              0) {
            const int error = errno;
            clear();
            throw std::system_error(
                error, std::generic_category(), "getsockname");
          }
          base_port_ = ntohs(address.sin_port);
          if (base_port_ + count - 1 > UINT16_MAX) {
            available = false;
            break;
          }
        }
      }
      if (available) {
        return;
      }
      clear();
    }
    throw std::runtime_error("No available test port block after 32 attempts");
  }

  ~ScopedPortRange() { clear(); }
  ScopedPortRange(const ScopedPortRange&) = delete;
  ScopedPortRange& operator=(const ScopedPortRange&) = delete;

  int32_t base_port() const { return base_port_; }

  int32_t release() {
    clear();
    return base_port_;
  }

 private:
  void clear() {
    for (int fd : sockets_) {
      close(fd);
    }
    sockets_.clear();
  }

  int32_t base_port_ = 0;
  std::vector<int> sockets_;
};

}  // namespace xllm::tests
