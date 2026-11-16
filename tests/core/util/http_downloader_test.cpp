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

#include "core/util/http_downloader.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(ParseHeadersJsonTest, EmptyStringReturnsEmptyMap) {
  EXPECT_TRUE(parse_headers_json("").empty());
  EXPECT_TRUE(parse_headers_json("{}").empty());
}

TEST(ParseHeadersJsonTest, ValidJsonParsesCorrectly) {
  const auto headers = parse_headers_json(
      R"({"Authorization":"Bearer token123","Referer":"https://example.com"})");
  ASSERT_EQ(headers.size(), 2);
  EXPECT_EQ(headers.at("Authorization"), "Bearer token123");
  EXPECT_EQ(headers.at("Referer"), "https://example.com");
}

TEST(ParseHeadersJsonTest, InvalidJsonReturnsEmptyMap) {
  EXPECT_TRUE(parse_headers_json("not-valid-json").empty());
}

}  // namespace
}  // namespace xllm
