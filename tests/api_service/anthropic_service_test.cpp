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

// Requires a running, tool-capable model server. Run manually with
// XLLM_TEST_BASE_URL and XLLM_TEST_MODEL, plus XLLM_TEST_API_KEY if needed:
// ./anthropic_service_test --gtest_also_run_disabled_tests
#include <brpc/channel.h>
#include <brpc/controller.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace xllm {
namespace {

struct TestConfig {
  std::string base_url;
  std::string model;
  std::string api_key;
};

TestConfig load_test_config() {
  const char* url = std::getenv("XLLM_TEST_BASE_URL");
  const char* model = std::getenv("XLLM_TEST_MODEL");
  const char* key = std::getenv("XLLM_TEST_API_KEY");
  return {url ? url : "http://127.0.0.1:9977",
          model ? model : "my_model",
          key ? key : "xllm-test-123456"};
}

using StreamEvents = std::vector<std::pair<std::string, nlohmann::json>>;

class HttpClient final {
 public:
  explicit HttpClient(const TestConfig& config) : api_key_(config.api_key) {}

  int init_channel(const TestConfig& config) {
    brpc::ChannelOptions options;
    options.protocol = brpc::PROTOCOL_HTTP;
    options.connection_type = brpc::CONNECTION_TYPE_POOLED;
    options.timeout_ms = 60000;
    options.max_retry = 0;
    return channel_.Init(config.base_url.c_str(), &options);
  }

  ::testing::AssertionResult post(const nlohmann::json& body,
                                  nlohmann::json& response) {
    std::string raw;
    const auto status = post_raw(body, /*streaming=*/false, raw);
    if (!status) {
      return status;
    }
    try {
      response = nlohmann::json::parse(raw);
    } catch (const nlohmann::json::parse_error& error) {
      return ::testing::AssertionFailure() << error.what() << ", body=" << raw;
    }
    if (!response.is_object() || !response.contains("content") ||
        !response["content"].is_array() || response["content"].empty()) {
      return ::testing::AssertionFailure()
             << "Invalid message response: " << raw;
    }
    return ::testing::AssertionSuccess();
  }

  ::testing::AssertionResult post_stream(const nlohmann::json& body,
                                         StreamEvents& events) {
    std::string raw;
    const auto status = post_raw(body, /*streaming=*/true, raw);
    if (!status) {
      return status;
    }
    std::istringstream stream(raw);
    std::string line;
    std::string event_type;
    bool saw_done = false;
    while (std::getline(stream, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (line.empty()) {
        event_type.clear();
        continue;
      }
      if (line.rfind("event: ", 0) == 0) {
        event_type = line.substr(7);
        continue;
      }
      if (line.rfind("data: ", 0) != 0) {
        continue;
      }
      const std::string data = line.substr(6);
      if (data == "[DONE]") {
        if (saw_done || events.empty() ||
            events.back().first != "message_stop") {
          return ::testing::AssertionFailure() << "Unexpected SSE terminator";
        }
        saw_done = true;
        continue;
      }
      if (saw_done) {
        return ::testing::AssertionFailure() << "SSE data after terminator";
      }
      try {
        events.emplace_back(event_type, nlohmann::json::parse(data));
      } catch (const nlohmann::json::parse_error& error) {
        return ::testing::AssertionFailure()
               << "Malformed SSE data: " << error.what();
      }
    }
    if (!saw_done) {
      return ::testing::AssertionFailure() << "Missing SSE terminator";
    }
    return ::testing::AssertionSuccess();
  }

 private:
  ::testing::AssertionResult post_raw(const nlohmann::json& body,
                                      bool streaming,
                                      std::string& raw) {
    brpc::Controller controller;
    controller.http_request().uri() = "/v1/messages";
    controller.http_request().set_method(brpc::HTTP_METHOD_POST);
    controller.http_request().set_content_type("application/json");
    controller.http_request().SetHeader("Authorization", "Bearer " + api_key_);
    controller.http_request().SetHeader("anthropic-version", "2023-06-01");
    controller.request_attachment().append(body.dump());
    channel_.CallMethod(nullptr, &controller, nullptr, nullptr, nullptr);
    raw = controller.response_attachment().to_string();
    if (controller.Failed() ||
        controller.http_response().status_code() != 200) {
      return ::testing::AssertionFailure()
             << "status=" << controller.http_response().status_code()
             << ", error=" << controller.ErrorText() << ", body=" << raw;
    }
    const std::string expected_type =
        streaming ? "text/event-stream" : "application/json";
    if (controller.http_response().content_type().find(expected_type) != 0) {
      return ::testing::AssertionFailure()
             << "Unexpected content type: "
             << controller.http_response().content_type();
    }
    return ::testing::AssertionSuccess();
  }

  brpc::Channel channel_;
  std::string api_key_;
};

nlohmann::json weather_tool() {
  return {
      {"name", "get_weather"},
      {"description", "Get the current weather in a given location"},
      {"input_schema",
       {{"type", "object"},
        {"properties",
         {{"city", {{"type", "string"}}},
          {"unit", {{"type", "string"}, {"enum", {"celsius", "fahrenheit"}}}}}},
        {"required", {"city", "unit"}}}}};
}

nlohmann::json time_tool() {
  return {{"name", "get_time"},
          {"description", "Get the current time in a given location"},
          {"input_schema",
           {{"type", "object"},
            {"properties", {{"city", {{"type", "string"}}}}},
            {"required", {"city"}}}}};
}

void expect_stream_lifecycle(const StreamEvents& events) {
  ASSERT_FALSE(events.empty());
  EXPECT_EQ(events.front().first, "message_start");
  EXPECT_EQ(events.back().first, "message_stop");
  for (const auto& [type, data] : events) {
    ASSERT_TRUE(data.contains("type"));
    EXPECT_EQ(data["type"], type);
    EXPECT_NE(type, "error");
  }
  auto cursor = events.begin();
  for (const std::string& type : {"message_start",
                                  "content_block_start",
                                  "content_block_delta",
                                  "content_block_stop",
                                  "message_delta",
                                  "message_stop"}) {
    cursor = std::find_if(cursor, events.end(), [&type](const auto& event) {
      return event.first == type;
    });
    ASSERT_NE(cursor, events.end()) << "Missing or out-of-order " << type;
    ++cursor;
  }
}

class DISABLED_AnthropicServerFeaturesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_ = load_test_config();
    client_ = std::make_unique<HttpClient>(config_);
    ASSERT_EQ(client_->init_channel(config_), 0)
        << "Failed to initialize Anthropic test HTTP channel for "
        << config_.base_url;
  }

  nlohmann::json make_request(const std::string& prompt,
                              int32_t max_tokens = 64) const {
    return {{"model", config_.model},
            {"max_tokens", max_tokens},
            {"messages", {{{"role", "user"}, {"content", prompt}}}},
            {"temperature", 0}};
  }

  TestConfig config_;
  std::unique_ptr<HttpClient> client_;
};

TEST_F(DISABLED_AnthropicServerFeaturesTest, ToolChoiceAuto) {
  auto request = make_request("What is the weather like in Paris?", 2048);
  request["tools"] = {weather_tool(), time_tool()};
  nlohmann::json response;
  ASSERT_TRUE(client_->post(request, response));
  EXPECT_TRUE(std::any_of(response["content"].begin(),
                          response["content"].end(),
                          [](const auto& block) {
                            return block.value("type", "") == "text" ||
                                   block.value("type", "") == "tool_use";
                          }));
}

TEST_F(DISABLED_AnthropicServerFeaturesTest, ToolChoiceAny) {
  auto request = make_request("Tell me something about Berlin.", 2048);
  request["tools"] = {weather_tool(), time_tool()};
  request["tool_choice"] = {{"type", "any"}};
  nlohmann::json response;
  ASSERT_TRUE(client_->post(request, response));
  EXPECT_EQ(response["stop_reason"], "tool_use");
  const auto block = std::find_if(
      response["content"].begin(),
      response["content"].end(),
      [](const auto& item) { return item.value("type", "") == "tool_use"; });
  ASSERT_NE(block, response["content"].end());
  EXPECT_TRUE((*block)["name"] == "get_weather" ||
              (*block)["name"] == "get_time");
}

TEST_F(DISABLED_AnthropicServerFeaturesTest, ToolUseBlockStructure) {
  auto request =
      make_request("Get the weather in San Francisco in celsius", 2048);
  request["tools"] = {weather_tool(), time_tool()};
  request["tool_choice"] = {{"type", "tool"}, {"name", "get_weather"}};
  nlohmann::json response;
  ASSERT_TRUE(client_->post(request, response));
  EXPECT_EQ(response["stop_reason"], "tool_use");
  const auto block = std::find_if(
      response["content"].begin(),
      response["content"].end(),
      [](const auto& item) { return item.value("type", "") == "tool_use"; });
  ASSERT_NE(block, response["content"].end());
  EXPECT_EQ((*block)["name"], "get_weather");
  ASSERT_TRUE((*block)["id"].is_string());
  EXPECT_FALSE((*block)["id"].get<std::string>().empty());
  const auto& input = (*block)["input"];
  ASSERT_TRUE(input.is_object());
  EXPECT_TRUE(input["city"].is_string());
  EXPECT_TRUE(input["unit"] == "celsius" || input["unit"] == "fahrenheit");
}

TEST_F(DISABLED_AnthropicServerFeaturesTest, StopReasonMaxTokens) {
  auto request =
      make_request("Write a long essay about the history of computing.", 1);
  nlohmann::json response;
  ASSERT_TRUE(client_->post(request, response));
  EXPECT_EQ(response["stop_reason"], "max_tokens");
}

TEST_F(DISABLED_AnthropicServerFeaturesTest, ChatCompletion) {
  auto request =
      make_request("What is the capital of France? Answer in a few words.");
  nlohmann::json response;
  ASSERT_TRUE(client_->post(request, response));
  ASSERT_TRUE(response["id"].is_string());
  EXPECT_FALSE(response["id"].get<std::string>().empty());
  EXPECT_EQ(response["type"], "message");
  EXPECT_EQ(response["role"], "assistant");
  EXPECT_TRUE(response["content"][0]["text"].is_string());
  EXPECT_EQ(response["model"], config_.model);
  EXPECT_TRUE(response["stop_reason"].is_string());
  EXPECT_TRUE(response["usage"]["input_tokens"].is_number_integer());
  EXPECT_TRUE(response["usage"]["output_tokens"].is_number_integer());
}

TEST_F(DISABLED_AnthropicServerFeaturesTest, ChatCompletionStream) {
  auto request = make_request("What is the capital of France?");
  request["stream"] = true;
  StreamEvents events;
  ASSERT_TRUE(client_->post_stream(request, events));
  expect_stream_lifecycle(events);
}

// Retains the streaming function-call integration risk, rather than accepting
// a text-only response that never exercises tool argument deltas.
TEST_F(DISABLED_AnthropicServerFeaturesTest, FunctionCallingStreamingSimple) {
  auto request =
      make_request("What is the temperature in Paris in celsius?", 2048);
  request["tools"] = {weather_tool()};
  request["tool_choice"] = {{"type", "tool"}, {"name", "get_weather"}};
  request["stream"] = true;
  StreamEvents events;
  ASSERT_TRUE(client_->post_stream(request, events));
  ASSERT_NO_FATAL_FAILURE(expect_stream_lifecycle(events));
  const auto start =
      std::find_if(events.begin(), events.end(), [](const auto& event) {
        return event.first == "content_block_start" &&
               event.second["content_block"].value("type", "") == "tool_use";
      });
  ASSERT_NE(start, events.end());
  EXPECT_EQ(start->second["content_block"]["name"], "get_weather");
  const int32_t index = start->second["index"].get<int32_t>();
  std::string arguments;
  for (const auto& [type, data] : events) {
    if (type != "content_block_delta" || data.value("index", -1) != index ||
        data["delta"].value("type", "") != "input_json_delta") {
      continue;
    }
    arguments += data["delta"]["partial_json"].get<std::string>();
  }
  const auto input = nlohmann::json::parse(arguments);
  EXPECT_TRUE(input["city"].is_string());
  EXPECT_TRUE(input["unit"] == "celsius" || input["unit"] == "fahrenheit");
  const auto delta =
      std::find_if(events.begin(), events.end(), [](const auto& event) {
        return event.first == "message_delta";
      });
  ASSERT_NE(delta, events.end());
  EXPECT_EQ(delta->second["delta"]["stop_reason"], "tool_use");
}

}  // namespace
}  // namespace xllm
