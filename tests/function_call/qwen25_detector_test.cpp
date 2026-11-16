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

#include "qwen25_detector.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "core_types.h"
#include "function_call_parser.h"

namespace xllm {
namespace function_call {

class Qwen25TestBase : public ::testing::Test {
 protected:
  void SetUp() override {
    nlohmann::json weather_params = {
        {"type", "object"},
        {"properties",
         {{"location",
           {{"type", "string"},
            {"description", "The city and state, e.g. San Francisco, CA"}}},
          {"unit", {{"type", "string"}, {"enum", {"celsius", "fahrenheit"}}}}}},
        {"required", {"location"}}};

    JsonFunction weather_func("get_current_weather",
                              "Get the current weather in a given location",
                              weather_params);

    nlohmann::json calculator_params = {
        {"type", "object"},
        {"properties",
         {{"expression",
           {{"type", "string"},
            {"description", "Mathematical expression to evaluate"}}}}},
        {"required", {"expression"}}};

    JsonFunction calculator_func(
        "calculate", "Calculate mathematical expressions", calculator_params);

    tools_ = {JsonTool("function", weather_func),
              JsonTool("function", calculator_func)};
  }

  std::vector<JsonTool> tools_;
};

class Qwen25DetectorTest : public Qwen25TestBase {
 protected:
  void SetUp() override {
    Qwen25TestBase::SetUp();
    detector_ = std::make_unique<Qwen25Detector>();
  }

  std::unique_ptr<Qwen25Detector> detector_;
};

class Qwen25StreamingTest : public Qwen25TestBase {
 protected:
  StreamingParseResult parse_chunks(const std::vector<std::string>& chunks) {
    FunctionCallParser parser(tools_, "qwen25");
    StreamingParseResult accumulated;
    accumulated.calls.reserve(2);
    for (const std::string& chunk : chunks) {
      const StreamingParseResult result =
          parser.parse_streaming_increment(chunk);
      accumulated.normal_text += result.normal_text;
      for (const ToolCallItem& call : result.calls) {
        EXPECT_GE(call.tool_index, 0);
        if (call.tool_index < 0) {
          continue;
        }
        while (accumulated.calls.size() <=
               static_cast<size_t>(call.tool_index)) {
          accumulated.calls.emplace_back();
        }
        ToolCallItem& current = accumulated.calls[call.tool_index];
        if (call.name.has_value()) {
          EXPECT_FALSE(current.name.has_value()) << "Duplicate tool name event";
          current.name = call.name;
        }
        current.parameters += call.parameters;
      }
    }
    return accumulated;
  }
};

// Test has_tool_call method
TEST_F(Qwen25DetectorTest, HasToolCallDetection) {
  // Test text containing tool calls
  EXPECT_TRUE(detector_->has_tool_call("<tool_call>\n"));
  EXPECT_TRUE(
      detector_->has_tool_call("Previous text <tool_call>\nFollowing content"));
  EXPECT_TRUE(detector_->has_tool_call(
      "<tool_call>\n{\"name\": \"test\"}\n</tool_call>"));

  // Test text not containing tool calls
  EXPECT_FALSE(detector_->has_tool_call(""));
  EXPECT_FALSE(detector_->has_tool_call("Regular text"));
  EXPECT_FALSE(detector_->has_tool_call("tool_call without brackets"));
  EXPECT_FALSE(detector_->has_tool_call("<tool_call without newline"));
}

// Test trim_whitespace method (indirectly tested through public interface)
TEST_F(Qwen25DetectorTest, TrimWhitespaceHandling) {
  std::string text_with_whitespace =
      "  \t\nPrevious text\r\n  <tool_call>\n  {\"name\": "
      "\"get_current_weather\", \"arguments\": {\"location\": \"Beijing\"}}  "
      "\n</tool_call>  \t\r\n";

  auto result = detector_->detect_and_parse(text_with_whitespace, tools_);

  // Verify normal text is correctly trimmed
  EXPECT_EQ(result.normal_text, "Previous text");

  // Verify tool call is correctly parsed
  ASSERT_EQ(result.calls.size(), 1);
}

// Test single tool call parsing
TEST_F(Qwen25DetectorTest, SingleToolCallParsing) {
  std::string text =
      "Please help me check the weather <tool_call>\n{\"name\": "
      "\"get_current_weather\", \"arguments\": {\"location\": \"Beijing\", "
      "\"unit\": \"celsius\"}}\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Please help me check the weather");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];

  ASSERT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_current_weather");

  // Verify parameter JSON
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "Beijing");
  EXPECT_EQ(params["unit"], "celsius");
}

// Test multiple tool calls parsing
TEST_F(Qwen25DetectorTest, MultipleToolCallsParsing) {
  std::string text =
      "Please help me check the weather and calculate an expression "
      "<tool_call>\n{\"name\": \"get_current_weather\", \"arguments\": "
      "{\"location\": \"Shanghai\"}}\n</tool_call>\n<tool_call>\n{\"name\": "
      "\"calculate\", \"arguments\": {\"expression\": \"2 + 3 * "
      "4\"}}\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text,
            "Please help me check the weather and calculate an expression");
  ASSERT_EQ(result.calls.size(), 2);

  // Verify first tool call
  const auto& call1 = result.calls[0];

  ASSERT_TRUE(call1.name.has_value());
  EXPECT_EQ(call1.name.value(), "get_current_weather");

  nlohmann::json params1 = nlohmann::json::parse(call1.parameters);
  EXPECT_EQ(params1["location"], "Shanghai");

  // Verify second tool call
  const auto& call2 = result.calls[1];

  ASSERT_TRUE(call2.name.has_value());
  EXPECT_EQ(call2.name.value(), "calculate");

  nlohmann::json params2 = nlohmann::json::parse(call2.parameters);
  EXPECT_EQ(params2["expression"], "2 + 3 * 4");
}

// Test invalid JSON handling
TEST_F(Qwen25DetectorTest, InvalidJsonHandling) {
  std::string text =
      "Test invalid JSON <tool_call>\n{\"name\": \"get_current_weather\", "
      "\"arguments\": {\"location\": \"Beijing\", invalid_json}}\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Test invalid JSON");
  EXPECT_EQ(result.calls.size(), 0);  // Invalid JSON should be ignored
}

// Test empty tool call content
TEST_F(Qwen25DetectorTest, EmptyToolCallContent) {
  std::string text = "Test empty content <tool_call>\n   \t\n  \n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Test empty content");
  EXPECT_EQ(result.calls.size(), 0);  // Empty content should be ignored
}

// Test incomplete tool call (only start tag)
TEST_F(Qwen25DetectorTest, IncompleteToolCall) {
  std::string text =
      "Incomplete tool call <tool_call>\n{\"name\": \"get_current_weather\"";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Incomplete tool call");
  EXPECT_EQ(result.calls.size(), 0);  // Incomplete calls should be ignored
}

// Test unknown tool name handling
TEST_F(Qwen25DetectorTest, UnknownToolName) {
  std::string text =
      "Unknown tool <tool_call>\n{\"name\": \"unknown_tool\", \"arguments\": "
      "{\"param\": \"value\"}}\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Unknown tool");
  // Base class will skip unknown tools, so should be 0 calls
  EXPECT_EQ(result.calls.size(), 0);
}

// Test case with only normal text
TEST_F(Qwen25DetectorTest, OnlyNormalText) {
  // Validate the public no-call fallback rather than a discarded detector
  // trim intermediate; every original input remains checked byte-for-byte.
  for (const std::string& text :
       {"This is a regular text without any tool calls.", "", "   \t\n\r   "}) {
    SCOPED_TRACE(text);
    FunctionCallParser parser(tools_, "qwen25");
    EXPECT_FALSE(parser.has_tool_call(text));
    const auto [content, calls] = parser.parse_non_stream(text);
    EXPECT_EQ(content, text);
    EXPECT_TRUE(calls.empty());
  }
}

TEST_F(Qwen25DetectorTest, ComplexNestedJsonParameters) {
  std::string text =
      "Complex parameter test <tool_call>\n{\"name\": \"get_current_weather\", "
      "\"arguments\": {\"location\": \"Beijing\", \"options\": "
      "{\"include_forecast\": true, \"days\": 7, \"details\": "
      "[\"temperature\", \"humidity\", \"wind\"]}}}\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Complex parameter test");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "Beijing");
  EXPECT_TRUE(params["options"]["include_forecast"]);
  EXPECT_EQ(params["options"]["days"], 7);
  EXPECT_EQ(params["options"]["details"].size(), 3);
}

// Test tool call in the middle of text
TEST_F(Qwen25DetectorTest, ToolCallInMiddleOfText) {
  std::string text =
      "Previous text <tool_call>\n{\"name\": \"calculate\", \"arguments\": "
      "{\"expression\": \"1+1\"}}\n</tool_call> Following text";

  auto result = detector_->detect_and_parse(text, tools_);

  // Note: According to implementation, only text before tool call is preserved
  // as normal_text
  EXPECT_EQ(result.normal_text, "Previous text");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];

  EXPECT_EQ(call.name.value(), "calculate");
}

// Test special characters handling
TEST_F(Qwen25DetectorTest, SpecialCharactersHandling) {
  std::string text =
      "Special characters test <tool_call>\n{\"name\": "
      "\"get_current_weather\", \"arguments\": {\"location\": \"New York "
      "City\", \"note\": \"Contains symbols！@#$%^&*()_+=\"}}\n</tool_call>";

  auto result = detector_->detect_and_parse(text, tools_);

  EXPECT_EQ(result.normal_text, "Special characters test");
  ASSERT_EQ(result.calls.size(), 1);

  const auto& call = result.calls[0];

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["location"], "New York City");
  EXPECT_EQ(params["note"], "Contains symbols！@#$%^&*()_+=");
}

// Test basic streaming functionality
TEST_F(Qwen25StreamingTest, BasicStreamingParsing) {
  // Simulate streaming chunks
  std::vector<std::string> chunks = {
      "I need to check the weather ",
      "<tool_call>\n",
      "{\"name\": \"get_current_weather\", ",
      "\"arguments\": {\"location\": \"Beijing\", ",
      "\"unit\": \"celsius\"}}\n",
      "</tool_call>"};

  const StreamingParseResult result = parse_chunks(chunks);
  EXPECT_EQ(result.normal_text, "I need to check the weather ");
  ASSERT_EQ(result.calls.size(), 1);
  EXPECT_EQ(result.calls[0].name, "get_current_weather");
  EXPECT_EQ(nlohmann::json::parse(result.calls[0].parameters),
            nlohmann::json({{"location", "Beijing"}, {"unit", "celsius"}}));
}

TEST_F(Qwen25StreamingTest, MultipleToolCallsStreaming) {
  // Simulate realistic token-level streaming chunks with multiple tool calls
  std::vector<std::string> chunks = {"Let",
                                     " me",
                                     " help",
                                     " you",
                                     " with",
                                     " weather",
                                     " and",
                                     " calculation",
                                     " ",
                                     "<tool_call>",
                                     "\n",
                                     "{",
                                     "\"name\"",
                                     ":",
                                     " \"",
                                     "get_current_weather",
                                     "\",",
                                     " ",
                                     "\"arguments\"",
                                     ":",
                                     " {",
                                     "\"location\"",
                                     ":",
                                     " \"",
                                     "Shanghai",
                                     "\"}}\n",
                                     "</tool_call>",
                                     "\n",
                                     "<tool_call>",
                                     "\n",
                                     "{",
                                     "\"name\"",
                                     ":",
                                     " \"",
                                     "calculate",
                                     "\",",
                                     " ",
                                     "\"arguments\"",
                                     ":",
                                     " {",
                                     "\"expression\"",
                                     ":",
                                     " \"",
                                     "2",
                                     " +",
                                     " ",
                                     "3",
                                     "\"}}\n",
                                     "</tool_call>"};

  const StreamingParseResult result = parse_chunks(chunks);
  EXPECT_EQ(result.normal_text,
            "Let me help you with weather and calculation ");
  ASSERT_EQ(result.calls.size(), 2);
  EXPECT_EQ(result.calls[0].name, "get_current_weather");
  EXPECT_EQ(nlohmann::json::parse(result.calls[0].parameters),
            nlohmann::json({{"location", "Shanghai"}}));
  EXPECT_EQ(result.calls[1].name, "calculate");
  EXPECT_EQ(nlohmann::json::parse(result.calls[1].parameters),
            nlohmann::json({{"expression", "2 + 3"}}));
}

}  // namespace function_call
}  // namespace xllm
