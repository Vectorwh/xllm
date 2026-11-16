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

#include "glm47_detector.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "function_call_parser.h"

namespace xllm {
namespace function_call {

class Glm47DetectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    detector_ = std::make_unique<Glm47Detector>();

    // Setup test tools
    nlohmann::json weather_params = {
        {"type", "object"},
        {"properties",
         {{"city",
           {{"type", "string"},
            {"description", "The city name, e.g. Beijing, Shanghai"}}},
          {"date",
           {{"type", "string"},
            {"description", "Date in YYYY-MM-DD format"}}}}},
        {"required", {"city"}}};

    JsonFunction weather_func("get_weather",
                              "Get the weather information for a given city",
                              weather_params);

    nlohmann::json calculator_params = {
        {"type", "object"},
        {"properties",
         {{"expression",
           {{"type", "string"},
            {"description", "Mathematical expression to evaluate"}}},
          {"precision",
           {{"type", "number"}, {"description", "Number of decimal places"}}}}},
        {"required", {"expression"}}};

    JsonFunction calculator_func(
        "calculate", "Calculate mathematical expressions", calculator_params);

    tools_ = {JsonTool("function", weather_func),
              JsonTool("function", calculator_func)};
  }

  std::unique_ptr<Glm47Detector> detector_;
  std::vector<JsonTool> tools_;
};

namespace {
// Concatenate the streamed argument fragments of one parse result.
std::string collect_streamed_args(const StreamingParseResult& result) {
  std::string arguments;
  for (const auto& call : result.calls) {
    EXPECT_GE(call.tool_index, 0);
    arguments += call.parameters;
  }
  return arguments;
}
}  // namespace

// Test has_tool_call method
TEST_F(Glm47DetectorTest, HasToolCallDetection) {
  // Test text containing tool calls
  EXPECT_TRUE(detector_->has_tool_call("<tool_call>"));
  EXPECT_TRUE(
      detector_->has_tool_call("Previous text <tool_call>Following content"));
  EXPECT_TRUE(detector_->has_tool_call(
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</"
      "arg_value></tool_call>"));

  // Test text not containing tool calls
  EXPECT_FALSE(detector_->has_tool_call(""));
  EXPECT_FALSE(detector_->has_tool_call("Regular text"));
  EXPECT_FALSE(detector_->has_tool_call("tool_call without brackets"));
  EXPECT_FALSE(detector_->has_tool_call("<tool_call without closing"));
}

// Test single tool call parsing (GLM-4.7 compact format)
TEST_F(Glm47DetectorTest, SingleToolCallParsing) {
  std::string text =
      "Please help me check the weather "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value>"
      "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>";

  FunctionCallParser parser(tools_, "glm47");
  const auto [normal_text, calls] = parser.parse_non_stream(text);

  EXPECT_EQ(normal_text, "Please help me check the weather");
  ASSERT_EQ(calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = calls[0];

  ASSERT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_weather");

  // Verify parameter JSON
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "北京");
  EXPECT_EQ(params["date"], "2024-06-27");
}

// Test multiple tool calls parsing (GLM-4.7 format)
TEST_F(Glm47DetectorTest, MultipleToolCallsParsing) {
  std::string text =
      "Please help me check the weather and calculate "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>上海</arg_value>"
      "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>"
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value>"
      "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>";

  FunctionCallParser parser(tools_, "glm47");
  const auto [normal_text, calls] = parser.parse_non_stream(text);

  EXPECT_EQ(normal_text, "Please help me check the weather and calculate");
  ASSERT_EQ(calls.size(), 2);  // Use ASSERT to stop test if this fails

  // Verify first tool call
  const auto& call1 = calls[0];

  ASSERT_TRUE(call1.name.has_value());
  EXPECT_EQ(call1.name.value(), "get_weather");

  nlohmann::json params1 = nlohmann::json::parse(call1.parameters);
  EXPECT_EQ(params1["city"], "上海");
  EXPECT_EQ(params1["date"], "2024-06-27");

  // Verify second tool call
  const auto& call2 = calls[1];

  ASSERT_TRUE(call2.name.has_value());
  EXPECT_EQ(call2.name.value(), "get_weather");

  nlohmann::json params2 = nlohmann::json::parse(call2.parameters);
  EXPECT_EQ(params2["city"], "北京");
  EXPECT_EQ(params2["date"], "2024-06-27");
}

TEST_F(Glm47DetectorTest, Glm52AutoParserUsesGlm5Format) {
  EXPECT_EQ(FunctionCallParser::get_parser_auto("auto", "glm_moe_dsa"), "glm5");
  EXPECT_EQ(FunctionCallParser::get_parser_auto("auto", "glm_moe_dsa_mtp"),
            "glm5");
}

TEST_F(Glm47DetectorTest, Glm52CompactXmlToolCallParsing) {
  std::string text =
      "Need current weather "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value>"
      "<arg_key>date</arg_key><arg_value>2026-06-26</arg_value></tool_call>";

  FunctionCallParser parser(tools_, "glm5");
  auto [normal_text, calls] = parser.parse_non_stream(text);

  EXPECT_EQ(normal_text, "Need current weather");
  ASSERT_EQ(calls.size(), 1);

  const ToolCallItem& call = calls[0];
  ASSERT_TRUE(call.name.has_value());
  EXPECT_EQ(call.name.value(), "get_weather");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
  EXPECT_EQ(params["date"], "2026-06-26");
}

TEST_F(Glm47DetectorTest, KeepsNormalTextBetweenAndAfterToolCalls) {
  std::string text =
      "First "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value></tool_call>"
      " middle "
      "<tool_call>calculate<arg_key>expression</arg_key><arg_value>1 + 2</"
      "arg_value></tool_call>"
      " done";

  FunctionCallParser parser(tools_, "glm47");
  const auto [normal_text, calls] = parser.parse_non_stream(text);

  EXPECT_EQ(normal_text, "First  middle  done");
  ASSERT_EQ(calls.size(), 2);
  EXPECT_EQ(calls[0].name.value(), "get_weather");
  EXPECT_EQ(calls[1].name.value(), "calculate");
  EXPECT_EQ(nlohmann::json::parse(calls[0].parameters),
            nlohmann::json({{"city", "Beijing"}}));
  EXPECT_EQ(nlohmann::json::parse(calls[1].parameters),
            nlohmann::json({{"expression", "1 + 2"}}));
}

TEST_F(Glm47DetectorTest, DropsTrailingIncompleteToolCallFromNormalText) {
  std::string text =
      "First "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value></tool_call>"
      " after "
      "<tool_call>calculate<arg_key>expression</arg_key><arg_value>1 +";

  FunctionCallParser parser(tools_, "glm47");
  const auto [normal_text, calls] = parser.parse_non_stream(text);

  EXPECT_EQ(normal_text, "First  after");
  ASSERT_EQ(calls.size(), 1);
  EXPECT_EQ(calls[0].name.value(), "get_weather");
  EXPECT_EQ(nlohmann::json::parse(calls[0].parameters),
            nlohmann::json({{"city", "Beijing"}}));
}

// Test number type coercion
TEST_F(Glm47DetectorTest, NumberTypeCoercion) {
  std::string text =
      "Calculate with precision "
      "<tool_call>calculate<arg_key>expression</arg_key><arg_value>3.14 * "
      "2</arg_value>"
      "<arg_key>precision</arg_key><arg_value>\"2\"</arg_value></tool_call>";

  FunctionCallParser parser(tools_, "glm47");
  const auto [normal_text, calls] = parser.parse_non_stream(text);

  EXPECT_EQ(normal_text, "Calculate with precision");
  ASSERT_EQ(calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = calls[0];
  EXPECT_EQ(call.name.value(), "calculate");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["expression"], "3.14 * 2");
  // precision should be parsed as number
  EXPECT_TRUE(params["precision"].is_number());
  EXPECT_EQ(params["precision"], 2);
}

// Test incomplete tool call
TEST_F(Glm47DetectorTest, IncompleteToolCall) {
  std::string text =
      "Incomplete tool call "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing";

  FunctionCallParser parser(tools_, "glm47");
  const auto [normal_text, calls] = parser.parse_non_stream(text);

  EXPECT_EQ(normal_text, text);
  EXPECT_EQ(calls.size(), 0);  // Incomplete calls should be ignored
}

// Test unknown tool name handling
TEST_F(Glm47DetectorTest, UnknownToolName) {
  std::string text =
      "Unknown tool "
      "<tool_call>unknown_tool<arg_key>param</arg_key><arg_value>value</"
      "arg_value></tool_call>";

  FunctionCallParser parser(tools_, "glm47");
  const auto [normal_text, calls] = parser.parse_non_stream(text);

  EXPECT_EQ(normal_text, text);
  // Base class will skip unknown tools
  EXPECT_EQ(calls.size(), 0);
}

TEST_F(Glm47DetectorTest, OnlyNormalText) {
  FunctionCallParser parser(tools_, "glm47");
  for (const std::string& text :
       {"This is a regular text without any tool calls.", ""}) {
    const auto [content, calls] = parser.parse_non_stream(text);
    EXPECT_EQ(content, text);
    EXPECT_TRUE(calls.empty());
  }
}

// Test complex nested JSON parameters
TEST_F(Glm47DetectorTest, ComplexNestedJsonParameters) {
  std::string text =
      "Complex parameter test "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value>"
      "<arg_key>options</arg_key><arg_value>{\"include_forecast\": true, "
      "\"days\": 7}</arg_value>"
      "</tool_call>";

  FunctionCallParser parser(tools_, "glm47");
  const auto [normal_text, calls] = parser.parse_non_stream(text);

  EXPECT_EQ(normal_text, "Complex parameter test");
  ASSERT_EQ(calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = calls[0];

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
  EXPECT_TRUE(params["options"]["include_forecast"]);
  EXPECT_EQ(params["options"]["days"], 7);
}

// Test special characters handling
TEST_F(Glm47DetectorTest, SpecialCharactersHandling) {
  std::string text =
      "Special characters test "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>New York "
      "City</arg_value>"
      "<arg_key>note</arg_key><arg_value>Contains "
      "symbols！@#$%^&*()_+=</arg_value>"
      "</tool_call>";

  FunctionCallParser parser(tools_, "glm47");
  const auto [normal_text, calls] = parser.parse_non_stream(text);

  EXPECT_EQ(normal_text, "Special characters test");
  ASSERT_EQ(calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = calls[0];

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "New York City");
  EXPECT_EQ(params["note"], "Contains symbols！@#$%^&*()_+=");
}

// Test whitespace handling in arg values
TEST_F(Glm47DetectorTest, WhitespaceHandlingInArgValues) {
  std::string text =
      "Whitespace test "
      "<tool_call>get_weather<arg_key>  city  </arg_key><arg_value>  Beijing  "
      "</arg_value>"
      "<arg_key>\t\ndate\r\n</arg_key><arg_value>\n\t2024-06-27\r\n</arg_value>"
      "</tool_call>";

  FunctionCallParser parser(tools_, "glm47");
  const auto [normal_text, calls] = parser.parse_non_stream(text);

  EXPECT_EQ(normal_text, "Whitespace test");
  ASSERT_EQ(calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = calls[0];
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");  // Whitespace should be trimmed
  EXPECT_EQ(params["date"], "2024-06-27");
}

// Test streaming parsing functionality
TEST_F(Glm47DetectorTest, StreamingParseBasicFunctionality) {
  FunctionCallParser parser(tools_, "glm47");
  std::string chunk1 = "<tool_call>get_weather";
  std::string chunk2 = "<arg_key>city</arg_key><arg_value>";
  std::string chunk3 = "Beijing</arg_value></tool_call>";

  // First chunk - function name not yet complete (no <arg_key> or </tool_call>)
  // Should wait for more data to avoid sending partial names
  auto result1 = parser.parse_streaming_increment(chunk1);
  EXPECT_EQ(result1.calls.size(), 0);

  // Second chunk - now we have <arg_key>, so function name is complete
  auto result2 = parser.parse_streaming_increment(chunk2);
  ASSERT_GE(result2.calls.size(), 1);
  ASSERT_TRUE(result2.calls[0].name.has_value());
  EXPECT_EQ(result2.calls[0].name.value(), "get_weather");

  // Third chunk - completes the tool call
  auto result3 = parser.parse_streaming_increment(chunk3);
  EXPECT_EQ(nlohmann::json::parse(collect_streamed_args(result1) +
                                  collect_streamed_args(result2) +
                                  collect_streamed_args(result3)),
            nlohmann::json({{"city", "Beijing"}}));
}

// Test streaming parse with normal text
TEST_F(Glm47DetectorTest, StreamingParseWithNormalText) {
  FunctionCallParser parser(tools_, "glm47");
  std::string chunk1 = "Please check the weather ";
  std::string chunk2 = "<tool_call>get_weather";
  std::string chunk3 =
      "<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>";

  // First chunk - normal text should be returned
  auto result1 = parser.parse_streaming_increment(chunk1);
  EXPECT_EQ(result1.normal_text, "Please check the weather ");
  EXPECT_EQ(result1.calls.size(), 0);

  // Second chunk - tool call start, but function name not yet complete
  // (no <arg_key> or </tool_call>), should wait for more data
  auto result2 = parser.parse_streaming_increment(chunk2);
  EXPECT_EQ(result2.calls.size(), 0);

  // Third chunk - now we have <arg_key>, function name is complete
  auto result3 = parser.parse_streaming_increment(chunk3);
  ASSERT_GE(result3.calls.size(), 1);
  EXPECT_EQ(result3.calls[0].name, "get_weather");
  EXPECT_EQ(nlohmann::json::parse(collect_streamed_args(result3)),
            nlohmann::json({{"city", "Tokyo"}}));
}

// The marker can share one increment with the tail of the normal text: MTP
// emits several tokens per step, and delayed streaming callbacks can carry
// several steps at once. That tail must still be streamed; dropping it
// truncates responses that do contain tool calls only in stream mode.
TEST_F(Glm47DetectorTest, StreamingKeepsNormalTextBeforeMarkerInSameChunk) {
  FunctionCallParser parser(tools_, "glm47");
  auto result1 = parser.parse_streaming_increment(
      "根据您的经营目标，我来为您生成全站智能推广的");
  EXPECT_EQ(result1.normal_text,
            "根据您的经营目标，我来为您生成全站智能推广的");

  // Marker and the tail of the sentence arrive together.
  auto result2 = parser.parse_streaming_increment(
      "投放方案。<tool_call>get_weather\n<arg_key>");
  EXPECT_EQ(result2.normal_text, "投放方案。");
  ASSERT_EQ(result2.calls.size(), 1);
  ASSERT_TRUE(result2.calls[0].name.has_value());
  EXPECT_EQ(result2.calls[0].name.value(), "get_weather");

  auto result3 =
      parser.parse_streaming_increment("city</arg_key>\n<arg_value>Beijing");
  auto result4 = parser.parse_streaming_increment("</arg_value>");
  auto result5 = parser.parse_streaming_increment("\n</tool_call>");

  EXPECT_EQ(result1.normal_text + result2.normal_text,
            "根据您的经营目标，我来为您生成全站智能推广的投放方案。");

  std::string streamed_args = collect_streamed_args(result3) +
                              collect_streamed_args(result4) +
                              collect_streamed_args(result5);
  ASSERT_FALSE(streamed_args.empty());
  nlohmann::json params = nlohmann::json::parse(streamed_args);
  EXPECT_EQ(params["city"], "Beijing");
}

// An increment ending with '<' is held back as a possible marker start. The
// held text must be flushed once the marker turns out to be a real tool call.
TEST_F(Glm47DetectorTest, StreamingKeepsNormalTextHeldByPotentialMarkerPrefix) {
  FunctionCallParser parser(tools_, "glm47");
  auto result1 = parser.parse_streaming_increment("First part ");
  EXPECT_EQ(result1.normal_text, "First part ");

  auto result2 = parser.parse_streaming_increment("second part.<");
  EXPECT_EQ(result2.calls.size(), 0);

  auto result3 = parser.parse_streaming_increment(
      "tool_call>get_weather<arg_key>city</arg_key>");
  ASSERT_GE(result3.calls.size(), 1);
  ASSERT_TRUE(result3.calls[0].name.has_value());
  EXPECT_EQ(result3.calls[0].name.value(), "get_weather");

  auto result4 = parser.parse_streaming_increment(
      "<arg_value>Beijing</arg_value></tool_call>");
  EXPECT_EQ(result1.normal_text + result2.normal_text + result3.normal_text +
                result4.normal_text,
            "First part second part.");
  std::string streamed_args =
      collect_streamed_args(result3) + collect_streamed_args(result4);
  ASSERT_FALSE(streamed_args.empty());
  nlohmann::json params = nlohmann::json::parse(streamed_args);
  EXPECT_EQ(params["city"], "Beijing");
}

// Normal text that follows a completed tool call shares the increment that
// carries the next marker, so it must be flushed as well.
TEST_F(Glm47DetectorTest, StreamingKeepsNormalTextBetweenToolCalls) {
  FunctionCallParser parser(tools_, "glm47");
  auto result1 = parser.parse_streaming_increment(
      "<tool_call>get_weather<arg_key>city</arg_key>"
      "<arg_value>Beijing</arg_value></tool_call>");
  ASSERT_GE(result1.calls.size(), 1);
  EXPECT_EQ(result1.calls[0].name.value(), "get_weather");
  EXPECT_EQ(nlohmann::json::parse(collect_streamed_args(result1)),
            nlohmann::json({{"city", "Beijing"}}));

  // The first call is now fully consumed by result1, so the inter-call text is
  // released together with the next call's name instead of waiting for the
  // increment that closes it. The order the client sees is unchanged: the first
  // call, then this text, then the second call.
  auto result2 = parser.parse_streaming_increment(
      "中间说明<tool_call>calculate<arg_key>expression</arg_key>");
  EXPECT_EQ(result2.normal_text, "中间说明");
  ASSERT_GE(result2.calls.size(), 1);
  ASSERT_TRUE(result2.calls[0].name.has_value());
  EXPECT_EQ(result2.calls[0].name.value(), "calculate");

  auto result3 = parser.parse_streaming_increment(
      "<arg_value>1 + 2</arg_value></tool_call>");
  EXPECT_TRUE(result3.normal_text.empty());
  std::string args3 = collect_streamed_args(result3);
  ASSERT_FALSE(args3.empty());
  EXPECT_EQ(nlohmann::json::parse(args3)["expression"], "1 + 2");
}

// Object/array arg values also end with '}'. Streaming must still emit the
// root-object closing brace; otherwise clients see truncated JSON like
// {"city":"Beijing","config":"{\"nested\": {\"deep\": \"value\"}}"
// (missing the final root '}').
TEST_F(Glm47DetectorTest, StreamingParseClosesRootAfterObjectArgValue) {
  FunctionCallParser parser(tools_, "glm47");
  // Keep name and args in separate increments: name is emitted on the first
  // <arg_key>, and argument JSON increments are only produced afterwards.
  std::string chunk1 = "<tool_call>get_weather";
  std::string chunk2 =
      "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
      "<arg_key>config</arg_key><arg_value>{\"nested\": {\"deep\": "
      "\"value\"}}</arg_value>";
  std::string chunk3 = "</tool_call>";

  auto result1 = parser.parse_streaming_increment(chunk1);
  EXPECT_EQ(result1.calls.size(), 0);

  auto result2 = parser.parse_streaming_increment(chunk2);
  ASSERT_GE(result2.calls.size(), 1);
  ASSERT_TRUE(result2.calls[0].name.has_value());
  EXPECT_EQ(result2.calls[0].name.value(), "get_weather");

  auto result3 = parser.parse_streaming_increment(chunk3);

  std::string streamed_args =
      collect_streamed_args(result2) + collect_streamed_args(result3);
  ASSERT_FALSE(streamed_args.empty());
  // Without the root closer, this parse fails. Streaming may emit object/array
  // arg values as escaped JSON strings; only require a complete root object.
  nlohmann::json params = nlohmann::json::parse(streamed_args);
  ASSERT_TRUE(params.is_object());
  EXPECT_EQ(params["city"], "Beijing");
  EXPECT_EQ(params["config"], R"({"nested": {"deep": "value"}})");
}

// Test invalid JSON in arg values
TEST_F(Glm47DetectorTest, InvalidJsonInArgValues) {
  std::string text =
      "Invalid JSON test "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value>"
      "<arg_key>config</arg_key><arg_value>{invalid json}</arg_value>"
      "</tool_call>";

  FunctionCallParser parser(tools_, "glm47");
  const auto [normal_text, calls] = parser.parse_non_stream(text);

  EXPECT_EQ(normal_text, "Invalid JSON test");
  ASSERT_EQ(calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = calls[0];
  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
  EXPECT_EQ(params["config"], "{invalid json}");  // Should be treated as string
}

// Test nested braces in JSON values
TEST_F(Glm47DetectorTest, NestedBracesInJsonValues) {
  std::string text =
      "Nested braces test "
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</"
      "arg_value>"
      "<arg_key>config</arg_key><arg_value>{\"nested\": {\"deep\": "
      "\"value\"}}</arg_value>"
      "</tool_call>";

  FunctionCallParser parser(tools_, "glm47");
  const auto [normal_text, calls] = parser.parse_non_stream(text);

  EXPECT_EQ(normal_text, "Nested braces test");
  ASSERT_EQ(calls.size(), 1);  // Use ASSERT to stop test if this fails

  const auto& call = calls[0];

  EXPECT_EQ(call.name.value(), "get_weather");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["city"], "Beijing");
  EXPECT_EQ(params["config"]["nested"]["deep"], "value");
}

// Regression test for issue #751: std::regex stack overflow on large payloads
// The old regex implementation with [\s\S]*? pattern caused O(n) recursion
// depth, leading to stack overflow on inputs larger than ~46KB (depending on
// stack size). The fix uses string::find() and substr() for O(1) stack usage.
TEST_F(Glm47DetectorTest, LargePayloadNoStackOverflow) {
  // Create a tool that accepts large content
  nlohmann::json write_params = {
      {"type", "object"},
      {"properties",
       {{"filename", {{"type", "string"}, {"description", "Filename"}}},
        {"content",
         {{"type", "string"}, {"description", "Content to write"}}}}},
      {"required", {"filename", "content"}}};

  JsonFunction write_func("write_file", "Write content to file", write_params);
  JsonTool write_tool("function", write_func);
  std::vector<JsonTool> tools = {write_tool};

  // Test with 50KB payload (larger than the ~46KB that caused stack overflow)
  std::string large_content(50000, 'A');
  std::string text =
      "Test "
      "<tool_call>write_file"
      "<arg_key>filename</arg_key><arg_value>test.txt</arg_value>"
      "<arg_key>content</arg_key><arg_value>" +
      large_content +
      "</arg_value>"
      "</tool_call>";

  // This would crash with stack overflow before the fix
  FunctionCallParser parser(tools, "glm47");
  const auto [normal_text, calls] = parser.parse_non_stream(text);

  EXPECT_EQ(normal_text, "Test");
  ASSERT_EQ(calls.size(), 1);

  const auto& call = calls[0];
  EXPECT_EQ(call.name.value(), "write_file");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["filename"], "test.txt");
  EXPECT_EQ(params["content"], large_content);
}

// Test with Chinese content to validate UTF-8 handling with large payloads
TEST_F(Glm47DetectorTest, LargeChinesePayloadNoStackOverflow) {
  nlohmann::json write_params = {
      {"type", "object"},
      {"properties",
       {{"filename", {{"type", "string"}}}, {"content", {{"type", "string"}}}}},
      {"required", {"filename", "content"}}};

  JsonFunction write_func("write_file", "Write content to file", write_params);
  JsonTool write_tool("function", write_func);
  std::vector<JsonTool> tools = {write_tool};

  // Generate ~50KB of Chinese content (each char is 3 bytes in UTF-8)
  std::string chinese_char = "测";  // 3 bytes in UTF-8
  std::string large_content;
  large_content.reserve(50000);
  for (int i = 0; i < 16667; ++i) {  // 16667 * 3 ≈ 50KB
    large_content += chinese_char;
  }

  std::string text =
      "<tool_call>write_file"
      "<arg_key>filename</arg_key><arg_value>中文.txt</arg_value>"
      "<arg_key>content</arg_key><arg_value>" +
      large_content +
      "</arg_value>"
      "</tool_call>";

  FunctionCallParser parser(tools, "glm47");
  const auto [normal_text, calls] = parser.parse_non_stream(text);

  ASSERT_EQ(calls.size(), 1);
  const auto& call = calls[0];
  EXPECT_EQ(call.name.value(), "write_file");

  nlohmann::json params = nlohmann::json::parse(call.parameters);
  EXPECT_EQ(params["filename"], "中文.txt");
  // Preserve every byte, not just the content length.
  EXPECT_EQ(params["content"], large_content);
}

// Test streaming with large payload
TEST_F(Glm47DetectorTest, StreamingLargePayloadNoStackOverflow) {
  const std::vector<JsonTool> tools = {JsonTool(
      "function",
      JsonFunction("write_file",
                   "Write content",
                   nlohmann::json({{"type", "object"},
                                   {"properties",
                                    {{"content", {{"type", "string"}}}}}})))};
  const std::string large_content(20000, 'X');
  const std::string full_text =
      "<tool_call>write_file<arg_key>content</arg_key>"
      "<arg_value>" +
      large_content + "</arg_value></tool_call>";
  FunctionCallParser parser(tools, "glm47");
  std::string arguments;
  int32_t name_events = 0;
  constexpr size_t kChunkSize = 1000;
  for (size_t offset = 0; offset < full_text.size(); offset += kChunkSize) {
    const StreamingParseResult result =
        parser.parse_streaming_increment(full_text.substr(offset, kChunkSize));
    EXPECT_TRUE(result.normal_text.empty());
    for (const ToolCallItem& call : result.calls) {
      EXPECT_EQ(call.tool_index, 0);
      if (call.name.has_value()) {
        ++name_events;
        EXPECT_EQ(call.name.value(), "write_file");
      }
      arguments += call.parameters;
    }
  }
  EXPECT_EQ(name_events, 1);
  EXPECT_EQ(nlohmann::json::parse(arguments),
            nlohmann::json({{"content", large_content}}));
}

TEST_F(Glm47DetectorTest, StreamingParseWithChineseCharactersSplit) {
  // Three-byte code points, four-byte code points, and surrounding ASCII all
  // cross incremental boundaries. Validate the argument bytes, not call count.
  for (const std::string& city : {"北京", "Tokyo 😊", "City: 北京"}) {
    SCOPED_TRACE(city);
    FunctionCallParser parser(tools_, "glm47");
    const std::string text =
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>" +
        city + "</arg_value></tool_call>";
    std::string arguments;
    int32_t name_events = 0;
    for (char byte : text) {
      const StreamingParseResult result =
          parser.parse_streaming_increment(std::string(1, byte));
      EXPECT_TRUE(result.normal_text.empty());
      for (const ToolCallItem& call : result.calls) {
        EXPECT_EQ(call.tool_index, 0);
        if (call.name.has_value()) {
          ++name_events;
          EXPECT_EQ(call.name.value(), "get_weather");
        }
        arguments += call.parameters;
      }
    }
    EXPECT_EQ(name_events, 1);
    EXPECT_EQ(nlohmann::json::parse(arguments),
              nlohmann::json({{"city", city}}));
  }
}

// Regression: a tool call whose name and closing tag arrive in the same
// increment used to emit the name and defer the arguments to the next
// increment. For the last tool call of a response there is no next increment,
// so the arguments were lost and the client saw an empty argument list.
TEST_F(Glm47DetectorTest, StreamingCompleteToolCallInSingleIncrement) {
  FunctionCallParser parser(tools_, "glm47");
  auto result = parser.parse_streaming_increment(
      "<tool_call>get_weather<arg_key>city</arg_key>"
      "<arg_value>Beijing</arg_value></tool_call>");

  ASSERT_FALSE(result.calls.empty());
  EXPECT_EQ(result.calls[0].name.value_or(""), "get_weather");
  // The arguments have to be in this very result, not in some later one.
  std::string args = collect_streamed_args(result);
  ASSERT_FALSE(args.empty());
  EXPECT_EQ(nlohmann::json::parse(args)["city"], "Beijing");
}

// The production shape: the last tool call closes inside the final increment.
// Its name, arguments and closing tag all arrive together, and nothing follows.
TEST_F(Glm47DetectorTest, StreamingLastToolCallArgsNotLost) {
  FunctionCallParser parser(tools_, "glm47");
  auto first = parser.parse_streaming_increment(
      "<tool_call>get_weather<arg_key>city</arg_key>"
      "<arg_value>Beijing</arg_value></tool_call>"
      "<tool_call>calculate<arg_key>expression</arg_key><arg_value>");
  ASSERT_FALSE(first.calls.empty());
  EXPECT_EQ(first.calls[0].name.value_or(""), "get_weather");
  EXPECT_EQ(nlohmann::json::parse(collect_streamed_args(first))["city"],
            "Beijing");

  // Final increment for the second call: no increment follows this one.
  auto second = parser.parse_streaming_increment("1+1</arg_value></tool_call>");

  std::string args = collect_streamed_args(second);
  ASSERT_FALSE(args.empty());
  EXPECT_EQ(nlohmann::json::parse(args)["expression"], "1+1");
}

}  // namespace function_call
}  // namespace xllm
