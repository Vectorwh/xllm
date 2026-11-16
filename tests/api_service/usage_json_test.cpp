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

#include <brpc/channel.h>
#include <brpc/server.h>
#include <gtest/gtest.h>
#include <json2pb/pb_to_json.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "api_service/stream_call.h"
#include "api_service/utils.h"
#include "chat.pb.h"
#include "completion.pb.h"
#include "xllm_service.pb.h"

namespace xllm {
namespace {

std::vector<JsonTool> make_test_tools() {
  return {JsonTool("function",
                   JsonFunction("get_weather",
                                "Get the weather",
                                nlohmann::json{{"type", "object"}}))};
}

class EmptyChoicesStreamService final : public proto::XllmAPIService {
 public:
  void CompletionsHttp(google::protobuf::RpcController* controller,
                       const proto::HttpRequest* /*request*/,
                       proto::HttpResponse* /*response*/,
                       google::protobuf::Closure* done) override {
    write_empty_choices<proto::CompletionRequest, proto::CompletionResponse>(
        static_cast<brpc::Controller*>(controller), done, "cmpl_usage");
  }

  void ChatCompletionsHttp(google::protobuf::RpcController* controller,
                           const proto::HttpRequest* /*request*/,
                           proto::HttpResponse* /*response*/,
                           google::protobuf::Closure* done) override {
    write_empty_choices<proto::ChatRequest, proto::ChatResponse>(
        static_cast<brpc::Controller*>(controller), done, "chatcmpl_usage");
  }

 private:
  template <typename Request, typename Response>
  static void write_empty_choices(brpc::Controller* controller,
                                  google::protobuf::Closure* done,
                                  const std::string& id) {
    Request request;
    request.set_stream(true);
    Response response;
    response.set_id(id);
    Usage usage;
    usage.num_prompt_tokens = 11;
    usage.num_generated_tokens = 2;
    usage.num_total_tokens = 13;
    usage.num_cached_tokens = 5;
    api_service::set_proto_usage(response.mutable_usage(), usage);
    StreamCall<Request, Response> call(controller,
                                       done,
                                       &request,
                                       &response,
                                       /*use_arena=*/true,
                                       /*is_http_request=*/true);
    EXPECT_TRUE(call.write(response, /*empty_array_field=*/"choices"));
    EXPECT_TRUE(call.finish());
  }
};

class EmptyChoicesStreamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(server_.AddService(&service_,
                                 brpc::SERVER_DOESNT_OWN_SERVICE,
                                 "v1/completions => CompletionsHttp, "
                                 "v1/chat/completions => ChatCompletionsHttp"),
              0);
    ASSERT_EQ(server_.Start("127.0.0.1:0", nullptr), 0);

    brpc::ChannelOptions options;
    options.protocol = brpc::PROTOCOL_HTTP;
    options.timeout_ms = 5000;
    options.max_retry = 0;
    const std::string address =
        "127.0.0.1:" + std::to_string(server_.listen_address().port);
    ASSERT_EQ(channel_.Init(address.c_str(), &options), 0);
  }

  void TearDown() override {
    server_.Stop(0);
    server_.Join();
  }

  void expect_empty_choices_event(const std::string& path,
                                  const std::string& expected_id) {
    brpc::Controller controller;
    controller.http_request().uri() = path;
    controller.http_request().set_method(brpc::HTTP_METHOD_POST);
    channel_.CallMethod(nullptr, &controller, nullptr, nullptr, nullptr);

    ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
    const std::string body = controller.response_attachment().to_string();
    const size_t event_end = body.find("\n\n");
    ASSERT_NE(event_end, std::string::npos);
    const std::string event = body.substr(0, event_end + 2);
    ASSERT_GE(event.size(), 8);
    EXPECT_EQ(event.substr(0, 6), "data: ");
    EXPECT_EQ(event.substr(event.size() - 2), "\n\n");

    const nlohmann::json payload =
        nlohmann::json::parse(event.substr(6, event.size() - 8));
    EXPECT_EQ(payload["id"], expected_id);
    ASSERT_TRUE(payload.contains("usage"));
    ASSERT_TRUE(payload["usage"].is_object());
    EXPECT_EQ(payload["usage"]["prompt_tokens"], 11);
    EXPECT_EQ(payload["usage"]["completion_tokens"], 2);
    EXPECT_EQ(payload["usage"]["total_tokens"], 13);
    ASSERT_TRUE(payload["usage"].contains("prompt_tokens_details"));
    EXPECT_EQ(payload["usage"]["prompt_tokens_details"]["cached_tokens"], 5);
    ASSERT_TRUE(payload.contains("choices"));
    ASSERT_TRUE(payload["choices"].is_array());
    EXPECT_TRUE(payload["choices"].empty());
    EXPECT_EQ(body.substr(event_end + 2), "data: [DONE]\n\n");
  }

 private:
  EmptyChoicesStreamService service_;
  brpc::Server server_;
  brpc::Channel channel_;
};

TEST(UsageJsonTest, ChatUsageSerializesOpenAICachedTokensField) {
  Usage usage;
  usage.num_prompt_tokens = 1024;
  usage.num_generated_tokens = 50;
  usage.num_total_tokens = 1074;
  usage.num_cached_tokens = 896;

  proto::ChatResponse response;
  api_service::set_proto_usage(response.mutable_usage(), usage);

  json2pb::Pb2JsonOptions options;
  options.bytes_to_base64 = false;
  options.jsonify_empty_array = true;
  options.always_print_primitive_fields = true;

  std::string json_text;
  std::string error_message;
  ASSERT_TRUE(json2pb::ProtoMessageToJson(
      response, &json_text, options, &error_message))
      << error_message;

  nlohmann::json json = nlohmann::json::parse(json_text);
  ASSERT_TRUE(json.contains("usage"));
  EXPECT_EQ(json["usage"]["prompt_tokens"], 1024);
  EXPECT_EQ(json["usage"]["completion_tokens"], 50);
  EXPECT_EQ(json["usage"]["total_tokens"], 1074);
  ASSERT_TRUE(json["usage"].contains("prompt_tokens_details"));
  EXPECT_EQ(json["usage"]["prompt_tokens_details"]["cached_tokens"], 896);
  EXPECT_EQ(json["usage"]["prompt_tokens_details"]["audio_tokens"], 0);
  ASSERT_TRUE(json["usage"].contains("completion_tokens_details"));
  EXPECT_EQ(json["usage"]["completion_tokens_details"]["reasoning_tokens"], 0);
  EXPECT_EQ(json["usage"]["completion_tokens_details"]["audio_tokens"], 0);
  EXPECT_EQ(json["usage"]["completion_tokens_details"].size(), 2);
}

TEST(UsageJsonTest, ChatUsagePrintsZeroCachedTokens) {
  Usage usage;
  usage.num_prompt_tokens = 12;
  usage.num_generated_tokens = 3;
  usage.num_total_tokens = 15;
  usage.num_cached_tokens = 0;

  proto::ChatResponse response;
  api_service::set_proto_usage(response.mutable_usage(), usage);

  json2pb::Pb2JsonOptions options;
  options.bytes_to_base64 = false;
  options.jsonify_empty_array = true;
  options.always_print_primitive_fields = true;

  std::string json_text;
  std::string error_message;
  ASSERT_TRUE(json2pb::ProtoMessageToJson(
      response, &json_text, options, &error_message))
      << error_message;

  nlohmann::json json = nlohmann::json::parse(json_text);
  ASSERT_TRUE(json["usage"].contains("prompt_tokens_details"));
  EXPECT_EQ(json["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
  EXPECT_EQ(json["usage"]["prompt_tokens_details"]["audio_tokens"], 0);
  ASSERT_TRUE(json["usage"].contains("completion_tokens_details"));
  EXPECT_EQ(json["usage"]["completion_tokens_details"]["reasoning_tokens"], 0);
  EXPECT_EQ(json["usage"]["completion_tokens_details"]["audio_tokens"], 0);
  EXPECT_EQ(json["usage"]["completion_tokens_details"].size(), 2);
}

TEST_F(EmptyChoicesStreamTest, CompletionWriteEmitsEmptyChoicesSseEvent) {
  expect_empty_choices_event("/v1/completions", "cmpl_usage");
}

TEST_F(EmptyChoicesStreamTest, ChatWriteEmitsEmptyChoicesSseEvent) {
  expect_empty_choices_event("/v1/chat/completions", "chatcmpl_usage");
}

TEST(ToolCallResultTest, KeepsNormalTextWhenNoToolCallMarkerExists) {
  const std::string text = "The weather is sunny.";

  const auto result =
      api_service::process_tool_calls(text, make_test_tools(), "glm45", "stop");

  EXPECT_FALSE(result.tool_calls.has_value());
  EXPECT_EQ(result.text, text);
  EXPECT_EQ(result.finish_reason, "stop");
}

TEST(ToolCallResultTest, KeepsNormalTextWhenMarkerDoesNotParse) {
  const std::string text =
      "The weather is sunny. <tool_call>unknown_tool\n"
      "<arg_key>location</arg_key>\n"
      "<arg_value>Beijing</arg_value>\n"
      "</tool_call>";

  const auto result =
      api_service::process_tool_calls(text, make_test_tools(), "glm45", "stop");

  EXPECT_FALSE(result.tool_calls.has_value());
  EXPECT_EQ(result.text, text);
  EXPECT_EQ(result.finish_reason, "stop");
}

TEST(ToolCallResultTest, PreservesParsedToolCall) {
  const auto result = api_service::process_tool_calls(
      "I will check. <tool_call>get_weather\n"
      "<arg_key>location</arg_key>\n"
      "<arg_value>Beijing</arg_value>\n"
      "</tool_call>",
      make_test_tools(),
      "glm45",
      "stop");

  ASSERT_TRUE(result.tool_calls.has_value());
  ASSERT_EQ(result.tool_calls->size(), 1);
  EXPECT_EQ(result.tool_calls->at(0).function().name(), "get_weather");
  EXPECT_EQ(result.finish_reason, "tool_calls");
}

TEST(ChatResponseJsonTest, OmitsEmptyToolCallsForNormalText) {
  auto* request = new proto::ChatRequest();
  request->set_stream(false);
  auto* response = new proto::ChatResponse();
  auto* choice = response->add_choices();
  auto* message = choice->mutable_message();
  message->set_role("assistant");
  message->set_content("The weather is sunny.");

  brpc::Controller controller;
  {
    StreamCall<proto::ChatRequest, proto::ChatResponse> call(
        &controller, brpc::DoNothing(), request, response);
    ASSERT_TRUE(call.write_and_finish(*response));
  }

  const nlohmann::json json =
      nlohmann::json::parse(controller.response_attachment().to_string());
  ASSERT_TRUE(json["choices"][0].contains("message"));
  EXPECT_FALSE(json["choices"][0]["message"].contains("tool_calls"));
}

TEST(ChatResponseJsonTest, KeepsNonEmptyToolCalls) {
  proto::ChatResponse response;
  auto* choice = response.add_choices();
  auto* message = choice->mutable_message();
  message->set_role("assistant");
  auto* tool_call = message->add_tool_calls();
  tool_call->set_id("call_test");
  tool_call->set_type("function");
  tool_call->mutable_function()->set_name("get_weather");
  tool_call->mutable_function()->set_arguments("{\"location\":\"Beijing\"}");

  json2pb::Pb2JsonOptions options;
  options.bytes_to_base64 = false;
  options.jsonify_empty_array = false;

  std::string json_text;
  std::string error_message;
  ASSERT_TRUE(json2pb::ProtoMessageToJson(
      response, &json_text, options, &error_message))
      << error_message;

  const nlohmann::json json = nlohmann::json::parse(json_text);
  const auto& tool_calls = json["choices"][0]["message"]["tool_calls"];
  ASSERT_TRUE(tool_calls.is_array());
  ASSERT_EQ(tool_calls.size(), 1);
  EXPECT_EQ(tool_calls[0]["function"]["name"], "get_weather");
}

TEST(AnthropicCallJsonTest, KeepsEmptyContentArray) {
  auto* request = new proto::AnthropicMessagesRequest();
  request->set_stream(false);
  auto* response = new proto::AnthropicMessagesResponse();

  brpc::Controller controller;
  {
    AnthropicCall call(&controller, brpc::DoNothing(), request, response);
    ASSERT_TRUE(call.write_and_finish(*response));
  }

  const nlohmann::json json =
      nlohmann::json::parse(controller.response_attachment().to_string());
  ASSERT_TRUE(json.contains("content"));
  EXPECT_TRUE(json["content"].is_array());
  EXPECT_TRUE(json["content"].empty());
}

}  // namespace
}  // namespace xllm
