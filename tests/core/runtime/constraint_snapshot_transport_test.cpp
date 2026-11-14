/* Copyright 2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "core/framework/config/execution_config.h"
#include "core/runtime/forward_shared_memory_manager.h"
#include "core/scheduler/disagg_pd_scheduler.h"

namespace xllm {
namespace {

class ConstraintSnapshotTransportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto& config = ExecutionConfig::get_instance();
    previous_contiguous_ = config.use_contiguous_input_buffer();
    // Exercise the real host SHM descriptor codec without device upload or
    // process-global NPU initialization; device materialization is separate.
    config.use_contiguous_input_buffer() = false;
  }
  void TearDown() override {
    ExecutionConfig::get_instance().use_contiguous_input_buffer() =
        previous_contiguous_;
  }

 private:
  bool previous_contiguous_ = false;
};

TEST_F(ConstraintSnapshotTransportTest,
       ForwardInputPackedRoundTripKeepsGrammarSpec) {
  // G9: the per-request grammar identity rides the packed input transport
  // (v8 descriptor), so a cross-process receiver can rebuild the grammar.
  ForwardInput input;
  input.json_object_state_snapshots.resize(1);
  auto& snapshot = input.json_object_state_snapshots[0];
  snapshot.enabled = true;
  snapshot.token_ids = {0, 1};
  GrammarSpec spec;
  spec.type = GrammarSpec::Type::JSON_SCHEMA;
  spec.source = R"({"type":"object"})";
  spec.strict = true;
  snapshot.grammar_spec = std::make_shared<const GrammarSpec>(std::move(spec));

  bool is_creator = false;
  auto shm_name =
      ForwardSharedMemoryManager::create_unique_name("batch_test_grammar_spec",
                                                     /*dp_group=*/0,
                                                     ForwardType::RAW_INPUT,
                                                     /*rank=*/0);
  ForwardSharedMemoryManager writer_manager(
      shm_name, 1 << 20, is_creator, ForwardType::RAW_INPUT);
  bool is_reader_creator = false;
  ForwardSharedMemoryManager reader_manager(
      shm_name, 1 << 20, is_reader_creator, ForwardType::RAW_INPUT);
  ASSERT_TRUE(writer_manager.input_write(input));

  ForwardInput round_trip;
  reader_manager.input_read(round_trip, torch::Device(torch::kCPU));
  ASSERT_EQ(round_trip.json_object_state_snapshots.size(), 1u);
  const auto& out = round_trip.json_object_state_snapshots[0];
  EXPECT_TRUE(out.enabled);
  EXPECT_EQ(out.token_ids, std::vector<int32_t>({0, 1}));
  ASSERT_NE(out.grammar_spec, nullptr);
  EXPECT_EQ(out.grammar_spec->type, GrammarSpec::Type::JSON_SCHEMA);
  EXPECT_TRUE(out.grammar_spec->strict);
  EXPECT_EQ(out.grammar_spec->source, R"({"type":"object"})");

  // Reuse the receive object for a builtin snapshot (empty source). resize
  // keeps the old entry alive, so the reader must explicitly clear its text.
  snapshot.grammar_spec = nullptr;
  snapshot.token_ids.clear();
  ASSERT_TRUE(writer_manager.input_write(input));
  ASSERT_TRUE(
      reader_manager.input_read(round_trip, torch::Device(torch::kCPU)));
  ASSERT_EQ(round_trip.json_object_state_snapshots.size(), 1u);
  EXPECT_EQ(round_trip.json_object_state_snapshots[0].grammar_spec, nullptr);
}

TEST_F(ConstraintSnapshotTransportTest,
       PackedGrammarSourceCannotReadPastDescriptor) {
  ForwardInput input;
  input.json_object_state_snapshots.resize(1);
  auto& snapshot = input.json_object_state_snapshots[0];
  snapshot.enabled = true;
  GrammarSpec spec;
  spec.type = GrammarSpec::Type::REGEX;
  spec.source = "g9-descriptor-boundary-sentinel";
  snapshot.grammar_spec = std::make_shared<const GrammarSpec>(std::move(spec));
  bool is_creator = false;
  const std::string name = ForwardSharedMemoryManager::create_unique_name(
      "batch_test_bad_grammar_length",
      /*dp_group=*/0,
      ForwardType::RAW_INPUT,
      /*rank=*/0);
  ForwardSharedMemoryManager writer(
      name, 1 << 20, is_creator, ForwardType::RAW_INPUT);
  bool is_reader_creator = false;
  ForwardSharedMemoryManager reader(
      name, 1 << 20, is_reader_creator, ForwardType::RAW_INPUT);
  ASSERT_TRUE(writer.input_write(input));
  char* base = static_cast<char*>(writer.base_address());
  const uint64_t bytes = ForwardSharedMemoryManager::input_size(input);
  const std::string& marker = snapshot.grammar_spec->source;
  char* source = std::search(base, base + bytes, marker.begin(), marker.end());
  ASSERT_NE(source, base + bytes);
  ASSERT_GE(source - base, static_cast<std::ptrdiff_t>(sizeof(uint64_t)));
  // Corrupt only the source's length word: it passes the 16-MiB application
  // cap but exceeds this actual descriptor, and must fail before copying.
  constexpr uint64_t kBadSourceLength = 1ULL << 20;
  std::memcpy(source - sizeof(uint64_t), &kBadSourceLength, sizeof(uint64_t));
  EXPECT_DEATH(
      {
        ForwardInput output;
        reader.input_read(output, torch::Device(torch::kCPU));
      },
      "grammar source exceeds descriptor");
}

TEST_F(ConstraintSnapshotTransportTest,
       PermanentConstraintRejectionIsTerminal) {
  EXPECT_TRUE(is_permanent_rejection(kDecodeConstraintRejectedStatusCode));
  EXPECT_TRUE(is_permanent_rejection(kDecodeAddNewPromptTooLongStatusCode));
  EXPECT_FALSE(is_permanent_rejection(404));
  EXPECT_FALSE(is_permanent_rejection(500));
}

}  // namespace
}  // namespace xllm
