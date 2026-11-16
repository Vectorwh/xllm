/* Copyright 2025-2026 The xLLM Authors.
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

#include "framework/tokenizer/fast_tokenizer.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "framework/tokenizer/tokenizer_args.h"
#include "framework/tokenizer/tokenizer_proxy.h"

namespace xllm {

std::string CreateTestTokenizerJson(const std::string& filepath,
                                    const std::string& truncation_json,
                                    const std::string& padding_json);

// Helper function to create a minimal valid tokenizer.json file for testing
// This creates a simple BPE tokenizer with a small vocabulary
std::string CreateTestTokenizerJson(const std::string& filepath) {
  return CreateTestTokenizerJson(filepath, "null", "null");
}

std::string CreateByteLevelTokenizerJson(const std::string& filepath) {
  const std::string tokenizer_json = R"({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true,
    "use_regex": false
  },
  "post_processor": null,
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": true,
    "trim_offsets": true,
    "use_regex": true
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "vocab": {
      "{": 0,
      "}": 1,
      "\"": 2,
      "\u0120": 3,
      "\u00e4\u00b8\u0122": 4,
      "\u00e5\u012e\u0139": 5,
      "city": 6,
      "\u00c3": 7,
      "\u00a9": 8
    },
    "merges": []
  }
})";

  std::ofstream file(filepath);
  if (!file.is_open()) {
    return "";
  }
  file << tokenizer_json;
  file.close();
  return filepath;
}

std::string CreateNonByteLevelTokenizerJson(const std::string& filepath) {
  CreateTestTokenizerJson(filepath);
  std::ifstream input(filepath);
  nlohmann::json tokenizer_json = nlohmann::json::parse(input);
  tokenizer_json["decoder"] = {
      {"type", "WordPiece"}, {"prefix", "##"}, {"cleanup", true}};
  tokenizer_json["model"]["vocab"]["\u00a1"] = 15;

  std::ofstream output(filepath);
  output << tokenizer_json.dump();
  return filepath;
}

std::string CreateTestTokenizerJson(const std::string& filepath,
                                    const std::string& truncation_json,
                                    const std::string& padding_json) {
  // Minimal valid tokenizer.json for testing
  // This is a simplified BPE tokenizer configuration compatible with
  // HuggingFace tokenizers
  const std::string tokenizer_json = R"({
  "version": "1.0",
  "truncation": __TRUNCATION__,
  "padding": __PADDING__,
  "added_tokens": [
    {"id": 0, "content": "<|bos|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<|eos|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "hello", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
    {"id": 3, "content": "world", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
    {"id": 4, "content": "test", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
  ],
  "normalizer": {
    "type": "NFC"
  },
  "pre_tokenizer": {
    "type": "Whitespace"
  },
  "post_processor": null,
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true,
    "use_regex": true
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "vocab": {
      "<|bos|>": 0,
      "<|eos|>": 1,
      "hello": 2,
      "world": 3,
      "test": 4,
      "h": 5,
      "e": 6,
      "l": 7,
      "o": 8,
      "w": 9,
      "r": 10,
      "d": 11,
      "t": 12,
      "s": 13,
      " ": 14
    },
    "merges": []
  }
})";
  std::string rendered = tokenizer_json;
  size_t pos = rendered.find("__TRUNCATION__");
  rendered.replace(pos, std::string("__TRUNCATION__").size(), truncation_json);
  pos = rendered.find("__PADDING__");
  rendered.replace(pos, std::string("__PADDING__").size(), padding_json);

  std::ofstream file(filepath);
  if (!file.is_open()) {
    return "";
  }
  file << rendered;
  file.close();
  return filepath;
}

class FastTokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a temporary directory for test files
    test_dir_ = std::filesystem::temp_directory_path() /
                ("fast_tokenizer_test_" + std::to_string(getpid()));
    std::filesystem::create_directories(test_dir_);

    // Create test tokenizer.json file
    tokenizer_json_path_ = test_dir_ / "tokenizer.json";
    CreateTestTokenizerJson(tokenizer_json_path_.string());
  }

  void TearDown() override {
    // Clean up test files
    if (std::filesystem::exists(test_dir_)) {
      std::filesystem::remove_all(test_dir_);
    }
  }

  std::filesystem::path test_dir_;
  std::filesystem::path tokenizer_json_path_;
};

// Test that BOS token is added when add_bos_token is true
TEST_F(FastTokenizerTest, AddBosToken) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = true;
  args.bos_token() = "<|bos|>";
  args.add_eos_token() = false;

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  ASSERT_TRUE(tokenizer.encode("hello world", &ids));
  EXPECT_EQ(ids, (std::vector<int32_t>{0, 2, 3}));
}

// Test that EOS token is added when add_eos_token is true
TEST_F(FastTokenizerTest, AddEosToken) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = false;
  args.add_eos_token() = true;
  args.eos_token() = "<|eos|>";

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  ASSERT_TRUE(tokenizer.encode("hello world", &ids));
  EXPECT_EQ(ids, (std::vector<int32_t>{2, 3, 1}));
}

// Test that both BOS and EOS tokens are added when both flags are true
TEST_F(FastTokenizerTest, AddBothBosAndEosTokens) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = true;
  args.bos_token() = "<|bos|>";
  args.add_eos_token() = true;
  args.eos_token() = "<|eos|>";

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  ASSERT_TRUE(tokenizer.encode("hello world", &ids));
  const std::vector<int32_t> expected = {0, 2, 3, 1};
  EXPECT_EQ(ids, expected);

  ASSERT_TRUE(tokenizer.encode("<|bos|> hello world <|eos|>", &ids));
  EXPECT_EQ(ids, expected);
}

// Test that no special tokens are added when both flags are false
TEST_F(FastTokenizerTest, NoSpecialTokens) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = false;
  args.add_eos_token() = false;

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  ASSERT_TRUE(tokenizer.encode("hello world", &ids));
  EXPECT_EQ(ids, (std::vector<int32_t>{2, 3}));
}

TEST_F(FastTokenizerTest, CallLevelFalseSuppressesConfiguredSpecialTokens) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = true;
  args.bos_token() = "<|bos|>";
  args.add_eos_token() = true;
  args.eos_token() = "<|eos|>";

  FastTokenizer tokenizer(args);
  std::vector<int32_t> ids_without_special_tokens;
  std::vector<int32_t> ids_with_special_tokens;
  ASSERT_TRUE(tokenizer.encode("hello",
                               &ids_without_special_tokens,
                               /*add_special_tokens=*/false));
  ASSERT_TRUE(tokenizer.encode("hello",
                               &ids_with_special_tokens,
                               /*add_special_tokens=*/true));

  EXPECT_EQ(ids_without_special_tokens, (std::vector<int32_t>{2}));
  EXPECT_EQ(ids_with_special_tokens, (std::vector<int32_t>{0, 2, 1}));
}

TEST_F(FastTokenizerTest, DecodeTokenPreservesByteLevelPieces) {
  const std::filesystem::path byte_level_path =
      test_dir_ / "byte_level_tokenizer.json";
  CreateByteLevelTokenizerJson(byte_level_path.string());

  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = byte_level_path.string();
  FastTokenizer tokenizer(args);

  EXPECT_EQ(tokenizer.decode_token(0), "{");
  EXPECT_EQ(tokenizer.decode_token(2), "\"");
  EXPECT_EQ(tokenizer.decode_token(3), " ");
  EXPECT_EQ(tokenizer.decode_token(4), "\xE4\xB8\x80");
  EXPECT_EQ(tokenizer.decode_token(5), "\xE5\x8C\x97");
  EXPECT_EQ(tokenizer.decode_token(7), "\xC3");
  EXPECT_EQ(tokenizer.decode_token(8), "\xA9");
  std::vector<std::string> pieces;
  ASSERT_TRUE(tokenizer.get_constraint_vocab(&pieces));
  ASSERT_EQ(pieces.size(), tokenizer.vocab_size());
  EXPECT_EQ(pieces[0], "{");
  EXPECT_EQ(pieces[1], "}");
  EXPECT_EQ(pieces[3], " ");
  EXPECT_EQ(pieces[7], "\xC3");
  EXPECT_EQ(pieces[8], "\xA9");

  const std::vector<int32_t> split_utf8_ids = {7, 8};
  EXPECT_EQ(tokenizer.decode(split_utf8_ids, /*skip_special_tokens=*/false),
            "\xC3\xA9");
}

TEST_F(FastTokenizerTest, DecodeTokenKeepsNonByteLevelPieces) {
  const std::filesystem::path non_byte_level_path =
      test_dir_ / "non_byte_level_tokenizer.json";
  CreateNonByteLevelTokenizerJson(non_byte_level_path.string());

  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = non_byte_level_path.string();
  FastTokenizer tokenizer(args);

  EXPECT_EQ(tokenizer.decode_token(15), "\xC2\xA1");
  std::vector<std::string> pieces;
  EXPECT_FALSE(tokenizer.get_constraint_vocab(&pieces));
}

TEST_F(FastTokenizerTest, ProxyForwardsDecodeToken) {
  const std::filesystem::path byte_level_path =
      test_dir_ / "byte_level_proxy_tokenizer.json";
  CreateByteLevelTokenizerJson(byte_level_path.string());

  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = byte_level_path.string();
  std::unique_ptr<Tokenizer> tokenizer = std::make_unique<FastTokenizer>(args);
  TokenizerProxy proxy(std::move(tokenizer));

  EXPECT_EQ(proxy.decode_token(3), " ");
  EXPECT_EQ(proxy.decode_token(4), "\xE4\xB8\x80");
}

TEST_F(FastTokenizerTest, EmptyConfiguredSpecialTokenIsSkippedIndependently) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = true;
  args.bos_token() = "";
  args.add_eos_token() = true;
  args.eos_token() = "<|eos|>";

  FastTokenizer missing_bos(args);
  std::vector<int32_t> ids;
  ASSERT_TRUE(missing_bos.encode("hello world", &ids));
  EXPECT_EQ(ids, (std::vector<int32_t>{2, 3, 1}));

  args.bos_token() = "<|bos|>";
  args.eos_token() = "";
  FastTokenizer missing_eos(args);
  ASSERT_TRUE(missing_eos.encode("hello world", &ids));
  EXPECT_EQ(ids, (std::vector<int32_t>{0, 2, 3}));
}

TEST_F(FastTokenizerTest, IgnoreTokenizerJsonPaddingAndTruncation) {
  std::filesystem::path padded_tokenizer_path =
      test_dir_ / "tokenizer_with_padding.json";
  ASSERT_FALSE(
      CreateTestTokenizerJson(
          padded_tokenizer_path.string(),
          R"({"direction":"Right","max_length":4,"strategy":"LongestFirst","stride":0})",
          R"({"strategy":{"Fixed":4},"direction":"Right","pad_to_multiple_of":null,"pad_id":0,"pad_type_id":0,"pad_token":"<|bos|>"})")
          .empty());

  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = padded_tokenizer_path.string();

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  ASSERT_TRUE(tokenizer.encode("hello", &ids));
  EXPECT_EQ(ids, (std::vector<int32_t>{2}));

  ASSERT_TRUE(tokenizer.encode("hello world test hello world", &ids));
  EXPECT_EQ(ids, (std::vector<int32_t>{2, 3, 4, 2, 3}))
      << "FastTokenizer should ignore tokenizer.json padding/truncation.";
}

}  // namespace xllm
