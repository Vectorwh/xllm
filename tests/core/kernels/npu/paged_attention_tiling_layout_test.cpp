/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/kernels/npu/paged_attention_tiling_layout.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace xllm::kernel::npu {
namespace {

TEST(PagedAttentionTilingLayoutTest, RejectsUnknownOrTruncatedAtbLayouts) {
  EXPECT_FALSE(parse_custom_paged_attention_tiling_layout({}).has_value());
  EXPECT_FALSE(
      parse_custom_paged_attention_tiling_layout(std::vector<uint32_t>(18, 0))
          .has_value());

  std::vector<uint32_t> words(128, 0);
  words[16] = 9;
  words[17] = 44;
  words[18] = 17;
  const PagedAttentionTilingLayout expected{/*buffer_words=*/128,
                                            /*header_words=*/44,
                                            /*row_stride_words=*/17,
                                            /*max_kv_seq_len_offset=*/22,
                                            /*kv_split_length_offset=*/23,
                                            /*kv_split_core_count_offset=*/24,
                                            /*row_kv_seq_len_offset=*/45,
                                            /*tiling_key=*/9};
  const auto layout = parse_custom_paged_attention_tiling_layout(words);
  ASSERT_TRUE(layout.has_value());
  EXPECT_EQ(*layout, expected);
  EXPECT_TRUE(paged_attention_tiling_required_words(*layout, /*num_rows=*/4)
                  .has_value());
  EXPECT_FALSE(paged_attention_tiling_required_words(*layout, /*num_rows=*/6)
                   .has_value());

  words[17] = 50;
  EXPECT_FALSE(parse_custom_paged_attention_tiling_layout(words).has_value());
  words[17] = 44;
  words[18] = 19;
  EXPECT_FALSE(parse_custom_paged_attention_tiling_layout(words).has_value());
}

}  // namespace
}  // namespace xllm::kernel::npu
