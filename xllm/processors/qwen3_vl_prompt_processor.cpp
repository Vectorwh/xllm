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

#include "processors/qwen3_vl_prompt_processor.h"

#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <optional>

namespace xllm {

namespace {

struct VisionGrid {
  int32_t frames;
  int32_t tokens_per_frame;
};

std::optional<VisionGrid> read_vision_grid(const MMDataItem& item,
                                           int32_t merge_size) {
  const bool image = item.is_type(MMType::IMAGE);
  const auto it = item.data().find(image ? "image_grid_thw" : "video_grid_thw");
  if (it == item.data().end() || merge_size <= 0 ||
      item.has(image ? "video_grid_thw" : "image_grid_thw")) {
    return std::nullopt;
  }
  const auto* grid = std::get_if<torch::Tensor>(&it->second);
  if (grid == nullptr || !grid->defined() || grid->dim() != 2 ||
      grid->size(0) != 1 || grid->size(1) != 3 ||
      (grid->scalar_type() != torch::kInt32 &&
       grid->scalar_type() != torch::kInt64)) {
    return std::nullopt;
  }
  const auto row = (*grid)[0];
  const int64_t t = row[0].item<int64_t>();
  const int64_t h = row[1].item<int64_t>();
  const int64_t w = row[2].item<int64_t>();
  constexpr int64_t kMaxGridSize = std::numeric_limits<int32_t>::max();
  if (t <= 0 || h <= 0 || w <= 0 || h > kMaxGridSize || w > kMaxGridSize ||
      h % merge_size != 0 || w % merge_size != 0 ||
      kMaxGridSize / t / (h / merge_size) < w / merge_size) {
    return std::nullopt;
  }
  if (item.is_embedded()) {
    const auto& value = item.data().at(get_embedding_key(item.type()));
    const auto* embedding = std::get_if<torch::Tensor>(&value);
    if (embedding == nullptr || !embedding->defined() || embedding->dim() < 1 ||
        embedding->size(0) != t * (h / merge_size) * (w / merge_size)) {
      return std::nullopt;
    }
  }
  return VisionGrid{static_cast<int32_t>(t),
                    static_cast<int32_t>((h / merge_size) * (w / merge_size))};
}

}  // namespace

Qwen3VLPromptProcessor::Qwen3VLPromptProcessor(const ModelArgs& args) {
  max_context_len_ = args.max_position_embeddings();
  merge_size_ = args.mm_spatial_merge_size() > 0 ? args.mm_spatial_merge_size()
                                                 : args.mm_image_merge_size();
  vision_start_token_id_ = args.vision_start_token_id();
  vision_end_token_id_ = args.vision_end_token_id();
  image_token_id_ = args.image_token_id();
  video_token_id_ = args.video_token_id();
  temporal_patch_size_ = args.mm_temporal_patch_size();
}

bool Qwen3VLPromptProcessor::process(std::string& prompt,
                                     const MMData& mm_data) {
  if (!mm_data.valid()) {
    return (vision_start_token_id_ <= 0 ||
            prompt.find(vision_start_token_) == std::string::npos) &&
           (vision_end_token_id_ <= 0 ||
            prompt.find(vision_end_token_) == std::string::npos) &&
           (image_token_id_ <= 0 ||
            prompt.find(image_token_) == std::string::npos) &&
           (video_token_id_ <= 0 ||
            prompt.find(video_token_) == std::string::npos);
  }
  if (!mm_data.hold<MMItemVec>() || mm_data.empty()) {
    return false;
  }
  const auto& items = mm_data.items<MMItemVec>();
  int64_t remaining = max_context_len_;
  std::vector<VisionGrid> grids;
  grids.reserve(items.size());
  for (const auto& item : items) {
    if (!item.is_type(MMType::IMAGE) && !item.is_type(MMType::VIDEO)) {
      return false;
    }
    const auto grid = read_vision_grid(item, merge_size_);
    if (!grid) {
      return false;
    }
    const int64_t visual_tokens =
        static_cast<int64_t>(grid->frames) * grid->tokens_per_frame;
    if (visual_tokens >= remaining) {
      return false;
    }
    remaining -= visual_tokens;
    grids.emplace_back(*grid);
  }

  size_t item_index = 0;
  std::string data;
  data.reserve(prompt.size());
  size_t begin = 0;
  auto pair = find_vision_token(prompt, begin);

  while (pair.second != std::string::npos) {
    if (item_index == items.size()) {
      return false;
    }
    const auto& grid = grids[item_index];
    const auto& item = items[item_index++];
    const bool is_image = pair.first == TokenType::IMAGE;
    if (!item.is_type(is_image ? MMType::IMAGE : MMType::VIDEO)) {
      return false;
    }
    if (is_image) {
      data.append(prompt, begin, pair.second - begin);

      int32_t token_num = grid.frames * grid.tokens_per_frame;
      while (token_num--) {
        data.append(image_token_);
      }

      begin = pair.second + image_token_.size();

    } else {
      const size_t pos = pair.second;
      const size_t vs_len = vision_start_token_.size();
      const size_t ve_len = vision_end_token_.size();
      const size_t vt_len = video_token_.size();

      size_t replace_begin = pos;
      size_t replace_end = pos + vt_len;

      if (pos >= vs_len &&
          prompt.compare(pos - vs_len, vs_len, vision_start_token_) == 0 &&
          prompt.compare(pos + vt_len, ve_len, vision_end_token_) == 0) {
        replace_begin = pos - vs_len;
        replace_end = pos + vt_len + ve_len;
      }

      data.append(prompt, begin, replace_begin - begin);

      const auto metadata = item.get_metadata<VideoMetadata>();
      if (!metadata || metadata->timestamps.empty() ||
          temporal_patch_size_ <= 0) {
        return false;
      }
      const int32_t num_frames = grid.frames;
      const auto selected = build_timestamps(
          metadata->timestamps, num_frames, temporal_patch_size_);
      for (int32_t idx = 0; idx < num_frames; ++idx) {
        data.append(format_timestamp_str(selected[idx]));
        data.append(vision_start_token_);
        int32_t num = grid.tokens_per_frame;

        while (num--) {
          data.append(video_token_);
        }
        data.append(vision_end_token_);
      }

      begin = replace_end;
    }

    pair = find_vision_token(prompt, begin);
  }

  if (item_index != items.size()) {
    return false;
  }
  if (begin < prompt.size()) data.append(prompt, begin, std::string::npos);
  prompt = std::move(data);
  return true;
}

bool Qwen3VLPromptProcessor::validate_text_tokens(
    const std::vector<int32_t>& token_ids) const {
  return std::none_of(token_ids.begin(), token_ids.end(), [this](int32_t id) {
    return id > 0 &&
           (id == vision_start_token_id_ || id == vision_end_token_id_ ||
            id == image_token_id_ || id == video_token_id_);
  });
}

bool Qwen3VLPromptProcessor::find_mm_spans(
    const std::vector<int32_t>& token_ids,
    MMData& mm_data) {
  if (!mm_data.valid()) {
    return validate_text_tokens(token_ids);
  }
  if (!mm_data.hold<MMItemVec>() || mm_data.empty() ||
      token_ids.size() > std::numeric_limits<int32_t>::max()) {
    return false;
  }

  const auto is_marker = [this](int32_t id) {
    return id == vision_start_token_id_ || id == vision_end_token_id_ ||
           id == image_token_id_ || id == video_token_id_;
  };
  auto cursor = token_ids.begin();
  for (auto& item : mm_data.items<MMItemVec>()) {
    const bool is_image = item.is_type(MMType::IMAGE);
    if (!is_image && !item.is_type(MMType::VIDEO)) {
      return false;
    }
    const auto grid = read_vision_grid(item, merge_size_);
    if (!grid) {
      return false;
    }
    // Images occupy one span. Video frames have separate timestamped spans.
    const int32_t spans = is_image ? 1 : grid->frames;
    const int32_t pads = is_image ? grid->frames * grid->tokens_per_frame
                                  : grid->tokens_per_frame;
    const int32_t pad_id = is_image ? image_token_id_ : video_token_id_;
    auto first_pad = cursor;
    for (int32_t span = 0; span < spans; ++span) {
      cursor = std::find_if(cursor, token_ids.end(), is_marker);
      if (cursor == token_ids.end() || *cursor != vision_start_token_id_) {
        return false;
      }
      ++cursor;
      if (token_ids.end() - cursor <= pads) {
        return false;
      }
      if (span == 0) {
        first_pad = cursor;
      }
      const auto end = cursor + pads;
      if (*end != vision_end_token_id_ ||
          !std::all_of(
              cursor, end, [pad_id](int32_t id) { return id == pad_id; })) {
        return false;
      }
      cursor = end + 1;
    }
    const auto last = cursor - 1;
    const int32_t offset = static_cast<int32_t>(first_pad - token_ids.begin());
    const int32_t length = static_cast<int32_t>(last - first_pad);
    auto& state = item.mutable_state();
    state.mutable_token_pos() = {offset, length};
    state.mutable_mm_token_num() = spans * pads;
    state.mutable_mm_token_mask() =
        is_image
            ? torch::ones({length}, torch::kBool)
            : torch::tensor(std::vector<int32_t>(first_pad, last)) == pad_id;
  }
  return std::none_of(cursor, token_ids.end(), is_marker);
}

std::pair<Qwen3VLPromptProcessor::TokenType, size_t>
Qwen3VLPromptProcessor::find_vision_token(const std::string& prompt,
                                          size_t begin) {
  auto img_pos = prompt.find(image_token_, begin);
  auto vid_pos = prompt.find(video_token_, begin);

  if (img_pos == std::string::npos && vid_pos == std::string::npos)
    return {TokenType::INVALID, std::string::npos};
  else if (vid_pos == std::string::npos)
    return {TokenType::IMAGE, img_pos};
  else if (img_pos == std::string::npos)
    return {TokenType::VIDEO, vid_pos};
  else
    return img_pos < vid_pos ? std::make_pair(TokenType::IMAGE, img_pos)
                             : std::make_pair(TokenType::VIDEO, vid_pos);
}

std::vector<double> Qwen3VLPromptProcessor::build_timestamps(
    const std::vector<double>& timestamps,
    size_t num_frames,
    int32_t merge_size) {
  CHECK_GT(merge_size, 0);

  if (timestamps.empty()) {
    return std::vector<double>(num_frames, 0.0);
  }

  std::vector<double> ts = timestamps;
  const size_t rem = ts.size() % static_cast<size_t>(merge_size);
  if (rem != 0) {
    ts.insert(ts.end(), static_cast<size_t>(merge_size) - rem, ts.back());
  }

  std::vector<double> out;
  out.reserve(ts.size() / static_cast<size_t>(merge_size));

  for (size_t i = 0; i < ts.size(); i += static_cast<size_t>(merge_size)) {
    out.push_back((ts[i] + ts[i + static_cast<size_t>(merge_size) - 1]) / 2.0);
  }

  if (out.size() > num_frames) {
    out.resize(num_frames);
  }
  while (out.size() < num_frames) {
    out.push_back(out.back());
  }

  return out;
}

std::string Qwen3VLPromptProcessor::format_timestamp_str(double timestamp) {
  char buffer[32];
  snprintf(buffer, sizeof(buffer), "<%.1f seconds>", timestamp);
  return buffer;
}

}  // namespace xllm
