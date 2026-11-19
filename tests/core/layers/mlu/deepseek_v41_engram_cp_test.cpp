/* Copyright 2026 The xLLM Authors.

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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <exception>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/layers/mlu/deepseek_v4/deepseek_v4_cp_context.h"
#include "core/layers/mlu/deepseek_v41/deepseek_v41_engram.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/util/net.h"

namespace xllm::layer {
namespace {

constexpr int32_t kSkip = 77;
constexpr std::chrono::seconds kChildTimeout{120};
using Weights = std::unordered_map<std::string, torch::Tensor>;

enum class ChildOutcome : int8_t { PASSED, SKIPPED, FAILED };

void verify_cp_engram(ProcessGroup& group,
                      ProcessGroup& tp_group,
                      ProcessGroup& cp_group,
                      const torch::Device& device) {
  const int32_t world_size = group.world_size();
  const int32_t cp_size = cp_group.world_size();
  const int32_t tp_size = tp_group.world_size();
  torch::InferenceMode inference;
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  ModelArgs args;
  args.hidden_size(8)
      .vocab_size(8)
      .hc_mult(2)
      .engram_layer_ids({0})
      .engram_num_embeddings({18})
      .engram_head_dim(32)
      .engram_n_heads(1)
      .engram_max_ngram_size(3)
      .engram_pad_token_id(0)
      .engram_compressed_vocab_size(8)
      .rms_norm_eps(1e-5);
  // CP selects query rows independently of the table's EP row partition.
  ProcessGroup single(/*rank=*/0, /*world_size=*/1, device);
  ParallelArgs cp_parallel(group.rank(), world_size, &group);
  cp_parallel.cp_size(cp_size)
      .tp_size(tp_size)
      .ep_size(world_size)
      .kv_split_size(1);
  cp_parallel.tp_group_ = &tp_group;
  cp_parallel.moe_ep_group_ = &group;
  cp_parallel.cp_group_ = &cp_group;
  ParallelArgs reference_parallel(/*rank=*/0, /*world_size=*/1, &single);
  reference_parallel.tp_group_ = &single;
  auto wkv = torch::zeros({24, 64}, torch::kBFloat16);
  auto channels = torch::arange(8, torch::kFloat32);
  // Zero keys fix the gate near 0.5; the two hash columns inject distinct
  // signed value directions. Missing either table shard exceeds the tolerance.
  wkv.slice(0, 16).slice(1, 0, 32).copy_(
      channels.remainder(2).mul(2).sub(1).div(32).unsqueeze(1));
  wkv.slice(0, 16).slice(1, 32, 64).copy_(channels.remainder(4)
                                              .ge(2)
                                              .to(torch::kFloat32)
                                              .mul(2)
                                              .sub(1)
                                              .div(32)
                                              .unsqueeze(1));
  const Weights weights{
      {"embed.weight",
       torch::arange(18)
           .add(0x30)
           .unsqueeze(1)
           .expand({18, 32})
           .contiguous()
           .to(torch::kUInt8)},
      {"embed.scale", torch::full({18, 1}, 127, torch::kUInt8)},
      {"wkv.weight", wkv},
      {"q_weight", torch::ones({2, 8}, torch::kBFloat16)},
      {"k_weight", torch::ones({2, 8}, torch::kBFloat16)},
  };
  const StateDict hash_weights({
      {"token_map", torch::arange(8, torch::kInt64)},
      {"multipliers", torch::tensor({{1, 3, 5}}, torch::kInt64)},
      {"primes", torch::tensor({{7, 11}}, torch::kInt64)},
      {"offsets", torch::tensor({{0, 7}}, torch::kInt64)},
  });
  for (const std::string hash_mode :
       {"cpu", "device_hash", "device_table", "shared"}) {
    const bool cpu_offload = hash_mode != "device_table";
    const bool shared = hash_mode == "shared";
    const std::string placement = cpu_offload ? "cpu" : "device";
    const std::string config =
        "{\"placement\":\"" + placement + "\",\"prefill_hash\":\"" +
        (hash_mode == "cpu" ? "cpu" : "device") + "\",\"replication\":\"" +
        (shared ? "per_node" : "none") + "\",\"communication\":\"";
    ModelConfig::get_instance().engram_config() = config + "allreduce\"}";
    DeepseekV41Engram baseline(
        ModelContext(cp_parallel, args, QuantArgs(), options), /*engram_id=*/0);
    ModelConfig::get_instance().engram_config() = R"({"placement":"device"})";
    DeepseekV41Engram reference(
        ModelContext(reference_parallel, args, QuantArgs(), options),
        /*engram_id=*/0);
    ModelConfig::get_instance().engram_config() =
        config + "reduce_scatter_allgather\"}";
    DeepseekV41Engram sharded(
        ModelContext(cp_parallel, args, QuantArgs(), options), /*engram_id=*/0);
    for (auto layer : {sharded, baseline, reference}) {
      layer->load_state_dict(StateDict(weights));
      layer->load_hash_state(hash_weights);
      layer->verify_loaded_weights();
    }
    // Neither request lengths nor per-rank row counts are rectangular. All
    // halves are nonempty, matching the real model's CP admission contract.
    const std::vector<std::vector<int32_t>> requests =
        shared ? std::vector<std::vector<int32_t>>{{127, 143, 16384}}
               : std::vector<std::vector<int32_t>>{
                     {127, 143}, {111, 159, 127}, {2 * cp_size, 1}};
    for (const std::vector<int32_t>& lengths : requests) {
      const int32_t total =
          std::accumulate(lengths.begin(), lengths.end(), int32_t{0});
      ModelInputParams input;
      input.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
      input.meta.num_sequences = static_cast<int32_t>(lengths.size());
      input.meta.actual_num_sequences = input.meta.num_sequences;
      input.attention.host.q_seq_lens = {0};
      input.attention.host.q_seq_lens.reserve(lengths.size() + 1);
      input.embedding.engram_lookback_token_ids.reserve(lengths.size() * 2);
      std::vector<int64_t> position_values;
      position_values.reserve(total);
      for (size_t seq = 0; seq < lengths.size(); ++seq) {
        input.attention.host.q_seq_lens.emplace_back(
            input.attention.host.q_seq_lens.back() + lengths[seq]);
        input.embedding.engram_lookback_token_ids.emplace_back(
            static_cast<int32_t>(seq) + 1);
        input.embedding.engram_lookback_token_ids.emplace_back(
            7 - static_cast<int32_t>(seq));
        for (int32_t row = 0; row < lengths[seq]; ++row) {
          position_values.emplace_back(3 + 2 * static_cast<int64_t>(seq) + row);
        }
      }
      const auto positions =
          torch::tensor(position_values, options.dtype(torch::kInt64));
      const auto cp = mlu_v4_cp::build_cp_context(
          cp_size, cp_group.rank(), &cp_group, lengths, positions);
      CHECK(cp.has_value());
      for (const auto& rows : cp->geometry.scatter_rows_by_rank) {
        CHECK(std::find(rows.begin(), rows.end(), cp->geometry.total_tokens) ==
              rows.end())
            << "Fixture must use CP without dummy rows";
      }
      CHECK_NE(cp->geometry.tokens_per_rank[0],
               cp->geometry.tokens_per_rank[1]);
      const auto tokens_cpu =
          (torch::arange(total, torch::kInt64) * 3 + lengths.front())
              .remainder(7)
              .add(1);
      const auto tokens = tokens_cpu.to(device);
      auto prepare_host = [&](DeepseekV41Engram& layer) {
        if (cpu_offload) {
          layer->prepare_embeddings(tokens_cpu, input, tokens);
        }
      };
      const auto hidden =
          (torch::arange(total * 2 * 8, torch::kFloat32).mul(0.19).cos() * 0.7)
              .reshape({total, 2, 8})
              .to(options);
      // Check the fixture against the small independent 2/3-gram hash formula:
      // global queries touch both physical table shards before local selection.
      std::vector<int32_t> shard_hits(world_size, 0);
      int64_t offset = 0;
      const auto token_values = tokens_cpu.accessor<int64_t, 1>();
      for (size_t seq = 0; seq < lengths.size(); ++seq) {
        int64_t previous = input.embedding.engram_lookback_token_ids[seq * 2];
        int64_t earlier =
            input.embedding.engram_lookback_token_ids[seq * 2 + 1];
        for (int32_t row = 0; row < lengths[seq]; ++row) {
          const int64_t token = token_values[offset++];
          const int64_t hash2 = (token ^ (previous * 3)) % 7;
          const int64_t hash3 =
              (token ^ (previous * 3) ^ (earlier * 5)) % 11 + 7;
          ++shard_hits[((hash2 + 1) * world_size - 1) / 18];
          ++shard_hits[((hash3 + 1) * world_size - 1) / 18];
          earlier = previous;
          previous = token;
        }
      }
      CHECK_GE(std::count_if(shard_hits.begin(),
                             shard_hits.end(),
                             [](int32_t hits) { return hits > 0; }),
               2);
      const auto expected = reference->forward(hidden, tokens, input);
      const auto local_hidden = mlu_v4_cp::shard_rows(hidden, *cp);
      // Use the same globally agreed layout as the real decoder. Compare both
      // collective policies with a full table and full-token reference.
      const auto layout = build_engram_cp_layout(*cp, tp_size);
      if (lengths.front() == 2 * cp_size && lengths.back() == 1) {
        // A full zigzag request keeps every front/back half nonempty. The
        // one-token request adds only to CP0: local sizes are [3,2,...]. At
        // TP2, every other CP rank's second TP shard contains only padding.
        CHECK_EQ(cp->geometry.tokens_per_rank.front(), 3);
        CHECK(std::all_of(cp->geometry.tokens_per_rank.begin() + 1,
                          cp->geometry.tokens_per_rank.end(),
                          [](int32_t rows) { return rows == 2; }));
        if (tp_size == 2) {
          CHECK_EQ(layout.rows_per_rank, 2);
          const int64_t valid_rows = std::clamp<int64_t>(
              layout.local_rows - tp_group.rank() * layout.rows_per_rank,
              0,
              layout.rows_per_rank);
          CHECK_EQ(valid_rows == 0, cp_group.rank() > 0 && tp_group.rank() == 1)
              << "Fixture must exercise a padding-only TP shard";
        }
      }
      std::unique_ptr<DeepseekV41EngramImpl::HashState> shared_state;
      if (shared) {
        // Shared tables shard heads across TP, independently of CP/EP. TP1
        // keeps both columns; TP2 saves 33 bytes/token and crosses 512KiB.
        const int64_t local_columns = (2 + tp_size - 1) / tp_size;
        const int64_t saved_bytes =
            static_cast<int64_t>(total) * (2 - local_columns) * 33;
        CHECK_EQ(saved_bytes >= 512 * 1024, tp_size > 1);
        shared_state = sharded->create_hash_state(total);
        shared_state->prepare_eager(tokens_cpu, input, tokens);
        shared_state->forward(tokens);
        CHECK_EQ(shared_state->lookup_packed_rows().size(0),
                 static_cast<int64_t>(total) * 2);
      } else {
        prepare_host(sharded);
      }
      const auto actual = sharded->forward(local_hidden,
                                           tokens,
                                           input,
                                           {shared_state.get(), {}, &layout},
                                           cp->local_row_indices);
      prepare_host(baseline);
      const auto baseline_actual = baseline->forward(local_hidden,
                                                     tokens,
                                                     input,
                                                     {nullptr, {}, &layout},
                                                     cp->local_row_indices);
      CHECK(torch::allclose(actual, baseline_actual, 0.02, 0.02));
      const auto expected_local =
          expected.index_select(0, cp->local_row_indices);
      CHECK(torch::allclose(actual, expected_local, 0.02, 0.02))
          << "Engram CP local rows differ: rank=" << group.rank()
          << " cpu_offload=" << cpu_offload;
      const auto restored = mlu_v4_cp::gather_restore(actual, *cp);
      CHECK(torch::allclose(restored, expected, 0.02, 0.02))
          << "Engram CP restore differs: cpu_offload=" << cpu_offload;
      CHECK_GT((expected - hidden).abs().max().item<float>(), 0.05f)
          << "Fixture must have a visible Engram contribution";
      // Configured CP may be inactive for a batch; no row selection/layout
      // must retain the complete result instead of scattering it.
      prepare_host(sharded);
      const auto replicated_actual = sharded->forward(hidden, tokens, input);
      CHECK(torch::allclose(replicated_actual, expected, 0.02, 0.02));
      // Refresh only the per-request history, preserving tokens and geometry.
      // This exposes implementations that hash local chunks as new sequences.
      for (auto& token : input.embedding.engram_lookback_token_ids) {
        token = (token + 2) % 7 + 1;
      }
      const auto next_expected = reference->forward(hidden, tokens, input);
      CHECK(!torch::allclose(next_expected, expected, 0.02, 0.02))
          << "Fixture must distinguish request histories";
      prepare_host(sharded);
      const auto next_actual = sharded->forward(local_hidden,
                                                tokens,
                                                input,
                                                {nullptr, {}, &layout},
                                                cp->local_row_indices);
      CHECK(
          torch::allclose(next_actual,
                          next_expected.index_select(0, cp->local_row_indices),
                          0.02,
                          0.02));
      CHECK(torch::allclose(mlu_v4_cp::gather_restore(next_actual, *cp),
                            next_expected,
                            0.02,
                            0.02));
    }
  }
}

int32_t run_child(int32_t rank,
                  int32_t cp_size,
                  int32_t tp_size,
                  const std::vector<int32_t>& ports) {
  const int32_t world_size = cp_size * tp_size;
  try {
    if (Platform::device_count() < world_size) {
      return kSkip;
    }
    Device xllm_device(rank);
    xllm_device.set_device();
    const torch::Device device = xllm_device.unwrap();
    auto group = create_process_group(rank,
                                      world_size,
                                      world_size,
                                      ports[0],
                                      /*trans=*/false,
                                      /*host=*/"127.0.0.1",
                                      /*group_name=*/"engram_cp_ep",
                                      device);
    CHECK(group != nullptr);
    ProcessGroup single(/*rank=*/0, /*world_size=*/1, device);
    std::unique_ptr<ProcessGroup> tp_group;
    std::unique_ptr<ProcessGroup> cp_group;
    if (tp_size > 1) {
      tp_group = create_process_group(rank,
                                      world_size,
                                      tp_size,
                                      ports[1 + rank / tp_size],
                                      /*trans=*/false,
                                      /*host=*/"127.0.0.1",
                                      /*group_name=*/"engram_cp_tp",
                                      device);
      cp_group = create_process_group(rank,
                                      world_size,
                                      cp_size,
                                      ports[1 + cp_size + rank % tp_size],
                                      /*trans=*/true,
                                      /*host=*/"127.0.0.1",
                                      /*group_name=*/"engram_cp_query",
                                      device);
    }
    verify_cp_engram(*group,
                     tp_group ? *tp_group : single,
                     cp_group ? *cp_group : *group,
                     device);
    xllm_device.synchronize_default_stream();
    return 0;
  } catch (const std::exception& error) {
    LOG(ERROR) << "Engram CP rank " << rank << " failed: " << error.what();
    return 1;
  }
}

ChildOutcome wait_children(const std::vector<pid_t>& pids) {
  const int32_t world_size = static_cast<int32_t>(pids.size());
  std::vector<bool> finished(world_size, false);
  int32_t remaining = world_size;
  int32_t skipped = 0;
  bool failed = false;
  const auto deadline = std::chrono::steady_clock::now() + kChildTimeout;
  while (remaining > 0 && !failed &&
         std::chrono::steady_clock::now() < deadline) {
    for (int32_t rank = 0; rank < world_size; ++rank) {
      if (finished[rank]) {
        continue;
      }
      int status = 0;
      const pid_t result = ::waitpid(pids[rank], &status, WNOHANG);
      if (result == 0 || (result < 0 && errno == EINTR)) {
        continue;
      }
      finished[rank] = true;
      --remaining;
      if (result > 0 && WIFEXITED(status) && WEXITSTATUS(status) == kSkip) {
        ++skipped;
        continue;
      }
      if (result < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        failed = true;
        LOG(ERROR) << "Engram CP rank " << rank << " failed, status=" << status;
      }
    }
    if (remaining > 0 && !failed) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  for (int32_t rank = 0; rank < world_size; ++rank) {
    if (finished[rank]) {
      continue;
    }
    failed = true;
    LOG(ERROR) << "Stopping stalled Engram CP rank " << rank;
    ::kill(pids[rank], SIGKILL);
    int status = 0;
    while (::waitpid(pids[rank], &status, 0) < 0 && errno == EINTR) {
    }
  }
  if (failed || (skipped != 0 && skipped != world_size)) {
    return ChildOutcome::FAILED;
  }
  return skipped == world_size ? ChildOutcome::SKIPPED : ChildOutcome::PASSED;
}

void verify_topology(int32_t cp_size, int32_t tp_size) {
  const int32_t world_size = cp_size * tp_size;
  std::vector<int32_t> ports;
  ports.reserve(1 + cp_size + tp_size);
  while (ports.size() < 1 + cp_size + tp_size) {
    const int32_t port = net::get_local_free_port();
    ASSERT_GT(port, 0);
    if (std::find(ports.begin(), ports.end(), port) == ports.end()) {
      ports.emplace_back(port);
    }
  }
  std::vector<pid_t> pids;
  pids.reserve(world_size);
  // The parent never initializes MLU; both ranks select their device after
  // fork.
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const pid_t pid = ::fork();
    if (pid == 0) {
      _exit(run_child(rank, cp_size, tp_size, ports));
    }
    if (pid < 0) {
      for (pid_t child : pids) {
        ::kill(child, SIGKILL);
        int status = 0;
        while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {
        }
      }
      FAIL() << "Cannot fork Engram CP rank " << rank << ": "
             << std::strerror(errno);
      return;
    }
    pids.emplace_back(pid);
  }
  const ChildOutcome outcome = wait_children(pids);
  if (outcome == ChildOutcome::SKIPPED) {
    GTEST_SKIP() << "Requires " << world_size << " visible MLU devices";
  }
  EXPECT_EQ(outcome, ChildOutcome::PASSED);
}

TEST(DeepseekV41EngramCpTest,
     TwoRanksMatchReplicatedForUnevenRequestsAndHistory) {
  verify_topology(/*cp_size=*/2, /*tp_size=*/1);
}

TEST(DeepseekV41EngramCpTest,
     Cp8MatchesAllreduceAndReplicatedForUnevenRequests) {
  verify_topology(/*cp_size=*/8, /*tp_size=*/1);
}

TEST(DeepseekV41EngramCpTest,
     Cp4Tp2MatchesAllreduceAndReplicatedForUnevenRequests) {
  verify_topology(/*cp_size=*/4, /*tp_size=*/2);
}

TEST(DeepseekV41EngramCpTest, LayoutPreservesUnevenRowsAndEmptyCpRanks) {
  // Exercise the layout contract independently of model CP admission, which
  // rejects batches requiring dummy query rows. TP neighbours must receive
  // consecutive portions of each CP rank, including its trailing padding.
  for (int32_t cp_size : {4, 8}) {
    for (int32_t tp_size : {1, 2}) {
      mlu_v4_cp::DeepseekV4CpContext context;
      context.geometry.total_tokens = 6;
      context.geometry.rows_by_rank = {{4, 0, 3}, {}, {1}, {2, 5}};
      context.geometry.rows_by_rank.resize(cp_size);
      auto source = torch::arange(6 * 2 * 3, torch::kFloat32).view({6, 2, 3});
      for (int32_t cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
        SCOPED_TRACE(::testing::Message() << "CP=" << cp_size << " TP="
                                          << tp_size << " cp_rank=" << cp_rank);
        context.cp_rank = cp_rank;
        context.local_row_indices = torch::tensor(
            context.geometry.rows_by_rank[cp_rank], torch::kInt64);
        const auto layout = build_engram_cp_layout(context, tp_size);
        const int64_t expected_rows_per_rank = (3 + tp_size - 1) / tp_size;
        EXPECT_EQ(layout.rows_per_rank, expected_rows_per_rank);
        EXPECT_EQ(layout.local_rows, context.local_row_indices.numel());
        EXPECT_EQ(layout.cp_rank, cp_rank);
        EXPECT_EQ(layout.cp_size, cp_size);
        EXPECT_EQ(layout.tp_size, tp_size);
        ASSERT_EQ(layout.pack_indices.numel(),
                  cp_size * tp_size * expected_rows_per_rank);
        EXPECT_TRUE(layout.pack_indices.ge(0).all().item<bool>());
        EXPECT_TRUE(layout.pack_indices.lt(6).all().item<bool>());
        auto packed = source.index_select(0, layout.pack_indices);
        auto rank_shards =
            packed.view({cp_size, tp_size, expected_rows_per_rank, 2, 3});
        // allgather_base's [TP,rows,C,D] output can flatten its first two
        // dimensions directly without changing the token order.
        auto local = rank_shards.select(0, cp_rank)
                         .flatten(0, 1)
                         .narrow(0, 0, layout.local_rows);
        EXPECT_TRUE(torch::equal(
            local, source.index_select(0, context.local_row_indices)));
      }
    }
  }
}

}  // namespace
}  // namespace xllm::layer
