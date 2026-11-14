/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/distributed_runtime/disagg_pd_service_impl.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/common/etcd_client.h"
#include "core/common/macros.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/service_config.h"
#include "core/scheduler/disagg_pd_scheduler.h"
#include "distributed_runtime/engine.h"
#include "framework/block/block_manager_pool.h"
#include "framework/model/model_args.h"
#include "framework/tokenizer/tokenizer.h"

namespace xllm {
namespace {

class FakeTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& /*text*/,
              std::vector<int32_t>* /*ids*/,
              bool /*add_special_tokens*/) const override {
    NOT_IMPLEMENTED();
  }

  std::string decode(const Slice<int32_t>& ids,
                     bool /*skip_special_tokens*/) const override {
    std::string text;
    for (int32_t id : ids) {
      text += std::to_string(id);
    }
    return text;
  }

  std::optional<int32_t> token_to_id(
      const std::string_view& /*token*/) const override {
    NOT_IMPLEMENTED();
  }

  std::string id_to_token(int32_t /*token*/) const override {
    NOT_IMPLEMENTED();
  }

  size_t vocab_size() const override { NOT_IMPLEMENTED(); }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>();
  }
};

// Minimal engine for the decode admission path: the failure classes under
// test return before any forward or tokenizer access, so every override is
// either a small stub or unreachable.
class FakeEngine final : public Engine {
 public:
  FakeEngine(int32_t num_blocks, int32_t block_size) {
    BlockManagerPool::Options options;
    options.num_blocks(num_blocks)
        .block_size(block_size)
        .enable_prefix_cache(true)
        .enable_disagg_pd(true);
    tokenizer_ = std::make_unique<FakeTokenizer>();
    block_manager_ = std::make_unique<BlockManagerPool>(options, /*dp_size=*/1);
  }

  ForwardOutput step(std::vector<Batch>& /*batch*/) override {
    NOT_IMPLEMENTED();
  }

  void update_last_step_result(std::vector<Batch>& /*batch*/) override {
    NOT_IMPLEMENTED();
  }

  const Tokenizer* tokenizer() const override { return tokenizer_.get(); }

  BlockManagerPool* block_manager_pool() const override {
    return block_manager_.get();
  }

  const ModelArgs& model_args() const override { return model_args_; }

  const TokenizerArgs& tokenizer_args() const override {
    static const TokenizerArgs args;
    return args;
  }

  std::vector<int64_t> get_active_activation_memory() const override {
    return {0};
  }

  bool init() override { return true; }

  bool link_cluster(const std::vector<uint64_t>& /*cluster_ids*/,
                    const std::vector<std::string>& /*addrs*/,
                    const std::vector<uint16_t>& /*ports*/,
                    int32_t /*src_dp_size*/,
                    int32_t /*src_kv_split_size*/) override {
    return true;
  }

  bool unlink_cluster(const std::vector<uint64_t>& /*cluster_ids*/,
                      const std::vector<std::string>& /*addrs*/,
                      const std::vector<uint16_t>& /*ports*/,
                      int32_t /*src_dp_size*/,
                      int32_t /*src_kv_split_size*/) override {
    return true;
  }

  bool pull_kv_blocks(
      int32_t /*src_dp_size*/,
      int32_t /*src_dp_rank*/,
      const std::vector<uint64_t>& /*src_cluster_ids*/,
      const std::vector<std::string>& /*src_addrs*/,
      int32_t /*dst_dp_rank*/,
      const std::vector<KVTransferMapping>& /*mappings*/) override {
    return true;
  }

 private:
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<BlockManagerPool> block_manager_;
  ModelArgs model_args_;
};

// The service impl's constructor requires an initialized XServiceClient;
// expose the protected no-services constructor without starting the
// dispatch thread or the RPC server.
class TestDisaggPDScheduler final : public DisaggPDScheduler {
 public:
  TestDisaggPDScheduler(Engine* engine, const Options& options)
      : DisaggPDScheduler(engine, options, /*start_services=*/false) {}
};

// Restores a process-global config field on scope exit, so one test's
// toggles cannot leak into the next case.
template <typename T>
class ScopedConfigValue final {
 public:
  ScopedConfigValue(T& field, T value) : field_(field), previous_(field) {
    field_ = std::move(value);
  }
  ~ScopedConfigValue() { field_ = std::move(previous_); }
  ScopedConfigValue(const ScopedConfigValue&) = delete;
  ScopedConfigValue& operator=(const ScopedConfigValue&) = delete;

 private:
  T& field_;
  T previous_;
};

}  // namespace

// Replace only registry I/O in this isolated executable, the same boundary
// pattern as xservice_client_test.cpp: DisaggPDServiceImpl's constructor
// requires an initialized XServiceClient, whose init() otherwise blocks on a
// reachable etcd master entry. These definitions satisfy EtcdClient's public
// boundary without pulling its production object, and no watch, registration
// or lookup performs any I/O.
EtcdClient::EtcdClient(const std::string& address, const std::string& space)
    : client_(address), etcd_addr_(address), etcd_namespace_prefix_(space) {}
EtcdClient::EtcdClient(const std::string& address,
                       const std::string&,
                       const std::string&,
                       const std::string& space)
    : EtcdClient(address, space) {}
EtcdClient::~EtcdClient() = default;
void EtcdClient::add_watch(const std::string&, Callback, bool) {}
void EtcdClient::remove_watch(const std::string&) {}
void EtcdClient::stop_watch() {}
bool EtcdClient::get_master_service(const std::string&, std::string* address) {
  *address = "127.0.0.1:9";
  return true;
}
bool EtcdClient::get_all_xservices(const std::string&,
                                   std::vector<std::string>*) {
  return false;
}
bool EtcdClient::register_instance(const std::string&,
                                   const std::string&,
                                   const int) {
  return true;
}
RegistrationLookupState EtcdClient::lookup_registration(const std::string&) {
  return RegistrationLookupState::kPresent;
}
bool EtcdClient::withdraw_registration(const std::string&) { return true; }

namespace {

class DisaggPDServiceImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize the process-global client once: the constructor of the
    // service under test LOG(FATAL)s without it. The stubbed EtcdClient
    // boundary above makes init() succeed without a real registry, and the
    // admission failures under test never exercise the client further.
    static const bool xservice_initialized =
        XServiceClient::get_instance()->init("http://127.0.0.1:9",
                                             "disagg-pd-admission-test");
    ASSERT_TRUE(xservice_initialized);
  }
};

TEST_F(DisaggPDServiceImplTest, AdmissionFailureStatusMatchesFailureClass) {
  // 422 (kDecodeConstraintRejectedStatusCode) makes the prefill scheduler
  // release the request's KV blocks as unreconcilable; before the fix every
  // nullptr from generate_request with a grammar_source present was reported
  // as 422, so unrelated failures (best_of without prefix cache, invalid VLM
  // metadata) were misdiagnosed as permanent constraint rejections. The
  // status must now follow the actual failure class.
  struct Case final {
    std::string name;
    bool best_of_without_prefix_cache;
    bool with_grammar_source;
    bool xgrammar_backend;
    int32_t expected_status;
  };
  const std::vector<Case> cases = {
      {"best_of>1 without prefix cache and a grammar stays a generic error",
       /*best_of_without_prefix_cache=*/true,
       /*with_grammar_source=*/true,
       /*xgrammar_backend=*/false,
       /*expected_status=*/500},
      {"constraint rebuild failure stays permanent",
       /*best_of_without_prefix_cache=*/false,
       /*with_grammar_source=*/true,
       /*xgrammar_backend=*/false,
       /*expected_status=*/kDecodeConstraintRejectedStatusCode},
      {"best_of>1 without prefix cache and no grammar stays a generic error",
       /*best_of_without_prefix_cache=*/true,
       /*with_grammar_source=*/false,
       /*xgrammar_backend=*/false,
       /*expected_status=*/500},
  };

  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
    DisaggPDScheduler::Options options;
    options.enable_pd_ooc(true)
        .enable_disagg_pd(true)
        .enable_schedule_overlap(false)
        .instance_role(InstanceRole::PREFILL)
        .max_tokens_per_batch(32)
        .max_seqs_per_batch(4)
        .max_tokens_per_chunk_for_prefill(32)
        .dp_size(1);
    TestDisaggPDScheduler scheduler(&engine, options);
    DisaggPDServiceImpl impl(&scheduler, &engine);

    // The best_of path requires the decode-side prefix cache to be off; the
    // constraint path requires the xgrammar backend to reject the spec (a
    // disabled backend in USE_XGRAMMAR builds, the #else branch otherwise).
    ScopedConfigValue<bool> prefix_cache_guard(
        KVCacheConfig::get_instance().enable_prefix_cache(),
        !c.best_of_without_prefix_cache);
    ScopedConfigValue<std::string> backend_guard(
        ServiceConfig::get_instance().structured_output_backend(),
        c.xgrammar_backend ? "xgrammar" : "builtin");

    proto::DisaggRequests requests;
    proto::DisaggRequest* req = requests.add_reqs();
    req->set_req_id("disagg-pd-admission-failure");
    req->set_prompt("prompt");
    req->set_n(1);
    req->set_best_of(c.best_of_without_prefix_cache ? 2 : 1);
    req->set_max_tokens(16);
    req->set_max_context_len(64);
    req->set_seq_capacity(64);
    req->set_offline(true);
    if (c.with_grammar_source) {
      req->set_grammar_type(
          static_cast<int32_t>(GrammarSpec::Type::JSON_SCHEMA));
      req->set_grammar_source(R"({"type":"object"})");
      req->set_grammar_strict(true);
    }

    proto::DisaggResponses responses;
    impl.decode_recv_new_requests(&requests, &responses);
    ASSERT_EQ(responses.resps_size(), 1);
    EXPECT_EQ(responses.resps(0).req_id(), req->req_id());
    EXPECT_EQ(responses.resps(0).status_code(), c.expected_status);
  }
}

}  // namespace
}  // namespace xllm
