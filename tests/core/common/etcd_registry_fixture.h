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

#pragma once

#include <glog/logging.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server_builder.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/etcd_schema.h"
#include "core/framework/config/distributed_config.h"
#include "proto/rpc.grpc.pb.h"
#include "tests/core/common/scoped_environment_variable.h"

namespace xllm::test {

// Serves the registry RPC boundary so a test can link the real EtcdClient:
// production client methods no longer need matching definitions in the test
// binary.
//
// Coverage boundary: the fixture implements only the RPCs EtcdClient issues,
// and the semantics the tests assert on follow a real etcd. Semantics the
// tests do not depend on are either explicitly fatal (an unsupported
// transaction shape aborts instead of degrading into retries) or documented
// as not modelled (watch events, lease expiry, auth); none of them is answered
// wrongly and silently. RPCs outside this set stay at gRPC's UNIMPLEMENTED.
class FakeRegistry final : public etcdserverpb::KV::Service,
                           public etcdserverpb::Lease::Service,
                           public etcdserverpb::Watch::Service {
 public:
  enum class Mode : int8_t {
    READY,
    REGISTRATION_FAILURE,
    MASTER_UNAVAILABLE,
  };

  explicit FakeRegistry(Mode mode = Mode::READY) : mode_(mode) {
    // Keep registry RPCs on loopback even when the runner configures an HTTP
    // proxy; the lifecycle cases poison grpc_proxy to cover this path. The
    // env guards snapshot these process-wide variables when this object is
    // constructed and restore them when it is destroyed.
    CHECK(no_grpc_proxy_guard_.set("127.0.0.1"));
    // init() reads these with util::get_optional_string_env(...).value_or(""),
    // so an empty value disables the auth path exactly like an unset one.
    CHECK(etcd_username_guard_.set(""));
    CHECK(etcd_password_guard_.set(""));
    const std::string& shard =
        DistributedConfig::get_instance().control_plane_shard_id();
    master_key_ = etcd_schema::master_service_key_for_shard(shard);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort(
        "127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(static_cast<etcdserverpb::KV::Service*>(this));
    builder.RegisterService(static_cast<etcdserverpb::Lease::Service*>(this));
    builder.RegisterService(static_cast<etcdserverpb::Watch::Service*>(this));
    server_ = builder.BuildAndStart();
    CHECK(server_ != nullptr) << "Failed to start the test registry.";
    endpoint_ = "127.0.0.1:" + std::to_string(port);
  }

  ~FakeRegistry() override {
    // Bound the shutdown: a leftover client stream must not hang the process
    // at exit.
    server_->Shutdown(std::chrono::system_clock::now() +
                      std::chrono::seconds(1));
    server_->Wait();
  }

  std::string address() const { return "http://" + endpoint_; }

  std::future<void> first_attempt() { return first_attempt_.get_future(); }

  int32_t attempts() const { return attempts_.load(); }

  grpc::Status Range(grpc::ServerContext* /*unused*/,
                     const etcdserverpb::RangeRequest* request,
                     etcdserverpb::RangeResponse* response) override {
    response->mutable_header()->set_revision(1);
    if (!request->range_end().empty()) {
      return scan_range(request, response);
    }
    if (request->key() == master_key_) {
      if (!master_visible()) {
        note_attempt();
        return grpc::Status(grpc::StatusCode::NOT_FOUND,
                            "The test master is unavailable.");
      }
      auto* value = response->add_kvs();
      value->set_key(master_key_);
      value->set_value(endpoint_);
      response->set_count(1);
      return grpc::Status::OK;
    }
    std::lock_guard<std::mutex> lock(values_mutex_);
    const auto it = values_.find(request->key());
    if (it != values_.end()) {
      *response->add_kvs() = it->second;
      response->set_count(1);
    }
    return grpc::Status::OK;
  }

  grpc::Status Put(grpc::ServerContext* /*unused*/,
                   const etcdserverpb::PutRequest* request,
                   etcdserverpb::PutResponse* response) override {
    if (request->key() == master_key_) {
      unsupported("write to the synthetic master entry");
    }
    std::lock_guard<std::mutex> lock(values_mutex_);
    auto& value = values_[request->key()];
    value.set_key(request->key());
    value.set_value(request->value());
    value.set_lease(request->lease());
    response->mutable_header()->set_revision(1);
    return grpc::Status::OK;
  }

  grpc::Status DeleteRange(
      grpc::ServerContext* /*unused*/,
      const etcdserverpb::DeleteRangeRequest* request,
      etcdserverpb::DeleteRangeResponse* response) override {
    std::lock_guard<std::mutex> lock(values_mutex_);
    response->set_deleted(values_.erase(request->key()));
    response->mutable_header()->set_revision(1);
    return grpc::Status::OK;
  }

  grpc::Status Txn(grpc::ServerContext* /*unused*/,
                   const etcdserverpb::TxnRequest* request,
                   etcdserverpb::TxnResponse* response) override {
    note_attempt();
    if (mode_ == Mode::REGISTRATION_FAILURE) {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                          "The test registration is rejected.");
    }
    std::lock_guard<std::mutex> lock(values_mutex_);
    for (const auto& compare : request->compare()) {
      if (compare.target() != etcdserverpb::Compare::CREATE ||
          compare.result() != etcdserverpb::Compare::EQUAL ||
          compare.create_revision() != 0) {
        unsupported("transaction compare other than a create check");
      }
      // The master entry is served from endpoint_ rather than values_, so a
      // create-if-absent check on it must still see it as present.
      const bool exists =
          values_.contains(compare.key()) ||
          (compare.key() == master_key_ && master_visible());
      if (exists) {
        response->set_succeeded(false);
        return grpc::Status::OK;
      }
    }
    for (const auto& operation : request->success()) {
      if (!operation.has_request_put()) {
        unsupported("transaction operation other than put");
      }
      const auto& put = operation.request_put();
      if (put.key() == master_key_) {
        unsupported("write to the synthetic master entry");
      }
      auto& value = values_[put.key()];
      value.set_key(put.key());
      value.set_value(put.value());
      value.set_lease(put.lease());
      response->add_responses()->mutable_response_put();
    }
    response->set_succeeded(true);
    response->mutable_header()->set_revision(1);
    return grpc::Status::OK;
  }

  // Leases never expire here, but each granted lease keeps its own TTL so a
  // later lookup cannot report another lease's value.
  grpc::Status LeaseGrant(grpc::ServerContext* /*unused*/,
                          const etcdserverpb::LeaseGrantRequest* request,
                          etcdserverpb::LeaseGrantResponse* response) override {
    const int64_t id = next_lease_.fetch_add(1);
    {
      std::lock_guard<std::mutex> lock(values_mutex_);
      lease_ttls_[id] = request->ttl();
    }
    response->set_id(id);
    response->set_ttl(request->ttl());
    return grpc::Status::OK;
  }

  grpc::Status LeaseKeepAlive(
      grpc::ServerContext* /*unused*/,
      grpc::ServerReaderWriter<etcdserverpb::LeaseKeepAliveResponse,
                               etcdserverpb::LeaseKeepAliveRequest>* stream)
      override {
    etcdserverpb::LeaseKeepAliveRequest request;
    while (stream->Read(&request)) {
      int64_t ttl = 0;
      {
        std::lock_guard<std::mutex> lock(values_mutex_);
        const auto it = lease_ttls_.find(request.id());
        if (it != lease_ttls_.end()) {
          ttl = it->second;
        }
      }
      etcdserverpb::LeaseKeepAliveResponse response;
      response.set_id(request.id());
      response.set_ttl(ttl);
      if (!stream->Write(response)) {
        break;
      }
    }
    return grpc::Status::OK;
  }

  grpc::Status LeaseTimeToLive(
      grpc::ServerContext* /*unused*/,
      const etcdserverpb::LeaseTimeToLiveRequest* request,
      etcdserverpb::LeaseTimeToLiveResponse* response) override {
    int64_t ttl = 0;
    {
      std::lock_guard<std::mutex> lock(values_mutex_);
      const auto it = lease_ttls_.find(request->id());
      if (it == lease_ttls_.end()) {
        return grpc::Status(grpc::StatusCode::NOT_FOUND,
                            "The test lease does not exist.");
      }
      ttl = it->second;
    }
    response->set_id(request->id());
    response->set_ttl(ttl);
    response->set_grantedttl(ttl);
    return grpc::Status::OK;
  }

  // Only the watch creation acknowledgement is sent; no events are delivered,
  // so event-driven callbacks are not modelled.
  grpc::Status Watch(
      grpc::ServerContext* /*unused*/,
      grpc::ServerReaderWriter<etcdserverpb::WatchResponse,
                               etcdserverpb::WatchRequest>* stream) override {
    etcdserverpb::WatchRequest request;
    while (stream->Read(&request)) {
      etcdserverpb::WatchResponse response;
      response.set_created(true);
      response.set_watch_id(1);
      if (!stream->Write(response)) {
        break;
      }
    }
    return grpc::Status::OK;
  }

 private:
  // Every read path asks the same question, so the answer lives in one place.
  bool master_visible() const { return mode_ != Mode::MASTER_UNAVAILABLE; }

  // Real etcd serves [key, range_end): the stored values plus the synthetic
  // master entry, in ascending key order. The master entry is not in values_
  // because it is answered from endpoint_ on single-key reads too.
  grpc::Status scan_range(const etcdserverpb::RangeRequest* request,
                          etcdserverpb::RangeResponse* response) {
    std::lock_guard<std::mutex> lock(values_mutex_);
    const bool serve_master =
        master_visible() && in_range(master_key_, request);
    std::vector<std::string> keys;
    keys.reserve(values_.size() + (serve_master ? 1u : 0u));
    for (const auto& entry : values_) {
      if (in_range(entry.first, request)) {
        keys.push_back(entry.first);
      }
    }
    if (serve_master) {
      keys.push_back(master_key_);
    }
    std::sort(keys.begin(), keys.end());
    for (const auto& key : keys) {
      if (serve_master && key == master_key_) {
        auto* value = response->add_kvs();
        value->set_key(master_key_);
        value->set_value(endpoint_);
        continue;
      }
      *response->add_kvs() = values_.at(key);
    }
    response->set_count(static_cast<int64_t>(keys.size()));
    return grpc::Status::OK;
  }

  static bool in_range(const std::string& key,
                       const etcdserverpb::RangeRequest* request) {
    if (key < request->key()) {
      return false;
    }
    // etcd treats a "\0" range_end as "from key to the end of the keyspace".
    // No current caller issues that form; it is modelled so an empty prefix
    // scan cannot silently answer wrongly.
    const std::string& end = request->range_end();
    if (end.size() == 1 && end[0] == '\0') {
      return true;
    }
    return key < end;
  }

  // Unsupported traffic means the fixture no longer matches the client. Abort
  // here instead of returning a retryable status that looks like a hang.
  [[noreturn]] static void unsupported(const char* what) {
    LOG(FATAL) << "Unsupported test registry request: " << what;
  }

  void note_attempt() {
    if (attempts_.fetch_add(1) == 0) {
      first_attempt_.set_value();
    }
  }

  const Mode mode_;
  std::string endpoint_;
  std::string master_key_;
  std::unique_ptr<grpc::Server> server_;
  mutable std::mutex values_mutex_;
  std::unordered_map<std::string, mvccpb::KeyValue> values_;
  std::atomic<int32_t> attempts_{0};
  std::promise<void> first_attempt_;
  tests::ScopedEnvironmentVariable no_grpc_proxy_guard_{"no_grpc_proxy"};
  tests::ScopedEnvironmentVariable etcd_username_guard_{"ETCD_USERNAME"};
  tests::ScopedEnvironmentVariable etcd_password_guard_{"ETCD_PASSWORD"};
  std::atomic<int64_t> next_lease_{1};
  // Granted TTL per lease id, guarded by values_mutex_.
  std::unordered_map<int64_t, int64_t> lease_ttls_;
};

}  // namespace xllm::test
