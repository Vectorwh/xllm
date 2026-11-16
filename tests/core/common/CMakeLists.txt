include(cc_library)
include(cc_test)

# The registry fixture is a shared test header, so its build requirements live
# here instead of being repeated in every consumer's CMakeLists.
cc_library(
  TESTONLY
  NAME etcd_registry_fixture
  HDRS etcd_registry_fixture.h
  INCLUDES
    ${CMAKE_BINARY_DIR}/third_party/etcd_cpp_apiv3/proto/gen
    ${CMAKE_BINARY_DIR}/third_party/etcd_cpp_apiv3/proto/gen/proto
  DEPS etcd-cpp-api
)

cc_test(
  NAME
    common_test
  CPU_ONLY
  SRCS
    flash_comm1_context_test.cpp
    instance_role_util_test.cpp
    metrics_test.cpp
    options_test.cpp
    rate_limiter_test.cpp
  DEPS
    :config
    common
    absl::synchronization
    absl::time
    GTest::gtest_main
    gflags::gflags
    glog::glog
    torch
)
