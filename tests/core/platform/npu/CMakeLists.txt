include(cc_test)

add_executable(npu_profiler_lifecycle_test
  profiler_lifecycle_test.cpp
  ${PROJECT_SOURCE_DIR}/xllm/core/platform/npu/npu_profiler.cpp
)
target_include_directories(npu_profiler_lifecycle_test PRIVATE
  ${COMMON_INCLUDE_DIRS}
)
target_link_libraries(npu_profiler_lifecycle_test PRIVATE
  torch
  torch_python
  pybind11::embed
  GTest::gtest_main
  glog::glog
)
add_dependencies(all_tests npu_profiler_lifecycle_test)
gtest_add_tests(TARGET npu_profiler_lifecycle_test
  TEST_LIST _npu_profiler_lifecycle_tests)
# Same hang guard as cc_test(): this binary embeds Python, so a stuck case
# would otherwise wait out CTest's 1500s default.
set_tests_properties(${_npu_profiler_lifecycle_tests}
  PROPERTIES TIMEOUT 300)

cc_test(
  NAME
    npu_profiler_test
  SRCS
    profiler_test.cpp
  DEPS
    :platform
    :config
    GTest::gtest_main
    glog::glog
)
