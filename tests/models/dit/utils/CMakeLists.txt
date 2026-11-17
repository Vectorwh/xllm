include(cc_test)

cc_test(
  NAME lanczos_resample_test
  CPU_ONLY
  SRCS
    lanczos_resample_test.cpp
    ${PROJECT_SOURCE_DIR}/xllm/models/dit/utils/lanczos_resample.cpp
  DEPS torch glog::glog GTest::gtest_main
)

cc_test(
  NAME
    cola_utils_test
  CPU_ONLY
  SRCS
    cola_utils_test.cpp
  DEPS
    torch
    GTest::gtest_main
)

cc_test(
  NAME minimax_h3_vae_tile_window_test
  CPU_ONLY
  SRCS minimax_h3_vae_tile_window_test.cpp
  DEPS torch glog::glog GTest::gtest_main
)
