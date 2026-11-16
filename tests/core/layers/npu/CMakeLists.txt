include(cc_test)

cc_test(
  NAME
    deepseek_v4_rotary_embedding_test
  CPU_ONLY
  SRCS
    deepseek_v4_rotary_embedding_tests.cpp
  DEPS
    :deepseek_v4_rotary_embedding_lib
    torch
    glog::glog
    GTest::gtest_main
)

cc_test(
  NAME
    dflash2_grouped_conv_test
  CPU_ONLY
  SRCS
    dflash2_grouped_conv_tests.cpp
  DEPS
    :common_layers
    torch
    glog::glog
    GTest::gtest_main
)
