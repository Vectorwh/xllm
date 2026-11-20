include(cc_test)

cc_test(
  NAME musa_attention_test
  SRCS attention_test.cpp
  DEPS :musa_layers GTest::gtest_main glog::glog
  CPU_ONLY
)

cc_test(
  NAME musa_prefill_test
  SRCS prefill_test.cpp
  DEPS :musa_layers :platform :config torch_musa GTest::gtest_main glog::glog
  ENVIRONMENT "FLASHINFER_OPS_PATH=$<TARGET_FILE_DIR:xllm>/musa_ops"
  RUN_SERIAL
)
target_link_libraries(musa_prefill_test PRIVATE
  "$<LINK_GROUP:RESCAN,platform,config,util>")

cc_test(
  NAME
    musa_qwen3_vision_attention_test
  SRCS
    qwen3_vision_attention_test.cpp
  DEPS
    :musa_layers
    :platform
    :config
    torch_musa
    GTest::gtest_main
    glog::glog
  RUN_SERIAL
)
target_link_libraries(musa_qwen3_vision_attention_test PRIVATE
  "$<LINK_GROUP:RESCAN,platform,config,util>")

cc_test(
  NAME
    musa_qwen3_vision_layer_test
  SRCS
    qwen3_vision_layer_test.cpp
  DEPS
    :layers
    :platform
    :config
    torch_musa
    GTest::gtest_main
    glog::glog
  RUN_SERIAL
)
target_link_libraries(musa_qwen3_vision_layer_test PRIVATE
  "$<LINK_GROUP:RESCAN,platform,config,util>")

cc_test(
  NAME musa_pcp_context_test
  SRCS pcp_context_test.cpp
  DEPS :musa_layers :platform :config GTest::gtest_main glog::glog
  CPU_ONLY
)
target_link_libraries(musa_pcp_context_test PRIVATE
  "$<LINK_GROUP:RESCAN,platform,config,util>")
