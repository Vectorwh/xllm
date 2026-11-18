add_subdirectory(utils)

# QwenImage21 is registered only by the NPU model backend. CPU_ONLY keeps
# backend macros, so these tests still require that backend's shared layers.
if(NOT USE_NPU)
  return()
endif()

include(cc_test)
# CPU_ONLY skips device bootstrap; retain backend macros to match the
# ProcessGroup and ModelContext libraries used by VaeParallelMixin.
cc_test(
  NAME qwen_image_21_structure_test
  CPU_ONLY
  SRCS qwen_image_21_structure_test.cpp
  DEPS torch common_layers state_dict parallel_state glog::glog proto::xllm_proto GTest::gtest_main
)

cc_test(
  NAME qwen_image_21_vae_test
  CPU_ONLY
  SRCS qwen_image_21_vae_test.cpp
  # model_context.cpp also contains the ATB-owning constructor in NPU builds.
  DEPS torch model_context parallel_state config
       $<$<BOOL:${USE_NPU}>:npu_layers>
       glog::glog GTest::gtest_main
)

if(TARGET qwen_image_21_vae_test)
  # ModelContext pulls in the config/platform/util static-library cycle.
  target_link_libraries(qwen_image_21_vae_test PRIVATE
    "$<LINK_GROUP:RESCAN,config,platform,util>")
endif()

cc_test(
  NAME qwen_image_21_npu_test
  SRCS qwen_image_21_npu_test.cpp
  DEPS torch common_layers state_dict config parallel_state glog::glog GTest::gtest_main
)
