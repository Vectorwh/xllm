add_subdirectory(utils)

include(cc_test)
# CPU_ONLY skips device bootstrap; retain backend macros to match the
# ProcessGroup and ModelContext libraries used by VaeParallelMixin.
cc_test(
  NAME qwen_image_21_structure_test
  CPU_ONLY
  SRCS qwen_image_21_structure_test.cpp
  DEPS torch common_layers state_dict parallel_state glog::glog proto::xllm_proto GTest::gtest_main
)

if(NOT USE_NPU)
  # Native execution tests use CPU tensors; NPU builds select fused operators.
  cc_test(
    NAME qwen_image_21_native_test
    CPU_ONLY
    SRCS qwen_image_21_native_test.cpp
    DEPS torch common_layers state_dict parallel_state glog::glog proto::xllm_proto GTest::gtest_main
  )
endif()

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

if(USE_NPU)
  cc_test(
    NAME qwen_image_21_npu_test
    SRCS qwen_image_21_npu_test.cpp
    DEPS torch common_layers state_dict config parallel_state glog::glog GTest::gtest_main
  )
endif()
