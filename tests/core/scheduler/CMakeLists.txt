include(cc_test)

add_subdirectory(profile)

# Exercise response-pool ordering without the scheduler's global device setup.
cc_test(
  NAME
    request_phase_output_test
  CPU_ONLY
  SRCS
    request_phase_output_test.cpp
    ${PROJECT_SOURCE_DIR}/xllm/core/scheduler/async_response_processor.cpp
  DEPS
    :config
    :request
    :util
    GTest::gtest_main
)
target_link_libraries(request_phase_output_test PUBLIC
    Python::Python
    $<$<BOOL:${USE_NPU}>:ascendcl>
    $<$<BOOL:${USE_NPU}>:hccl>
    $<$<BOOL:${USE_NPU}>:c_sec>
    $<$<BOOL:${USE_NPU}>:nnopbase>)

cc_test(
  NAME
    scheduler_test
  SRCS
    disagg_pd_scheduler_test.cpp
    continuous_scheduler_test.cpp
    fixed_steps_scheduler_test.cpp
    scheduler_policy_test.cpp
    linear_prefix_admission_policy_test.cpp
  DEPS
    :config
    :scheduler
    :xllm_server
    :block
    GTest::gtest_main
    $<$<BOOL:${USE_NPU}>:nnopbase>
)
# BatchFactory fixes DP size on first use; keep DP=2 restore scenarios isolated.
cc_test(
  NAME
    pd_decode_restore_test
  SRCS
    pd_decode_restore_test.cpp
  DEPS
    :config
    :scheduler
    :xllm_server
    :block
    GTest::gtest_main
    $<$<BOOL:${USE_NPU}>:nnopbase>
)
cc_test(
  NAME
    pd_inflight_lifetime_test
  CPU_ONLY
  SRCS
    pd_inflight_lifetime_test.cpp
  DEPS
    :config
    :scheduler
    :xllm_server
    :block
    GTest::gtest_main
    $<$<BOOL:${USE_NPU}>:nnopbase>
)
# Isolate process-global response metrics without starting model workers.
cc_test(
  NAME
    dit_scheduler_test
  CPU_ONLY
  SRCS
    dit_scheduler_test.cpp
  DEPS
    :config
    :scheduler
    :xllm_server
    :block
    GTest::gtest_main
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${USE_NPU}>:torch_python>
    $<$<BOOL:${USE_NPU}>:nnopbase>
)
set_tests_properties(
  DiTResponseMetricsTest.OutputBuildEndsBeforeCallback
  PROPERTIES TIMEOUT 60)

foreach(test_target scheduler_test pd_decode_restore_test pd_inflight_lifetime_test dit_scheduler_test)
  target_link_libraries(${test_target}
                        PUBLIC
                        Python::Python
                        $<$<BOOL:${USE_NPU}>:ascendcl>
                        $<$<BOOL:${USE_NPU}>:hccl>
                        $<$<BOOL:${USE_NPU}>:c_sec>)
  if(USE_NPU)
    # Keep target dependencies so CMake repeats the complete static-library SCC.
    # A partial RESCAN group can introduce API/runtime references too late.
    target_link_libraries(${test_target} PRIVATE :xtensor :xllm_server)
  elseif(test_target STREQUAL "dit_scheduler_test")
    # Rescan execution/request archives without merging shared target cycles.
    # batch.cpp consumes request_timing, so that archive belongs in the rescan
    # group too; otherwise libbatch.a's later repetition reintroduces its
    # symbols after the last request_timing scan.
    # Runtime and block objects also consume provider and prefix-cache symbols.
    target_link_libraries(${test_target} PRIVATE
      "$<LINK_GROUP:RESCAN,xtensor,xllm_server,xgrammar_constraint,$<TARGET_FILE:runtime>,$<TARGET_FILE:batch>,$<TARGET_FILE:request>,$<TARGET_FILE:request_timing>,$<TARGET_FILE:kv_cache>,$<TARGET_FILE:block>,$<TARGET_FILE:block_hasher>,$<TARGET_FILE:provider_contract>,$<TARGET_FILE:prefix_cache>,$<TARGET_FILE:prefix_cache_journal>>")
  else()
    target_link_libraries(${test_target} PRIVATE
      "$<LINK_GROUP:RESCAN,xtensor,xllm_server>")
  endif()
endforeach()

# Register extracted multiline cases even with source-scanning GTest discovery.
if(TARGET scheduler_test)
  foreach(case_filter IN ITEMS
    VlmPDMetadataTest.OocDispatchRejectsMediaBeforeOnlineOrOfflineSerialization
    VlmPDBestOfTest.MissingPrefixCacheRejectsFreshCandidatesAndReleasesDonor
  )
    if(NOT TEST "${case_filter}")
      add_test(NAME "${case_filter}"
        COMMAND "$<TARGET_FILE:scheduler_test>" "--gtest_filter=${case_filter}")
      set_tests_properties("${case_filter}" PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        LABELS "target:scheduler_test"
        SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]")
    endif()
  endforeach()
endif()
