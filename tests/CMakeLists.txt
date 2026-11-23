if(NOT BUILD_TESTING)
  return()
endif()

add_test(NAME grouped_gtest_runner_test
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/grouped_gtest_runner_test.py")
set_tests_properties(grouped_gtest_runner_test PROPERTIES LABELS cpu)

if(USE_NPU)
  # These source-level metadata tests stub platform kernels and use CPU tensors.
  # Expose them through setup.py test without the device bootstrap targets.
  add_custom_target(cache_resource_metadata_python_test)
  add_test(
    NAME cache_resource_metadata_python_test
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}"
      "TORCH_DEVICE_BACKEND_AUTOLOAD=0"
      "${Python3_EXECUTABLE}" -m pytest -q
      "${CMAKE_CURRENT_SOURCE_DIR}/python/test_dsa_metadata.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/test_csa_attention.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/test_csa2_block_metadata.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/test_decode_acl_graph.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/test_decode_acl_graph_empty_dp.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/test_decode_acl_graph_dspark_verify.py"
  )
  set_tests_properties(cache_resource_metadata_python_test PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    LABELS "cpu;target:cache_resource_metadata_python_test")

  add_test(
    NAME python_executor_test
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}"
      "XLLM_BATCH_TEST_BINARY=$<TARGET_FILE:batch_test>"
      "${Python3_EXECUTABLE}" -m pytest -q "${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  set_tests_properties(
    python_executor_test
    PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )
endif()

add_subdirectory(api_service)
add_subdirectory(core)
add_subdirectory(function_call)
add_subdirectory(models)
add_subdirectory(parser)
add_subdirectory(processors)
