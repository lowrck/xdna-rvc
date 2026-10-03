add_executable(xdna_rvc_tests
    test_main.cpp
    test_backend_discovery.cpp
    test_session_report.cpp
    test_model_session.cpp
    test_resampler.cpp
)
target_link_libraries(xdna_rvc_tests PRIVATE xdna_rvc_core xdna_rvc_doctest)
target_compile_definitions(xdna_rvc_tests PRIVATE
    XDNA_RVC_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")
xdna_rvc_copy_ort_runtime(xdna_rvc_tests)

# Tests that need an AMD XDNA 2 NPU with the Ryzen AI runtime are tagged with the
# doctest test-suite "hardware". They skip themselves (with a message) when the
# VitisAI EP is unavailable, and can be excluded entirely with
#   xdna_rvc_tests --test-suite-exclude=hardware
add_test(NAME xdna_rvc_unit COMMAND xdna_rvc_tests --test-suite-exclude=hardware
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
add_test(NAME xdna_rvc_hardware COMMAND xdna_rvc_tests --test-suite=hardware
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(xdna_rvc_hardware PROPERTIES LABELS hardware)
