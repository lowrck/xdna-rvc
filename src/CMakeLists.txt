# Core library shared by the CLI, the GUI and the tests.
add_library(xdna_rvc_core STATIC
    util/log.cpp
    util/system_info.cpp
    inference/backend_kind.cpp
    inference/npu_detect.cpp
    inference/ort_runtime.cpp
    inference/session_report.cpp
    inference/cpu_backend.cpp
    inference/directml_backend.cpp
    inference/xdna2_backend.cpp
    inference/backend_factory.cpp
    inference/model_session.cpp
)
target_include_directories(xdna_rvc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(xdna_rvc_core
    PUBLIC
        xdna_rvc_options
        onnxruntime::onnxruntime
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        xdna_rvc_miniaudio)
target_compile_definitions(xdna_rvc_core PUBLIC XDNA_RVC_ORT_FLAVOR="${XDNA_RVC_ORT_FLAVOR_RESOLVED}")
if(WIN32)
    target_link_libraries(xdna_rvc_core PUBLIC setupapi)
endif()

# CLI: diagnostics, offline conversion, benchmarking.
add_executable(xdna-rvc-cli
    app/cli_main.cpp
    app/cli_context.cpp
    app/cmd_providers.cpp
)
target_link_libraries(xdna-rvc-cli PRIVATE xdna_rvc_core CLI11::CLI11)
xdna_rvc_copy_ort_runtime(xdna-rvc-cli)
