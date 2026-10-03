cmake_minimum_required(VERSION 3.24)

project(xdna_rvc
    VERSION 0.1.0
    DESCRIPTION "Realtime RVC voice changer for AMD Ryzen AI (XDNA 2) NPUs"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(XDNA_RVC_BUILD_GUI "Build the ImGui desktop application (xdna-rvc)" ON)
option(XDNA_RVC_BUILD_TESTS "Build the C++ unit tests" ON)
option(XDNA_RVC_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
set(XDNA_RVC_ASIO_SDK_DIR "" CACHE PATH
    "Path to the Steinberg ASIO SDK (Windows only). Leave empty to build without ASIO.")

include(CompilerOptions)
include(OnnxRuntime)
include(Dependencies)

add_subdirectory(src)

if(XDNA_RVC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

message(STATUS "")
message(STATUS "xdna-rvc configuration summary")
message(STATUS "  ONNX Runtime flavor : ${XDNA_RVC_ORT_FLAVOR_RESOLVED}")
message(STATUS "  ONNX Runtime root   : ${XDNA_RVC_ORT_ROOT_RESOLVED}")
message(STATUS "  GUI                 : ${XDNA_RVC_BUILD_GUI}")
message(STATUS "  Tests               : ${XDNA_RVC_BUILD_TESTS}")
message(STATUS "  ASIO SDK            : ${XDNA_RVC_ASIO_SDK_DIR}")
message(STATUS "")
