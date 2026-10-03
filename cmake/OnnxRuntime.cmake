# Locates (or downloads) the ONNX Runtime C/C++ package and exposes it as the
# imported target onnxruntime::onnxruntime.
#
# Flavors (XDNA_RVC_ORT_FLAVOR):
#   auto      Windows: "ryzenai" if RYZEN_AI_INSTALLATION_PATH is set, else "directml".
#             Other platforms: "cpu".
#   ryzenai   The ONNX Runtime shipped with AMD Ryzen AI Software. Contains the
#             VitisAIExecutionProvider (XDNA NPU), plus CPU (and DirectML in the AMD build).
#             Requires RYZEN_AI_INSTALLATION_PATH (set by the Ryzen AI installer).
#   directml  Microsoft.ML.OnnxRuntime.DirectML NuGet package (Windows only) + DirectML.dll.
#   cpu       Official Microsoft release archive (CPU execution provider only).
#   custom    Use XDNA_RVC_ORT_ROOT, which must contain include/ and lib/.
#
# Every flavor exposes the same C++ API; which execution providers actually exist
# is decided at runtime with Ort::GetAvailableProviders() and reported by
# `xdna-rvc-cli providers`.

set(XDNA_RVC_ORT_FLAVOR "auto" CACHE STRING "ONNX Runtime package flavor: auto|ryzenai|directml|cpu|custom")
set_property(CACHE XDNA_RVC_ORT_FLAVOR PROPERTY STRINGS auto ryzenai directml cpu custom)
set(XDNA_RVC_ORT_ROOT "" CACHE PATH "ONNX Runtime root for XDNA_RVC_ORT_FLAVOR=custom")

set(_ort_version "1.30.0")
set(_ort_dml_version "1.24.4")       # Microsoft.ML.OnnxRuntime.DirectML (DirectML EP is in maintenance mode)
set(_directml_version "1.15.4")      # Microsoft.AI.DirectML redistributable
set(_ort_download_dir "${CMAKE_BINARY_DIR}/_deps/onnxruntime")

set(_flavor "${XDNA_RVC_ORT_FLAVOR}")
if(_flavor STREQUAL "auto")
    if(WIN32)
        if(DEFINED ENV{RYZEN_AI_INSTALLATION_PATH} AND EXISTS "$ENV{RYZEN_AI_INSTALLATION_PATH}")
            set(_flavor "ryzenai")
        else()
            set(_flavor "directml")
        endif()
    else()
        set(_flavor "cpu")
    endif()
endif()

function(_xdna_rvc_download url sha256 dest)
    if(NOT EXISTS "${dest}")
        message(STATUS "Downloading ${url}")
        file(DOWNLOAD "${url}" "${dest}"
            EXPECTED_HASH SHA256=${sha256}
            SHOW_PROGRESS
            STATUS _status)
        list(GET _status 0 _code)
        if(NOT _code EQUAL 0)
            list(GET _status 1 _msg)
            file(REMOVE "${dest}")
            message(FATAL_ERROR "Download failed (${_msg}): ${url}\n"
                "If this machine is offline, download the file manually to ${dest} "
                "or configure with -DXDNA_RVC_ORT_FLAVOR=custom -DXDNA_RVC_ORT_ROOT=<dir>.")
        endif()
    endif()
endfunction()

set(_ort_runtime_files "")

if(_flavor STREQUAL "ryzenai")
    if(NOT WIN32)
        message(FATAL_ERROR "XDNA_RVC_ORT_FLAVOR=ryzenai is only supported on Windows. "
            "AMD's Ryzen AI deployment package (VitisAI EP DLLs) is Windows-only.")
    endif()
    if(NOT DEFINED ENV{RYZEN_AI_INSTALLATION_PATH})
        message(FATAL_ERROR
            "XDNA_RVC_ORT_FLAVOR=ryzenai requires the RYZEN_AI_INSTALLATION_PATH environment variable.\n"
            "Install AMD Ryzen AI Software (1.8.0 or newer) and the NPU driver (>= 32.0.203.280), "
            "then reopen your terminal so the variable is visible. See docs/amd_xdna_setup.md.")
    endif()
    file(TO_CMAKE_PATH "$ENV{RYZEN_AI_INSTALLATION_PATH}" _rai)
    set(_ort_root "${_rai}/onnxruntime")
    # Layout used by AMD's own C++ samples (RyzenAI-SW CNN-examples/.../CMakeLists.txt).
    set(_ort_include "${_ort_root}/include/onnxruntime/core/session")
    set(_ort_lib "${_ort_root}/lib/onnxruntime.lib")
    if(NOT EXISTS "${_ort_include}/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "Ryzen AI ONNX Runtime headers not found at ${_ort_include}. "
            "Is RYZEN_AI_INSTALLATION_PATH (${_rai}) pointing at a complete Ryzen AI install?")
    endif()
    if(NOT EXISTS "${_ort_lib}")
        message(FATAL_ERROR "Ryzen AI onnxruntime.lib not found at ${_ort_lib}.")
    endif()
    # Deployment DLLs listed in the Ryzen AI 1.8 "Application Packaging Requirements".
    # Not every file exists in every release; missing optional ones are skipped.
    foreach(_dll
            onnxruntime.dll onnxruntime_providers_shared.dll onnxruntime_providers_vitisai.dll
            onnxruntime_vitisai_ep.dll dyn_dispatch_core.dll aiecompiler_client.dll
            onnxruntime_vitis_ai_custom_ops.dll DirectML.dll vaiml.dll
            ryzenai_onnx_utils.dll zlib.dll zstd.dll)
        if(EXISTS "${_rai}/deployment/${_dll}")
            list(APPEND _ort_runtime_files "${_rai}/deployment/${_dll}")
        endif()
    endforeach()
    if(NOT EXISTS "${_rai}/deployment/onnxruntime_providers_vitisai.dll")
        message(WARNING "onnxruntime_providers_vitisai.dll was not found in ${_rai}/deployment. "
            "The VitisAI execution provider will be unavailable at runtime.")
    endif()

elseif(_flavor STREQUAL "directml")
    if(NOT WIN32)
        message(FATAL_ERROR "XDNA_RVC_ORT_FLAVOR=directml is only supported on Windows.")
    endif()
    set(_pkg "${_ort_download_dir}/Microsoft.ML.OnnxRuntime.DirectML.${_ort_dml_version}.nupkg")
    _xdna_rvc_download(
        "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/${_ort_dml_version}"
        "57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681" "${_pkg}")
    set(_dml_pkg "${_ort_download_dir}/Microsoft.AI.DirectML.${_directml_version}.nupkg")
    _xdna_rvc_download(
        "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/${_directml_version}"
        "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9" "${_dml_pkg}")
    set(_ort_root "${_ort_download_dir}/ort-dml-${_ort_dml_version}")
    if(NOT EXISTS "${_ort_root}/build/native/include/onnxruntime_cxx_api.h")
        file(ARCHIVE_EXTRACT INPUT "${_pkg}" DESTINATION "${_ort_root}")
    endif()
    set(_dml_root "${_ort_download_dir}/directml-${_directml_version}")
    if(NOT EXISTS "${_dml_root}/bin/x64-win/DirectML.dll")
        file(ARCHIVE_EXTRACT INPUT "${_dml_pkg}" DESTINATION "${_dml_root}")
    endif()
    set(_ort_include "${_ort_root}/build/native/include")
    set(_ort_lib "${_ort_root}/runtimes/win-x64/native/onnxruntime.lib")
    list(APPEND _ort_runtime_files
        "${_ort_root}/runtimes/win-x64/native/onnxruntime.dll"
        "${_ort_root}/runtimes/win-x64/native/onnxruntime_providers_shared.dll"
        "${_dml_root}/bin/x64-win/DirectML.dll")

elseif(_flavor STREQUAL "cpu")
    if(WIN32)
        set(_name "onnxruntime-win-x64-${_ort_version}")
        set(_archive "${_ort_download_dir}/${_name}.zip")
        _xdna_rvc_download(
            "https://github.com/microsoft/onnxruntime/releases/download/v${_ort_version}/${_name}.zip"
            "c6ba983baf5681af108599675d2a89c2d145512d02de28aed0bff177cd0ba949" "${_archive}")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(_name "onnxruntime-linux-x64-${_ort_version}")
        set(_archive "${_ort_download_dir}/${_name}.tgz")
        _xdna_rvc_download(
            "https://github.com/microsoft/onnxruntime/releases/download/v${_ort_version}/${_name}.tgz"
            "a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd" "${_archive}")
    else()
        message(FATAL_ERROR "No prebuilt ONNX Runtime download is configured for "
            "${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}. Use -DXDNA_RVC_ORT_FLAVOR=custom.")
    endif()
    if(NOT EXISTS "${_ort_download_dir}/${_name}/include/onnxruntime_cxx_api.h")
        file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_ort_download_dir}")
    endif()
    set(_ort_root "${_ort_download_dir}/${_name}")
    set(_ort_include "${_ort_root}/include")
    if(WIN32)
        set(_ort_lib "${_ort_root}/lib/onnxruntime.lib")
        list(APPEND _ort_runtime_files "${_ort_root}/lib/onnxruntime.dll"
                                       "${_ort_root}/lib/onnxruntime_providers_shared.dll")
    else()
        set(_ort_lib "${_ort_root}/lib/libonnxruntime.so")
        file(GLOB _sos "${_ort_root}/lib/libonnxruntime*.so*")
        list(APPEND _ort_runtime_files ${_sos})
    endif()

elseif(_flavor STREQUAL "custom")
    if(NOT XDNA_RVC_ORT_ROOT OR NOT EXISTS "${XDNA_RVC_ORT_ROOT}")
        message(FATAL_ERROR "XDNA_RVC_ORT_FLAVOR=custom requires -DXDNA_RVC_ORT_ROOT=<dir>.")
    endif()
    set(_ort_root "${XDNA_RVC_ORT_ROOT}")
    find_path(_ort_include onnxruntime_cxx_api.h
        PATHS "${_ort_root}/include" "${_ort_root}/include/onnxruntime"
              "${_ort_root}/include/onnxruntime/core/session" "${_ort_root}/build/native/include"
        NO_DEFAULT_PATH REQUIRED)
    find_library(_ort_lib NAMES onnxruntime
        PATHS "${_ort_root}/lib" "${_ort_root}/runtimes/win-x64/native" NO_DEFAULT_PATH REQUIRED)
    if(WIN32)
        file(GLOB _ort_runtime_files "${_ort_root}/lib/*.dll" "${_ort_root}/runtimes/win-x64/native/*.dll")
    else()
        file(GLOB _ort_runtime_files "${_ort_root}/lib/libonnxruntime*.so*")
    endif()
else()
    message(FATAL_ERROR "Unknown XDNA_RVC_ORT_FLAVOR '${XDNA_RVC_ORT_FLAVOR}'")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
if(WIN32)
    list(GET _ort_runtime_files 0 _ort_dll)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_IMPLIB "${_ort_lib}"
        IMPORTED_LOCATION "${_ort_dll}"
        INTERFACE_INCLUDE_DIRECTORIES "${_ort_include}")
else()
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${_ort_lib}"
        IMPORTED_NO_SONAME FALSE
        INTERFACE_INCLUDE_DIRECTORIES "${_ort_include}")
endif()

set(XDNA_RVC_ORT_FLAVOR_RESOLVED "${_flavor}")
set(XDNA_RVC_ORT_ROOT_RESOLVED "${_ort_root}")
set(XDNA_RVC_ORT_RUNTIME_FILES "${_ort_runtime_files}")

# Copies the ONNX Runtime runtime libraries next to an executable so that the
# exact package we compiled against is the one loaded (important on Windows, where
# C:\Windows\System32\onnxruntime.dll may otherwise be picked up).
function(xdna_rvc_copy_ort_runtime target)
    foreach(_f ${XDNA_RVC_ORT_RUNTIME_FILES})
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_f}" "$<TARGET_FILE_DIR:${target}>"
            VERBATIM)
    endforeach()
    if(NOT WIN32)
        set_target_properties(${target} PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
    endif()
endfunction()
