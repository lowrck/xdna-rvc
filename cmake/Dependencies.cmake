# Third-party dependencies, fetched at pinned versions with FetchContent.
# Licenses are recorded in THIRD_PARTY_LICENSES.md.

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# --- spdlog (MIT), bundled fmt -------------------------------------------------
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE
    SYSTEM)
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)

# --- nlohmann/json (MIT) -------------------------------------------------------
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    GIT_SHALLOW TRUE
    SYSTEM)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)

# --- CLI11 (BSD-3-Clause) ------------------------------------------------------
FetchContent_Declare(CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.7.2
    GIT_SHALLOW TRUE
    SYSTEM)
set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(CLI11_INSTALL OFF CACHE BOOL "" FORCE)

# --- miniaudio (public domain / MIT-0) -----------------------------------------
# Used for device I/O (WASAPI shared/exclusive on Windows; ALSA/PulseAudio/JACK on
# Linux for development) and WAV file decode/encode.
FetchContent_Declare(miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG 0.11.25
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR _no_cmake_)   # we only need the single header

FetchContent_MakeAvailable(spdlog nlohmann_json CLI11 miniaudio)

add_library(xdna_rvc_miniaudio STATIC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/miniaudio_impl.c")
target_include_directories(xdna_rvc_miniaudio SYSTEM PUBLIC "${miniaudio_SOURCE_DIR}")
target_compile_definitions(xdna_rvc_miniaudio PUBLIC MA_NO_GENERATION MA_NO_ENGINE MA_NO_NODE_GRAPH MA_NO_RESOURCE_MANAGER)
if(UNIX AND NOT APPLE)
    target_link_libraries(xdna_rvc_miniaudio PUBLIC ${CMAKE_DL_LIBS} m Threads::Threads)
endif()

# --- doctest (MIT) for unit tests --------------------------------------------------
if(XDNA_RVC_BUILD_TESTS)
    FetchContent_Declare(doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        GIT_TAG v2.5.3
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR _no_cmake_
        SYSTEM)
    FetchContent_MakeAvailable(doctest)
    add_library(xdna_rvc_doctest INTERFACE)
    target_include_directories(xdna_rvc_doctest SYSTEM INTERFACE "${doctest_SOURCE_DIR}")
endif()

# --- GUI: Dear ImGui (MIT) + GLFW (zlib) + OpenGL --------------------------------
if(XDNA_RVC_BUILD_GUI)
    find_package(OpenGL)
    if(NOT OpenGL_FOUND)
        message(WARNING "OpenGL not found; disabling the GUI (XDNA_RVC_BUILD_GUI=OFF).")
        set(XDNA_RVC_BUILD_GUI OFF CACHE BOOL "" FORCE)
    endif()
endif()

if(XDNA_RVC_BUILD_GUI)
    find_package(glfw3 3.3 QUIET)
    if(NOT glfw3_FOUND)
        FetchContent_Declare(glfw
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG 3.5.1
            GIT_SHALLOW TRUE
            SYSTEM)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(glfw)
    endif()

    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.92.9b
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR _no_cmake_)
    FetchContent_MakeAvailable(imgui)

    add_library(xdna_rvc_imgui STATIC
        "${imgui_SOURCE_DIR}/imgui.cpp"
        "${imgui_SOURCE_DIR}/imgui_draw.cpp"
        "${imgui_SOURCE_DIR}/imgui_tables.cpp"
        "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
        "${imgui_SOURCE_DIR}/imgui_demo.cpp"
        "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
        "${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp")
    target_include_directories(xdna_rvc_imgui SYSTEM PUBLIC
        "${imgui_SOURCE_DIR}" "${imgui_SOURCE_DIR}/backends")
    target_link_libraries(xdna_rvc_imgui PUBLIC glfw OpenGL::GL)
endif()
