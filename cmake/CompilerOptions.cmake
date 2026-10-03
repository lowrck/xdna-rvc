# Common compiler settings applied through the xdna_rvc_options interface target.

add_library(xdna_rvc_options INTERFACE)

if(MSVC)
    target_compile_options(xdna_rvc_options INTERFACE
        /W4 /permissive- /utf-8 /Zc:__cplusplus /MP
        /wd4100   # unreferenced formal parameter (common in callbacks)
    )
    target_compile_definitions(xdna_rvc_options INTERFACE
        NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS UNICODE _UNICODE)
    if(XDNA_RVC_WARNINGS_AS_ERRORS)
        target_compile_options(xdna_rvc_options INTERFACE /WX)
    endif()
else()
    target_compile_options(xdna_rvc_options INTERFACE
        -Wall -Wextra -Wpedantic -Wno-unused-parameter)
    if(XDNA_RVC_WARNINGS_AS_ERRORS)
        target_compile_options(xdna_rvc_options INTERFACE -Werror)
    endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(xdna_rvc_options INTERFACE Threads::Threads)
