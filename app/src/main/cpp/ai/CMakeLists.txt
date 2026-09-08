# AI JNI：llama.cpp（Vulkan GPU）JNI 封装
cmake_minimum_required(VERSION 3.22)
project(ai_jni CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp / ggml 构建选项：只要 llama 静态库 + Vulkan 后端
set(LLAMA_BUILD_IS_DEV        OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON        OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS         OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS         OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER        OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP           OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL                OFF CACHE BOOL "" FORCE)
set(LLAMA_METAL               OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS         OFF CACHE BOOL "" FORCE)

# Vulkan：由 ggml 的 add_subdirectory(vendor) 内部编译 shader
set(GGML_VULKAN               ON  CACHE BOOL "" FORCE)
# NDK 自带 glslc（shader-tools），显式指定避免 FindVulkan 找不到
if(NOT Vulkan_GLSLC_EXECUTABLE)
    if(EXISTS "$ENV{ANDROID_NDK}/shader-tools/windows-x86_64/glslc.exe")
        set(Vulkan_GLSLC_EXECUTABLE "$ENV{ANDROID_NDK}/shader-tools/windows-x86_64/glslc.exe")
    elseif(EXISTS "C:/Users/29268/AppData/Local/Android/Sdk/ndk/27.1.12297006/shader-tools/windows-x86_64/glslc.exe")
        set(Vulkan_GLSLC_EXECUTABLE "C:/Users/29268/AppData/Local/Android/Sdk/ndk/27.1.12297006/shader-tools/windows-x86_64/glslc.exe")
    endif()
endif()
# vulkan.hpp 等头文件来自 vendored Vulkan-Headers（NDK 只有 C 头）
set(Vulkan_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-headers/include" CACHE PATH "" FORCE)
# 满足 ggml-vulkan 的 find_package(SPIRV-Headers) 检查（include 由下方 include_directories 提供）
set(SPIRV-Headers_DIR "${CMAKE_CURRENT_SOURCE_DIR}/spirv-headers-config" CACHE PATH "" FORCE)
# host 端 vulkan-shaders-gen 强制用 MSVC（MinGW fork() 多线程死锁）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/host-msvc.toolchain.cmake")
    set(GGML_VULKAN_SHADERS_GEN_TOOLCHAIN "${CMAKE_CURRENT_SOURCE_DIR}/host-msvc.toolchain.cmake" CACHE FILEPATH "" FORCE)
endif()
# 优先使用仓库内置的预编译 host 工具（见 prebuilt/），跳过现场编译
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/vulkan-shaders-gen.exe")
    set(GGML_VULKAN_SHADERS_GEN_PREBUILT "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/vulkan-shaders-gen.exe" CACHE FILEPATH "" FORCE)
endif()
# ExternalProject 配置 host 工具时显式传入 ninja（gradle 环境无全局 PATH）
list(APPEND VULKAN_SHADER_GEN_CMAKE_ARGS "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}")

# ggml-vulkan 的 spirv.hpp 直接指向 vendored SPIRV-Headers
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/spirv-headers-include")

add_subdirectory(llama.cpp)

add_library(ai_llm SHARED ai_llm_jni.cpp)
target_link_libraries(ai_llm PRIVATE llama android log)
