cmake_minimum_required(VERSION 3.18.1)
project(crrc_ncnn LANGUAGES CXX)

if(NOT DEFINED CRRC_NCNN_ANDROID_ROOT)
    message(FATAL_ERROR "CRRC_NCNN_ANDROID_ROOT is required")
endif()

set(ncnn_DIR
    "${CRRC_NCNN_ANDROID_ROOT}/${ANDROID_ABI}/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

add_library(crrc_ncnn SHARED ncnn_fastener_jni.cpp)
if(CRRC_NCNN_EXACT_MATH)
    target_compile_definitions(crrc_ncnn PRIVATE CRRC_NCNN_EXACT_MATH=1)
endif()
target_compile_features(crrc_ncnn PRIVATE cxx_std_11)
target_compile_options(crrc_ncnn PRIVATE -Wall -Wextra -Werror)
target_link_libraries(crrc_ncnn PRIVATE ncnn android log)
install(TARGETS crrc_ncnn LIBRARY DESTINATION .)
