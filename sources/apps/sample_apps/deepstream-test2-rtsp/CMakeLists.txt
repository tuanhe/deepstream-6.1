cmake_minimum_required(VERSION 3.10)

project(deepstream-test2-rtsp)

set(CUDA_VER 11.6)
set(NVDS_VERSION 6.1)

message("-- DeepStream Version: ${NVDS_VERSION}")
message("-- CMAKE_HOST_SYSTEM_PROCESSOR: ${CMAKE_HOST_SYSTEM_PROCESSOR}")

if("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
    message("-- aarch64 Jetson TEGRA")
    set(CMAKE_C_STANDARD 11) 
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPLATFORM_TEGRA")
    # If you want to use C++ on Jetson, rename the source file from .c to .cpp
    # set(CMAKE_CXX_STANDARD 11)
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPLATFORM_TEGRA")

elseif("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    message("-- X86 dGPU")
    set(CMAKE_CXX_STANDARD 17)
    # set(CMAKE_C_STANDARD 11)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(DSAPP REQUIRED gstreamer-1.0)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/../../../includes
        /usr/local/cuda-${CUDA_VER}/include
        ${DSAPP_INCLUDE_DIRS}
)

link_directories(
        /usr/local/cuda-${CUDA_VER}/lib64
        /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/
)

AUX_SOURCE_DIRECTORY(.  ds_app_source_files)
#AUX_SOURCE_DIRECTORY(../../apps-common/src  ds_app_common_source_files)
#AUX_SOURCE_DIRECTORY(../../apps-common/src/deepstream-yaml  ds_app_common_yaml_source_files)

list(APPEND ds_app_sources ${ds_app_source_files} )

add_executable(${CMAKE_PROJECT_NAME} ${ds_app_sources})

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${DSAPP_LIBRARIES}
                                                     cudart 
                                                     nvdsgst_meta 
                                                     nvds_meta 
                                                     nvdsgst_helper
                                                     nvdsgst_smartrecord 
                                                     nvds_utils 
                                                     nvds_msgbroker 
                                                     m
                                                     yaml-cpp
                                                     cuda
                                                     gstrtspserver-1.0
                                                     nvds_yml_parser)
