cmake_minimum_required(VERSION 3.10)

project(dsrclpublisher)

set(CUDA_VER 11.4)
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
pkg_check_modules(DSRCL REQUIRED gstreamer-1.0 gstreamer-base-1.0)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ../../includes
        /usr/local/cuda-${CUDA_VER}/include
        ${DSRCL_INCLUDE_DIRS}
        /usr/lib/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux-gnu/glib-2.0/include
        /opt/ros/foxy/include
)

link_directories(
        /usr/local/cuda-${CUDA_VER}/lib64
        /opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib/
        /opt/ros/foxy/lib
)

AUX_SOURCE_DIRECTORY(.  gst_dsrcl_source_files)

list(APPEND gst_dsrcl_source ${gst_dsrcl_source_files})

add_library(${CMAKE_PROJECT_NAME} SHARED ${gst_dsrcl_source})

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${DSAPP_LIBRARIES}
                                                     cudart 
                                                     dl
                                                     nppc
                                                     nppig
                                                     npps
                                                     nppicc
                                                     nppidei
                                                     nvdsgst_meta 
                                                     nvds_meta 
                                                     nvdsgst_helper
                                                     nvdsgst_smartrecord 
                                                     nvds_utils 
                                                     nvds_msgbroker
                                                     nvbufsurface
                                                     nvbufsurftransform
                                                     rclcpp
                                                     gstbase-1.0
                                                     )