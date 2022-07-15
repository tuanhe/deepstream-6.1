cmake_minimum_required(VERSION 3.5)
project(dsrcl)

set(CUDA_VER 11.4)
set(NVDS_VERSION 6.1)

message("-- DeepStream Version: ${NVDS_VERSION}")
message("-- CMAKE_HOST_SYSTEM_PROCESSOR: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message("-- PROJECT_SOURCE_DIR: - ${PROJECT_SOURCE_DIR}")

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(dsmsg REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(DSRCL REQUIRED gstreamer-1.0 gstreamer-base-1.0)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ../../../includes
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

set(ds_dependency_libraries
    ${DSAPP_LIBRARIES}
    nvdsgst_helper
    nvdsgst_meta
    nvds_meta
    nvbufsurface
    nvbufsurftransform
    gstbase-1.0
)

add_library(dsrclpublisher SHARED src/gstdsrclpublisher.cpp)
ament_target_dependencies(dsrclpublisher rclcpp dsmsg)
target_link_libraries(dsrclpublisher ${ds_dependency_libraries})

ament_package()
