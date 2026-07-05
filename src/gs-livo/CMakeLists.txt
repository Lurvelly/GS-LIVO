cmake_minimum_required(VERSION 3.16)
project(fast_livo LANGUAGES CUDA CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")

find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
  message(FATAL_ERROR "CUDA not found. Install CUDA Toolkit before building GS-LIVO.")
endif()
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")

set(PYTHON_EXECUTABLE /usr/bin/python3 CACHE FILEPATH "Python executable used by catkin")
find_package(PythonLibs REQUIRED)

get_filename_component(GS_LIVO_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(GS_LIVO_LIBTORCH_DIR "" CACHE PATH "Path to a libtorch or Python torch installation prefix")
if(GS_LIVO_LIBTORCH_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${GS_LIVO_LIBTORCH_DIR}")
  if(EXISTS "${GS_LIVO_LIBTORCH_DIR}/share/cmake/Torch/TorchConfig.cmake")
    set(Torch_DIR "${GS_LIVO_LIBTORCH_DIR}/share/cmake/Torch" CACHE PATH "Torch CMake package directory" FORCE)
  elseif(EXISTS "${GS_LIVO_LIBTORCH_DIR}/torch/share/cmake/Torch/TorchConfig.cmake")
    set(Torch_DIR "${GS_LIVO_LIBTORCH_DIR}/torch/share/cmake/Torch" CACHE PATH "Torch CMake package directory" FORCE)
  endif()
endif()

if(NOT Torch_DIR)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import torch, sys; sys.stdout.write(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE _torch_python_result
    OUTPUT_VARIABLE _torch_python_prefix
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(_torch_python_result EQUAL 0 AND _torch_python_prefix)
    list(APPEND CMAKE_PREFIX_PATH "${_torch_python_prefix}")
    message(STATUS "Using Torch CMake prefix from Python: ${_torch_python_prefix}")
  endif()
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64|ARM|AARCH64)")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -ffast-math")
    message(STATUS "Using 64-bit ARM optimizations")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
    message(STATUS "Using 32-bit ARM optimizations")
  endif()
  add_definitions(-DARM_ARCH)
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops")
  message(STATUS "Using x86 optimizations")
  add_definitions(-DX86_ARCH)
endif()

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

include(ProcessorCount)
ProcessorCount(N)
message(STATUS "Processor count: ${N}")
if(N GREATER 4)
  set(PROC_NUM 4)
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
elseif(N GREATER 1)
  set(PROC_NUM ${N})
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
else()
  set(PROC_NUM 1)
  add_definitions(-DMP_PROC_NUM=1)
endif()
message(STATUS "Mapping threads: ${PROC_NUM}")

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
  add_compile_options(${OpenMP_CXX_FLAGS})
endif()

find_package(mimalloc QUIET)
if(mimalloc_FOUND)
  message(STATUS "mimalloc found")
else()
  message(STATUS "mimalloc not found, proceeding without it")
endif()

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  message_generation
  eigen_conversions
  vikit_common
  vikit_ros
  cv_bridge
  image_transport
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)
find_library(Sophus_LIBRARY NAMES Sophus libSophus PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
if(Sophus_LIBRARY)
  set(Sophus_LIBRARIES ${Sophus_LIBRARY})
elseif(TARGET Sophus::Sophus)
  set(Sophus_LIBRARIES Sophus::Sophus)
endif()

set(GS_LIVO_LIB3DGS_DIR "${GS_LIVO_SOURCE_ROOT}/lib3dgs" CACHE PATH "Path to the bundled lib3dgs source")
if(NOT EXISTS "${GS_LIVO_LIB3DGS_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "lib3dgs source not found at ${GS_LIVO_LIB3DGS_DIR}")
endif()
if(NOT TARGET 3dgs_lib)
  add_subdirectory("${GS_LIVO_LIB3DGS_DIR}" "${CMAKE_BINARY_DIR}/lib3dgs")
endif()

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime cv_bridge vikit_common vikit_ros image_transport
  DEPENDS EIGEN3 PCL OpenCV Sophus
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${GS_LIVO_LIB3DGS_DIR}/includes
  ${GS_LIVO_LIB3DGS_DIR}/cuda_rasterizer
  ${GS_LIVO_SOURCE_ROOT}/simple-knn
  ${GS_LIVO_SOURCE_ROOT}/tinyply/source
  ${GS_LIVO_SOURCE_ROOT}/json/include
  ${GS_LIVO_SOURCE_ROOT}/glm
  ${GS_LIVO_SOURCE_ROOT}/eigen
  include
)

add_library(vio src/vio.cpp src/frame.cpp src/visual_point.cpp)
target_include_directories(vio PUBLIC
  ${GS_LIVO_LIB3DGS_DIR}/includes
  ${GS_LIVO_LIB3DGS_DIR}/cuda_rasterizer
  ${GS_LIVO_SOURCE_ROOT}/simple-knn
  ${GS_LIVO_SOURCE_ROOT}/tinyply/source
)
target_link_libraries(vio PUBLIC 3dgs_lib ${Sophus_LIBRARIES} ${TORCH_LIBRARIES})

add_library(lio src/voxel_map.cpp)
add_library(pre src/preprocess.cpp)
add_library(imu_proc src/IMU_Processing.cpp)
add_library(laser_mapping src/LIVMapper.cpp)
target_link_libraries(laser_mapping PUBLIC 3dgs_lib ${Sophus_LIBRARIES} ${TORCH_LIBRARIES})

add_executable(fastlivo_mapping src/main.cpp)
target_link_libraries(fastlivo_mapping
  laser_mapping
  vio
  lio
  pre
  imu_proc
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
  ${TORCH_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fastlivo_mapping OpenMP::OpenMP_CXX)
endif()

if(mimalloc_FOUND)
  target_link_libraries(fastlivo_mapping mimalloc)
endif()
