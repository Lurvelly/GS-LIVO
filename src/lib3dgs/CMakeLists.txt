cmake_minimum_required(VERSION 3.24)
project(3dgs LANGUAGES CUDA CXX)

get_filename_component(GS_LIVO_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

include(ProcessorCount)
ProcessorCount(total_cores)
if(NOT DEFINED total_cores OR total_cores EQUAL 0)
  set(total_cores 1)
  set(used_cores 1)
elseif(total_cores GREATER 2)
  math(EXPR used_cores "${total_cores} - 2")
else()
  set(used_cores ${total_cores})
endif()
set(ENV{MAKEFLAGS} "-j${used_cores}")
message(STATUS "Building lib3dgs with ${used_cores} out of ${total_cores} available cores")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS "11.7")
  message(FATAL_ERROR "GS-LIVO requires CUDA 11.7 or higher")
endif()

find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(yaml-cpp REQUIRED)
if(TARGET yaml-cpp::yaml-cpp)
  set(GS_LIVO_YAML_CPP_LIB yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
  set(GS_LIVO_YAML_CPP_LIB yaml-cpp)
elseif(YAML_CPP_LIBRARIES)
  set(GS_LIVO_YAML_CPP_LIB ${YAML_CPP_LIBRARIES})
else()
  find_library(GS_LIVO_YAML_CPP_LIB NAMES yaml-cpp REQUIRED)
endif()

if(NOT TARGET simple-knn)
  add_subdirectory("${GS_LIVO_SOURCE_ROOT}/simple-knn" "${CMAKE_BINARY_DIR}/simple-knn")
endif()

if(NOT TARGET tinyply)
  set(BUILD_TESTS OFF CACHE BOOL "Build tinyply tests" FORCE)
  add_subdirectory("${GS_LIVO_SOURCE_ROOT}/tinyply" "${CMAKE_BINARY_DIR}/tinyply")
endif()

find_package(Eigen3 3.4 QUIET)
if(NOT TARGET Eigen3::Eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${GS_LIVO_SOURCE_ROOT}/eigen"
  )
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
  set(JSON_BuildTests OFF CACHE BOOL "Build nlohmann_json tests" FORCE)
  set(JSON_Install OFF CACHE BOOL "Install nlohmann_json" FORCE)
  add_subdirectory("${GS_LIVO_SOURCE_ROOT}/json" "${CMAKE_BINARY_DIR}/nlohmann_json")
endif()

if(NOT TARGET glm::glm)
  set(GLM_TEST_ENABLE OFF CACHE BOOL "Build glm tests" FORCE)
  add_subdirectory("${GS_LIVO_SOURCE_ROOT}/glm" "${CMAKE_BINARY_DIR}/glm")
endif()

set(HEADERS
  includes/read_utils.cuh
  includes/gaussian.cuh
  includes/camera.cuh
  includes/loss_monitor.cuh
  includes/image.cuh
  includes/stb_image.h
  includes/stb_image_resize.h
  includes/point_cloud.cuh
  includes/scene.cuh
  includes/scene_info.cuh
  includes/camera_utils.cuh
  includes/general_utils.cuh
  includes/sh_utils.cuh
  includes/parameters.cuh
  includes/loss_utils.cuh
  includes/render_utils.cuh
  includes/rasterizer.cuh
  includes/rasterize_points.cuh
  includes/camera_info.cuh
  includes/debug_utils.cuh
  cuda_rasterizer/backward.h
  cuda_rasterizer/forward.h
  cuda_rasterizer/auxiliary.h
  cuda_rasterizer/rasterizer_impl.h
  cuda_rasterizer/rasterizer.h
  cuda_rasterizer/serialization.h
)

set(SOURCES
  src/read_utils.cu
  src/gaussian.cu
  src/camera.cu
  src/loss_monitor.cu
  src/image.cu
  src/scene.cu
  src/camera_utils.cu
  src/general_utils.cu
  src/parameters.cu
  src/rasterizer.cu
  src/rasterize_points.cu
  cuda_rasterizer/backward.cu
  cuda_rasterizer/forward.cu
  cuda_rasterizer/rasterizer_impl.cu
)

add_library(3dgs_lib SHARED ${SOURCES} ${HEADERS})
set_target_properties(3dgs_lib PROPERTIES
  CUDA_ARCHITECTURES native
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

target_include_directories(3dgs_lib PUBLIC
  ${PROJECT_SOURCE_DIR}/includes
  ${PROJECT_SOURCE_DIR}/cuda_rasterizer
  ${GS_LIVO_SOURCE_ROOT}/simple-knn
  ${GS_LIVO_SOURCE_ROOT}/tinyply/source
  ${OpenCV_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(3dgs_lib
  PUBLIC
  ${TORCH_LIBRARIES}
  CUDA::cudart
  CUDA::curand
  CUDA::cublas
  simple-knn
  tinyply
  Eigen3::Eigen
  nlohmann_json::nlohmann_json
  glm::glm
  Threads::Threads
  ${GS_LIVO_YAML_CPP_LIB}
  PRIVATE
  ${OpenCV_LIBS}
)

install(TARGETS 3dgs_lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/includes/ DESTINATION include)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/cuda_rasterizer/ DESTINATION include)
