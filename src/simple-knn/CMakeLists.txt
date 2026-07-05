cmake_minimum_required(VERSION 3.18)
project(simple_knn LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

add_library(simple-knn STATIC
  spatial.cu
  simple_knn.cu
)

set_target_properties(simple-knn PROPERTIES
  CUDA_ARCHITECTURES native
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

target_include_directories(simple-knn PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(simple-knn PUBLIC
  ${TORCH_LIBRARIES}
  CUDA::cudart
)
