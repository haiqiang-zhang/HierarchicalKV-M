cmake_minimum_required(VERSION 3.18)
project(merlin_ycsb LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)





foreach(cuda_arch ${sm})
  list(APPEND cuda_arch_list ${cuda_arch})
  message(STATUS "Assign GPU architecture (sm=${cuda_arch})")
endforeach()

list(LENGTH cuda_arch_list cuda_arch_list_length)
if(cuda_arch_list_length EQUAL 0)
  list(APPEND cuda_arch_list "86")
  message(STATUS "Assign default GPU architecture sm=86")
endif()

foreach(cuda_arch ${cuda_arch_list})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${cuda_arch},code=sm_${cuda_arch}")
endforeach()

find_package(nlohmann_json 3.12.0 REQUIRED)

include_directories(../include)

# Create a library from ycsb.cc.cu that both executables can use
add_library(ycsb_lib STATIC ycsb.cc.cu)
target_compile_features(ycsb_lib PUBLIC cxx_std_17 cuda_std_17)
set_target_properties(ycsb_lib PROPERTIES CUDA_ARCHITECTURES OFF)

# Main YCSB benchmark executable
add_executable(single_ycsb single_ycsb.cc.cu)
target_compile_features(single_ycsb PUBLIC cxx_std_17 cuda_std_17)
set_target_properties(single_ycsb PROPERTIES CUDA_ARCHITECTURES OFF)
target_link_libraries(single_ycsb PRIVATE ycsb_lib)

# Batch YCSB benchmark executable
add_executable(batch_ycsb batch_ycsb.cc.cu)
target_compile_features(batch_ycsb PUBLIC cxx_std_17 cuda_std_17)
set_target_properties(batch_ycsb PROPERTIES CUDA_ARCHITECTURES OFF)
target_link_libraries(batch_ycsb PRIVATE ycsb_lib)
target_link_libraries(batch_ycsb PRIVATE nlohmann_json::nlohmann_json)