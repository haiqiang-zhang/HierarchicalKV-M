cmake_minimum_required(VERSION 3.18)
project(merlin_hashtable_python LANGUAGES CXX CUDA)


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


find_package(Python REQUIRED COMPONENTS Interpreter Development)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

include_directories(
    ../include
)



# Define the Python module
pybind11_add_module(merlin_hashtable_python bindings.cc.cu)
target_compile_features(merlin_hashtable_python PUBLIC cxx_std_17 cuda_std_17)
set_target_properties(merlin_hashtable_python PROPERTIES CUDA_ARCHITECTURES OFF)


execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
  OUTPUT_VARIABLE PYTHON_SITE
  OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "PYTHON_SITE: ${PYTHON_SITE}")


# install the module to python site-packages
install(TARGETS merlin_hashtable_python DESTINATION ${PYTHON_SITE}/merlin_hashtable_python)