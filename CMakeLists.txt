cmake_minimum_required(VERSION 3.14)
project(matrix_mult)

find_package(OpenMP REQUIRED)
include_directories(include)
add_executable(main main.cpp src/matrix_mul.cpp)
target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
set(CMAKE_CXX_FLAGS "-O3")