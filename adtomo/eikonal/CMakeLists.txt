cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(eik_test2D)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(eik2D Eikonal2D.cpp)
target_link_libraries(eik2D "${TORCH_LIBRARIES}")
set_property(TARGET eik2D PROPERTY CXX_STANDARD 17)
