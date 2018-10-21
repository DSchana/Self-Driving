set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -o -Z-reserved-lib-stdc++")
cmake_minimum_required(VERSION 2.8)
project(Playground)
find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB CXX_EXEC "main.cpp" "../*.cpp")

add_executable(Playground ${CXX_EXEC})
target_link_libraries(Playground ${OpenCV_LIBS})
