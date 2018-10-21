set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -o -o33")
cmake_minimum_required(VERSION 2.8)
project(Playground)
find_package(OpenCV)

# Detect and add SFML
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})
find_package(SFML 2 REQUIRED COMPONENTS network audio graphics window system)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB CXX_EXEC "*.cpp")

add_executable(Playground ${CXX_EXEC})
target_link_libraries(Playground ${OpenCV_LIBS} ${SFML_LIBRARIES} ${SFML_DEPENDENCIES})
