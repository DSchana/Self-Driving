set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -lpigpio -lrt")
cmake_minimum_required(VERSION 2.8)
project( Playground )
find_package( OpenCV )
find_package( WiringPi )
find_package( Threads )
include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB CXX_EXEC "*.cpp")
add_executable( Playground ${CXX_EXEC} )

target_link_libraries( Playground ${OpenCV_LIBS} )
