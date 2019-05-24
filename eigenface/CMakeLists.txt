cmake_minimum_required(VERSION 3.11)

project(engenface)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DRS})
add_executable(main eigenface.cpp)
target_link_libraries(main ${OpenCV_LIBS})