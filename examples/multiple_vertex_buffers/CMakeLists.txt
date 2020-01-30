cmake_minimum_required(VERSION 3.12)

message(STATUS "Mulitple Vertex Buffers example included")

add_executable(mul_vbs ${CMAKE_CURRENT_LIST_DIR}/mul_vbs.cpp)
target_link_libraries(mul_vbs common)