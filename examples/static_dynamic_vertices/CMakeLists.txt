cmake_minimum_required(VERSION 3.12)

message(STATUS "Static and Dynamic Vertices example included")

add_executable(variable_vertex_buffers ${CMAKE_CURRENT_LIST_DIR}/variable_vertex_buffers.cpp)
target_link_libraries(variable_vertex_buffers common)