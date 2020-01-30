cmake_minimum_required(VERSION 3.12)

message(STATUS "Pong example included")

add_executable(pong ${CMAKE_CURRENT_LIST_DIR}/pong.cpp)
target_link_libraries(pong common)