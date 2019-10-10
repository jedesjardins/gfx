cmake_minimum_required(VERSION 3.12)

message(STATUS "Resize example included")

add_executable(resize ${CMAKE_CURRENT_LIST_DIR}/resize.cpp)
target_link_libraries(resize common)