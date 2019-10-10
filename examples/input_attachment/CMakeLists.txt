cmake_minimum_required(VERSION 3.12)

message(STATUS "Input Attachment example included")

add_executable(input_attachment ${CMAKE_CURRENT_LIST_DIR}/input_attachment.cpp)
target_link_libraries(input_attachment common)