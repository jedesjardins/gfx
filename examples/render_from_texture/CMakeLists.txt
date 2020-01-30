cmake_minimum_required(VERSION 3.12)

message(STATUS "Render from Texture example included")

add_executable(render_from_texture ${CMAKE_CURRENT_LIST_DIR}/render_from_texture.cpp)
target_link_libraries(render_from_texture common)