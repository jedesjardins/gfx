cmake_minimum_required(VERSION 3.12)

message(STATUS "Static and Dynamic Vertices example included")

FetchContent_Declare(
	stb
	GIT_REPOSITORY https://github.com/nothings/stb.git
)

FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
	FetchContent_Populate(stb)
	message(STATUS "stb source dir: ${stb_SOURCE_DIR}")
	message(STATUS "stb binary dir: ${stb_BINARY_DIR}")
	message(STATUS "stb include dir: ${stb_SOURCE_DIR}")
endif()

add_executable(variable_vertex_buffers ${CMAKE_CURRENT_LIST_DIR}/variable_vertex_buffers.cpp)
target_include_directories(variable_vertex_buffers PUBLIC $<BUILD_INTERFACE:${stb_SOURCE_DIR}>)
target_link_libraries(variable_vertex_buffers common)