cmake_minimum_required(VERSION 3.12)
project(gfx_examples)

message(STATUS "Examples included")

FetchContent_Declare(
	Log
	GIT_REPOSITORY https://github.com/jedesjardins/log.git
)

FetchContent_GetProperties(Log)
if(NOT Log_POPULATED)
	FetchContent_Populate(Log)
	message(STATUS "Log source dir: ${log_SOURCE_DIR}")
	message(STATUS "Log binary dir: ${log_BINARY_DIR}")
	message(STATUS "Log include dir: ${log_SOURCE_DIR}/include")
	add_subdirectory(${log_SOURCE_DIR} ${log_BINARY_DIR})
endif()

FetchContent_Declare(
	glm
	GIT_REPOSITORY "https://github.com/g-truc/glm"
	GIT_TAG 0.9.9.5
)

FetchContent_GetProperties(glm)
if(NOT glm_POPULATED)
	FetchContent_Populate(glm)
	message(STATUS "glm source dir: ${glm_SOURCE_DIR}")
	message(STATUS "glm binary dir: ${glm_BINARY_DIR}")
	message(STATUS "glm include dir: ${glm_SOURCE_DIR}/include")

	set(BUILD_STATIC_LIBS OFF CACHE BOOL "" FORCE)
	set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
	set(GLM_TEST_ENABLE OFF CACHE BOOL "" FORCE)
	add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR})
endif()

add_compile_definitions(RESOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/")

add_library(common STATIC ${CMAKE_CURRENT_LIST_DIR}/common.cpp)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(common JED::gfx JED::log glm)

add_subdirectory(input_attachment)
add_subdirectory(multiple_vertex_buffers)
add_subdirectory(pong)
add_subdirectory(render_from_texture)
add_subdirectory(resize)
add_subdirectory(static_dynamic_vertices)
add_subdirectory(imgui)