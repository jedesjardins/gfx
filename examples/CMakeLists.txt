cmake_minimum_required(VERSION 3.12)
project(gfx_examples)

message(STATUS "Examples included")

# Modeled after Catch2 examples directory

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

# gather all example files
set(SOURCES_SINGLE_FILE
	# list test files here
	variable_vertex_buffers.cpp
	pong.cpp
	mul_vbs.cpp
)

# get base names of each file (strips off .cpp)
string(REPLACE ".cpp" "" BASENAMES_SINGLE_FILE "${SOURCES_SINGLE_FILE}")

# gather list of single file targets
set(TARGETS_SINGLE_FILE ${BASENAMES_SINGLE_FILE} )

add_compile_definitions(RESOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/")
message(STATUS "current source dir: ${CMAKE_CURRENT_SOURCE_DIR}")

foreach(name ${TARGETS_SINGLE_FILE})
    add_executable(${name} ${CMAKE_CURRENT_LIST_DIR}/${name}.cpp)
    target_include_directories(${name} PUBLIC $<BUILD_INTERFACE:${stb_SOURCE_DIR}>)
    target_link_libraries(${name} JED::gfx JED::log glm)
endforeach()