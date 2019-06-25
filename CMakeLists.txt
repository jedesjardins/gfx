cmake_minimum_required(VERSION 3.12)
project(gfx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GFX_BUILD_EXAMPLES "Build documentation examples" OFF)
option(GFX_BUILD_TESTS "Build documentation tests" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "DEBUG" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

include(FetchContent)

FetchContent_Declare(
	Cmd
	GIT_REPOSITORY "https://github.com/jedesjardins/cmd"
	GIT_TAG develop
)

FetchContent_GetProperties(Cmd)
if(NOT Cmd_POPULATED)
	FetchContent_Populate(Cmd)
	message(STATUS "Cmd source dir: ${cmd_SOURCE_DIR}")
	message(STATUS "Cmd binary dir: ${cmd_BINARY_DIR}")
	message(STATUS "Cmd include dir: ${cmd_SOURCE_DIR}/include")
	add_subdirectory(${cmd_SOURCE_DIR} ${cmd_BINARY_DIR})
endif()

#[[
FetchContent_Declare(
	glfw
	GIT_REPOSITORY "https://github.com/glfw/glfw"
	GIT_TAG latest
)

FetchContent_GetProperties(glfw)
if(NOT glfw_POPULATED)
	FetchContent_Populate(glfw)
	message(STATUS "glfw source dir: ${glfw_SOURCE_DIR}")
	message(STATUS "glfw binary dir: ${glfw_BINARY_DIR}")
	message(STATUS "glfw include dir: ${glfw_SOURCE_DIR}/include")

	set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
	set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
	set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
	add_subdirectory(${glfw_SOURCE_DIR} ${glfw_BINARY_DIR})
endif()
]]

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
	rapidjson
	GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
)

FetchContent_GetProperties(rapidjson)
if(NOT rapidjson_POPULATED)
	FetchContent_Populate(rapidjson)
	message(STATUS "rapidjson source dir: ${rapidjson_SOURCE_DIR}")
	message(STATUS "rapidjson binary dir: ${rapidjson_BINARY_DIR}")
	message(STATUS "rapidjson include dir: ${rapidjson_SOURCE_DIR}/include")

	set(RAPIDJSON_BUILD_DOC OFF CACHE BOOL "" FORCE)
	set(RAPIDJSON_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
	set(RAPIDJSON_BUILD_TESTS OFF CACHE BOOL "" FORCE)
	add_subdirectory(${rapidjson_SOURCE_DIR} ${rapidjson_BINARY_DIR})
endif()

add_library(Gfx INTERFACE)
target_include_directories(Gfx
	INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
	$<BUILD_INTERFACE:${rapidjson_SOURCE_DIR}/include>
)

find_package(vulkan REQUIRED)
find_package(glfw3 REQUIRED)

target_link_libraries(Gfx INTERFACE
	JED::cmd
	Vulkan::Vulkan
	glfw
	glm
)

add_library(JED::gfx ALIAS Gfx)

if(GFX_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

if(GFX_BUILD_TESTS)
	add_subdirectory(tests)
endif()