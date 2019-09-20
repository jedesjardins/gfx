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

add_library(Gfx INTERFACE)
target_include_directories(Gfx
	INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
	$<BUILD_INTERFACE:${rapidjson_SOURCE_DIR}/include>
	$<BUILD_INTERFACE:${stb_SOURCE_DIR}>
)

find_package(vulkan REQUIRED)
find_package(glfw3 REQUIRED)

target_link_libraries(Gfx INTERFACE
	JED::cmd
	Vulkan::Vulkan
	glfw
)

add_library(JED::gfx ALIAS Gfx)

if(GFX_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

if(GFX_BUILD_TESTS)
	add_subdirectory(tests)
endif()