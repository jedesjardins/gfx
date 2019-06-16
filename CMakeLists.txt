cmake_minimum_required(VERSION 3.12)
project(gfx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GFX_BUILD_EXAMPLES "Build documentation examples" OFF)
option(GFX_BUILD_TESTS "Build documentation tests" OFF)

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

add_library(gfx INTERFACE)
target_include_directories(gfx
	INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
	$<BUILD_INTERFACE:${cmd_SOURCE_DIR}/include>
)

find_package(vulkan REQUIRED)
find_package(glfw3 REQUIRED)

target_link_libraries(gfx INTERFACE
	Vulkan::Vulkan
	glfw
)

add_library(JED::gfx ALIAS gfx)

if(GFX_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

if(GFX_BUILD_TESTS)
	add_subdirectory(tests)
endif()