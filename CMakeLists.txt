cmake_minimum_required(VERSION 3.12)
project(gfx)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GFX_BUILD_EXAMPLES "Build documentation examples" OFF)
option(GFX_BUILD_TESTS "Build documentation tests" OFF)

find_package(vulkan REQUIRED)
find_package(glfw3 REQUIRED)

add_library(gfx INTERFACE)
target_include_directories(gfx
	INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
)

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