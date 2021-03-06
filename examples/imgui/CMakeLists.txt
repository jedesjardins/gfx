cmake_minimum_required(VERSION 3.12)

message(STATUS "Imgui example included")

#Imgui
FetchContent_Declare(
	Imgui
	GIT_REPOSITORY "https://github.com/ocornut/imgui"
)

FetchContent_GetProperties(Imgui)
if(NOT Imgui_POPULATED)
	FetchContent_Populate(Imgui)
	message(STATUS "${imgui_SOURCE_DIR} ${imgui_BINARY_DIR}")
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

list(APPEND SOURCES
	main.cpp
	${imgui_SOURCE_DIR}/imgui.cpp
	${imgui_SOURCE_DIR}/imgui_draw.cpp
	${imgui_SOURCE_DIR}/imgui_widgets.cpp
	${imgui_SOURCE_DIR}/examples/imgui_impl_glfw.cpp
	${imgui_SOURCE_DIR}/examples/imgui_impl_glfw.h
)

add_compile_definitions(IMGUI_USER_CONFIG="my_imgui_config.hpp")

add_executable(imgui ${SOURCES})

target_include_directories(imgui PRIVATE
	$<BUILD_INTERFACE:${stb_SOURCE_DIR}>
	${imgui_SOURCE_DIR}
	${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(imgui common)