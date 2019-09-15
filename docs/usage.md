# Usage


## CMake

This library is probably easiest to use if you're integrating it into an existing CMake build system. It can automatically fetch the gfx library and its dependencies for you and put them on your include path.


### Automatically retrieving the library

You can use Fetchcontent to get the library and then populate the build fields.

```
include(FetchContent)

FetchContent_Declare(
	Gfx
	GIT_REPOSITORY "https://github.com/jedesjardins/gfx"
	GIT_TAG realease-v0.1
)

FetchContent_GetProperties(Gfx)
if(NOT Gfx_POPULATED)
	FetchContent_Populate(Gfx)
	add_subdirectory(${gfx_SOURCE_DIR} ${gfx_BINARY_DIR})
endif()
```

This will define an interface library JED::gfx, which you can link to your executable to add the header files to your build system with:

```
target_link_libraries(${name} JED::gfx JED::log)
```


### Dependencies

At this moment, the gfx library depends on a few other libraries:
+ Vulkan (of course)
	+ The gfx CMakeLists.txt looks for this with `find_package(vulkan REQUIRED)` so it needs to be able to be found by CMake
+ [GLFW](https://www.glfw.org/) - the windowing library
	+ The gfx CMakeLists.txt looks for this with `find_package(glfw3 REQUIRED)` so it needs to be able to be found by CMake
	+ Plans are to remove this eventually
+ [cmd](https://github.com/jedesjardins/cmd) - another JED library for creating and queuing actions
+ [rapidjson](https://github.com/Tencent/rapidjson) - parses the json configuration file
+ [stb](https://github.com/nothings/stb) - for reading png files


### Compiling

Both stb and cmd are single header files that have the function definitions behind header guards. So in exactly one file you need to
```
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION // so that any other includes of stb_image.h doesn't double define stb functions
```
and
```
#define JED_GFX_IMPLEMENTATION
#include "gfx/render_device.hpp"
#undef JED_GFX_IMPLEMENTATION // so that any other includes of gfx/render_device.hpp doesn't double define gfx functions
```

## Non-CMake builds

You'll need to gather the resources yourself and make sure they are on your include path.
