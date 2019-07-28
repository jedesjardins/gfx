

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#define JED_GFX_IMPLEMENTATION
#include "gfx/render_device.hpp"
#undef JED_GFX_IMPLEMENTATION

#define JED_CMD_IMPLEMENTATION
#include "cmd/cmd.hpp"
#undef JED_CMD_IMPLEMENTATION

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION

int main() {

	if (glfwInit() == GLFW_FALSE)
    {
        // error
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    auto window = glfwCreateWindow(600, 400, "Vulkan", nullptr, nullptr);

    auto render_config = gfx::RenderConfig{
        .config_filename = "../examples/example_renderer_config.json"};

    render_config.init();

    gfx::Renderer renderer{window};

    renderer.init(render_config);

    renderer.quit();

	glfwDestroyWindow(window);

    glfwTerminate();
}