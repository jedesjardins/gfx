
#include "gfx/render_device.hpp"

#include <vulkan/vulkan.h>
#include <glfw/glfw3.h>

#include <iostream>

int main()
{
    if (glfwInit() == GLFW_FALSE)
    {
        // error
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    auto window = glfwCreateWindow(600, 400, "Vulkan", nullptr, nullptr);

    auto render_device = gfx::RenderDevice{window};

    std::cout << render_device.init("Example") << "\n";


    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        render_device.drawFrame();
    }

    render_device.quit();

    glfwDestroyWindow(window);

    glfwTerminate();
}