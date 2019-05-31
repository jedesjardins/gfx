
#include "gfx/render_device.hpp"

#include <vulkan/vulkan.h>
#include <glfw/glfw3.h>

#include <iostream>

std::vector<Vertex> obj1_vertices{{{-.5f, .5f, 0.5f}, {1.f, 0.f, 0.f}},
                             {{.5f, .5f, 0.5f}, {1.f, 0.f, 0.f}},
                             {{.5f, -.5f, 0.5f}, {1.f, 0.f, 0.f}},
                             {{-.5f, -.5f, 0.5f}, {1.f, 0.f, 0.f}}};

std::vector<uint32_t> obj1_indices{0, 1, 2, 0, 2, 3};

std::vector<Vertex> obj2_vertices{{{-1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
                             {{0.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
                             {{0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}},
                             {{-1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}}};

std::vector<uint32_t> obj2_indices{0, 1, 2, 0, 2, 3};

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


    std::vector<Object> objects{2};

    render_device.createStaticObject(
        objects[0], obj1_vertices.size(), obj1_vertices.data(), obj1_indices.size(), obj1_indices.data());

    render_device.createStaticObject(
        objects[1], obj2_vertices.size(), obj2_vertices.data(), obj2_indices.size(), obj2_indices.data());


    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        render_device.drawFrame(objects.size(), objects.data());
    }

    render_device.waitForIdle();

    render_device.destroyStaticObject(objects[0]);
    render_device.destroyStaticObject(objects[1]);

    render_device.quit();

    glfwDestroyWindow(window);

    glfwTerminate();
}