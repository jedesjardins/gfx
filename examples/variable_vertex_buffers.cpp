
#include "gfx/render_device.hpp"

#include <vulkan/vulkan.h>
#include <glfw/glfw3.h>

#include <iostream>

std::vector<Vertex> obj1_vertices{{{0.f, 1.f, 0.5f}, {1.f, 0.f, 0.f}},
                                  {{1.f, 1.f, 0.5f}, {1.f, 0.f, 0.f}},
                                  {{1.f, 0.f, 0.5f}, {1.f, 0.f, 0.f}},
                                  {{0.f, 0.f, 0.5f}, {1.f, 0.f, 0.f}}};

std::vector<Vertex> obj2_vertices{{{-1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
                                  {{0.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
                                  {{0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}},
                                  {{-1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}}};

std::vector<Vertex> obj3_vertices{{{0.f, 0.f, 0.f}, {0.f, 0.f, 1.f}},
                                  {{1.f, 0.f, 0.f}, {0.f, 0.f, 1.f}},
                                  {{1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}},
                                  {{0.f, -1.f, 0.f}, {0.f, 0.f, 1.f}}};

std::vector<Vertex> obj4_vertices{{{-1.f, 0.f, 0.f}, {0.f, .5f, .5f}},
                                  {{0.f, 0.f, 0.f}, {0.f, .5f, .5f}},
                                  {{0.f, -1.f, 0.f}, {0.f, .5f, .5f}},
                                  {{-1.f, -1.f, 0.f}, {0.f, .5f, .5f}}};

std::vector<uint32_t> obj_indices{0, 1, 2, 0, 2, 3};

int main()
{
    if (glfwInit() == GLFW_FALSE)
    {
        // error
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    auto window = glfwCreateWindow(600, 400, "Vulkan", nullptr, nullptr);

    auto render_device = gfx::RenderDevice{window};

    std::cout << render_device.init("Example", 8, 12) << "\n";

    std::vector<Object> objects{4};

    render_device.createStaticObject(objects[0],
                                     obj1_vertices.size(),
                                     obj1_vertices.data(),
                                     obj_indices.size(),
                                     obj_indices.data());

    objects[0].transform = glm::scale(glm::mat4{1.f}, glm::vec3{.5f, .5f, .5f});

    render_device.createStaticObject(objects[1],
                                     obj2_vertices.size(),
                                     obj2_vertices.data(),
                                     obj_indices.size(),
                                     obj_indices.data());

    objects[2].type          = ObjectType::STREAMED;
    objects[2].d_vertex_data = StreamedVertexData{.vertex_count = 4,
                                                  .vertices     = obj3_vertices.data(),
                                                  .index_count  = obj_indices.size(),
                                                  .indices      = obj_indices.data()};

    objects[3].type          = ObjectType::STREAMED;
    objects[3].d_vertex_data = StreamedVertexData{.vertex_count = 4,
                                                  .vertices     = obj4_vertices.data(),
                                                  .index_count  = obj_indices.size(),
                                                  .indices      = obj_indices.data()};

    uint32_t i = 0;

    glm::mat4 view = glm::scale(glm::mat4(1.0), glm::vec3(1.f, -1.f, 1.f));

    auto opt_view_handle = render_device.newUniform();

    gfx::UniformHandle view_handle = opt_view_handle.value();

    while (!glfwWindowShouldClose(window))
    {
        render_device.startFrame();

        glfwPollEvents();

        render_device.updateUniform(view_handle, view);

        if (++i % 10 == 0)
        {
            obj1_vertices[0].pos.y -= .01f;

            render_device.updateStaticObject(objects[0],
                                             obj1_vertices.size(),
                                             obj1_vertices.data(),
                                             obj_indices.size(),
                                             obj_indices.data());
        }

        obj3_vertices[0].pos.y -= .01f;
        obj4_vertices[0].pos.y -= .01f;
        render_device.drawFrame(1, &view_handle, objects.size(), objects.data());
    }

    render_device.waitForIdle();

    render_device.destroyStaticObject(objects[0]);
    render_device.destroyStaticObject(objects[1]);

    render_device.quit();

    glfwDestroyWindow(window);

    glfwTerminate();
}