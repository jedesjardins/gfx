
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#define JED_LOG_IMPLEMENTATION
#include "log/logger.hpp"
#undef JED_LOG_IMPLEMENTATION

#define JED_GFX_IMPLEMENTATION
#include "gfx/render_device.hpp"
#undef JED_GFX_IMPLEMENTATION

#define JED_CMD_IMPLEMENTATION
#include "cmd/cmd.hpp"
#undef JED_CMD_IMPLEMENTATION

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <utility>
#include <algorithm>

std::vector<glm::vec2> left_pos_vec{{-1.f, -1.f}, {0.f, 0.f}, {-1.f, 1.f}};

std::vector<glm::vec2> right_pos_vec{{1.f, -1.f}, {0.f, 0.f}, {1.f, 1.f}};

std::vector<glm::vec3> color_vec{{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}};

std::vector<uint32_t> indices{0, 1, 2};

int main()
{
    get_console_sink()->set_level(spdlog::level::info);
    get_file_sink()->set_level(spdlog::level::trace);
    get_logger()->set_level(spdlog::level::debug);

    LOG_INFO("Starting Multiple VertexBuffers test");

    if (glfwInit() == GLFW_FALSE)
    {
        LOG_ERROR("GLFW didn't initialize correctly");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    auto window = glfwCreateWindow(600, 400, "mul_vb", nullptr, nullptr);

    auto render_device = gfx::Renderer{window};

    auto render_config = gfx::RenderConfig{.config_filename = RESOURCE_PATH "mul_vbs.json"};

    render_config.init();

    if (!render_device.init(render_config))
    {
        LOG_ERROR("Couldn't initialize the Renderer");
        return 0;
    }

    auto pipeline = render_device.get_pipeline_handle("mul_vbs_shader").value();

    // Both vertex bindings in their own buffer
    auto pos_buffer = render_device
                          .create_buffer(
                              left_pos_vec.size() * sizeof(glm::vec2),
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                          .value();

    render_device.update_buffer(pos_buffer, left_pos_vec.size() * sizeof(glm::vec2), left_pos_vec.data());

    auto color_buffer = render_device
                            .create_buffer(color_vec.size() * sizeof(glm::vec3),
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                               | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                            .value();

    render_device.update_buffer(
        color_buffer, color_vec.size() * sizeof(glm::vec3), color_vec.data());


    // Both vertex bindings in one buffer
    std::array<float, 3 * 2 + 3 * 3> vertices;

    float * vec_1 = vertices.data();
    float * vec_2 = vec_1 + (right_pos_vec.size() * 2);

    memcpy(vec_1, right_pos_vec.data(), right_pos_vec.size() * sizeof(glm::vec2));
    memcpy(vec_2, color_vec.data(), color_vec.size() * sizeof(glm::vec3));

    auto vertex_buffer = render_device
                             .create_buffer(vertices.size() * sizeof(float),
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                                | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                             .value();

    render_device.update_buffer(vertex_buffer, vertices.size() * sizeof(float), vertices.data());

    auto index_buffer = render_device
                            .create_buffer(
                                indices.size() * sizeof(uint32_t),
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                            .value();

    render_device.update_buffer(index_buffer, indices.size() * sizeof(uint32_t), indices.data());

    bool draw_success{true};
    while (!glfwWindowShouldClose(window) && draw_success)
    {
        glfwPollEvents();

        // draw with different buffers
        gfx::DrawParameters params_1{};

        std::array<gfx::BufferHandle, 2> vertex_buffers_1{pos_buffer, color_buffer};
        std::array<VkDeviceSize, 2>      vertex_buffer_offsets_1{0, 0};

        params_1.pipeline = pipeline;

        params_1.vertex_buffer_count   = 2;
        params_1.vertex_buffers        = vertex_buffers_1.data();
        params_1.vertex_buffer_offsets = vertex_buffer_offsets_1.data();

        params_1.index_buffer        = index_buffer;
        params_1.index_buffer_offset = 0;
        params_1.index_count         = 3;

        render_device.draw(params_1);


        // draw with the same buffer twice
        gfx::DrawParameters params_2{};

        std::array<gfx::BufferHandle, 2> vertex_buffers_2{vertex_buffer, vertex_buffer};
        std::array<VkDeviceSize, 2>      vertex_buffer_offsets_2{0, right_pos_vec.size() * sizeof(glm::vec2)};

        params_2.pipeline = pipeline;

        params_2.vertex_buffer_count   = 2;
        params_2.vertex_buffers        = vertex_buffers_2.data();
        params_2.vertex_buffer_offsets = vertex_buffer_offsets_2.data();

        params_2.index_buffer        = index_buffer;
        params_2.index_buffer_offset = 0;
        params_2.index_count         = 3;

        render_device.draw(params_2);

        draw_success = render_device.submit_frame();
    }

    render_device.wait_for_idle();

    render_device.quit();

    glfwDestroyWindow(window);

    glfwTerminate();

    LOG_INFO("Stopping Pong\n");
}