
#include "log/logger.hpp"
#include "gfx/renderer.hpp"
#include "common.hpp"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

gfx::ErrorCode readFile(char const * file_name, std::vector<char> & buffer);

bool pressed_esc = false;

void key_callback(GLFWwindow * window, int key, int scancode, int action, int mods)
{
    switch (key)
    {
    case GLFW_KEY_ESCAPE:
        pressed_esc = action != GLFW_RELEASE;
        break;
    }
}

struct POS_UV
{
    glm::vec2 pos;
    glm::vec2 uv;
};

int main()
{
    get_console_sink()->set_level(spdlog::level::info);
    get_file_sink()->set_level(spdlog::level::trace);
    get_logger()->set_level(spdlog::level::debug);

    LOG_INFO("Starting Input Attachment test");

    if (glfwInit() == GLFW_FALSE)
    {
        LOG_ERROR("GLFW didn't initialize correctly");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    auto window = glfwCreateWindow(600, 400, "mul_vb", nullptr, nullptr);

    glfwSetKeyCallback(window, key_callback);

    auto render_config = gfx::RenderConfig{};

    if (render_config.init(RESOURCE_PATH "input_attachment/input_attachment_config.json", readFile)
        != gfx::ErrorCode::NONE)
    {
        LOG_ERROR("Couldn't initialize the Render Configuration");
        return 0;
    }

    auto renderer = gfx::Renderer{window};

    if (!renderer.init(render_config))
    {
        LOG_ERROR("Couldn't initialize the Renderer");
        return 0;
    }

    auto to_texture_pipeline   = renderer.get_pipeline_handle("initial_shader").value();
    auto from_texture_pipeline = renderer.get_pipeline_handle("blit_shader").value();

    auto texture_uniform = make_texture_uniform_from_attachment(
        renderer, "us_input_attachment", "a_input_color");

    // object vertices
    std::vector<glm::vec2> object_vertices = {{-.9f, -.9f}, {0.f, -.9f}, {0.f, 0.f}, {-.9f, 0.f}};

    auto object_vertex_buffer = renderer
                                    .create_buffer(object_vertices.size() * sizeof(glm::vec2),
                                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                                       | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                                    .value();

    renderer.update_buffer(
        object_vertex_buffer, object_vertices.size() * sizeof(glm::vec2), object_vertices.data());

    // texture mapped vertices
    std::vector<POS_UV> texture_blit_vertices = {{{-1.f, -1.f}, {0.f, 0.f}},
                                                 {{1.f, -1.f}, {1.f, 0.f}},
                                                 {{1.f, 1.f}, {1.f, 1.f}},
                                                 {{-1.f, 1.f}, {0.f, 1.f}}};

    auto blit_vertex_buffer = renderer
                                  .create_buffer(texture_blit_vertices.size() * sizeof(POS_UV),
                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                                     | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                                  .value();

    renderer.update_buffer(blit_vertex_buffer,
                           texture_blit_vertices.size() * sizeof(POS_UV),
                           texture_blit_vertices.data());

    // indices
    std::vector<uint32_t> indices = {0, 1, 2, 0, 2, 3};

    auto index_buffer = renderer
                            .create_buffer(
                                indices.size() * sizeof(uint32_t),
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                            .value();

    renderer.update_buffer(index_buffer, indices.size() * sizeof(uint32_t), indices.data());

    bool draw_success{true};
    while (!glfwWindowShouldClose(window) && draw_success && !pressed_esc)
    {
        glfwPollEvents();

        VkDeviceSize zero_offset = 0;

        // draw the object
        gfx::DrawParameters object_params{};

        object_params.pipeline = to_texture_pipeline;

        object_params.vertex_buffer_count   = 1;
        object_params.vertex_buffers        = &object_vertex_buffer;
        object_params.vertex_buffer_offsets = &zero_offset;

        object_params.index_buffer        = index_buffer;
        object_params.index_buffer_offset = 0;
        object_params.index_count         = 3;

        object_params.push_constant_size = 0;
        object_params.push_constant_data = nullptr;

        object_params.uniform_count = 0;
        object_params.uniforms      = nullptr;

        object_params.scissor  = nullptr;
        object_params.viewport = nullptr;

        renderer.draw(object_params);

        // draw the texture to the screen
        gfx::DrawParameters texture_params{};

        texture_params.pipeline = from_texture_pipeline;

        texture_params.vertex_buffer_count   = 1;
        texture_params.vertex_buffers        = &blit_vertex_buffer;
        texture_params.vertex_buffer_offsets = &zero_offset;

        texture_params.index_buffer        = index_buffer;
        texture_params.index_buffer_offset = 0;
        texture_params.index_count         = indices.size();

        texture_params.push_constant_size = 0;
        texture_params.push_constant_data = nullptr;

        texture_params.uniform_count = 1;
        texture_params.uniforms      = &texture_uniform;

        texture_params.scissor  = nullptr;
        texture_params.viewport = nullptr;

        renderer.draw(texture_params);

        draw_success = renderer.submit_frame();
    }

    renderer.wait_for_idle();

    renderer.quit();

    glfwDestroyWindow(window);

    glfwTerminate();
}