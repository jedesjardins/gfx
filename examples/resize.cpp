
#include "log/logger.hpp"
#include "gfx/renderer.hpp"
#include "cmd/cmd.hpp"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <utility>
#include <algorithm>

bool pressed_f   = false;
bool pressed_m   = false;
bool pressed_esc = false;

float scale   = 1.5f;
bool  resized = false;

void key_callback(GLFWwindow * window, int key, int scancode, int action, int mods)
{
    switch (key)
    {
    case GLFW_KEY_F:
        pressed_f = action == GLFW_RELEASE;
        break;
    case GLFW_KEY_M:
        pressed_m = action != GLFW_RELEASE;
        break;
    case GLFW_KEY_ESCAPE:
        pressed_esc = action != GLFW_RELEASE;
        break;
    }
}

void fb_callback(GLFWwindow * window, int width, int height)
{
    scale   = (float)(width) / height;
    resized = true;
}

void set_scale(GLFWwindow * window)
{
    int width{0}, height{0};

    glfwGetFramebufferSize(window, &width, &height);

    scale = (float)(width) / height;
}

std::vector<glm::vec3> vertices{{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}};

std::vector<uint32_t> indices{0, 1, 2, 0, 2, 3};

int main()
{
    get_console_sink()->set_level(spdlog::level::info);
    get_file_sink()->set_level(spdlog::level::trace);
    get_logger()->set_level(spdlog::level::debug);

    LOG_INFO("Starting Resize");

    if (glfwInit() == GLFW_FALSE)
    {
        LOG_ERROR("GLFW didn't initialize correctly");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    auto window = glfwCreateWindow(600, 400, "Resize", nullptr, nullptr);

    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, fb_callback);

    auto render_device = gfx::Renderer{window};

    auto render_config = gfx::RenderConfig{.config_filename = RESOURCE_PATH "resize_config.json"};

    render_config.init();

    if (!render_device.init(render_config))
    {
        LOG_ERROR("Couldn't initialize the Renderer");
        return 0;
    }

    auto opt_layout_handle = render_device.get_uniform_layout_handle("ul_camera_matrix");

    set_scale(window);

    LOG_INFO("Scale {}", scale);

    glm::mat4 view = glm::ortho(0.f, 60.f, 0.f, 60.f / scale);

    gfx::UniformHandle view_handle = render_device
                                         .new_uniform(opt_layout_handle.value(),
                                                      sizeof(glm::mat4),
                                                      glm::value_ptr(view))
                                         .value();

    auto pipeline_handle = render_device.get_pipeline_handle("square_shader").value();

    auto vertex_buffer = render_device
                             .create_buffer(vertices.size() * sizeof(glm::vec3),
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                                | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                             .value();

    render_device.update_buffer(
        vertex_buffer, vertices.size() * sizeof(glm::vec3), vertices.data());

    auto index_buffer = render_device
                            .create_buffer(
                                indices.size() * sizeof(uint32_t),
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                            .value();

    render_device.update_buffer(index_buffer, indices.size() * sizeof(uint32_t), indices.data());

    glm::mat4 transform = glm::translate(glm::mat4{1.0f}, glm::vec3{10.f, 10.f, 0.f});

    bool draw_success{true};
    bool windowed{true};
    while (!glfwWindowShouldClose(window) && draw_success)
    {
        pressed_f = false;
        glfwPollEvents();

        if (pressed_f)
        {
            if (windowed)
            {
                glfwMaximizeWindow(window);

                windowed = false;
            }
            else
            {
                glfwRestoreWindow(window);

                windowed = true;
            }
        }

        if (resized)
        {
            view = glm::ortho(0.f, 60.f, 0.f, 60.f / scale);
            render_device.update_uniform(view_handle, sizeof(glm::mat4), glm::value_ptr(view));
        }

        gfx::DrawParameters params{};

        VkDeviceSize vertex_buffer_offset = 0;

        params.pipeline = pipeline_handle;

        params.vertex_buffer_count   = 1;
        params.vertex_buffers        = &vertex_buffer;
        params.vertex_buffer_offsets = &vertex_buffer_offset;

        params.index_buffer        = index_buffer;
        params.index_buffer_offset = 0;
        params.index_count         = indices.size();

        params.push_constant_size = sizeof(glm::mat4);
        params.push_constant_data = glm::value_ptr(transform);

        params.uniform_count = 1;
        params.uniforms      = &view_handle;

        params.scissor  = nullptr;
        params.viewport = nullptr;

        render_device.draw(params);

        draw_success = render_device.submit_frame();

        draw_success &= !pressed_esc;
        LOG_INFO("Running");
    }

    render_device.wait_for_idle();

    render_device.quit();

    glfwDestroyWindow(window);

    glfwTerminate();

    LOG_INFO("Stopping Resize\n");
}