
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

#include <variant>
#include <iostream>
#include <numeric>
#include <thread>

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        auto bindingDescription = VkVertexInputBindingDescription{
            .binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        auto attributeDescriptions = std::array<VkVertexInputAttributeDescription, 2>{
            VkVertexInputAttributeDescription{.binding  = 0,
                                              .location = 0,
                                              .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                              .offset   = offsetof(Vertex, pos)},
            VkVertexInputAttributeDescription{.binding  = 0,
                                              .location = 1,
                                              .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                              .offset   = offsetof(Vertex, color)}};

        return attributeDescriptions;
    }
};

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

struct Material
{
    gfx::PipelineHandle pipeline{0};
    glm::mat4           transform{1.0f};
};

std::optional<Material> make_material(gfx::Renderer & render_device,
                                      std::string     pipeline_name,
                                      glm::mat4       transform)
{
    auto opt_pipeline_handle = render_device.get_pipeline_handle(pipeline_name);

    if (!opt_pipeline_handle)
    {
        return std::nullopt;
    }

    LOG_DEBUG("PipelineHandle for {} is {}", pipeline_name, opt_pipeline_handle.value());

    return Material{opt_pipeline_handle.value(), transform};
}

struct StaticVertexData // can be edited with a
{
    gfx::BufferHandle vertexbuffer;
    gfx::BufferHandle indexbuffer;
    size_t            index_count;
};

struct StreamedVertexData
{
    size_t     vertex_count;
    Vertex *   vertices;
    size_t     index_count;
    uint32_t * indices;
};

enum class ObjectType
{
    NONE,
    STATIC,
    STREAMED
};

class Object
{
public:
    Object(gfx::Renderer &     render_device,
           ObjectType          type,
           uint32_t            vertex_count,
           Vertex *            vertices,
           uint32_t            index_count,
           uint32_t *          indices,
           std::string const & pipeline_name,
           glm::mat4           transform = glm::mat4{1.0f})
    : material{make_material(render_device, pipeline_name, transform).value()}
    {
        if (type == ObjectType::STATIC)
        {
            initStaticObject(render_device, vertex_count, vertices, index_count, indices);
        }
        else if (type == ObjectType::STREAMED)
        {
            initDynamicObject(render_device, vertex_count, vertices, index_count, indices);
        }
    }

    bool initStaticObject(gfx::Renderer & render_device,
                          uint32_t        vertex_count,
                          Vertex *        vertices,
                          uint32_t        index_count,
                          uint32_t *      indices)
    {
        auto static_vertex_data        = StaticVertexData{};
        static_vertex_data.index_count = index_count;

        vertex_data = static_vertex_data;

        auto & static_data = std::get<StaticVertexData>(vertex_data);

        auto opt_vertexbuffer = render_device.create_buffer(
            vertex_count * sizeof(Vertex),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        static_data.vertexbuffer = opt_vertexbuffer.value();
        render_device.update_buffer(
            static_data.vertexbuffer, vertex_count * sizeof(Vertex), vertices);

        auto opt_indexbuffer = render_device.create_buffer(
            index_count * sizeof(uint32_t),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        static_data.indexbuffer = opt_indexbuffer.value();
        render_device.update_buffer(
            static_data.indexbuffer, index_count * sizeof(uint32_t), indices);

        return true;
    }

    void initDynamicObject(gfx::Renderer & render_device,
                           uint32_t        vertex_count,
                           Vertex *        vertices,
                           uint32_t        index_count,
                           uint32_t *      indices)
    {
        vertex_data = StreamedVertexData{.vertex_count = vertex_count,
                                         .vertices     = vertices,
                                         .index_count  = index_count,
                                         .indices      = indices};
    }

    void updateStaticObject(gfx::Renderer & render_device,
                            uint32_t        vertex_count,
                            Vertex *        vertices,
                            uint32_t        index_count,
                            uint32_t *      indices)
    {
        assert(std::holds_alternative<StaticVertexData>(vertex_data));

        auto & static_data = std::get<StaticVertexData>(vertex_data);

        render_device.update_buffer(
            static_data.vertexbuffer, vertex_count * sizeof(Vertex), vertices);
        render_device.update_buffer(
            static_data.indexbuffer, index_count * sizeof(uint32_t), indices);
    }

    void destroyStaticObject(gfx::Renderer & render_device)
    {
        assert(std::holds_alternative<StaticVertexData>(vertex_data));

        auto & static_data = std::get<StaticVertexData>(vertex_data);

        gfx::BufferHandle buffers[2] = {static_data.vertexbuffer, static_data.indexbuffer};

        render_device.delete_buffers(2, buffers);
    }

    void draw(gfx::Renderer & render_device)
    {
        if (std::holds_alternative<StaticVertexData>(vertex_data))
        {
            auto & static_data = std::get<StaticVertexData>(vertex_data);

            render_device.draw(material.pipeline,
                               material.transform,
                               static_data.vertexbuffer,
                               0,
                               static_data.indexbuffer,
                               0,
                               static_data.index_count);
        }
        else if (std::holds_alternative<StreamedVertexData>(vertex_data))
        {
            auto & streamed_data = std::get<StreamedVertexData>(vertex_data);

            render_device.draw(material.pipeline,
                               material.transform,
                               streamed_data.vertex_count * sizeof(Vertex),
                               streamed_data.vertices,
                               streamed_data.index_count,
                               streamed_data.indices);
        }
    }

    Material & getMaterial()
    {
        return material;
    }

private:
    Material                                           material;
    std::variant<StaticVertexData, StreamedVertexData> vertex_data;
};

class RawClock
{
public:
    RawClock(): start_time{Clock::now()}
    {}

    void Resume()
    {
        paused = false;

        start_time = Clock::now();
    }
    void Pause()
    {
        auto end_time = Clock::now();

        std::chrono::duration<double, std::milli> mdt = end_time - start_time;

        paused_dt += mdt.count();
        paused = true;
    }
    void Clear()
    {
        paused_dt  = 0;
        start_time = Clock::now();
    }
    double Read()
    {
        if (paused)
        {
            return paused_dt;
        }

        auto end_time = Clock::now();

        std::chrono::duration<double, std::milli> mdt = end_time - start_time;

        return paused_dt + mdt.count();
    }
    double Restart()
    {
        auto dt = paused_dt;

        if (!paused)
        {
            auto end_time = Clock::now();

            std::chrono::duration<double, std::milli> mdt = end_time - start_time;
            dt += mdt.count();
        }

        paused     = false;
        paused_dt  = 0;
        start_time = Clock::now();

        return dt;
    }

private:
    using Clock    = std::chrono::high_resolution_clock;
    using Duration = std::chrono::time_point<Clock>;

    Duration start_time;

    double paused_dt{0};
    bool   paused = false;

}; // class RawClock

int main()
{
    get_console_sink()->set_level(spdlog::level::warn);
    get_file_sink()->set_level(spdlog::level::trace);
    get_logger()->set_level(spdlog::level::debug);

    LOG_INFO("Starting Example");

    if (glfwInit() == GLFW_FALSE)
    {
        LOG_ERROR("GLFW didn't initialize correctly");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    auto window = glfwCreateWindow(600, 400, "Vulkan", nullptr, nullptr);

    auto render_device = gfx::Renderer{window};

    auto render_config = gfx::RenderConfig{
        .config_filename = "../examples/example_renderer_config.json"};

    render_config.init();

    if (!render_device.init(render_config))
    {
        LOG_ERROR("Couldn't initialize the Renderer");
        return 0;
    }

    std::vector<Object> objects{};

    auto opt_texture = render_device.create_texture("../sword.png");
    if (!opt_texture)
    {
        LOG_ERROR("Couldn't create texture, exiting program!!");
        return 0;
    }

    auto texture = opt_texture.value();
    LOG_DEBUG("Created texture {}", texture);

    objects.emplace_back(render_device,
                         ObjectType::STATIC,
                         obj1_vertices.size(),
                         obj1_vertices.data(),
                         obj_indices.size(),
                         obj_indices.data(),
                         "colored_texture_shader");

    objects[0].getMaterial().transform = glm::scale(glm::mat4{1.f}, glm::vec3{.5f, .5f, .5f});

    objects.emplace_back(render_device,
                         ObjectType::STATIC,
                         obj2_vertices.size(),
                         obj2_vertices.data(),
                         obj_indices.size(),
                         obj_indices.data(),
                         "simple_texture_shader");

    objects.emplace_back(render_device,
                         ObjectType::STREAMED,
                         obj3_vertices.size(),
                         obj3_vertices.data(),
                         obj_indices.size(),
                         obj_indices.data(),
                         "colored_texture_shader");

    objects.emplace_back(render_device,
                         ObjectType::STREAMED,
                         obj4_vertices.size(),
                         obj4_vertices.data(),
                         obj_indices.size(),
                         obj_indices.data(),
                         "colored_texture_shader");

    uint32_t frame_number = 0;

    // Camera View Uniform
    glm::mat4 view = glm::scale(glm::mat4(1.0), glm::vec3(1.f, -1.f, 1.f));

    auto opt_layout_handle = render_device.get_uniform_layout_handle("ul_camera_matrix");
    if (!opt_layout_handle)
    {
        LOG_ERROR("Couldn't get UniformLayoutHandle for \"ul_camera_matrix\" UniformLayout");
        return 0;
    }

    auto opt_view_handle = render_device.new_uniform(
        opt_layout_handle.value(), sizeof(glm::mat4), glm::value_ptr(view));
    if (!opt_view_handle)
    {
        LOG_ERROR("Couldn't create uniform for camera view");
        return 0;
    }
    gfx::UniformHandle view_handle = opt_view_handle.value();

    // Texture Sampler Uniform
    opt_layout_handle = render_device.get_uniform_layout_handle("ul_texture");
    if (!opt_layout_handle)
    {
        LOG_ERROR("Couldn't get UniformLayoutHandle for \"ul_texture\" UniformLayout");
        return 0;
    }

    auto opt_sampler_handle = render_device.new_uniform(opt_layout_handle.value(), texture);
    gfx::UniformHandle sampler_handle = opt_sampler_handle.value();

    auto clock = RawClock{};

    clock.Clear();

    std::vector<double> frame_times{10, 16.6};
    uint32_t            frameIndex{0};

    bool draw_success{true};

    LOG_INFO("Starting loop");

    while (!glfwWindowShouldClose(window) && draw_success)
    {
        glfwPollEvents();

        if (++frame_number % 10 == 0)
        {
            view *= glm::vec4(1.f, -1.f, 1.f, 1.f);

            render_device.update_uniform(
                view_handle, sizeof(glm::mat4), static_cast<void *>(glm::value_ptr(view)));

            obj1_vertices[0].pos.y -= .01f;

            objects[0].updateStaticObject(render_device,
                                          obj1_vertices.size(),
                                          obj1_vertices.data(),
                                          obj_indices.size(),
                                          obj_indices.data());
        }

        obj3_vertices[0].pos.y -= .01f;
        obj4_vertices[0].pos.y -= .01f;

        for (uint32_t i = 0; i < objects.size(); ++i)
        {
            objects[i].draw(render_device);
        }

        std::array<gfx::UniformHandle, 2> uniforms = {view_handle, sampler_handle};

        draw_success = render_device.draw_frame(2, uniforms.data());

        frame_times[++frameIndex % frame_times.size()] = clock.Restart();

        double sum_time = std::accumulate(frame_times.cbegin(), frame_times.cend(), 0.0);

        LOG_TRACE("Frame time {}", sum_time / frame_times.size());
    }

    render_device.wait_for_idle();

    render_device.delete_uniforms(1, &view_handle);
    render_device.delete_textures(1, &texture);
    objects[0].destroyStaticObject(render_device);
    objects[1].destroyStaticObject(render_device);

    render_device.quit();

    glfwDestroyWindow(window);

    glfwTerminate();

    LOG_INFO("Stopping Example\n");
}