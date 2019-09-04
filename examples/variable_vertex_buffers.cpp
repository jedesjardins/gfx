
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <variant>
#include <iostream>
#include <numeric>

#define JED_GFX_IMPLEMENTATION
#include "gfx/render_device.hpp"
#undef JED_GFX_IMPLEMENTATION

#define JED_CMD_IMPLEMENTATION
#include "cmd/cmd.hpp"
#undef JED_CMD_IMPLEMENTATION

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION

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
    Object(gfx::Renderer & render_device,
           ObjectType      type,
           uint32_t        vertex_count,
           Vertex *        vertices,
           uint32_t        index_count,
           uint32_t *      indices)
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
                               streamed_data.vertex_count,
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
    if (glfwInit() == GLFW_FALSE)
    {
        // error
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    auto window = glfwCreateWindow(600, 400, "Vulkan", nullptr, nullptr);

    auto render_device = gfx::Renderer{window};

    auto render_config = gfx::RenderConfig{
        .config_filename = "../examples/example_renderer_config.json"};

    render_config.init();

    std::cout << "Render Device Init\n";
    if (!render_device.init(render_config))
    {
        throw std::runtime_error("Couldn't initialize Vulkan!");
    }

    std::vector<Object> objects{};

    std::cout << "Create texture\n";
    auto texture = render_device.create_texture("../sword.png");

    std::cout << "Create first object\n";
    objects.emplace_back(render_device,
                         ObjectType::STATIC,
                         obj1_vertices.size(),
                         obj1_vertices.data(),
                         obj_indices.size(),
                         obj_indices.data());

    objects[0].getMaterial().transform = glm::scale(glm::mat4{1.f}, glm::vec3{.5f, .5f, .5f});

    std::cout << "Create second object\n";
    objects.emplace_back(render_device,
                         ObjectType::STATIC,
                         obj2_vertices.size(),
                         obj2_vertices.data(),
                         obj_indices.size(),
                         obj_indices.data());

    std::cout << "Create third object\n";
    objects.emplace_back(render_device,
                         ObjectType::STREAMED,
                         obj3_vertices.size(),
                         obj3_vertices.data(),
                         obj_indices.size(),
                         obj_indices.data());

    std::cout << "Create fourth object\n";
    objects.emplace_back(render_device,
                         ObjectType::STREAMED,
                         obj4_vertices.size(),
                         obj4_vertices.data(),
                         obj_indices.size(),
                         obj_indices.data());

    uint32_t frame_number = 0;

    glm::mat4 view = glm::scale(glm::mat4(1.0), glm::vec3(1.f, -1.f, 1.f));

    auto opt_view_handle = render_device.new_uniform(0, sizeof(glm::mat4), glm::value_ptr(view));
    gfx::UniformHandle view_handle = opt_view_handle.value();

    auto               opt_sampler_handle = render_device.new_uniform(1, texture);
    gfx::UniformHandle sampler_handle     = opt_sampler_handle.value();

    auto clock = RawClock{};

    clock.Clear();

    std::vector<double> frame_times{10, 16.6};
    uint32_t            frameIndex{0};

    while (!glfwWindowShouldClose(window))
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

        render_device.draw_frame(2, uniforms.data());

        frame_times[++frameIndex % frame_times.size()] = clock.Restart();

        double sum_time = std::accumulate(frame_times.cbegin(), frame_times.cend(), 0.0);

        // std::cout << sum_time/frame_times.size() << "\n";
    }

    render_device.wait_for_idle();

    std::cout << "delete texture\n";
    render_device.delete_textures(1, &texture);

    std::cout << "delete first object\n";
    objects[0].destroyStaticObject(render_device);
    std::cout << "delete second object\n";
    objects[1].destroyStaticObject(render_device);

    std::cout << "pump\n";
    render_device.quit();

    glfwDestroyWindow(window);

    glfwTerminate();
}