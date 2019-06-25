
#include "gfx/render_device.hpp"

#define JED_CMD_IMPLEMENTATION
#include "cmd/cmd.hpp"
#undef JED_CMD_IMPLEMENTATION

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <variant>
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

struct Material
{
    gfx::PipelineHandle pipeline{0};
    glm::mat4           transform{1.0f};
};

struct StaticVertexData // can be edited with a
{
    VkBuffer       vertexbuffer;
    VkDeviceMemory vertexbuffer_memory;
    VkDeviceSize   vertexbuffer_offset;

    VkBuffer       indexbuffer;
    VkDeviceMemory indexbuffer_memory;
    size_t         indexbuffer_offset;
    size_t         indexbuffer_count;
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
    Object(gfx::RenderDevice & render_device,
           ObjectType          type,
           uint32_t            vertex_count,
           Vertex *            vertices,
           uint32_t            index_count,
           uint32_t *          indices)
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

    bool initStaticObject(gfx::RenderDevice & render_device,
                          uint32_t            vertex_count,
                          Vertex *            vertices,
                          uint32_t            index_count,
                          uint32_t *          indices)
    {
        vertex_data = StaticVertexData{.vertexbuffer        = VK_NULL_HANDLE,
                                       .vertexbuffer_memory = VK_NULL_HANDLE,
                                       .vertexbuffer_offset = 0,
                                       .indexbuffer         = VK_NULL_HANDLE,
                                       .indexbuffer_memory  = VK_NULL_HANDLE,
                                       .indexbuffer_offset  = 0,
                                       .indexbuffer_count   = index_count};

        auto & static_data = std::get<StaticVertexData>(vertex_data);

        if (!render_device.createVertexbuffer(
                static_data.vertexbuffer, static_data.vertexbuffer_memory, vertex_count, vertices))
        {
            return false;
        }

        if (!render_device.createIndexbuffer(
                static_data.indexbuffer, static_data.indexbuffer_memory, index_count, indices))
        {
            return false;
        }

        return true;
    }

    void initDynamicObject(gfx::RenderDevice & render_device,
                           uint32_t            vertex_count,
                           Vertex *            vertices,
                           uint32_t            index_count,
                           uint32_t *          indices)
    {
        vertex_data = StreamedVertexData{.vertex_count = vertex_count,
                                         .vertices     = vertices,
                                         .index_count  = index_count,
                                         .indices      = indices};
    }

    void updateStaticObject(gfx::RenderDevice & render_device,
                            uint32_t            vertex_count,
                            Vertex *            vertices,
                            uint32_t            index_count,
                            uint32_t *          indices)
    {
        assert(std::holds_alternative<StaticVertexData>(vertex_data));

        auto & static_data = std::get<StaticVertexData>(vertex_data);

        render_device.updateDeviceLocalBuffers(static_data.vertexbuffer,
                                               static_data.indexbuffer,
                                               vertex_count,
                                               vertices,
                                               index_count,
                                               indices);
    }

    void destroyStaticObject(gfx::RenderDevice & render_device)
    {
        assert(std::holds_alternative<StaticVertexData>(vertex_data));

        auto & static_data = std::get<StaticVertexData>(vertex_data);

        render_device.destroyDeviceLocalBuffers(static_data.vertexbuffer,
                                                static_data.vertexbuffer_memory,
                                                static_data.indexbuffer,
                                                static_data.indexbuffer_memory);
    }

    void draw(gfx::RenderDevice & render_device)
    {
        if (std::holds_alternative<StaticVertexData>(vertex_data))
        {
            auto & static_data = std::get<StaticVertexData>(vertex_data);

            render_device.draw(material.pipeline,
                               material.transform,
                               static_data.vertexbuffer,
                               static_data.vertexbuffer_offset,
                               static_data.indexbuffer,
                               static_data.indexbuffer_offset,
                               static_data.indexbuffer_count);
        }
        else if (std::holds_alternative<StreamedVertexData>(vertex_data))
        {
            auto & streamed_data = std::get<StreamedVertexData>(vertex_data);

            render_device.dynamicDraw(material.pipeline,
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

    auto render_device = gfx::RenderDevice{window};

    auto render_config
        = gfx::RenderConfig{
            .config_filename = "../examples/example_renderer_config.json",
            .uniform_layouts = {gfx::UniformLayout{
                .binding       = {.binding            = 0,
                            .descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                            .descriptorCount    = 1,
                            .stageFlags         = VK_SHADER_STAGE_VERTEX_BIT,
                            .pImmutableSamplers = nullptr},
                .uniform_count = 1}},
            .pipelines       = {gfx::Pipeline{.vertex_shader     = 0,
                                        .fragment_shader   = 1,
                                        .vertex_bindings   = {0},
                                        .vertex_attributes = {0, 1},
                                        .uniform_layouts   = {},
                                        .push_constants    = {0},
                                        .renderpass        = 0,
                                        .subpass           = 0}},
            .push_constants    = {VkPushConstantRange{
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(glm::mat4)}},
            .vertex_bindings   = {VkVertexInputBindingDescription{
                .binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX}},
            .vertex_attributes = {
                VkVertexInputAttributeDescription{.binding  = 0,
                                                  .location = 0,
                                                  .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                                  .offset   = offsetof(Vertex, pos)},
                VkVertexInputAttributeDescription{.binding  = 0,
                                                  .location = 1,
                                                  .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                                  .offset   = offsetof(Vertex, color)}}};

    render_config.init();

    if (!render_device.init(render_config))
    {
        throw std::runtime_error("Couldn't initialize Vulkan!");
    }

    std::vector<Object> objects{};

    objects.emplace_back(render_device,
                         ObjectType::STATIC,
                         obj1_vertices.size(),
                         obj1_vertices.data(),
                         obj_indices.size(),
                         obj_indices.data());

    objects[0].getMaterial().transform = glm::scale(glm::mat4{1.f}, glm::vec3{.5f, .5f, .5f});

    objects.emplace_back(render_device,
                         ObjectType::STATIC,
                         obj2_vertices.size(),
                         obj2_vertices.data(),
                         obj_indices.size(),
                         obj_indices.data());

    objects.emplace_back(render_device,
                         ObjectType::STREAMED,
                         obj3_vertices.size(),
                         obj3_vertices.data(),
                         obj_indices.size(),
                         obj_indices.data());

    objects.emplace_back(render_device,
                         ObjectType::STREAMED,
                         obj4_vertices.size(),
                         obj4_vertices.data(),
                         obj_indices.size(),
                         obj_indices.data());

    uint32_t i = 0;

    glm::mat4 view = glm::scale(glm::mat4(1.0), glm::vec3(1.f, -1.f, 1.f));

    auto opt_view_handle = render_device.newUniform();

    gfx::UniformHandle view_handle = opt_view_handle.value();

    auto clock = RawClock{};

    clock.Clear();

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        render_device.updateUniform(view_handle, view);

        if (++i % 10 == 0)
        {
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

        render_device.drawFrame(1, &view_handle);

        // std::cout << clock.Restart() << "\n";
    }

    render_device.waitForIdle();

    objects[0].destroyStaticObject(render_device);
    objects[1].destroyStaticObject(render_device);

    render_device.quit();

    glfwDestroyWindow(window);

    glfwTerminate();
}