
#include "log/logger.hpp"
#include "gfx/renderer.hpp"
#include "common.hpp"

#include "imgui.h"
#include "examples/imgui_impl_glfw.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION

#include <glm/glm.hpp>

gfx::TextureHandle    g_font_texture;
gfx::UniformHandle    g_font_uniform;

bool escape = false;

void key_callback(GLFWwindow * window, int key, int scancode, int action, int mods)
{
    switch (key)
    {
    case GLFW_KEY_ESCAPE:
        escape = action != GLFW_RELEASE;
        break;
    }
}

std::optional<gfx::TextureHandle> create_texture(gfx::Renderer & renderer,
                                                 char const *    texture_path)
{
    int       texWidth, texHeight, texChannels;
    stbi_uc * pixels = stbi_load(texture_path, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

    if (!pixels)
    {
        LOG_ERROR("Failed to load texture image {}", texture_path);
        return std::nullopt;
    }

    auto texture = renderer.create_texture(texWidth, texHeight, 4, pixels);

    stbi_image_free(pixels);

    return texture;
}

//
// Imgui Helper Functions
//

struct ImguiPushConstant
{
    glm::vec2 scale;
    glm::vec2 translate;
};

void ImGui_Implgfx_CreateFontsTexture(gfx::Renderer & renderer)
{
    unsigned char * pixels;
    int             width  = 0;
    int             height = 0;

    ImGuiIO & io = ImGui::GetIO();
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    // size_t upload_size = width*height*4*sizeof(char);

    LOG_INFO("Size of texture {}", width * height * 4);

    g_font_texture = renderer.create_texture(width, height, 4, pixels).value();

    g_font_uniform = make_texture_uniform(renderer, "us_texture", g_font_texture);
}

void ImGui_Implgfx_RenderDrawData(gfx::Renderer & renderer, ImDrawData * draw_data)
{
    // copy vertices into vertex and index buffers

    // setup render state?

    auto pipeline_handle = renderer.get_pipeline_handle("imgui_pipeline").value();

    ImguiPushConstant push_constant;
    push_constant.scale     = {2.0f / draw_data->DisplaySize.x, 2.0f / draw_data->DisplaySize.y};
    push_constant.translate = {-1.0f - draw_data->DisplayPos.x * push_constant.scale.x,
                               -1.0f - draw_data->DisplayPos.y * push_constant.scale.y};

    int fb_width  = (int)(draw_data->DisplaySize.x * draw_data->FramebufferScale.x);
    int fb_height = (int)(draw_data->DisplaySize.y * draw_data->FramebufferScale.y);
    if (fb_width <= 0 || fb_height <= 0 || draw_data->TotalVtxCount == 0)
        return;

    // create host visible buffer for vertex and index buffer

    VkDeviceSize vertex_size = draw_data->TotalVtxCount * sizeof(ImDrawVert);
    VkDeviceSize index_size  = draw_data->TotalIdxCount * sizeof(ImDrawIdx);

    auto vertex_buffer_handle = renderer
                                    .create_buffer(vertex_size,
                                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                       | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                    .value();
    auto index_buffer_handle = renderer
                                   .create_buffer(index_size,
                                                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                      | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                   .value();

    char * mapped_vertices = reinterpret_cast<char *>(
        renderer.map_buffer(vertex_buffer_handle).value());
    char * mapped_indices = reinterpret_cast<char *>(
        renderer.map_buffer(index_buffer_handle).value());

    size_t vertex_offset = 0;
    size_t index_offset  = 0;
    for (int n = 0; n < draw_data->CmdListsCount; n++)
    {
        const ImDrawList * cmd_list = draw_data->CmdLists[n];

        memcpy(mapped_vertices + vertex_offset,
               cmd_list->VtxBuffer.Data,
               cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy(mapped_indices + index_offset,
               cmd_list->IdxBuffer.Data,
               cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));

        vertex_offset += cmd_list->VtxBuffer.Size * sizeof(ImDrawVert);
        index_offset += cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx);
    }

    ImVec2 clip_off   = draw_data->DisplayPos;
    ImVec2 clip_scale = draw_data->FramebufferScale;

    int global_vtx_offset = 0;
    int global_idx_offset = 0;
    for (int n = 0; n < draw_data->CmdListsCount; n++)
    {
        const ImDrawList * cmd_list = draw_data->CmdLists[n];

        for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++)
        {
            const ImDrawCmd * pcmd = &cmd_list->CmdBuffer[cmd_i];
            if (pcmd->UserCallback != NULL)
            {
                /*
                // User callback, registered via ImDrawList::AddCallback()
                // (ImDrawCallback_ResetRenderState is a special callback value used by the user to
                request the renderer to reset render state.) if (pcmd->UserCallback ==
                ImDrawCallback_ResetRenderState) ImGui_ImplVulkan_SetupRenderState(draw_data,
                command_buffer, rb, fb_width, fb_height); else pcmd->UserCallback(cmd_list, pcmd);
                */
                LOG_INFO("Not Drawing");
            }
            else
            {
                ImVec4 clip_rect;
                clip_rect.x = (pcmd->ClipRect.x - clip_off.x) * clip_scale.x;
                clip_rect.y = (pcmd->ClipRect.y - clip_off.y) * clip_scale.y;
                clip_rect.z = (pcmd->ClipRect.z - clip_off.x) * clip_scale.x;
                clip_rect.w = (pcmd->ClipRect.w - clip_off.y) * clip_scale.y;

                if (clip_rect.x < fb_width && clip_rect.y < fb_height && clip_rect.z >= 0.0f
                    && clip_rect.w >= 0.0f)
                {
                    // Negative offsets are illegal for vkCmdSetScissor
                    if (clip_rect.x < 0.0f)
                        clip_rect.x = 0.0f;
                    if (clip_rect.y < 0.0f)
                        clip_rect.y = 0.0f;

                    // Apply scissor/clipping rectangle
                    VkRect2D scissor;
                    scissor.offset.x      = (int32_t)(clip_rect.x);
                    scissor.offset.y      = (int32_t)(clip_rect.y);
                    scissor.extent.width  = (uint32_t)(clip_rect.z - clip_rect.x);
                    scissor.extent.height = (uint32_t)(clip_rect.w - clip_rect.y);

                    VkDeviceSize vertex_buffer_offset = (pcmd->VtxOffset + global_vtx_offset)
                                                        * sizeof(ImDrawVert);

                    gfx::DrawParameters params{};

                    params.pipeline = pipeline_handle;

                    params.vertex_buffer_count   = 1;
                    params.vertex_buffers        = &vertex_buffer_handle;
                    params.vertex_buffer_offsets = &vertex_buffer_offset;

                    params.index_buffer        = index_buffer_handle;
                    params.index_buffer_offset = pcmd->IdxOffset + global_idx_offset;
                    params.index_count         = pcmd->ElemCount;

                    params.push_constant_size = sizeof(ImguiPushConstant);
                    params.push_constant_data = &push_constant;

                    if (pcmd->TextureId != nullptr)
                    {
                        LOG_INFO("Texture is set");
                        params.uniform_count = 1;
                        params.uniforms      = static_cast<gfx::UniformHandle*>(pcmd->TextureId);
                    }
                    else
                    {
                        LOG_INFO("Texture is not set");
                        params.uniform_count = 1;
                        params.uniforms      = &g_font_uniform;
                    }

                    params.scissor  = &scissor;
                    params.viewport = nullptr;

                    renderer.draw(params);
                }
            }
        }
        global_idx_offset += cmd_list->IdxBuffer.Size;
        global_vtx_offset += cmd_list->VtxBuffer.Size;
    }

    std::array<gfx::BufferHandle, 2> delete_buffers = {vertex_buffer_handle, index_buffer_handle};

    renderer.delete_buffers(delete_buffers.size(), delete_buffers.data());
}

void ImGui_Implgfx_RenderFrame(gfx::Renderer & renderer)
{
    ImGui_Implgfx_RenderDrawData(renderer, ImGui::GetDrawData());
}

int main()
{
    get_console_sink()->set_level(spdlog::level::info);
    get_file_sink()->set_level(spdlog::level::debug);
    get_logger()->set_level(spdlog::level::debug);

    LOG_INFO("ImGui example");

    if (glfwInit() == GLFW_FALSE)
    {
        LOG_ERROR("GLFW didn't initialize correctly");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    auto window = glfwCreateWindow(600, 400, "Gfx+Imgui", nullptr, nullptr);

    glfwSetKeyCallback(window, key_callback);

    auto render_config = gfx::RenderConfig{};

    if (render_config.init(RESOURCE_PATH "imgui/imgui_gfx_config.json", readFile)
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

    // setup objects here

    ImGui::CreateContext();
    ImGuiIO & io = ImGui::GetIO();

    ImGui::StyleColorsDark();

    // Probably should pass false here, so I can set callbacks myself
    ImGui_ImplGlfw_InitForVulkan(window, true);

    // upload font here
    ImGui_Implgfx_CreateFontsTexture(renderer);

    gfx::TextureHandle sword_texture = create_texture(renderer, RESOURCE_PATH "sword.png").value();

    gfx::UniformHandle sword_uniform = make_texture_uniform(renderer, "us_texture", sword_texture);

    bool show_window = true;
    while (!glfwWindowShouldClose(window) && !escape)
    {
        glfwPollEvents();

        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (show_window)
        {
            ImGui::Begin("Window", &show_window);
            ImGui::Text("Text");
            if (ImGui::Button("Button"))
            {
                ImGui::Text("Pressed");
            }

            ImGui::Image(static_cast<void *>(&sword_uniform), ImVec2(50.0, 50.0));

            ImGui::End();
        }

        ImGui::Render();

        ImGui_Implgfx_RenderFrame(renderer);

        renderer.submit_frame();
    }

    renderer.wait_for_idle();

    renderer.quit();

    glfwDestroyWindow(window);

    glfwTerminate();

    LOG_INFO("Stopping ImGui\n");
}