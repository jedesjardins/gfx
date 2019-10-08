

#define JED_LOG_IMPLEMENTATION
#include "log/logger.hpp"
#undef JED_LOG_IMPLEMENTATION

#define JED_GFX_IMPLEMENTATION
#include "gfx/renderer.hpp"
#undef JED_GFX_IMPLEMENTATION

#define JED_CMD_IMPLEMENTATION
#include "cmd/cmd.hpp"
#undef JED_CMD_IMPLEMENTATION

#include "common.hpp"

#include <fstream>

gfx::ErrorCode readFile(char const * file_name, std::vector<char> & buffer)
{
    LOG_DEBUG("Reading file {}", file_name);
    std::ifstream file(file_name, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        LOG_ERROR("Failed to open file {}", file_name);
        return gfx::ErrorCode::FILE_ERROR;
    }

    size_t            file_size = (size_t)file.tellg();
    buffer.resize(file_size);

    file.seekg(0);
    file.read(buffer.data(), file_size);

    file.close();

    return gfx::ErrorCode::NONE;
}

BufferUniform make_matrix_uniform(gfx::Renderer & renderer, std::string const& set_name, glm::mat4 & view_matrix)
{

    auto uniform = BufferUniform{};

    uniform.buffer_handle = renderer
                      .create_buffer(
                          sizeof(glm::mat4),
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                      .value();

    renderer.update_buffer(uniform.buffer_handle, sizeof(glm::mat4), static_cast<void *>(glm::value_ptr(view_matrix)));

    auto opt_set_handle = renderer.get_uniform_set_handle("us_camera_matrix").value();

    gfx::BufferWrite buffer_info{};
    buffer_info.buffer = uniform.buffer_handle;
    buffer_info.offset = 0;
    buffer_info.size = sizeof(glm::mat4);

    gfx::UniformWrite write_info{};
    write_info.first_array_element = 0;
    write_info.buffer_write_count = 1;
    write_info.buffer_writes= &buffer_info;

    uniform.uniform_handle = renderer.create_uniform(opt_set_handle, 1, &write_info).value();

    return uniform;
}