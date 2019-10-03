

#define JED_LOG_IMPLEMENTATION
#include "log/logger.hpp"
#undef JED_LOG_IMPLEMENTATION

#define JED_GFX_IMPLEMENTATION
#include "gfx/renderer.hpp"
#undef JED_GFX_IMPLEMENTATION

#define JED_CMD_IMPLEMENTATION
#include "cmd/cmd.hpp"
#undef JED_CMD_IMPLEMENTATION

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