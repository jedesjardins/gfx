#ifndef JED_GFX_GFX_HELPER_HPP
#define JED_GFX_GFX_HELPER_HPP

#ifndef LOG_TRACE
#define LOG_TRACE(...)
#endif

#ifndef LOG_DEBUG
#define LOG_DEBUG(...)
#endif

#ifndef LOG_INFO
#define LOG_INFO(...)
#endif

#ifndef LOG_WARN
#define LOG_WARN(...)
#endif

#ifndef LOG_ERROR
#define LOG_ERROR(...)
#endif

#ifndef LOG_CRITICAL
#define LOG_CRITICAL(...)
#endif

#define VK_CHECK_RESULT(op, error_message)          \
    do                                              \
    {                                               \
        VkResult result = op;                       \
        if (result != VK_SUCCESS)                   \
        {                                           \
            LOG_CRITICAL("{} {} {} {} {} {} {}",    \
                         error_message,             \
                         "with value",              \
                         gfx::error_string(result), \
                         "in",                      \
                         __FILE__,                  \
                         "at line",                 \
                         __LINE__);                 \
            assert(result == VK_SUCCESS);           \
            return ErrorCode::VULKAN_ERROR;         \
        }                                           \
    } while (0)

namespace gfx
{
char const * error_string(VkResult error_code);

enum class ErrorCode
{
    NONE,
    VULKAN_ERROR,
    API_ERROR,
    JSON_ERROR,
    FILE_ERROR
};

enum class Format
{
    USE_DEPTH,
    USE_COLOR
};

//
//  HANDLES
//

using CommandbufferHandle   = size_t;
using RenderPassHandle      = size_t;
using SubpassHandle         = size_t;
using AttachmentHandle      = size_t;
using FramebufferHandle     = size_t;
using UniformLayoutHandle   = size_t;
using PushConstantHandle    = size_t;
using VertexBindingHandle   = size_t;
using VertexAttributeHandle = size_t;
using ShaderHandle          = size_t;
using PipelineHandle        = size_t;
using BufferHandle          = size_t;
using TextureHandle         = size_t;

struct UniformHandle
{
    uint64_t uniform_layout_id : 32;
    uint64_t uniform_id : 32;
}; // struct UniformHandle

}; // namespace gfx

#endif