#ifndef JED_GFX_GFX_HELPER_HPP
#define JED_GFX_GFX_HELPER_HPP

#include <cstddef>
#include <cstdint>

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

#ifndef COMMANDS_PER_BUCKET
#define COMMANDS_PER_BUCKET 8
#endif

#ifndef DESCRIPTORS_PER_POOL
#define DESCRIPTORS_PER_POOL 8
#endif

namespace gfx
{

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
using PushConstantHandle    = size_t;
using VertexBindingHandle   = size_t;
using VertexAttributeHandle = size_t;
using ShaderHandle          = size_t;
using PipelineHandle        = size_t;
using BufferHandle          = size_t;
using TextureHandle         = size_t;

using UniformSetHandle = uint16_t;

struct UniformHandle
{
    UniformSetHandle set;
    uint16_t         generation;
    uint32_t         uniform;
};

}; // namespace gfx

#endif