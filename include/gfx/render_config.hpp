
#ifndef JED_GFX_RENDER_CONFIG_HPP
#define JED_GFX_RENDER_CONFIG_HPP

#include <vulkan/vulkan.h>

#include "gfx/gfx_helper.hpp"

#include "rapidjson/document.h"

namespace gfx
{
//
//  CONFIGURATION STRUCTURES
//

struct AttachmentConfig
{
    Format                format;
    VkImageUsageFlags     usage;
    VkSampleCountFlagBits multisamples;
    bool                  is_swapchain_image;
    bool                  use_swapchain_size;
    VkExtent2D            extent;

    friend bool operator==(AttachmentConfig const & lhs, AttachmentConfig const & rhs);
    friend bool operator!=(AttachmentConfig const & lhs, AttachmentConfig const & rhs);
};

struct SubpassInfo
{
    std::vector<VkAttachmentReference>   color_attachments;
    std::vector<VkAttachmentReference>   color_resolve_attachments;
    std::vector<VkAttachmentReference>   input_attachments;
    std::vector<uint32_t>                preserve_attachments;
    std::optional<VkAttachmentReference> depth_stencil_attachment;

    VkSampleCountFlagBits multisamples;

    friend bool operator==(SubpassInfo const & lhs, SubpassInfo const & rhs);
    friend bool operator!=(SubpassInfo const & lhs, SubpassInfo const & rhs);
};

struct RenderPassConfig
{
    std::unordered_map<std::string, size_t>                  attachments;
    std::unordered_map<std::string, VkAttachmentDescription> descriptions;
    std::unordered_map<std::string, VkClearValue>            clear_values;
    std::unordered_map<std::string, SubpassHandle>           subpass_handles;
    std::vector<SubpassInfo>                                 subpasses;
    std::vector<VkSubpassDependency>                         subpass_dependencies;

    friend bool operator==(RenderPassConfig const & lhs, RenderPassConfig const & rhs);
    friend bool operator!=(RenderPassConfig const & lhs, RenderPassConfig const & rhs);
};

struct PipelineConfig
{
    std::string vertex_shader_name;
    std::string fragment_shader_name;

    std::vector<std::string> vertex_binding_names;
    std::vector<std::string> vertex_attribute_names;

    std::vector<std::string> uniform_layout_names;
    std::vector<std::string> push_constant_names;

    std::string render_pass;
    std::string subpass;

    bool blendable;
    bool tests_depth;

    std::vector<VkDynamicState> dynamic_state;
};

/*
struct UniformConfig
{
    size_t                       max_uniform_count;
    VkDescriptorSetLayoutBinding layout_binding;
};
*/

using ReadFileFn = ErrorCode (*)(char const * file_name, std::vector<char> & buffer);

struct RenderConfig
{
    ReadFileFn read_file;

    char const * config_filename;

    char const * window_name;

    std::unordered_map<std::string, RenderPassConfig> render_pass_configs;

    std::vector<std::string> render_pass_order;

    std::unordered_map<std::string, AttachmentConfig> attachment_configs;

    /*
    std::unordered_map<std::string, UniformConfig> uniform_configs;
    */

    std::unordered_map<std::string, VkDescriptorSetLayoutBinding> uniform_bindings;

    std::unordered_map<std::string, std::vector<std::string>> uniform_sets;

    std::unordered_map<std::string, VkPushConstantRange> push_constants;

    std::unordered_map<std::string, VkVertexInputBindingDescription> vertex_bindings;

    std::unordered_map<std::string, VkVertexInputAttributeDescription> vertex_attributes;

    std::unordered_map<std::string, std::string> shader_names;

    std::unordered_map<std::string, PipelineConfig> pipeline_configs;

    ErrorCode init(char const * file_name, ReadFileFn read_file_fn);
};

}; // namespace gfx

#endif

//
// IMPLEMENTATION
//

#ifdef JED_GFX_IMPLEMENTATION

namespace gfx
{
bool operator==(VkExtent2D const & lhs, VkExtent2D const & rhs)
{
    return (lhs.width == rhs.width) && (lhs.height == rhs.height);
}

bool operator!=(VkExtent2D const & lhs, VkExtent2D const & rhs)
{
    return !(lhs == rhs);
}

bool operator==(VkAttachmentDescription const & lhs, VkAttachmentDescription const & rhs)
{
    return lhs.flags == rhs.flags && lhs.format == rhs.format && lhs.samples == rhs.samples
           && lhs.loadOp == rhs.loadOp && lhs.storeOp == rhs.storeOp
           && lhs.stencilLoadOp == rhs.stencilLoadOp && lhs.stencilStoreOp == rhs.stencilStoreOp
           && lhs.initialLayout == rhs.initialLayout && lhs.finalLayout == rhs.finalLayout;
}

bool operator!=(VkAttachmentDescription const & lhs, VkAttachmentDescription const & rhs)
{
    return !(lhs == rhs);
}

bool operator==(VkAttachmentReference const & lhs, VkAttachmentReference const & rhs)
{
    return lhs.attachment == rhs.attachment && lhs.layout == rhs.layout;
}

bool operator!=(VkAttachmentReference const & lhs, VkAttachmentReference const & rhs)
{
    return !(lhs == rhs);
}

bool operator==(SubpassInfo const & lhs, SubpassInfo const & rhs)
{
    // color attachments
    if (lhs.color_attachments.size() != rhs.color_attachments.size())
    {
        return false;
    }

    for (uint32_t i = 0; i < lhs.color_attachments.size(); ++i)
    {
        if (lhs.color_attachments[i] != rhs.color_attachments[i])
        {
            return false;
        }
    }

    // color resolve attachments
    if (lhs.color_resolve_attachments.size() != rhs.color_resolve_attachments.size())
    {
        return false;
    }

    for (uint32_t i = 0; i < lhs.color_resolve_attachments.size(); ++i)
    {
        if (lhs.color_resolve_attachments[i] != rhs.color_resolve_attachments[i])
        {
            return false;
        }
    }

    // preserve attachments
    if (lhs.preserve_attachments.size() != rhs.preserve_attachments.size())
    {
        return false;
    }

    for (uint32_t i = 0; i < lhs.preserve_attachments.size(); ++i)
    {
        if (lhs.preserve_attachments[i] != rhs.preserve_attachments[i])
        {
            return false;
        }
    }

    // input attachments
    if (lhs.input_attachments.size() != rhs.input_attachments.size())
    {
        return false;
    }

    for (uint32_t i = 0; i < lhs.input_attachments.size(); ++i)
    {
        if (lhs.input_attachments[i] != rhs.input_attachments[i])
        {
            return false;
        }
    }

    // depth attachment
    if (lhs.depth_stencil_attachment && lhs.depth_stencil_attachment)
    {
        return lhs.depth_stencil_attachment.value() == rhs.depth_stencil_attachment.value();
    }
    else
    {
        return !lhs.depth_stencil_attachment && !lhs.depth_stencil_attachment;
    }

    return true;
}

bool operator!=(SubpassInfo const & lhs, SubpassInfo const & rhs)
{
    return !(lhs == rhs);
}

bool operator==(VkSubpassDependency const & lhs, VkSubpassDependency const & rhs)
{
    return lhs.srcSubpass == rhs.srcSubpass && lhs.dstSubpass == rhs.dstSubpass
           && lhs.srcStageMask == rhs.srcStageMask && lhs.dstStageMask == rhs.dstStageMask
           && lhs.srcAccessMask == rhs.srcAccessMask && lhs.dstAccessMask == rhs.dstAccessMask
           && lhs.dependencyFlags == rhs.dependencyFlags;
}

bool operator!=(VkSubpassDependency const & lhs, VkSubpassDependency const & rhs)
{
    return !(lhs == rhs);
}

bool operator==(RenderPassConfig const & lhs, RenderPassConfig const & rhs)
{
    if (lhs.subpasses.size() != rhs.subpasses.size())
    {
        return false;
    }

    for (uint32_t i = 0; i < lhs.subpasses.size(); ++i)
    {
        if (lhs.subpasses[i] != rhs.subpasses[i])
        {
            return false;
        }
    }

    if (lhs.subpass_dependencies.size() != rhs.subpass_dependencies.size())
    {
        return false;
    }

    for (uint32_t i = 0; i < lhs.subpass_dependencies.size(); ++i)
    {
        if (lhs.subpass_dependencies[i] != rhs.subpass_dependencies[i])
        {
            return false;
        }
    }

    return true;
}

bool operator!=(RenderPassConfig const & lhs, RenderPassConfig const & rhs)
{
    return !(lhs == rhs);
}

//
// CONFIGURATION CODE
//

std::optional<VkDynamicState> getVkDynamicState(std::string const & state_name)
{
#define MAP_PAIR(value)                  \
    {                                    \
#value, VK_DYNAMIC_STATE_##value \
    }

    static std::unordered_map<std::string, VkDynamicState> states{MAP_PAIR(VIEWPORT),
                                                                  MAP_PAIR(SCISSOR)};

#undef MAP_PAIR

    auto state = states.find(state_name);
    assert(state != states.end());
    if (state == states.end())
    {
        return std::nullopt;
    }

    return state->second;
}

std::optional<VkSampleCountFlagBits> getVkSampleCountFlagBits(size_t num_samples)
{
#define SWITCH_CASE(value) \
    case value:            \
        return VK_SAMPLE_COUNT_##value##_BIT
    switch (num_samples)
    {
        SWITCH_CASE(64);
        SWITCH_CASE(32);
        SWITCH_CASE(16);
        SWITCH_CASE(8);
        SWITCH_CASE(4);
        SWITCH_CASE(2);
        SWITCH_CASE(1);
    default:
        return std::nullopt;
    };
#undef SWITCH_CASE
}

std::optional<VkImageLayout> getVkImageLayout(std::string const & layout_name)
{
#define MAP_PAIR(value)                 \
    {                                   \
#value, VK_IMAGE_LAYOUT_##value \
    }

    static std::unordered_map<std::string, VkImageLayout> layouts{
        MAP_PAIR(UNDEFINED),
        MAP_PAIR(GENERAL),
        MAP_PAIR(COLOR_ATTACHMENT_OPTIMAL),
        MAP_PAIR(DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        MAP_PAIR(DEPTH_STENCIL_READ_ONLY_OPTIMAL),
        MAP_PAIR(SHADER_READ_ONLY_OPTIMAL),
        MAP_PAIR(TRANSFER_SRC_OPTIMAL),
        MAP_PAIR(TRANSFER_DST_OPTIMAL),
        MAP_PAIR(PREINITIALIZED),
        MAP_PAIR(DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL),
        MAP_PAIR(DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL),
        MAP_PAIR(PRESENT_SRC_KHR),
        MAP_PAIR(SHARED_PRESENT_KHR)};

#undef MAP_PAIR

    auto layout = layouts.find(layout_name);
    assert(layout != layouts.end());
    if (layout == layouts.end())
    {
        LOG_ERROR("Couldn't find VkImageLayout {}", layout_name);

        return std::nullopt;
    }

    return layout->second;
}

std::optional<VkImageUsageFlagBits> getVkImageUsageFlagBits(std::string const & usage_name)
{
#define MAP_PAIR(value)                      \
    {                                        \
#value, VK_IMAGE_USAGE_##value##_BIT \
    }

    static std::unordered_map<std::string, VkImageUsageFlagBits> bits{
        MAP_PAIR(TRANSFER_SRC),
        MAP_PAIR(TRANSFER_DST),
        MAP_PAIR(SAMPLED),
        MAP_PAIR(STORAGE),
        MAP_PAIR(COLOR_ATTACHMENT),
        MAP_PAIR(DEPTH_STENCIL_ATTACHMENT),
        MAP_PAIR(TRANSIENT_ATTACHMENT),
        MAP_PAIR(INPUT_ATTACHMENT),
    };

#undef MAP_PAIR

    auto bit = bits.find(usage_name);
    assert(bit != bits.end());
    if (bit == bits.end())
    {
        LOG_ERROR("Couldn't find VkImageUsageFlagBit {}", usage_name);
        return std::nullopt;
    }

    return bit->second;
}

std::optional<VkAttachmentLoadOp> getVkAttachmentLoadOp(std::string const & op_name)
{
#define MAP_PAIR(value)                       \
    {                                         \
#value, VK_ATTACHMENT_LOAD_OP_##value \
    }

    static std::unordered_map<std::string, VkAttachmentLoadOp> ops{
        MAP_PAIR(LOAD), MAP_PAIR(CLEAR), MAP_PAIR(DONT_CARE)};

#undef MAP_PAIR

    auto op = ops.find(op_name);
    assert(op != ops.end());
    if (op == ops.end())
    {
        LOG_ERROR("Couldn't find VkAttachmentLoadOp {}", op_name);

        return std::nullopt;
    }

    return op->second;
}

std::optional<VkAttachmentStoreOp> getVkAttachmentStoreOp(std::string const & op_name)
{
#define MAP_PAIR(value)                        \
    {                                          \
#value, VK_ATTACHMENT_STORE_OP_##value \
    }

    static std::unordered_map<std::string, VkAttachmentStoreOp> ops{MAP_PAIR(STORE),
                                                                    MAP_PAIR(DONT_CARE)};

#undef MAP_PAIR

    auto op = ops.find(op_name);
    assert(op != ops.end());
    if (op == ops.end())
    {
        LOG_ERROR("Couldn't find VkAttachmentStoreOp {}", op_name);
        return std::nullopt;
    }

    return op->second;
}

std::optional<VkPipelineStageFlagBits> getVkPipelineStageFlagBit(std::string const & bit_name)
{
#define MAP_PAIR(value)                         \
    {                                           \
#value, VK_PIPELINE_STAGE_##value##_BIT \
    }

    static std::unordered_map<std::string, VkPipelineStageFlagBits> bits = {
        MAP_PAIR(TOP_OF_PIPE),
        MAP_PAIR(DRAW_INDIRECT),
        MAP_PAIR(VERTEX_INPUT),
        MAP_PAIR(VERTEX_SHADER),
        MAP_PAIR(FRAGMENT_SHADER),
        MAP_PAIR(EARLY_FRAGMENT_TESTS),
        MAP_PAIR(LATE_FRAGMENT_TESTS),
        MAP_PAIR(COLOR_ATTACHMENT_OUTPUT),
        MAP_PAIR(COMPUTE_SHADER),
        MAP_PAIR(TRANSFER),
        MAP_PAIR(BOTTOM_OF_PIPE),
        MAP_PAIR(HOST),
        MAP_PAIR(ALL_GRAPHICS),
        MAP_PAIR(ALL_COMMANDS)};

#undef MAP_PAIR

    auto bit = bits.find(bit_name);
    assert(bit != bits.end());
    if (bit == bits.end())
    {
        LOG_ERROR("Couldn't find VkPipelineStageFlagBit {}", bit_name);
        return std::nullopt;
    }

    return bit->second;
}

std::optional<VkAccessFlagBits> getVkAccessFlagBit(std::string const & bit_name)
{
#define MAP_PAIR(value)                 \
    {                                   \
#value, VK_ACCESS_##value##_BIT \
    }

    static std::unordered_map<std::string, VkAccessFlagBits> bits = {
        MAP_PAIR(INDIRECT_COMMAND_READ),
        MAP_PAIR(INDEX_READ),
        MAP_PAIR(VERTEX_ATTRIBUTE_READ),
        MAP_PAIR(UNIFORM_READ),
        MAP_PAIR(INPUT_ATTACHMENT_READ),
        MAP_PAIR(SHADER_READ),
        MAP_PAIR(SHADER_WRITE),
        MAP_PAIR(COLOR_ATTACHMENT_READ),
        MAP_PAIR(COLOR_ATTACHMENT_WRITE),
        MAP_PAIR(DEPTH_STENCIL_ATTACHMENT_READ),
        MAP_PAIR(DEPTH_STENCIL_ATTACHMENT_WRITE),
        MAP_PAIR(TRANSFER_READ),
        MAP_PAIR(TRANSFER_WRITE),
        MAP_PAIR(HOST_READ),
        MAP_PAIR(HOST_WRITE),
        MAP_PAIR(MEMORY_READ),
        MAP_PAIR(MEMORY_WRITE)};

#undef MAP_PAIR

    auto bit = bits.find(bit_name);
    assert(bit != bits.end());
    if (bit == bits.end())
    {
        LOG_ERROR("Couldn't find VkPipelineStageFlagBit {}", bit_name);
        return std::nullopt;
    }

    return bit->second;
}

std::optional<VkShaderStageFlagBits> getVkShaderStageFlagBit(std::string const & flag_name)
{
#define MAP_PAIR(value)                       \
    {                                         \
#value, VK_SHADER_STAGE_##value##_BIT \
    }

    static std::unordered_map<std::string, VkShaderStageFlagBits> flags{
        MAP_PAIR(VERTEX),
        MAP_PAIR(FRAGMENT),
        MAP_PAIR(COMPUTE),
        MAP_PAIR(TESSELLATION_CONTROL),
        MAP_PAIR(TESSELLATION_EVALUATION)};

#undef MAP_PAIR

    auto flag = flags.find(flag_name);
    assert(flag != flags.end());
    if (flag == flags.end())
    {
        LOG_ERROR("Couldn't find VkShaderStageFlagBits {}", flag_name);
        return std::nullopt;
    }

    return flag->second;
}

std::optional<VkDescriptorType> getVkDescriptorType(std::string const & type_name)
{
#define MAP_PAIR(value)                    \
    {                                      \
#value, VK_DESCRIPTOR_TYPE_##value \
    }

    static std::unordered_map<std::string, VkDescriptorType> types{MAP_PAIR(COMBINED_IMAGE_SAMPLER),
                                                                   MAP_PAIR(SAMPLER),
                                                                   MAP_PAIR(SAMPLED_IMAGE),
                                                                   MAP_PAIR(STORAGE_IMAGE),
                                                                   MAP_PAIR(UNIFORM_TEXEL_BUFFER),
                                                                   MAP_PAIR(STORAGE_TEXEL_BUFFER),
                                                                   MAP_PAIR(UNIFORM_BUFFER),
                                                                   MAP_PAIR(STORAGE_BUFFER),
                                                                   MAP_PAIR(UNIFORM_BUFFER_DYNAMIC),
                                                                   MAP_PAIR(STORAGE_BUFFER_DYNAMIC),
                                                                   MAP_PAIR(INPUT_ATTACHMENT)};

#undef MAP_PAIR

    auto type = types.find(type_name);
    assert(type != types.end());
    if (type == types.end())
    {
        LOG_ERROR("Couldn't find VkDescriptorType {}", type_name);
        return std::nullopt;
    }

    return type->second;
}

std::optional<VkFormat> getVkFormat(std::string const & format_name)
{
#define MAP_PAIR(value)           \
    {                             \
#value, VK_FORMAT_##value \
    }
    static std::unordered_map<std::string, VkFormat> formats{MAP_PAIR(R32G32_SFLOAT),
                                                             MAP_PAIR(R32G32B32_SFLOAT),
                                                             MAP_PAIR(R32G32B32A32_SFLOAT),
                                                             MAP_PAIR(R8G8B8A8_UNORM)};

#undef MAP_PAIR

    auto format = formats.find(format_name);
    assert(format != formats.end());
    if (format == formats.end())
    {
        LOG_ERROR("Couldn't find VkFormat {}", format_name);
        return std::nullopt;
    }

    return format->second;
}

#define CALL_FUNC(var, func) var.func()

#ifdef NDEBUG

#define CHECK_JSON_TYPE(document, type_func)                            \
    do                                                                  \
    {                                                                   \
        if (!CALL_FUNC(document, type_func))                            \
        {                                                               \
            LOG_ERROR("Field failed {} check.", #document, #type_func); \
            return ErrorCode::JSON_ERROR;                               \
        }                                                               \
    } while (0)

#define CHECK_JSON_FIELD(document, name, type_func)                               \
    do                                                                            \
    {                                                                             \
        if (!document.HasMember(#name) || !CALL_FUNC(document[#name], type_func)) \
        {                                                                         \
            LOG_ERROR("Field {} was missing or {} failed.", #name, #type_func);   \
            return ErrorCode::JSON_ERROR;                                         \
        }                                                                         \
    } while (0)
#else

#define CHECK_JSON_TYPE(document, type_func) \
    assert(CALL_FUNC(document, type_func) && "Field failed " #type_func " check.")

#define CHECK_JSON_FIELD(document, name, type_func)                           \
    assert(document.HasMember(#name) && CALL_FUNC(document[#name], type_func) \
           && "Field " #name " was missing or " #type_func " failed.")
#endif

ErrorCode initAttachmentDescription(rapidjson::Value const &  document,
                                    VkAttachmentDescription & description)
{
    CHECK_JSON_TYPE(document, IsObject);

    CHECK_JSON_FIELD(document, initial_layout, IsString);
    auto opt_initial_layout = getVkImageLayout(document["initial_layout"].GetString());
    if (!opt_initial_layout)
    {
        LOG_ERROR("Couldn't initialize initial_layout with {}",
                  document["initial_layout"].GetString());
        return ErrorCode::JSON_ERROR;
    }
    description.initialLayout = opt_initial_layout.value();

    CHECK_JSON_FIELD(document, final_layout, IsString);
    auto opt_final_layout = getVkImageLayout(document["final_layout"].GetString());
    if (!opt_final_layout)
    {
        LOG_ERROR("Couldn't initialize final_layout with {}", document["final_layout"].GetString());
        return ErrorCode::JSON_ERROR;
    }
    description.finalLayout = opt_final_layout.value();

    CHECK_JSON_FIELD(document, load_op, IsString);
    auto opt_load_op = getVkAttachmentLoadOp(document["load_op"].GetString());
    if (!opt_load_op)
    {
        LOG_ERROR("Couldn't initialize load_op with {}", document["load_op"].GetString());
        return ErrorCode::JSON_ERROR;
    }
    description.loadOp = opt_load_op.value();

    CHECK_JSON_FIELD(document, store_op, IsString);
    auto opt_store_op = getVkAttachmentStoreOp(document["store_op"].GetString());
    if (!opt_store_op)
    {
        LOG_ERROR("Couldn't initialize store_op with {}", document["store_op"].GetString());
        return ErrorCode::JSON_ERROR;
    }
    description.storeOp = opt_store_op.value();

    if (document.HasMember("stencil_load_op"))
    {
        CHECK_JSON_TYPE(document["stencil_load_op"], IsString);
        auto opt_stencil_load = getVkAttachmentLoadOp(document["stencil_load_op"].GetString());
        if (!opt_stencil_load)
        {
            LOG_ERROR("Couldn't initialize stencil_load_op with {}",
                      document["stencil_load_op"].GetString());
            return ErrorCode::JSON_ERROR;
        }
        description.stencilLoadOp = opt_stencil_load.value();
    }
    else
    {
        description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    }

    if (document.HasMember("stencil_store_op"))
    {
        CHECK_JSON_TYPE(document["stencil_store_op"], IsString);
        auto opt_stencil_store = getVkAttachmentStoreOp(document["stencil_store_op"].GetString());
        if (!opt_stencil_store)
        {
            LOG_ERROR("Couldn't initialize stencil_store_op with {}",
                      document["stencil_store_op"].GetString());
            return ErrorCode::JSON_ERROR;
        }
        description.stencilStoreOp = opt_stencil_store.value();
    }
    else
    {
        description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    }

    return ErrorCode::NONE;
}

ErrorCode initAttachmentReference(rapidjson::Value &                        document,
                                  VkAttachmentReference &                   reference,
                                  std::unordered_map<std::string, size_t> & attachment_indices)
{
    CHECK_JSON_TYPE(document, IsObject);

    CHECK_JSON_FIELD(document, attachment_name, IsString);

    auto attachment_index_iter = attachment_indices.find(document["attachment_name"].GetString());
    if (attachment_index_iter != attachment_indices.end())
    {
        reference.attachment = attachment_index_iter->second;
    }
    else if (strcmp(document["attachment_name"].GetString(), "UNUSED") == 0)
    {
        reference.attachment = VK_ATTACHMENT_UNUSED;
    }
    else
    {
        LOG_ERROR("Couldn't find preserve_attachment {} in framebuffer",
                  document["attachment_name"].GetString());
        return ErrorCode::JSON_ERROR;
    }

    CHECK_JSON_FIELD(document, layout, IsString);

    auto opt_layout = getVkImageLayout(document["layout"].GetString());
    if (!opt_layout)
    {
        LOG_ERROR("Couldn't initialize layout with {}", document["layout"].GetString());
        return ErrorCode::JSON_ERROR;
    }
    reference.layout = opt_layout.value();

    return ErrorCode::NONE;
}

ErrorCode initClearValue(rapidjson::Value const & document, VkClearValue & value)
{
    CHECK_JSON_TYPE(document, IsObject);

    if (document.HasMember("color"))
    {
        value.color = {0.f, 0.f, 0.f, 1.f};

        auto & color = document["color"];

        CHECK_JSON_TYPE(color, IsArray);
        assert(color.Size() == 4);
        if (color.Size() != 4)
        {
            LOG_ERROR("Clear Color arrays must have 4 elements for RGBA");
            return ErrorCode::JSON_ERROR;
        }

        size_t i = 0;
        for (auto const & comp_value: color.GetArray())
        {
            CHECK_JSON_TYPE(comp_value, IsNumber);
            if (comp_value.IsInt())
            {
                value.color.float32[i] = comp_value.GetInt();
            }
            else if (comp_value.IsDouble())
            {
                value.color.float32[i] = comp_value.GetDouble();
            }
            else
            {
                LOG_ERROR("Clear Color Component values must be a double or integer");
                return ErrorCode::JSON_ERROR;
            }

            if (value.color.float32[i] > 1.0 || value.color.float32[i] < 0.0)
            {
                LOG_WARN(
                    "Clear Color Component values must be clamped between 0.0 and 1.0 (inclusive)");
            }

            ++i;
        }
        LOG_DEBUG("Parsed VkClearValue {} {} {} {}",
                  value.color.float32[0],
                  value.color.float32[1],
                  value.color.float32[2],
                  value.color.float32[3]);
    }
    else if (document.HasMember("depth_stencil"))
    {
        value.depthStencil = {1.f, 0};

        auto & depth_stencil = document["depth_stencil"];

        CHECK_JSON_TYPE(depth_stencil, IsArray);
        assert(depth_stencil.Size() == 2);
        if (depth_stencil.Size() != 2)
        {
            LOG_ERROR("Clear Depth Stencil arrays must have 2 elements for depth and stencil");
            return ErrorCode::JSON_ERROR;
        }

        CHECK_JSON_TYPE(depth_stencil[0], IsDouble);
        value.depthStencil.depth = depth_stencil[0].GetDouble();
        if (value.depthStencil.depth > 1.0 || value.depthStencil.depth < 0.0)
        {
            LOG_WARN("Depth values must be clamped between 0.0 and 1.0 (inclusive)");
        }

        CHECK_JSON_TYPE(depth_stencil[1], IsUint);
        value.depthStencil.stencil = depth_stencil[1].GetUint();

        LOG_DEBUG(
            "Parsed VkClearValue {} {}", value.depthStencil.depth, value.depthStencil.stencil);
    }

    return ErrorCode::NONE;
}

ErrorCode initStageFlags(rapidjson::Value & document, VkPipelineStageFlags & stage_flags)
{
    CHECK_JSON_TYPE(document, IsArray);

    for (auto & stage_name: document.GetArray())
    {
        CHECK_JSON_TYPE(stage_name, IsString);

        auto opt_flag = getVkPipelineStageFlagBit(stage_name.GetString());
        if (!opt_flag)
        {
            LOG_ERROR("Stage Flag {} wasn't recognized", stage_name.GetString());
            return ErrorCode::JSON_ERROR;
        }

        stage_flags = static_cast<VkPipelineStageFlags>(stage_flags | opt_flag.value());
    }

    return ErrorCode::NONE;
}

std::optional<uint32_t> initSubpassIndex(
    rapidjson::Value &                                     document,
    std::unordered_map<std::string, SubpassHandle> const & subpass_handles)
{
    assert(document.IsString());

    char const * subpass = document.GetString();

    if (strcmp(subpass, "EXTERNAL_SUBPASS") == 0)
    {
        return VK_SUBPASS_EXTERNAL;
    }
    else
    {
        auto iter = subpass_handles.find(subpass);
        if (iter != subpass_handles.end())
        {
            return iter->second;
        }
    }

    return std::nullopt;
}

ErrorCode initVkExtent2D(rapidjson::Value & document, VkExtent2D & extent)
{
    CHECK_JSON_TYPE(document, IsObject);

    CHECK_JSON_FIELD(document, width, IsUint);
    extent.width = document["width"].GetUint();

    CHECK_JSON_FIELD(document, height, IsUint);
    extent.height = document["height"].GetUint();

    return ErrorCode::NONE;
}

ErrorCode initVkDescriptorSetLayoutBinding(rapidjson::Value &             document,
                                           VkDescriptorSetLayoutBinding & layout)
{
    CHECK_JSON_TYPE(document, IsObject);

    CHECK_JSON_FIELD(document, binding, IsUint);
    layout.binding = document["binding"].GetUint();

    CHECK_JSON_FIELD(document, descriptor_type, IsString);
    auto opt_descriptor_type = getVkDescriptorType(document["descriptor_type"].GetString());
    if (!opt_descriptor_type)
    {
        LOG_ERROR("Couldn't initialize descriptor_type with {}",
                  document["descriptor_type"].GetString());
        return ErrorCode::JSON_ERROR;
    }
    layout.descriptorType = opt_descriptor_type.value();

    CHECK_JSON_FIELD(document, descriptor_count, IsUint);
    layout.descriptorCount = document["descriptor_count"].GetUint();

    CHECK_JSON_FIELD(document, stage, IsArray);
    layout.stageFlags = 0;
    for (auto & stage: document["stage"].GetArray())
    {
        CHECK_JSON_TYPE(stage, IsString);
        auto opt_shader_stage = getVkShaderStageFlagBit(stage.GetString());
        if (!opt_shader_stage)
        {
            LOG_ERROR("Couldn't initialize stage with {}", stage.GetString());
            return ErrorCode::JSON_ERROR;
        }

        layout.stageFlags |= opt_shader_stage.value();
    }

    return ErrorCode::NONE;
}

ErrorCode initAccessFlags(rapidjson::Value & document, VkAccessFlags & access_flags)
{
    CHECK_JSON_TYPE(document, IsArray);

    for (auto & access_name: document.GetArray())
    {
        CHECK_JSON_TYPE(access_name, IsString);

        auto opt_access_bit = getVkAccessFlagBit(access_name.GetString());
        if (!opt_access_bit)
        {
            LOG_ERROR("Access Flag {} wasn't recognized", access_name.GetString());
            return ErrorCode::JSON_ERROR;
        }

        access_flags = static_cast<VkAccessFlagBits>(access_flags | opt_access_bit.value());
    }

    return ErrorCode::NONE;
}

ErrorCode initDependency(rapidjson::Value &                                     document,
                         VkSubpassDependency &                                  dependency,
                         std::unordered_map<std::string, SubpassHandle> const & subpass_handles)
{
    CHECK_JSON_TYPE(document, IsObject);

    if (document.HasMember("src_subpass"))
    {
        auto opt_subpass_index = initSubpassIndex(document["src_subpass"], subpass_handles);
        if (!opt_subpass_index)
        {
            LOG_ERROR("src_subpass in subpass dependency couldn't be found");
            return ErrorCode::JSON_ERROR;
        }

        dependency.srcSubpass = opt_subpass_index.value();
    }

    if (document.HasMember("dst_subpass"))
    {
        auto opt_subpass_index = initSubpassIndex(document["dst_subpass"], subpass_handles);
        if (!opt_subpass_index)
        {
            LOG_ERROR("dst_subpass in subpass dependency couldn't be found");
            return ErrorCode::JSON_ERROR;
        }

        dependency.dstSubpass = opt_subpass_index.value();
    }

    if (document.HasMember("src_stage_mask"))
    {
        auto error = initStageFlags(document["src_stage_mask"], dependency.srcStageMask);
        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Subpass Dependency src_stage_mask couldn't be initialized");
            return error;
        }
    }

    if (document.HasMember("dst_stage_mask"))
    {
        auto error = initStageFlags(document["dst_stage_mask"], dependency.dstStageMask);
        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Subpass Dependency dst_stage_mask couldn't be initialized");
            return error;
        }
    }

    if (document.HasMember("src_access_mask"))
    {
        auto error = initAccessFlags(document["src_access_mask"], dependency.srcAccessMask);
        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Couldn't read configuration for src_access_mask in Subpass Dependency");
            return error;
        }
    }

    if (document.HasMember("dst_access_mask"))
    {
        auto error = initAccessFlags(document["dst_access_mask"], dependency.dstAccessMask);
        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Couldn't read configuration for src_access_mask in Subpass Dependency");
            return error;
        }
    }

    return ErrorCode::NONE;
}

ErrorCode initVkPushConstantRange(rapidjson::Value & document, VkPushConstantRange & push_constant)
{
    CHECK_JSON_TYPE(document, IsObject);

    CHECK_JSON_FIELD(document, offset, IsUint);
    push_constant.offset = document["offset"].GetUint();

    CHECK_JSON_FIELD(document, size, IsUint);
    push_constant.size = document["size"].GetUint();

    CHECK_JSON_FIELD(document, stage, IsArray);
    push_constant.stageFlags = 0;
    for (auto & stage: document["stage"].GetArray())
    {
        CHECK_JSON_TYPE(stage, IsString);
        auto opt_stage_flags = getVkShaderStageFlagBit(stage.GetString());
        if (!opt_stage_flags)
        {
            LOG_ERROR("Couldn't initialize stage with {}", stage.GetString());
        }

        push_constant.stageFlags |= opt_stage_flags.value();
    }

    return ErrorCode::NONE;
}

ErrorCode initVkVertexInputBindingDescription(rapidjson::Value &                document,
                                              VkVertexInputBindingDescription & vertex_binding)
{
    CHECK_JSON_TYPE(document, IsObject);

    CHECK_JSON_FIELD(document, binding_slot, IsUint);
    vertex_binding.binding = document["binding_slot"].GetUint();

    CHECK_JSON_FIELD(document, stride, IsUint);
    vertex_binding.stride = document["stride"].GetUint();

    CHECK_JSON_FIELD(document, input_rate, IsString);
    std::string input_rate = document["input_rate"].GetString();
    if (strcmp(input_rate.c_str(), "PER_VERTEX") == 0)
    {
        vertex_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    }
    else if (strcmp(input_rate.c_str(), "PER_INSTANCE") == 0)
    {
        vertex_binding.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    }
    else
    {
        LOG_ERROR("Vertex Binding input_rate {} not recognized", input_rate);
        return ErrorCode::JSON_ERROR;
    }

    return ErrorCode::NONE;
}

ErrorCode initVkVertexInputAttributeDescription(
    rapidjson::Value &                                               document,
    VkVertexInputAttributeDescription &                              attribute,
    std::unordered_map<std::string, VkVertexInputBindingDescription> vertex_bindings)
{
    CHECK_JSON_TYPE(document, IsObject);

    CHECK_JSON_FIELD(document, vertex_binding_name, IsString);
    auto iter = vertex_bindings.find(document["vertex_binding_name"].GetString());
    if (iter == vertex_bindings.end())
    {
        LOG_ERROR("Couldn't find Vertex Binding {} referenced in Vertex Attribute",
                  document["vertex_binding_name"].GetString());
        return ErrorCode::JSON_ERROR;
    }
    attribute.binding = iter->second.binding;

    CHECK_JSON_FIELD(document, location, IsUint);
    attribute.location = document["location"].GetUint();

    CHECK_JSON_FIELD(document, offset, IsUint);
    attribute.offset = document["offset"].GetUint();

    CHECK_JSON_FIELD(document, format, IsString);
    auto opt_format = getVkFormat(document["format"].GetString());
    if (!opt_format)
    {
        LOG_ERROR("Couldn't initialize format with {}", document["format"].GetString());
        return ErrorCode::JSON_ERROR;
    }
    attribute.format = opt_format.value();

    return ErrorCode::NONE;
}

//
// CONFIG STRUCT INITIALIZERS
//

ErrorCode init(rapidjson::Value & document, AttachmentConfig & attachment_config)
{
    CHECK_JSON_TYPE(document, IsObject);

    CHECK_JSON_FIELD(document, format, IsString);
    char const * format_str = document["format"].GetString();
    if (strcmp(format_str, "color") == 0)
    {
        attachment_config.format = Format::USE_COLOR;
    }
    else if (strcmp(format_str, "depth") == 0)
    {
        attachment_config.format = Format::USE_DEPTH;
    }

    CHECK_JSON_FIELD(document, usage, IsArray);
    attachment_config.usage = static_cast<VkImageUsageFlags>(0);
    for (auto & usage_bit_name: document["usage"].GetArray())
    {
        CHECK_JSON_TYPE(usage_bit_name, IsString);

        auto opt_usage = getVkImageUsageFlagBits(usage_bit_name.GetString());
        if (!opt_usage)
        {
            LOG_ERROR("Couldn't initialize usage with {}", usage_bit_name.GetString());
            return ErrorCode::JSON_ERROR;
        }

        attachment_config.usage |= opt_usage.value();
    }

    if (document.HasMember("multisamples"))
    {
        CHECK_JSON_TYPE(document["multisamples"], IsUint);
        auto opt_multisamples = getVkSampleCountFlagBits(document["multisamples"].GetUint());
        if (!opt_multisamples)
        {
            LOG_ERROR("Couldn't initialize multisamples with {}",
                      document["multisamples"].GetUint());
            return ErrorCode::JSON_ERROR;
        }

        attachment_config.multisamples = opt_multisamples.value();
    }
    else
    {
        attachment_config.multisamples = getVkSampleCountFlagBits(1).value();
    }

    if (document.HasMember("is_swapchain_image"))
    {
        CHECK_JSON_TYPE(document["is_swapchain_image"], IsBool);
        attachment_config.is_swapchain_image = document["is_swapchain_image"].GetBool();
    }
    else
    {
        attachment_config.is_swapchain_image = false;
    }

    if (!attachment_config.is_swapchain_image && document.HasMember("size"))
    {
        attachment_config.use_swapchain_size = false;

        // TODO: change initVkExtent2D to return an error
        auto error = initVkExtent2D(document["size"], attachment_config.extent);
        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Attachment Extent failed to initialize");
            return error;
        }
    }
    else
    {
        attachment_config.use_swapchain_size = true;
    }

    return ErrorCode::NONE;
}

ErrorCode init(rapidjson::Value &                        document,
               SubpassInfo &                             subpass_info,
               std::unordered_map<std::string, size_t> & attachment_indices)
{
    CHECK_JSON_TYPE(document, IsObject);

    if (document.HasMember("multisamples"))
    {
        CHECK_JSON_TYPE(document["multisamples"], IsUint);

        auto opt_multisamples = getVkSampleCountFlagBits(document["multisamples"].GetUint());
        if (!opt_multisamples)
        {
            LOG_ERROR("Couldn't initialize multisamples with {}",
                      document["multisamples"].GetUint());
            return ErrorCode::JSON_ERROR;
        }

        subpass_info.multisamples = opt_multisamples.value();
    }
    else
    {
        subpass_info.multisamples = getVkSampleCountFlagBits(1).value();
    }

    if (document.HasMember("color_attachments"))
    {
        CHECK_JSON_TYPE(document["color_attachments"], IsArray);

        subpass_info.color_attachments.reserve(document["color_attachments"].GetArray().Size());
        for (auto & ca: document["color_attachments"].GetArray())
        {
            subpass_info.color_attachments.push_back({});

            auto error = initAttachmentReference(
                ca, subpass_info.color_attachments.back(), attachment_indices);
            if (error != ErrorCode::NONE)
            {
                LOG_DEBUG("Couldn't initialize color_attachments");
                return error;
            }
        }
    }

    if (document.HasMember("color_resolve_attachments"))
    {
        CHECK_JSON_TYPE(document["color_resolve_attachments"], IsArray);

        subpass_info.color_resolve_attachments.reserve(
            document["color_resolve_attachments"].GetArray().Size());

        for (auto & cra: document["color_resolve_attachments"].GetArray())
        {
            subpass_info.color_resolve_attachments.push_back({});

            auto error = initAttachmentReference(
                cra, subpass_info.color_resolve_attachments.back(), attachment_indices);
            if (error != ErrorCode::NONE)
            {
                LOG_DEBUG("Couldn't initialize color_resolve_attachments");
                return error;
            }
        }
    }

    assert(subpass_info.color_attachments.size() == subpass_info.color_resolve_attachments.size()
           || subpass_info.color_resolve_attachments.size() == 0);
    if (subpass_info.color_attachments.size() != subpass_info.color_resolve_attachments.size()
        && subpass_info.color_resolve_attachments.size() != 0)
    {
        LOG_ERROR("Subpass's color_attachments need to map 1-to-1 with it's "
                  "color_resolve_attachments, or color_resolve_attachments must be empty");
        return ErrorCode::JSON_ERROR;
    }

    if (document.HasMember("input_attachments"))
    {
        CHECK_JSON_TYPE(document["input_attachments"], IsArray);

        subpass_info.input_attachments.reserve(document["input_attachments"].GetArray().Size());
        for (auto & ia: document["input_attachments"].GetArray())
        {
            subpass_info.input_attachments.push_back({});
            auto error = initAttachmentReference(
                ia, subpass_info.input_attachments.back(), attachment_indices);
            if (error != ErrorCode::NONE)
            {
                LOG_DEBUG("Couldn't initialize input_attachments");
                return error;
            }
        }
    }

    if (document.HasMember("preserve_attachments"))
    {
        CHECK_JSON_TYPE(document["preserve_attachments"], IsArray);

        for (auto & pa: document["preserve_attachments"].GetArray())
        {
            CHECK_JSON_TYPE(pa, IsString);

            auto attachment_index_iter = attachment_indices.find(pa.GetString());
            if (attachment_index_iter != attachment_indices.end())
            {
                subpass_info.preserve_attachments.push_back(attachment_index_iter->second);
            }
            else
            {
                LOG_ERROR("Couldn't find preserve_attachment {} in framebuffer", pa.GetString());
                return ErrorCode::JSON_ERROR;
            }
        }
    }

    if (document.HasMember("depth_stencil_attachment"))
    {
        VkAttachmentReference depth_stencil_reference;

        auto error = initAttachmentReference(
            document["depth_stencil_attachment"], depth_stencil_reference, attachment_indices);

        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Couldn't initialize depth_stencil_attachment");
            return error;
        }

        LOG_DEBUG("Initialized depth_stencil_attachment with {}",
                  depth_stencil_reference.attachment);

        subpass_info.depth_stencil_attachment = depth_stencil_reference;
    }
    else
    {
        subpass_info.depth_stencil_attachment = std::nullopt;
    }

    return ErrorCode::NONE;
}

ErrorCode init(rapidjson::Value & document, RenderPassConfig & render_pass_config)
{
    CHECK_JSON_TYPE(document, IsObject);

    CHECK_JSON_FIELD(document, framebuffer, IsArray);
    size_t attachment_index{0};
    for (auto & ad: document["framebuffer"].GetArray())
    {
        CHECK_JSON_TYPE(ad, IsObject);

        CHECK_JSON_FIELD(ad, attachment_name, IsString);
        std::string name = ad["attachment_name"].GetString();

        LOG_DEBUG("Render Pass uses {} at attachment_index {}", name, attachment_index);

        render_pass_config.attachments[name] = attachment_index++;

        auto error = initAttachmentDescription(ad, render_pass_config.descriptions[name]);
        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Couldn't initialize render pass attachment description");
            return error;
        }

        if (ad.HasMember("clear_value"))
        {
            auto error = initClearValue(
                ad["clear_value"],
                render_pass_config.clear_values[ad["attachment_name"].GetString()]);

            if (error != ErrorCode::NONE)
            {
                LOG_DEBUG("Couldn't initialize Render Pass Clear Value");
            }
        }
    }

    CHECK_JSON_FIELD(document, subpasses, IsArray);
    for (auto & sp: document["subpasses"].GetArray())
    {
        assert(sp.IsObject());
        CHECK_JSON_FIELD(sp, name, IsString);

        SubpassHandle handle = render_pass_config.subpasses.size();

        SubpassInfo info{};

        auto error = init(sp, info, render_pass_config.attachments);
        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Subpass {} failed to initialize", sp["name"].GetString());
            return error;
        }

        render_pass_config.subpass_handles[sp["name"].GetString()] = handle;
        render_pass_config.subpasses.push_back(info);
    }

    CHECK_JSON_FIELD(document, subpass_dependencies, IsArray);
    auto const & json_dependency_array = document["subpass_dependencies"].GetArray();
    render_pass_config.subpass_dependencies.resize(json_dependency_array.Size());
    for (size_t i = 0; i < json_dependency_array.Size(); ++i)
    {
        auto & sp_d      = render_pass_config.subpass_dependencies[i];
        auto & json_sp_d = json_dependency_array[i];

        auto error = initDependency(json_sp_d, sp_d, render_pass_config.subpass_handles);

        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Couldn't read configuration for Subpass Dependency at index {}", i);
        }
    }

    return ErrorCode::NONE;
}

ErrorCode init(rapidjson::Value &                                   document,
               PipelineConfig &                                     pipeline_config,
               std::unordered_map<std::string, std::string> const & shader_names)
{
    CHECK_JSON_TYPE(document, IsObject);

    CHECK_JSON_FIELD(document, vertex_shader_name, IsString);
    pipeline_config.vertex_shader_name = document["vertex_shader_name"].GetString();

    CHECK_JSON_FIELD(document, fragment_shader_name, IsString);
    pipeline_config.fragment_shader_name = document["fragment_shader_name"].GetString();

    CHECK_JSON_FIELD(document, vertex_bindings, IsArray);
    for (auto const & vbi: document["vertex_bindings"].GetArray())
    {
        CHECK_JSON_TYPE(vbi, IsString);
        pipeline_config.vertex_binding_names.push_back(vbi.GetString());
    }

    CHECK_JSON_FIELD(document, vertex_attributes, IsArray);
    for (auto const & vai: document["vertex_attributes"].GetArray())
    {
        CHECK_JSON_TYPE(vai, IsString);
        pipeline_config.vertex_attribute_names.push_back(vai.GetString());
    }

    CHECK_JSON_FIELD(document, uniform_sets, IsArray);
    for (auto const & uli: document["uniform_sets"].GetArray())
    {
        CHECK_JSON_TYPE(uli, IsString);
        pipeline_config.uniform_layout_names.push_back(uli.GetString());
    }

    CHECK_JSON_FIELD(document, push_constants, IsArray);
    for (auto const & pci: document["push_constants"].GetArray())
    {
        CHECK_JSON_TYPE(pci, IsString);
        pipeline_config.push_constant_names.push_back(pci.GetString());
    }

    CHECK_JSON_FIELD(document, render_pass, IsString);
    pipeline_config.render_pass = document["render_pass"].GetString();

    CHECK_JSON_FIELD(document, subpass, IsString);
    pipeline_config.subpass = document["subpass"].GetString();

    if (document.HasMember("blendable"))
    {
        CHECK_JSON_TYPE(document["blendable"], IsBool);
        pipeline_config.blendable = document["blendable"].GetBool();
    }
    else
    {
        pipeline_config.blendable = true;
    }

    if (document.HasMember("tests_depth"))
    {
        CHECK_JSON_TYPE(document["tests_depth"], IsBool);
        pipeline_config.tests_depth = document["tests_depth"].GetBool();
    }
    else
    {
        pipeline_config.tests_depth = false;
    }

    if (document.HasMember("dynamic_state"))
    {
        CHECK_JSON_TYPE(document["dynamic_state"], IsArray);

        pipeline_config.dynamic_state.reserve(document["dynamic_state"].GetArray().Size());
        for (auto const & state: document["dynamic_state"].GetArray())
        {
            CHECK_JSON_TYPE(state, IsString);

            auto opt_state = getVkDynamicState(state.GetString());
            if (!opt_state)
            {
                LOG_ERROR("Couldn't initialize dynamic_state with {}", state.GetString());
                return ErrorCode::JSON_ERROR;
            }

            pipeline_config.dynamic_state.push_back(opt_state.value());
        }
    }

    return ErrorCode::NONE;
}

/*
ErrorCode init(rapidjson::Value & document, UniformConfig & uniform_config)
{
    CHECK_JSON_TYPE(document, IsObject);

    CHECK_JSON_FIELD(document, max_count, IsUint);
    uniform_config.max_uniform_count = document["max_count"].GetUint();
    assert(uniform_config.max_uniform_count != 0);

    if (uniform_config.max_uniform_count == 0)
    {
        LOG_ERROR("Uniform Layout cannot have a max_count of 0");
        return ErrorCode::JSON_ERROR;
    }

    auto error = initVkDescriptorSetLayoutBinding(document, uniform_config.layout_binding);
    if (error != ErrorCode::NONE)
    {
        LOG_DEBUG("Couldn't read configuration for Uniform Layout");
        return error;
    }

    return ErrorCode::NONE;
}
*/

ErrorCode RenderConfig::init(char const * file_name, ReadFileFn read_file_fn)
{
    namespace rj = rapidjson;

    LOG_INFO("Initializing Render Configuration");

    assert(read_file_fn != nullptr);

    config_filename = file_name;
    read_file       = read_file_fn;

    rj::Document document;

    std::vector<char> config_json;

    if (file_name == nullptr)
    {
        LOG_ERROR("File path was not defined");
        return ErrorCode::API_ERROR;
    }

    if (read_file == nullptr)
    {
        LOG_ERROR("ReadFileFn was not defined");
        return ErrorCode::API_ERROR;
    }

    if (read_file(config_filename, config_json) != ErrorCode::NONE)
    {
        LOG_ERROR("Couldn't read configuration file {}", config_filename);
        return ErrorCode::FILE_ERROR;
    }

    // auto config_json = readFile(config_filename);
    config_json.push_back('\0');

    if (document.Parse(config_json.data()).HasParseError())
    {
        LOG_ERROR("Couldn't parse Render Configuration json: \"{}\"", config_json.data());
        return ErrorCode::JSON_ERROR;
    }
    else
    {
        LOG_DEBUG("Parsed Render Configuration file {}", config_filename);
    }

    CHECK_JSON_TYPE(document, IsObject);

    CHECK_JSON_FIELD(document, window_name, IsString);
    window_name = document["window_name"].GetString();

    CHECK_JSON_FIELD(document, attachments, IsArray);
    for (auto & a: document["attachments"].GetArray())
    {
        CHECK_JSON_TYPE(a, IsObject);

        CHECK_JSON_FIELD(a, name, IsString);

        auto error = gfx::init(a, attachment_configs[a["name"].GetString()]);
        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Couldn't read configuration for Attachment {}", a["name"].GetString());
            return error;
        }
    }
    LOG_DEBUG("Created {} Attachments", attachment_configs.size());

    CHECK_JSON_FIELD(document, render_passes, IsArray);
    for (auto & rp: document["render_passes"].GetArray())
    {
        CHECK_JSON_TYPE(rp, IsObject);
        CHECK_JSON_FIELD(rp, name, IsString);

        auto error = gfx::init(rp, render_pass_configs[rp["name"].GetString()]);
        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Couldn't read configuration for Render Pass {}", rp["name"].GetString());
            return error;
        }
    }

    CHECK_JSON_FIELD(document, render_pass_order, IsArray);
    for (auto & render_pass_name: document["render_pass_order"].GetArray())
    {
        CHECK_JSON_TYPE(render_pass_name, IsString);
        render_pass_order.push_back(render_pass_name.GetString());
    }

    CHECK_JSON_FIELD(document, shaders, IsArray);
    for (auto & s: document["shaders"].GetArray())
    {
        CHECK_JSON_TYPE(s, IsObject);
        CHECK_JSON_FIELD(s, name, IsString);
        CHECK_JSON_FIELD(s, file, IsString);

        shader_names[s["name"].GetString()] = s["file"].GetString();
    }

    CHECK_JSON_FIELD(document, uniforms, IsArray);
    for(auto & u: document["uniforms"].GetArray())
    {
        CHECK_JSON_TYPE(u, IsObject);
        CHECK_JSON_FIELD(u, name, IsString);

        VkDescriptorSetLayoutBinding binding;

        auto error = initVkDescriptorSetLayoutBinding(u, uniform_bindings[u["name"].GetString()]);
        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Couldn't read configuration for Uniform");
            return error;
        }
    }

    CHECK_JSON_FIELD(document, uniform_sets, IsArray);
    for (auto & us:  document["uniform_sets"].GetArray())
    {
        CHECK_JSON_TYPE(us, IsObject);
        CHECK_JSON_FIELD(us, name, IsString);
        CHECK_JSON_FIELD(us, uniforms, IsArray);

        auto & uniform_set = uniform_sets[us["name"].GetString()];

        if (us["uniforms"].GetArray().Size() == 0)
        {
            LOG_ERROR("Uniform Set {} must have at least one uniform", us["name"].GetString());
            return ErrorCode::JSON_ERROR;
        }

        for (auto & u: us["uniforms"].GetArray())
        {
            CHECK_JSON_TYPE(u, IsString);

            auto uniform_iter = uniform_bindings.find(u.GetString());
            if (uniform_iter == uniform_bindings.end())
            {
                LOG_ERROR("Couldn't find Uniform {} for Uniform Set {}", u.GetString(), us["name"].GetString());
                return ErrorCode::JSON_ERROR;
            }

            uniform_set.push_back(u.GetString());
        }
    }

    /*
    CHECK_JSON_FIELD(document, uniform_layouts, IsArray);
    for (auto & ul: document["uniform_layouts"].GetArray())
    {
        CHECK_JSON_TYPE(ul, IsObject);
        CHECK_JSON_FIELD(ul, name, IsString);

        auto error = gfx::init(ul, uniform_configs[ul["name"].GetString()]);
        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Couldn't read configuration for Uniform Layout {}", ul["name"].GetString());
            return error;
        }
    }
    */

    CHECK_JSON_FIELD(document, push_constants, IsArray);
    for (auto & pc: document["push_constants"].GetArray())
    {
        CHECK_JSON_TYPE(pc, IsObject);
        CHECK_JSON_FIELD(pc, name, IsString);

        auto error = initVkPushConstantRange(pc, push_constants[pc["name"].GetString()]);
        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Couldn't read the configuration for Push Constant {}",
                      pc["name"].GetString());
            return error;
        }
    }

    CHECK_JSON_FIELD(document, vertex_bindings, IsArray);
    for (auto & vb: document["vertex_bindings"].GetArray())
    {
        CHECK_JSON_TYPE(vb, IsObject);
        CHECK_JSON_FIELD(vb, name, IsString);

        auto error = initVkVertexInputBindingDescription(vb,
                                                         vertex_bindings[vb["name"].GetString()]);
        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Couldn't read the configuration for Vertex Binding {}",
                      vb["name"].GetString());
            return error;
        }
    }

    CHECK_JSON_FIELD(document, vertex_attributes, IsArray);
    for (auto & va: document["vertex_attributes"].GetArray())
    {
        CHECK_JSON_TYPE(va, IsObject);
        CHECK_JSON_FIELD(va, name, IsString);

        auto error = initVkVertexInputAttributeDescription(
            va, vertex_attributes[va["name"].GetString()], vertex_bindings);

        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Couldn't read the configuration for Vertex Attribute {}",
                      va["name"].GetString());
            return error;
        }
    }

    CHECK_JSON_FIELD(document, pipelines, IsArray);
    for (auto & p: document["pipelines"].GetArray())
    {
        CHECK_JSON_TYPE(p, IsObject);
        CHECK_JSON_FIELD(p, name, IsString);

        auto error = gfx::init(p, pipeline_configs[p["name"].GetString()], shader_names);
        if (error != ErrorCode::NONE)
        {
            LOG_DEBUG("Couldn't read configuration for Pipeline {}", p["name"].GetString());
            return error;
        }
    }

    return ErrorCode::NONE;
}

}; // namespace gfx

#endif