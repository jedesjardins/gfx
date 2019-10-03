
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
    std::vector<VkAttachmentReference> color_attachments;
    std::vector<VkAttachmentReference> color_resolve_attachments;
    std::vector<VkAttachmentReference> input_attachments;
    std::vector<uint32_t>              preserve_attachments;
    VkAttachmentReference              depth_stencil_attachment;

    VkSampleCountFlagBits multisamples;

    friend bool operator==(SubpassInfo const & lhs, SubpassInfo const & rhs);
    friend bool operator!=(SubpassInfo const & lhs, SubpassInfo const & rhs);
};

struct RenderpassConfig
{
    std::unordered_map<std::string, size_t>                  attachments;
    std::unordered_map<std::string, VkAttachmentDescription> descriptions;
    std::unordered_map<std::string, VkClearValue>            clear_values;
    std::unordered_map<std::string, SubpassHandle>           subpass_handles;
    std::vector<SubpassInfo>                                 subpasses;
    std::vector<VkSubpassDependency>                         subpass_dependencies;

    friend bool operator==(RenderpassConfig const & lhs, RenderpassConfig const & rhs);
    friend bool operator!=(RenderpassConfig const & lhs, RenderpassConfig const & rhs);
};

struct PipelineConfig
{
    std::string vertex_shader_name;
    std::string fragment_shader_name;

    std::vector<std::string> vertex_binding_names;
    std::vector<std::string> vertex_attribute_names;

    std::vector<std::string> uniform_layout_names;
    std::vector<std::string> push_constant_names;

    std::string renderpass;
    std::string subpass;

    bool blendable;
    bool tests_depth;

    std::vector<VkDynamicState> dynamic_state;
};

struct UniformConfig
{
    size_t                       max_uniform_count;
    VkDescriptorSetLayoutBinding layout_binding;
};

struct RenderConfig
{
    char const * config_filename;

    char const * window_name;

    std::unordered_map<std::string, RenderpassConfig> renderpass_configs;

    std::vector<std::string> renderpass_order;

    std::unordered_map<std::string, AttachmentConfig> attachment_configs;

    std::unordered_map<std::string, UniformConfig> uniform_configs;

    std::unordered_map<std::string, VkPushConstantRange> push_constants;

    std::unordered_map<std::string, VkVertexInputBindingDescription> vertex_bindings;

    std::unordered_map<std::string, VkVertexInputAttributeDescription> vertex_attributes;

    std::unordered_map<std::string, std::string> shader_names;

    std::unordered_map<std::string, PipelineConfig> pipeline_configs;

    void init();
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
    if (lhs.depth_stencil_attachment != rhs.depth_stencil_attachment)
    {
        return true;
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

bool operator==(RenderpassConfig const & lhs, RenderpassConfig const & rhs)
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

bool operator!=(RenderpassConfig const & lhs, RenderpassConfig const & rhs)
{
    return !(lhs == rhs);
}

//
// CONFIGURATION CODE
//

VkDynamicState getVkDynamicState(std::string const & state_name)
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
        return static_cast<VkDynamicState>(0);
    }

    return state->second;
}

VkSampleCountFlagBits getVkSampleCountFlagBits(size_t num_samples)
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
        return VK_SAMPLE_COUNT_1_BIT;
    };
#undef SWITCH_CASE
}

VkImageLayout getVkImageLayout(std::string const & layout_name)
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

        return VK_IMAGE_LAYOUT_UNDEFINED;
    }

    return layout->second;
}

VkImageUsageFlagBits getVkImageUsageFlagBits(std::string const & usage_name)
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
        return static_cast<VkImageUsageFlagBits>(0);
    }

    return bit->second;
}

VkAttachmentLoadOp getVkAttachmentLoadOp(std::string const & op_name)
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

        return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    }

    return op->second;
}

VkAttachmentStoreOp getVkAttachmentStoreOp(std::string const & op_name)
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
        return VK_ATTACHMENT_STORE_OP_DONT_CARE;
    }

    return op->second;
}

VkPipelineStageFlagBits getVkPipelineStageFlagBit(std::string const & bit_name)
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
        return static_cast<VkPipelineStageFlagBits>(0);
    }

    return bit->second;
}

VkAccessFlagBits getVkAccessFlagBit(std::string const & bit_name)
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
        return static_cast<VkAccessFlagBits>(0);
    }

    return bit->second;
}

VkShaderStageFlagBits getVkShaderStageFlagBit(std::string const & flag_name)
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
        return static_cast<VkShaderStageFlagBits>(0);
    }

    return flag->second;
}

VkDescriptorType getVkDescriptorType(std::string const & type_name)
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
        return static_cast<VkDescriptorType>(0);
    }

    return type->second;
}

VkFormat getVkFormat(std::string const & format_name)
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
        return static_cast<VkFormat>(0);
    }

    return format->second;
}

VkAttachmentDescription initAttachmentDescription(rapidjson::Value const & document)
{
    assert(document.IsObject());

    VkAttachmentDescription description{};

    assert(document.HasMember("initial_layout"));
    assert(document["initial_layout"].IsString());
    description.initialLayout = getVkImageLayout(document["initial_layout"].GetString());

    assert(document.HasMember("final_layout"));
    assert(document["final_layout"].IsString());
    description.finalLayout = getVkImageLayout(document["final_layout"].GetString());

    assert(document.HasMember("load_op"));
    assert(document["load_op"].IsString());
    description.loadOp = getVkAttachmentLoadOp(document["load_op"].GetString());

    assert(document.HasMember("store_op"));
    assert(document["store_op"].IsString());
    description.storeOp = getVkAttachmentStoreOp(document["store_op"].GetString());

    if (document.HasMember("stencil_load_op"))
    {
        assert(document["stencil_load_op"].IsString());
        description.stencilLoadOp = getVkAttachmentLoadOp(document["stencil_load_op"].GetString());
    }
    else
    {
        description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    }

    if (document.HasMember("stencil_store_op"))
    {
        assert(document["stencil_store_op"].IsString());
        description.stencilStoreOp = getVkAttachmentStoreOp(
            document["stencil_store_op"].GetString());
    }
    else
    {
        description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    }

    return description;
}

VkAttachmentReference initAttachmentReference(
    rapidjson::Value &                        document,
    std::unordered_map<std::string, size_t> & attachment_indices)
{
    assert(document.IsObject());

    VkAttachmentReference reference{};

    assert(document.HasMember("attachment_name"));
    assert(document["attachment_name"].IsString());

    auto attachment_index_iter = attachment_indices.find(document["attachment_name"].GetString());
    if (attachment_index_iter != attachment_indices.end())
    {
        reference.attachment = attachment_index_iter->second;
    }
    else
    {
        assert(strcmp(document["attachment_name"].GetString(), "UNUSED") == 0);
        reference.attachment = VK_ATTACHMENT_UNUSED;
    }

    assert(document.HasMember("layout"));
    assert(document["layout"].IsString());
    reference.layout = getVkImageLayout(document["layout"].GetString());

    return reference;
}

VkClearValue initClearValue(rapidjson::Value const & document)
{
    VkClearValue value;

    assert(document.IsObject());

    if (document.HasMember("color"))
    {
        value.color = {0.f, 0.f, 0.f, 1.f};

        assert(document["color"].IsArray());

        size_t i = 0;
        for (auto const & comp_value: document["color"].GetArray())
        {
            assert(comp_value.IsNumber());
            if (comp_value.IsInt())
            {
                value.color.float32[i] = comp_value.GetInt();
            }
            else if (comp_value.IsDouble())
            {
                value.color.float32[i] = comp_value.GetDouble();
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

        assert(depth_stencil.IsArray());
        assert(depth_stencil.Size() == 2);

        assert(depth_stencil[0].IsNumber());
        if (depth_stencil[0].IsDouble())
        {
            value.depthStencil.depth = depth_stencil[0].GetDouble();
        }
        else if (depth_stencil[0].IsUint())
        {
            value.depthStencil.depth = depth_stencil[0].GetUint();
        }

        assert(depth_stencil[1].IsUint());
        value.depthStencil.stencil = depth_stencil[1].GetUint();

        LOG_DEBUG(
            "Parsed VkClearValue {} {}", value.depthStencil.depth, value.depthStencil.stencil);
    }

    return value;
}

VkPipelineStageFlagBits initStageFlags(rapidjson::Value & document)
{
    assert(document.IsArray());

    VkPipelineStageFlagBits stage_flags{};

    for (auto & stage_name: document.GetArray())
    {
        assert(stage_name.IsString());
        stage_flags = static_cast<VkPipelineStageFlagBits>(
            stage_flags | getVkPipelineStageFlagBit(stage_name.GetString()));
    }

    return stage_flags;
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

VkExtent2D initVkExtent2D(rapidjson::Value & document)
{
    VkExtent2D extent{};
    assert(document.IsObject());
    assert(document.HasMember("width") && document["width"].IsUint());
    extent.width = document["width"].GetUint();

    assert(document.HasMember("height") && document["height"].IsUint());
    extent.height = document["height"].GetUint();

    LOG_DEBUG("Read extent {} {}", extent.width, extent.height);

    return extent;
}

VkDescriptorSetLayoutBinding initVkDescriptorSetLayoutBinding(rapidjson::Value & document)
{
    assert(document.IsObject());

    VkDescriptorSetLayoutBinding layout{};

    assert(document.HasMember("binding"));
    assert(document["binding"].IsInt());
    layout.binding = document["binding"].GetInt();

    assert(document.HasMember("descriptor_type"));
    assert(document["descriptor_type"].IsString());
    layout.descriptorType = getVkDescriptorType(document["descriptor_type"].GetString());

    assert(document.HasMember("descriptor_count"));
    assert(document["descriptor_count"].IsInt());
    layout.descriptorCount = document["descriptor_count"].GetInt();

    assert(document.HasMember("stage"));
    assert(document["stage"].IsArray());
    layout.stageFlags = 0;
    for (auto & stage: document["stage"].GetArray())
    {
        assert(stage.IsString());
        layout.stageFlags |= getVkShaderStageFlagBit(stage.GetString());
    }

    return layout;
}

VkAccessFlagBits initAccessFlags(rapidjson::Value & document)
{
    assert(document.IsArray());

    VkAccessFlagBits access_flags{};

    for (auto & access_name: document.GetArray())
    {
        assert(access_name.IsString());
        access_flags = static_cast<VkAccessFlagBits>(access_flags
                                                     | getVkAccessFlagBit(access_name.GetString()));
    }

    return access_flags;
}

VkSubpassDependency initDependency(
    rapidjson::Value &                                     document,
    std::unordered_map<std::string, SubpassHandle> const & subpass_handles)
{
    assert(document.IsObject());

    VkSubpassDependency dependency{};

    if (document.HasMember("src_subpass"))
    {
        dependency.srcSubpass = initSubpassIndex(document["src_subpass"], subpass_handles).value();
    }

    if (document.HasMember("dst_subpass"))
    {
        dependency.dstSubpass = initSubpassIndex(document["dst_subpass"], subpass_handles).value();
    }

    if (document.HasMember("src_stage_mask"))
    {
        dependency.srcStageMask = initStageFlags(document["src_stage_mask"]);
    }

    if (document.HasMember("dst_stage_mask"))
    {
        dependency.dstStageMask = initStageFlags(document["dst_stage_mask"]);
    }

    if (document.HasMember("src_access_mask"))
    {
        dependency.srcAccessMask = initAccessFlags(document["src_access_mask"]);
    }

    if (document.HasMember("dst_access_mask"))
    {
        dependency.dstAccessMask = initAccessFlags(document["dst_access_mask"]);
    }

    return dependency;
}

VkPushConstantRange initVkPushConstantRange(rapidjson::Value & document)
{
    assert(document.IsObject());

    VkPushConstantRange push_constant;

    assert(document.HasMember("offset"));
    assert(document["offset"].IsInt());
    push_constant.offset = document["offset"].GetInt();

    assert(document.HasMember("size"));
    assert(document["size"].IsInt());
    push_constant.size = document["size"].GetInt();

    assert(document.HasMember("stage"));
    assert(document["stage"].IsArray());
    push_constant.stageFlags = 0;
    for (auto & stage: document["stage"].GetArray())
    {
        assert(stage.IsString());
        push_constant.stageFlags |= getVkShaderStageFlagBit(stage.GetString());
    }

    return push_constant;
}

VkVertexInputBindingDescription initVkVertexInputBindingDescription(rapidjson::Value & document)
{
    assert(document.IsObject());

    VkVertexInputBindingDescription vertex_binding;

    assert(document.HasMember("binding_slot"));
    assert(document["binding_slot"].IsUint());
    vertex_binding.binding = document["binding_slot"].GetUint();

    assert(document.HasMember("stride"));
    assert(document["stride"].IsUint());
    vertex_binding.stride = document["stride"].GetUint();

    assert(document.HasMember("input_rate"));
    assert(document["input_rate"].IsString());
    std::string input_rate = document["input_rate"].GetString();
    if (strcmp(input_rate.c_str(), "PER_VERTEX") == 0)
    {
        vertex_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    }
    else if (strcmp(input_rate.c_str(), "PER_INSTANCE") == 0)
    {
        vertex_binding.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    }

    return vertex_binding;
}

VkVertexInputAttributeDescription initVkVertexInputAttributeDescription(
    rapidjson::Value &                                               document,
    std::unordered_map<std::string, VkVertexInputBindingDescription> vertex_bindings)
{
    assert(document.IsObject());

    VkVertexInputAttributeDescription attribute;

    assert(document.HasMember("vertex_binding_name"));
    assert(document["vertex_binding_name"].IsString());
    attribute.binding = vertex_bindings[document["vertex_binding_name"].GetString()].binding;

    assert(document.HasMember("location"));
    assert(document["location"].IsInt());
    attribute.location = document["location"].GetInt();

    assert(document.HasMember("offset"));
    assert(document["offset"].IsInt());
    attribute.offset = document["offset"].GetInt();

    assert(document.HasMember("format"));
    assert(document["format"].IsString());
    attribute.format = getVkFormat(document["format"].GetString());

    return attribute;
}

//
// CONFIG STRUCT INITIALIZERS
//

void init(rapidjson::Value & document, AttachmentConfig & attachment_config)
{
    assert(document.IsObject());

    assert(document.HasMember("format"));
    assert(document["format"].IsString());
    char const * format_str = document["format"].GetString();
    if (strcmp(format_str, "color") == 0)
    {
        attachment_config.format = Format::USE_COLOR;
    }
    else if (strcmp(format_str, "depth") == 0)
    {
        attachment_config.format = Format::USE_DEPTH;
    }

    attachment_config.usage = static_cast<VkImageUsageFlags>(0);
    assert(document.HasMember("usage"));
    assert(document["usage"].IsArray());
    for (auto & usage_bit_name: document["usage"].GetArray())
    {
        assert(usage_bit_name.IsString());
        attachment_config.usage |= getVkImageUsageFlagBits(usage_bit_name.GetString());
    }

    if (document.HasMember("multisamples"))
    {
        assert(document["multisamples"].IsUint());
        attachment_config.multisamples = getVkSampleCountFlagBits(
            document["multisamples"].GetUint());
    }
    else
    {
        attachment_config.multisamples = getVkSampleCountFlagBits(1);
    }

    if (document.HasMember("is_swapchain_image"))
    {
        assert(document["is_swapchain_image"].IsBool());
        attachment_config.is_swapchain_image = document["is_swapchain_image"].GetBool();
    }
    else
    {
        attachment_config.is_swapchain_image = false;
    }

    if (!attachment_config.is_swapchain_image && document.HasMember("size"))
    {
        attachment_config.use_swapchain_size = false;

        assert(document["size"].IsObject());
        attachment_config.extent = initVkExtent2D(document["size"]);
    }
    else
    {
        attachment_config.use_swapchain_size = true;
    }
}

void init(rapidjson::Value &                        document,
          SubpassInfo &                             subpass_info,
          std::unordered_map<std::string, size_t> & attachment_indices)
{
    assert(document.IsObject());

    if (document.HasMember("multisamples"))
    {
        assert(document["multisamples"].IsUint());
        subpass_info.multisamples = getVkSampleCountFlagBits(document["multisamples"].GetUint());
    }
    else
    {
        subpass_info.multisamples = getVkSampleCountFlagBits(1);
    }

    if (document.HasMember("color_attachments"))
    {
        assert(document["color_attachments"].IsArray());
        for (auto & ca: document["color_attachments"].GetArray())
        {
            subpass_info.color_attachments.push_back(
                initAttachmentReference(ca, attachment_indices));
        }
    }

    if (document.HasMember("color_resolve_attachments"))
    {
        assert(document["color_resolve_attachments"].IsArray());
        for (auto & cra: document["color_resolve_attachments"].GetArray())
        {
            subpass_info.color_resolve_attachments.push_back(
                initAttachmentReference(cra, attachment_indices));
        }
    }

    assert(subpass_info.color_attachments.size() == subpass_info.color_resolve_attachments.size()
           || subpass_info.color_resolve_attachments.size() == 0);

    if (document.HasMember("input_attachments"))
    {
        assert(document["input_attachments"].IsArray());
        for (auto & ia: document["input_attachments"].GetArray())
        {
            subpass_info.input_attachments.push_back(
                initAttachmentReference(ia, attachment_indices));
        }
    }

    if (document.HasMember("preserve_attachments"))
    {
        assert(document["preserve_attachments"].IsArray());
        for (auto & pa: document["preserve_attachments"].GetArray())
        {
            assert(pa.IsString());

            auto attachment_index_iter = attachment_indices.find(pa.GetString());
            if (attachment_index_iter != attachment_indices.end())
            {
                subpass_info.preserve_attachments.push_back(attachment_index_iter->second);
            }
            else
            {
                LOG_ERROR("Couldn't find index in framebuffer for preserve attachment {}",
                          pa.GetString());
            }
        }
    }

    if (document.HasMember("depth_stencil_attachment"))
    {
        subpass_info.depth_stencil_attachment = initAttachmentReference(
            document["depth_stencil_attachment"], attachment_indices);
    }
    else
    {
        subpass_info.depth_stencil_attachment = VkAttachmentReference{
            .attachment = VK_ATTACHMENT_UNUSED, .layout = VK_IMAGE_LAYOUT_UNDEFINED};
    }
}

void init(rapidjson::Value & document, RenderpassConfig & renderpass_config)
{
    assert(document.IsObject());

    assert(document.HasMember("framebuffer"));
    assert(document["framebuffer"].IsArray());
    size_t attachment_index{0};
    for (auto & ad: document["framebuffer"].GetArray())
    {
        assert(ad.IsObject());
        assert(ad.HasMember("attachment_name"));
        assert(ad["attachment_name"].IsString());
        std::string name = ad["attachment_name"].GetString();

        renderpass_config.attachments[name]  = attachment_index++;
        renderpass_config.descriptions[name] = initAttachmentDescription(ad);

        if (ad.HasMember("clear_value"))
        {
            renderpass_config.clear_values[ad["attachment_name"].GetString()] = initClearValue(
                ad["clear_value"]);
        }
    }

    assert(document.HasMember("subpasses"));
    assert(document["subpasses"].IsArray());

    for (auto & sp: document["subpasses"].GetArray())
    {
        assert(sp.IsObject());
        assert(sp.HasMember("name"));
        assert(sp["name"].IsString());

        SubpassHandle handle = renderpass_config.subpasses.size();

        SubpassInfo info{};
        init(sp, info, renderpass_config.attachments);

        renderpass_config.subpass_handles[sp["name"].GetString()] = handle;
        renderpass_config.subpasses.push_back(info);
    }

    assert(document.HasMember("subpass_dependencies"));
    assert(document["subpass_dependencies"].IsArray());

    for (auto & spd: document["subpass_dependencies"].GetArray())
    {
        renderpass_config.subpass_dependencies.push_back(
            initDependency(spd, renderpass_config.subpass_handles));
    }
}

void init(rapidjson::Value &                                   document,
          PipelineConfig &                                     pipeline_config,
          std::unordered_map<std::string, std::string> const & shader_names)
{
    assert(document.IsObject());

    assert(document.HasMember("vertex_shader_name"));
    assert(document["vertex_shader_name"].IsString());
    pipeline_config.vertex_shader_name = document["vertex_shader_name"].GetString();

    assert(document.HasMember("fragment_shader_name"));
    assert(document["fragment_shader_name"].IsString());
    pipeline_config.fragment_shader_name = document["fragment_shader_name"].GetString();

    assert(document.HasMember("vertex_bindings"));
    assert(document["vertex_bindings"].IsArray());
    for (auto const & vbi: document["vertex_bindings"].GetArray())
    {
        assert(vbi.IsString());
        pipeline_config.vertex_binding_names.push_back(vbi.GetString());
    }

    assert(document.HasMember("vertex_attributes"));
    assert(document["vertex_attributes"].IsArray());
    for (auto const & vai: document["vertex_attributes"].GetArray())
    {
        assert(vai.IsString());
        pipeline_config.vertex_attribute_names.push_back(vai.GetString());
    }

    assert(document.HasMember("uniform_layouts"));
    assert(document["uniform_layouts"].IsArray());
    for (auto const & uli: document["uniform_layouts"].GetArray())
    {
        assert(uli.IsString());
        pipeline_config.uniform_layout_names.push_back(uli.GetString());
    }

    assert(document.HasMember("push_constants"));
    assert(document["push_constants"].IsArray());
    for (auto const & pci: document["push_constants"].GetArray())
    {
        assert(pci.IsString());
        pipeline_config.push_constant_names.push_back(pci.GetString());
    }

    assert(document.HasMember("renderpass"));
    assert(document["renderpass"].IsString());
    pipeline_config.renderpass = document["renderpass"].GetString();

    assert(document.HasMember("subpass"));
    assert(document["subpass"].IsString());
    pipeline_config.subpass = document["subpass"].GetString();

    if (document.HasMember("blendable"))
    {
        assert(document["blendable"].IsBool());
        pipeline_config.blendable = document["blendable"].GetBool();
    }
    else
    {
        pipeline_config.blendable = true;
    }

    if (document.HasMember("tests_depth"))
    {
        assert(document["tests_depth"].IsBool());
        pipeline_config.tests_depth = document["tests_depth"].GetBool();
    }
    else
    {
        pipeline_config.tests_depth = false;
    }

    if (document.HasMember("dynamic_state"))
    {
        assert(document["dynamic_state"].IsArray());

        for (auto const & state: document["dynamic_state"].GetArray())
        {
            assert(state.IsString());
            LOG_DEBUG("Pushing state {}", state.GetString());
            pipeline_config.dynamic_state.push_back(getVkDynamicState(state.GetString()));
        }
    }
}

void init(rapidjson::Value & document, UniformConfig & uniform_config)
{
    assert(document.IsObject());
    assert(document.HasMember("max_count"));
    assert(document["max_count"].IsUint());

    uniform_config.max_uniform_count = document["max_count"].GetUint();
    assert(uniform_config.max_uniform_count != 0);

    uniform_config.layout_binding = initVkDescriptorSetLayoutBinding(document);
}

void RenderConfig::init()
{
    namespace rj = rapidjson;

    rj::Document document;

    auto config_json = readFile(config_filename);
    config_json.push_back('\0');

    if (document.Parse(config_json.data()).HasParseError())
    {
        LOG_ERROR("Parse error on json data: \"{}\"", config_json.data());
        return;
    }
    else
    {
        LOG_DEBUG("Parsed file {} in RenderConfig", config_filename);
    }

    assert(document.IsObject());

    assert(document.HasMember("window_name"));
    assert(document["window_name"].IsString());
    window_name = document["window_name"].GetString();

    assert(document.HasMember("attachments"));
    assert(document["attachments"].IsArray());

    for (auto & a: document["attachments"].GetArray())
    {
        assert(a.IsObject());
        assert(a.HasMember("name"));
        assert(a["name"].IsString());

        gfx::init(a, attachment_configs[a["name"].GetString()]);
    }

    assert(document.HasMember("renderpasses"));
    assert(document["renderpasses"].IsArray());

    for (auto & rp: document["renderpasses"].GetArray())
    {
        assert(rp.IsObject());
        assert(rp.HasMember("name"));
        assert(rp["name"].IsString());

        gfx::init(rp, renderpass_configs[rp["name"].GetString()]);
    }

    assert(document.HasMember("renderpass_order"));
    assert(document["renderpass_order"].IsArray());
    for (auto & renderpass_name: document["renderpass_order"].GetArray())
    {
        assert(renderpass_name.IsString());
        renderpass_order.push_back(renderpass_name.GetString());
    }

    assert(document.HasMember("shaders"));
    assert(document["shaders"].IsArray());

    for (auto & s: document["shaders"].GetArray())
    {
        assert(s.IsObject());
        assert(s.HasMember("name"));
        assert(s["name"].IsString());

        assert(s.HasMember("file"));
        assert(s["file"].IsString());

        shader_names[s["name"].GetString()] = s["file"].GetString();
    }

    assert(document.HasMember("uniform_layouts"));
    assert(document["uniform_layouts"].IsArray());

    for (auto & ul: document["uniform_layouts"].GetArray())
    {
        assert(ul.IsObject());
        assert(ul.HasMember("name"));
        assert(ul["name"].IsString());

        gfx::init(ul, uniform_configs[ul["name"].GetString()]);
    }

    assert(document.HasMember("push_constants"));
    assert(document["push_constants"].IsArray());

    for (auto & pc: document["push_constants"].GetArray())
    {
        assert(pc.IsObject());
        assert(pc.HasMember("name"));
        assert(pc["name"].IsString());

        push_constants[pc["name"].GetString()] = initVkPushConstantRange(pc);
    }

    assert(document.HasMember("vertex_bindings"));
    assert(document["vertex_bindings"].IsArray());

    for (auto & vb: document["vertex_bindings"].GetArray())
    {
        assert(vb.IsObject());
        assert(vb.HasMember("name"));
        assert(vb["name"].IsString());

        vertex_bindings[vb["name"].GetString()] = initVkVertexInputBindingDescription(vb);
    }

    assert(document.HasMember("vertex_attributes"));
    assert(document["vertex_attributes"].IsArray());

    for (auto & va: document["vertex_attributes"].GetArray())
    {
        assert(va.IsObject());
        assert(va.HasMember("name"));
        assert(va["name"].IsString());

        vertex_attributes[va["name"].GetString()] = initVkVertexInputAttributeDescription(
            va, vertex_bindings);
    }

    assert(document.HasMember("pipelines"));
    assert(document["pipelines"].IsArray());

    for (auto & p: document["pipelines"].GetArray())
    {
        assert(p.IsObject());
        assert(p.HasMember("name"));
        assert(p["name"].IsString());

        gfx::init(p, pipeline_configs[p["name"].GetString()], shader_names);
    }
}

}; // namespace gfx

#endif