#ifndef JED_GFX_RENDER_DEVICE_HPP
#define JED_GFX_RENDER_DEVICE_HPP

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <array>
#include <set>
#include <iostream>
#include <fstream>
#include <variant>
#include <optional>
#include <unordered_map>

#include "cmd/cmd.hpp"

#include "rapidjson/document.h"

#include "stb_image.h"

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription                  getBindingDescription();
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions();
};

std::vector<char> readFile(std::string const & filename);

namespace gfx
{
class RenderDevice;
class RenderConfig;

using CommandbufferHandle   = size_t;
using RenderpassHandle      = size_t;
using AttachmentHandle      = size_t;
using FramebufferHandle     = size_t;
using UniformLayoutHandle   = size_t;
using PushConstantHandle    = size_t;
using VertexBindingHandle   = size_t;
using VertexAttributeHandle = size_t;
using ShaderHandle          = size_t;
using PipelineHandle        = size_t;

struct UniformHandle
{
    uint64_t uniform_layout_id : 32;
    uint64_t uniform_id : 32;
};

bool operator==(VkAttachmentDescription const & lhs, VkAttachmentDescription const & rhs);
bool operator!=(VkAttachmentDescription const & lhs, VkAttachmentDescription const & rhs);

bool operator==(VkAttachmentReference const & lhs, VkAttachmentReference const & rhs);
bool operator!=(VkAttachmentReference const & lhs, VkAttachmentReference const & rhs);

bool operator==(VkSubpassDependency const & lhs, VkSubpassDependency const & rhs);
bool operator!=(VkSubpassDependency const & lhs, VkSubpassDependency const & rhs);

enum class Format
{
    USE_DEPTH,
    USE_COLOR
};

struct AttachmentConfig
{
    Format format;
    bool   multisampled;
    bool   is_swapchain_image;

    void init(rapidjson::Value & document);

    friend bool operator==(AttachmentConfig const & lhs, AttachmentConfig const & rhs);
    friend bool operator!=(AttachmentConfig const & lhs, AttachmentConfig const & rhs);
};

struct Texture
{
    VkDeviceMemory vk_image_memory{VK_NULL_HANDLE};
    VkImage        vk_image{VK_NULL_HANDLE};
    VkImageView    vk_image_view{VK_NULL_HANDLE};
};

struct SampledTexture
{
    Texture   texture;
    VkSampler sampler;
};

struct FramebufferConfig
{
    std::vector<AttachmentHandle> attachments;
    uint32_t                      width;
    uint32_t                      height;
    uint32_t                      depth;

    void init(rapidjson::Value & document);
};

struct SubpassInfo
{
    std::vector<VkAttachmentReference> color_attachments;
    VkAttachmentReference              color_resolve_attachment;
    VkAttachmentReference              depth_stencil_attachment;

    void init(rapidjson::Value & document);

    friend bool operator==(SubpassInfo const & lhs, SubpassInfo const & rhs);
    friend bool operator!=(SubpassInfo const & lhs, SubpassInfo const & rhs);
};

struct RenderpassConfig
{
    FramebufferConfig                    framebuffer_config;
    std::vector<VkAttachmentDescription> descriptions;
    std::vector<SubpassInfo>             subpasses;
    std::vector<VkSubpassDependency>     subpass_dependencies;

    void init(rapidjson::Value & document);

    friend bool operator==(RenderpassConfig const & lhs, RenderpassConfig const & rhs);
    friend bool operator!=(RenderpassConfig const & lhs, RenderpassConfig const & rhs);
};

struct MappedBuffer
{
    void *         data;
    VkBuffer       buffer;
    VkDeviceMemory memory;
    VkDeviceSize   memory_size;
    size_t         offset;

    size_t copy(size_t size, void const * src_data);

    void reset();
};

struct DynamicUniformBuffer
{
    VkDescriptorSet const &   vk_descriptorset;
    std::vector<MappedBuffer> uniform_buffers;
};

struct UniformLayout
{
    VkDescriptorSetLayoutBinding binding;

    VkDescriptorSetLayout        vk_descriptorset_layout{VK_NULL_HANDLE};
    VkDescriptorPool             vk_descriptor_pool{VK_NULL_HANDLE};
    size_t                       uniform_count;
    VkDeviceSize                 uniform_size;
    std::vector<VkDescriptorSet> vk_descriptorsets; // currentFrame
    std::vector<MappedBuffer>    uniform_buffers;   // currentFrame

    void init(rapidjson::Value & document);

    std::optional<UniformHandle> newUniform();

    std::optional<DynamicUniformBuffer> getUniform(UniformHandle handle);
};

struct PipelineConfig
{
    ShaderHandle vertex_shader;
    ShaderHandle fragment_shader;

    std::vector<VertexBindingHandle>   vertex_bindings;
    std::vector<VertexAttributeHandle> vertex_attributes;

    std::vector<UniformLayoutHandle> uniform_layouts;
    std::vector<PushConstantHandle>  push_constants;

    RenderpassHandle renderpass;
    uint32_t         subpass;

    void init(rapidjson::Value & document);
};

struct Pipeline
{
    VkPipeline       vk_pipeline;
    VkPipelineLayout vk_pipeline_layout;
};

struct RenderConfig
{
    char const * config_filename;

    char const * window_name;

    size_t dynamic_vertices_count;

    size_t dynamic_indices_count;

    size_t staging_buffer_size;

    std::vector<RenderpassConfig> renderpass_configs;

    std::vector<AttachmentConfig> attachment_configs;

    std::vector<UniformLayout> uniform_layouts;

    std::vector<VkPushConstantRange> push_constants;

    std::vector<VkVertexInputBindingDescription> vertex_bindings;

    std::vector<VkVertexInputAttributeDescription> vertex_attributes;

    std::vector<std::string> shader_names;

    std::vector<PipelineConfig> pipeline_configs;

    void init();
};

struct Draw
{
    static cmd::BackendDispatchFunction const DISPATCH_FUNCTION;

    VkCommandBuffer  commandbuffer;
    VkBuffer         vertexbuffer;
    VkDeviceSize     vertexbuffer_offset;
    VkBuffer         indexbuffer;
    VkDeviceSize     indexbuffer_offset;
    VkDeviceSize     indexbuffer_count;
    VkPipelineLayout pipeline_layout;
    glm::mat4        transform;
};
static_assert(std::is_pod<Draw>::value == true, "Draw must be a POD.");

void draw(void const * data);

struct Copy
{
    static cmd::BackendDispatchFunction const DISPATCH_FUNCTION;

    VkCommandBuffer commandbuffer;
    VkBuffer        srcBuffer;
    VkBuffer        dstBuffer;
    VkDeviceSize    srcOffset;
    VkDeviceSize    dstOffset;
    VkDeviceSize    size;
};
static_assert(std::is_pod<Copy>::value == true, "Copy must be a POD.");

void copy(void const * data);

struct CopyToImage
{
    static cmd::BackendDispatchFunction const DISPATCH_FUNCTION;

    VkCommandBuffer commandbuffer;
    VkBuffer        srcBuffer;
    VkDeviceSize    srcOffset;
    VkImage         dstImage;
    uint32_t        width;
    uint32_t        height;
};
static_assert(std::is_pod<Copy>::value == true, "Copy must be a POD.");

void copyToImage(void const * data);

struct SetImageLayout
{
    static cmd::BackendDispatchFunction const DISPATCH_FUNCTION;

    VkCommandBuffer      commandbuffer;
    VkAccessFlags        srcAccessMask;
    VkAccessFlags        dstAccessMask;
    VkImageLayout        oldLayout;
    VkImageLayout        newLayout;
    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;
    VkImage              image;
    uint32_t             mipLevels;
    VkImageAspectFlags   aspectMask;
};
static_assert(std::is_pod<SetImageLayout>::value == true, "SetImageLayout must be a POD.");

void setImageLayout(void const * data);

class RenderDevice
{
public:
    RenderDevice(GLFWwindow * window_ptr);

    bool init(RenderConfig & render_config);

    void quit();

    void waitForIdle();

    void drawFrame(uint32_t uniform_count, UniformHandle * p_uniforms);

    std::optional<UniformHandle> newUniform();

    void updateUniform(UniformHandle handle, glm::mat4 const & data);

    void dynamicDraw(PipelineHandle    pipeline,
                     glm::mat4 const & transform,
                     uint32_t          vertex_count,
                     Vertex *          vertices,
                     uint32_t          index_count,
                     uint32_t *        indices);

    void draw(PipelineHandle    pipeline,
              glm::mat4 const & transform,
              VkBuffer          vertexbuffer,
              VkDeviceSize      vertexbuffer_offset,
              VkBuffer          indexbuffer,
              VkDeviceSize      indexbuffer_offset,
              VkDeviceSize      indexbuffer_count);

    bool createVertexbuffer(VkBuffer &       vertexbuffer,
                            VkDeviceMemory & vertexbuffer_memory,
                            uint32_t         vertex_count,
                            Vertex *         vertices);

    bool createIndexbuffer(VkBuffer &       indexbuffer,
                           VkDeviceMemory & indexbuffer_memory,
                           uint32_t         index_count,
                           uint32_t *       indices);

    void updateDeviceLocalBuffers(VkBuffer   vertexbuffer,
                                  VkBuffer   indexbuffer,
                                  uint32_t   vertex_count,
                                  Vertex *   vertices,
                                  uint32_t   index_count,
                                  uint32_t * indices);

    void destroyDeviceLocalBuffers(VkBuffer       vertexbuffer,
                                   VkDeviceMemory vertex_memory,
                                   VkBuffer       indexbuffer,
                                   VkDeviceMemory index_memory);

    void createTexture(char const * texture_path);

    void destroyTexture(Texture & texture);

private:
    void checkValidationLayerSupport();

    void getRequiredExtensions();

    // INSTANCE
    VkResult createInstance(char const * window_name);

    // VALIDATION LAYER DEBUG MESSAGER
    static VKAPI_ATTR VkBool32 VKAPI_CALL
                               debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
                                             VkDebugUtilsMessageTypeFlagsEXT              messageType,
                                             const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData,
                                             void *                                       pUserData);

    VkResult createDebugMessenger();

    VkResult createDebugUtilsMessengerEXT(const VkDebugUtilsMessengerCreateInfoEXT * pCreateInfo,
                                          const VkAllocationCallbacks *              pAllocator,
                                          VkDebugUtilsMessengerEXT * pDebugMessenger);

    void cleanupDebugUtilsMessengerEXT(VkDebugUtilsMessengerEXT      debugMessenger,
                                       const VkAllocationCallbacks * pAllocator);

    // SURFACE
    VkResult createSurface();

    // PHYSICAL DEVICE
    struct PhysicalDeviceInfo
    {
        std::vector<VkQueueFamilyProperties> queue_family_properties{};
        int32_t                              present_queue{-1};
        int32_t                              graphics_queue{-1};
        int32_t                              transfer_queue{-1};

        bool queues_complete() const;

        std::vector<VkExtensionProperties> available_extensions{};

        bool has_required_extensions{false};

        VkSurfaceCapabilitiesKHR        capabilities{};
        std::vector<VkSurfaceFormatKHR> formats{};
        std::vector<VkPresentModeKHR>   presentModes{};

        bool swapchain_adequate() const;

        VkPhysicalDeviceFeatures features{};

        VkPhysicalDeviceProperties properties{};

        VkSampleCountFlagBits msaa_samples{VK_SAMPLE_COUNT_1_BIT};
    };

    bool pickPhysicalDevice();

    bool isDeviceSuitable(VkPhysicalDevice device, PhysicalDeviceInfo & device_info);

    void findQueueFamilies(VkPhysicalDevice device, PhysicalDeviceInfo & device_info);

    bool checkDeviceExtensionSupport(VkPhysicalDevice device, PhysicalDeviceInfo & device_info);

    void querySwapChainSupport(VkPhysicalDevice device, PhysicalDeviceInfo & device_info);

    void getMaxUsableSampleCount();

    // LOGICAL DEVICE
    VkResult createLogicalDevice();

    // SWAPCHAIN
    VkResult createSwapChain();

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(
        std::vector<VkSurfaceFormatKHR> const & availableFormats);

    VkPresentModeKHR chooseSwapPresentMode(
        const std::vector<VkPresentModeKHR> & availablePresentModes);

    VkExtent2D chooseSwapExtent(VkSurfaceCapabilitiesKHR const & capabilities);

    // SWAPCHAIN IMAGES
    void getSwapChainImages();

    VkResult createSwapChainImageViews();

    VkResult createImageView(VkImage            image,
                             VkImageView &      image_view,
                             VkFormat           format,
                             VkImageAspectFlags aspectFlags,
                             uint32_t           mipLevels);

    // RENDER PASS & ATTACHMENT DESCRIPTIONS
    VkResult createRenderPass();

    VkResult createFramebuffer(RenderpassConfig const &     renderpass_config,
                               VkRenderPass const &         renderpass,
                               std::vector<VkFramebuffer> & framebuffers);

    VkFormat findDepthFormat();

    VkFormat findSupportedFormat(const std::vector<VkFormat> & candidates,
                                 VkImageTiling                 tiling,
                                 VkFormatFeatureFlags          features);

    VkResult createUniformLayouts();

    // SHADERS
    VkResult createShaders();

    VkResult createShaderModule(std::vector<char> const & code, VkShaderModule & shaderModule);

    // GRAPHICS PIPELINE
    VkResult createGraphicsPipeline();

    // COMMAND POOL
    VkResult createCommandPool();

    VkResult createAttachments();

    VkResult createImage(uint32_t              width,
                         uint32_t              height,
                         uint32_t              mipLevels,
                         VkSampleCountFlagBits numSamples,
                         VkFormat              format,
                         VkImageTiling         tiling,
                         VkImageUsageFlags     usage,
                         VkMemoryPropertyFlags properties,
                         VkImage &             image,
                         VkDeviceMemory &      imageMemory);

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    void transitionImageLayout(VkImage       image,
                               VkFormat      format,
                               VkImageLayout oldLayout,
                               VkImageLayout newLayout,
                               uint32_t      mipLevels);

    bool hasStencilComponent(VkFormat format);

    VkCommandBuffer beginSingleTimeCommands();

    void endSingleTimeCommands(VkCommandBuffer commandBuffer);

    // FRAMEBUFFER
    VkResult createFramebuffers(RenderConfig & render_config);

    VkResult createBuffer(VkDeviceSize          size,
                          VkBufferUsageFlags    usage,
                          VkMemoryPropertyFlags properties,
                          VkBuffer &            buffer,
                          VkDeviceMemory &      bufferMemory);

    void copyBuffer(VkBuffer     srcBuffer,
                    VkDeviceSize srcOffset,
                    VkBuffer     dstBuffer,
                    VkDeviceSize dstOffset,
                    VkDeviceSize size);

    // COMMANDBUFFERS
    // TODO: rewrite to return VkResult
    VkResult createCommandbuffers();

    // COMMANDBUFFER
    VkResult createCommandbuffer(uint32_t        image_index,
                                 uint32_t        uniform_count,
                                 UniformHandle * p_uniforms);

    // SYNC OBJECTS
    VkResult createSyncObjects();

    VkResult createDynamicObjectResources(size_t dynamic_vertices_count,
                                          size_t dynamic_indices_count);

    VkResult createStagingObjectResources(size_t staging_buffer_size);

    GLFWwindow * window{nullptr};

#ifdef NDEBUG
    bool use_validation{false};
#else
    bool use_validation{true};
#endif
    bool                            validation_supported{false};
    const std::vector<char const *> validation_layers{"VK_LAYER_KHRONOS_validation"};

    std::vector<char const *>       required_extensions{};
    const std::vector<const char *> required_device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    VkInstance instance{VK_NULL_HANDLE};

    VkDebugUtilsMessengerEXT debug_messager{VK_NULL_HANDLE};

    VkSurfaceKHR surface{VK_NULL_HANDLE};

    VkPhysicalDevice   physical_device{VK_NULL_HANDLE};
    PhysicalDeviceInfo physical_device_info;

    VkDevice logical_device;

    VkQueue graphics_queue;
    VkQueue present_queue;
    VkQueue transfer_queue;

    VkFormat   swapchain_image_format;
    VkFormat   depth_format;
    VkExtent2D swapchain_extent;

    VkSwapchainKHR           swapchain;
    uint32_t                 swapchain_image_count;
    std::vector<VkImage>     swapchain_images;
    std::vector<VkImageView> swapchain_image_views;

    std::vector<VkSemaphore> image_available_semaphores;
    std::vector<VkSemaphore> render_finished_semaphores;
    std::vector<VkFence>     in_flight_fences;

    int32_t const MAX_FRAMES_IN_FLIGHT{2};
    int32_t const MAX_BUFFERED_RESOURCES{MAX_FRAMES_IN_FLIGHT + 1};
    uint32_t      currentImage{0};
    size_t        currentFrame{0};
    uint32_t      currentResource{0};

    bool framebuffer_resized;

    VkCommandPool                command_pool;
    std::vector<VkCommandBuffer> commandbuffers;
    std::vector<VkCommandBuffer> transfer_commandbuffers;

    std::vector<MappedBuffer> dynamic_mapped_vertices;
    std::vector<MappedBuffer> dynamic_mapped_indices;

    std::vector<MappedBuffer> staging_buffer;

    // need a way of handling the swapchain images
    // attachments that are used after this frame need to be double buffered etc (i.e. like
    // swapchain) if Store op is DONT_CARE, it doesn't need to be buffered
    std::vector<AttachmentConfig> attachment_configs;
    std::vector<Texture>          attachments;

    std::vector<RenderpassConfig>           renderpass_configs;
    std::vector<VkRenderPass>               renderpasses;
    std::vector<std::vector<VkFramebuffer>> framebuffers;

    std::vector<UniformLayout> uniform_layouts;

    std::vector<VkPushConstantRange> push_constants;

    std::vector<VkVertexInputBindingDescription> vertex_bindings;

    std::vector<VkVertexInputAttributeDescription> vertex_attributes;

    std::vector<std::string>    shader_names;
    std::vector<VkShaderModule> shaders;

    std::vector<PipelineConfig> pipeline_configs;
    std::vector<Pipeline>       pipelines;

    std::vector<cmd::CommandBucket<int>> buckets;
    std::vector<cmd::CommandBucket<int>> transfer_buckets;

    Texture         texture;
    VkSampler       sampler;
    VkDescriptorSet sampler_descriptor;

}; // class RenderDevice
}; // namespace gfx

#endif

#ifdef JED_GFX_IMPLEMENTATION

VkVertexInputBindingDescription Vertex::getBindingDescription()
{
    auto bindingDescription = VkVertexInputBindingDescription{
        .binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};

    return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 2> Vertex::getAttributeDescriptions()
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

std::vector<char> readFile(std::string const & filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file!");
    }

    size_t            fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

namespace gfx
{
//
//  RAPIDJSON INIT FUNCTIONS
//

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
    if (lhs.color_resolve_attachment != rhs.color_resolve_attachment)
    {
        return false;
    }

    if (lhs.depth_stencil_attachment != rhs.depth_stencil_attachment)
    {
        return false;
    }

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

VkImageLayout getVkImageLayout(std::string const & layout_name)
{
    static std::unordered_map<std::string, VkImageLayout> layouts{
        {"UNDEFINED", VK_IMAGE_LAYOUT_UNDEFINED},
        {"GENERAL", VK_IMAGE_LAYOUT_GENERAL},
        {"COLOR_ATTACHMENT_OPTIMAL", VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        {"DEPTH_STENCIL_ATTACHMENT_OPTIMAL", VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
        {"DEPTH_STENCIL_READ_ONLY_OPTIMAL", VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL},
        {"SHADER_READ_ONLY_OPTIMAL", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {"TRANSFER_SRC_OPTIMAL", VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
        {"TRANSFER_DST_OPTIMAL", VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL},
        {"PREINITIALIZED", VK_IMAGE_LAYOUT_PREINITIALIZED},
        {"DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL",
         VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL},
        {"DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL",
         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL},
        {"PRESENT_SRC_KHR", VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
        {"SHARED_PRESENT_KHR", VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR}};

    auto layout = layouts.find(layout_name);
    assert(layout != layouts.end());
    if (layout == layouts.end())
    {
        // return static_cast<VkPipelineStageFlagBits>(0);
    }

    return layout->second;
}

VkAttachmentLoadOp getVkAttachmentLoadOp(std::string const & op_name)
{
    static std::unordered_map<std::string, VkAttachmentLoadOp> ops{
        {"LOAD", VK_ATTACHMENT_LOAD_OP_LOAD},
        {"CLEAR", VK_ATTACHMENT_LOAD_OP_CLEAR},
        {"DONT_CARE", VK_ATTACHMENT_LOAD_OP_DONT_CARE}};

    auto op = ops.find(op_name);
    assert(op != ops.end());
    if (op == ops.end())
    {
        // return static_cast<VkPipelineStageFlagBits>(0);
    }

    return op->second;
}

VkAttachmentStoreOp getVkAttachmentStoreOp(std::string const & op_name)
{
    static std::unordered_map<std::string, VkAttachmentStoreOp> ops{
        {"STORE", VK_ATTACHMENT_STORE_OP_STORE}, {"DONT_CARE", VK_ATTACHMENT_STORE_OP_DONT_CARE}};

    auto op = ops.find(op_name);
    assert(op != ops.end());
    if (op == ops.end())
    {
        // return static_cast<VkPipelineStageFlagBits>(0);
    }

    return op->second;
}

VkPipelineStageFlagBits getVkPipelineStageFlagBit(std::string const & bit_name)
{
    static std::unordered_map<std::string, VkPipelineStageFlagBits> bits = {
        {"TOP_OF_PIPE", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT},
        {"DRAW_INDIRECT", VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT},
        {"VERTEX_INPUT", VK_PIPELINE_STAGE_VERTEX_INPUT_BIT},
        {"VERTEX_SHADER", VK_PIPELINE_STAGE_VERTEX_SHADER_BIT},
        {"FRAGMENT_SHADER", VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT},
        {"EARLY_FRAGMENT_TESTS", VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT},
        {"LATE_FRAGMENT_TESTS", VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT},
        {"COLOR_ATTACHMENT_OUTPUT", VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
        {"COMPUTE_SHADER", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT},
        {"TRANSFER", VK_PIPELINE_STAGE_TRANSFER_BIT},
        {"BOTTOM_OF_PIPE", VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT},
        {"HOST", VK_PIPELINE_STAGE_HOST_BIT},
        {"ALL_GRAPHICS", VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT},
        {"ALL_COMMANDS", VK_PIPELINE_STAGE_ALL_COMMANDS_BIT}};

    auto bit = bits.find(bit_name);
    assert(bit != bits.end());
    if (bit == bits.end())
    {
        // return static_cast<VkPipelineStageFlagBits>(0);
    }

    return bit->second;
}

VkAccessFlagBits getVkAccessFlagBit(std::string const & bit_name)
{
    static std::unordered_map<std::string, VkAccessFlagBits> bits = {
        {"INDIRECT_COMMAND_READ", VK_ACCESS_INDIRECT_COMMAND_READ_BIT},
        {"INDEX_READ", VK_ACCESS_INDEX_READ_BIT},
        {"VERTEX_ATTRIBUTE_READ", VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT},
        {"UNIFORM_READ", VK_ACCESS_UNIFORM_READ_BIT},
        {"INPUT_ATTACHMENT_READ", VK_ACCESS_INPUT_ATTACHMENT_READ_BIT},
        {"SHADER_READ", VK_ACCESS_SHADER_READ_BIT},
        {"SHADER_WRITE", VK_ACCESS_SHADER_WRITE_BIT},
        {"COLOR_ATTACHMENT_READ", VK_ACCESS_COLOR_ATTACHMENT_READ_BIT},
        {"COLOR_ATTACHMENT_WRITE", VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT},
        {"DEPTH_STENCIL_ATTACHMENT_READ", VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT},
        {"DEPTH_STENCIL_ATTACHMENT_WRITE", VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT},
        {"TRANSFER_READ", VK_ACCESS_TRANSFER_READ_BIT},
        {"TRANSFER_WRITE", VK_ACCESS_TRANSFER_WRITE_BIT},
        {"HOST_READ", VK_ACCESS_HOST_READ_BIT},
        {"HOST_WRITE", VK_ACCESS_HOST_WRITE_BIT},
        {"MEMORY_READ", VK_ACCESS_MEMORY_READ_BIT},
        {"MEMORY_WRITE", VK_ACCESS_MEMORY_WRITE_BIT}};

    auto bit = bits.find(bit_name);
    assert(bit != bits.end());
    if (bit == bits.end())
    {
        // return static_cast<VkPipelineStageFlagBits>(0);
    }

    return bit->second;
}

VkShaderStageFlagBits getVkShaderStageFlagBit(std::string const & flag_name)
{
    static std::unordered_map<std::string, VkShaderStageFlagBits> flags{
        {"VERTEX", VK_SHADER_STAGE_VERTEX_BIT},
        {"FRAGMENT", VK_SHADER_STAGE_FRAGMENT_BIT},
        {"COMPUTE", VK_SHADER_STAGE_COMPUTE_BIT},
        {"TESSELLATION_CONTROL", VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT},
        {"TESSELLATION_EVALUATION", VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT},
        {"ALL_GRAPHICS", VK_SHADER_STAGE_ALL_GRAPHICS},
        {"ALL", VK_SHADER_STAGE_ALL}};

    auto flag = flags.find(flag_name);
    assert(flag != flags.end());
    if (flag == flags.end())
    {
        // return static_cast<VkPipelineStageFlagBits>(0);
    }

    return flag->second;
}

VkDescriptorType getVkDescriptorType(std::string const & type_name)
{
    static std::unordered_map<std::string, VkDescriptorType> types{
        {"SAMPLER", VK_DESCRIPTOR_TYPE_SAMPLER},
        {"COMBINED_IMAGE_SAMPLER", VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
        {"SAMPLED_IMAGE", VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE},
        {"STORAGE_IMAGE", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        {"UNIFORM_TEXEL_BUFFER", VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER},
        {"STORAGE_TEXEL_BUFFER", VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER},
        {"UNIFORM_BUFFER", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
        {"STORAGE_BUFFER", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {"UNIFORM_BUFFER_DYNAMIC", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC},
        {"STORAGE_BUFFER_DYNAMIC", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC},
        {"INPUT_ATTACHMENT", VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT}};

    auto type = types.find(type_name);
    assert(type != types.end());
    if (type == types.end())
    {
        // return static_cast<VkPipelineStageFlagBits>(0);
    }

    return type->second;
}

VkFormat getVkFormat(std::string const & format_name)
{
    static std::unordered_map<std::string, VkFormat> formats{
        {"R32G32B32_SFLOAT", VK_FORMAT_R32G32B32_SFLOAT},
    };

    auto format = formats.find(format_name);
    assert(format != formats.end());
    if (format == formats.end())
    {
        // return static_cast<VkPipelineStageFlagBits>(0);
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

VkAttachmentReference initAttachmentReference(rapidjson::Value & document)
{
    assert(document.IsObject());

    VkAttachmentReference reference{};

    assert(document.HasMember("attachment_index"));
    assert(document["attachment_index"].IsInt());
    reference.attachment = document["attachment_index"].GetInt();

    assert(document.HasMember("layout"));
    assert(document["layout"].IsString());
    reference.layout = getVkImageLayout(document["layout"].GetString());

    return reference;
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

uint32_t initSubpassIndex(rapidjson::Value & document)
{
    assert(document.IsInt() || document.IsString());

    if (document.IsInt())
    {
        return document.GetInt();
    }
    else if (document.IsString())
    {
        assert(strcmp(document.GetString(), "EXTERNAL_SUBPASS") == 0);
        return VK_SUBPASS_EXTERNAL;
    }

    return 0;
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
    assert(document["stage"].IsString());
    layout.stageFlags = getVkShaderStageFlagBit(document["stage"].GetString());

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

VkSubpassDependency initDependency(rapidjson::Value & document)
{
    assert(document.IsObject());

    VkSubpassDependency dependency{};

    if (document.HasMember("src_subpass"))
    {
        dependency.srcSubpass = initSubpassIndex(document["src_subpass"]);
    }

    if (document.HasMember("dst_subpass"))
    {
        dependency.dstSubpass = initSubpassIndex(document["dst_subpass"]);
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

    assert(document.HasMember("stage"));
    assert(document["stage"].IsString());
    push_constant.stageFlags = getVkShaderStageFlagBit(document["stage"].GetString());

    assert(document.HasMember("offset"));
    assert(document["offset"].IsInt());
    push_constant.offset = document["offset"].GetInt();

    assert(document.HasMember("size"));
    assert(document["size"].IsInt());
    push_constant.size = document["size"].GetInt();

    return push_constant;
}

VkVertexInputBindingDescription initVkVertexInputBindingDescription(rapidjson::Value & document)
{
    assert(document.IsObject());

    VkVertexInputBindingDescription vertex_binding;

    assert(document.HasMember("binding"));
    assert(document["binding"].IsInt());
    vertex_binding.binding = document["binding"].GetInt();

    assert(document.HasMember("stride"));
    assert(document["stride"].IsInt());
    vertex_binding.stride = document["stride"].GetInt();

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

VkVertexInputAttributeDescription initVkVertexInputAttributeDescription(rapidjson::Value & document)
{
    assert(document.IsObject());

    VkVertexInputAttributeDescription attribute;

    assert(document.HasMember("binding"));
    assert(document["binding"].IsInt());
    attribute.binding = document["binding"].GetInt();

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

void AttachmentConfig::init(rapidjson::Value & document)
{
    assert(document.IsObject());

    assert(document.HasMember("format"));
    assert(document["format"].IsString());
    char const * format_str = document["format"].GetString();
    if (strcmp(format_str, "color") == 0)
    {
        format = Format::USE_COLOR;
    }
    else if (strcmp(format_str, "depth") == 0)
    {
        format = Format::USE_DEPTH;
    }

    assert(document.HasMember("multisampled"));
    assert(document["multisampled"].IsBool());
    multisampled = document["multisampled"].GetBool();

    if (document.HasMember("is_swapchain_image"))
    {
        assert(document["is_swapchain_image"].IsBool());
        is_swapchain_image = document["is_swapchain_image"].GetBool();
    }
    else
    {
        is_swapchain_image = false;
    }
}

void SubpassInfo::init(rapidjson::Value & document)
{
    assert(document.IsObject());

    if (document.HasMember("color_attachments"))
    {
        assert(document["color_attachments"].IsArray());
        for (auto & ca: document["color_attachments"].GetArray())
        {
            VkAttachmentReference reference = initAttachmentReference(ca);
            color_attachments.push_back(reference);
        }
    }

    if (document.HasMember("resolve_attachment"))
    {
        color_resolve_attachment = initAttachmentReference(document["resolve_attachment"]);
    }

    if (document.HasMember("depth_stencil_attachment"))
    {
        depth_stencil_attachment = initAttachmentReference(document["depth_stencil_attachment"]);
    }
}

void RenderpassConfig::init(rapidjson::Value & document)
{
    assert(document.IsObject());

    assert(document.HasMember("framebuffer"));
    framebuffer_config.init(document["framebuffer"]);

    assert(document["framebuffer"].IsObject());
    auto const & json_framebuffer = document["framebuffer"];
    assert(json_framebuffer.HasMember("attachments"));
    assert(json_framebuffer["attachments"].IsArray());
    for (auto & ad: json_framebuffer["attachments"].GetArray())
    {
        descriptions.push_back(initAttachmentDescription(ad));
    }

    assert(document.HasMember("subpasses"));
    assert(document["subpasses"].IsArray());

    for (auto & sp: document["subpasses"].GetArray())
    {
        SubpassInfo info{};
        info.init(sp);

        subpasses.push_back(info);
    }

    assert(document.HasMember("subpass_dependencies"));
    assert(document["subpass_dependencies"].IsArray());

    for (auto & spd: document["subpass_dependencies"].GetArray())
    {
        subpass_dependencies.push_back(initDependency(spd));
    }
}

void FramebufferConfig::init(rapidjson::Value & document)
{
    assert(document.IsObject());

    assert(document.HasMember("attachments"));
    assert(document["attachments"].IsArray());

    for (auto const & attachment: document["attachments"].GetArray())
    {
        assert(attachment.IsObject());
        auto const & att_obj = attachment.GetObject();

        assert(att_obj.HasMember("attachment"));
        assert(att_obj["attachment"].IsInt());
        attachments.push_back(att_obj["attachment"].GetInt());
    }
}

void UniformLayout::init(rapidjson::Value & document)
{
    assert(document.IsObject());
    binding = initVkDescriptorSetLayoutBinding(document);

    assert(document.HasMember("uniform_count"));
    assert(document["uniform_count"].IsInt());
    uniform_count = document["uniform_count"].GetInt();
}

std::optional<UniformHandle> UniformLayout::newUniform()
{
    if (uniform_buffers[0].offset >= uniform_buffers[0].memory_size)
    {
        return std::nullopt;
    }

    auto next_id = uniform_buffers[0].offset / uniform_size;

    for (auto & uniform_buffer: uniform_buffers)
    {
        uniform_buffer.offset += uniform_size;
    }

    return UniformHandle{.uniform_layout_id = 0, .uniform_id = next_id};
}

std::optional<DynamicUniformBuffer> UniformLayout::getUniform(UniformHandle handle)
{
    if (handle.uniform_id >= uniform_count)
    {
        return std::nullopt;
    }

    auto uniform = DynamicUniformBuffer{.vk_descriptorset = vk_descriptorsets[0],
                                        .uniform_buffers  = uniform_buffers};

    for (auto & uniform_buffer: uniform.uniform_buffers)
    {
        uniform_buffer.offset = handle.uniform_id * uniform_size;
    }

    return uniform;
}

void PipelineConfig::init(rapidjson::Value & document)
{
    assert(document.IsObject());

    assert(document.HasMember("vertex_shader"));
    assert(document["vertex_shader"].IsInt());
    vertex_shader = document["vertex_shader"].GetInt();

    assert(document.HasMember("fragment_shader"));
    assert(document["fragment_shader"].IsInt());
    fragment_shader = document["fragment_shader"].GetInt();

    assert(document.HasMember("vertex_bindings"));
    assert(document["vertex_bindings"].IsArray());
    for (auto const & vbi: document["vertex_bindings"].GetArray())
    {
        assert(vbi.IsInt());
        vertex_bindings.push_back(vbi.GetInt());
    }

    assert(document.HasMember("vertex_attributes"));
    assert(document["vertex_attributes"].IsArray());
    for (auto const & vai: document["vertex_attributes"].GetArray())
    {
        assert(vai.IsInt());
        vertex_attributes.push_back(vai.GetInt());
    }

    assert(document.HasMember("uniform_layouts"));
    assert(document["uniform_layouts"].IsArray());
    for (auto const & uli: document["uniform_layouts"].GetArray())
    {
        assert(uli.IsInt());
        uniform_layouts.push_back(uli.GetInt());
    }

    assert(document.HasMember("push_constants"));
    assert(document["push_constants"].IsArray());
    for (auto const & pci: document["push_constants"].GetArray())
    {
        assert(pci.IsInt());
        push_constants.push_back(pci.GetInt());
    }

    assert(document.HasMember("renderpass"));
    assert(document["renderpass"].IsInt());
    renderpass = document["renderpass"].GetInt();

    assert(document.HasMember("subpass"));
    assert(document["subpass"].IsInt());
    subpass = document["subpass"].GetInt();
}

void RenderConfig::init()
{
    namespace rj = rapidjson;

    rj::Document document;

    auto config_json = readFile(config_filename);
    config_json.push_back('\0');

    if (document.Parse(config_json.data()).HasParseError())
    {
        std::cout << "\"" << config_json.data() << "\"\n";
        std::cout << "Parse error\n";
        return;
    }
    else
    {
        std::cout << "Parsed okay\n";
    }

    assert(document.IsObject());

    assert(document.HasMember("window_name"));
    assert(document["window_name"].IsString());
    window_name = document["window_name"].GetString();

    assert(document.HasMember("dynamic_vertices_count"));
    assert(document["dynamic_vertices_count"].IsNumber());
    assert(document["dynamic_vertices_count"].IsInt());
    dynamic_vertices_count = document["dynamic_vertices_count"].GetInt();

    assert(document.HasMember("dynamic_indices_count"));
    assert(document["dynamic_indices_count"].IsNumber());
    assert(document["dynamic_indices_count"].IsInt());
    dynamic_indices_count = document["dynamic_indices_count"].GetInt();

    assert(document.HasMember("staging_buffer_size"));
    assert(document["staging_buffer_size"].IsNumber());
    assert(document["staging_buffer_size"].IsInt());
    staging_buffer_size = document["staging_buffer_size"].GetInt();

    assert(document.HasMember("attachments"));
    assert(document["attachments"].IsArray());

    for (auto & a: document["attachments"].GetArray())
    {
        AttachmentConfig attachment_config{};
        attachment_config.init(a);
        attachment_configs.push_back(attachment_config);
    }

    assert(document.HasMember("renderpasses"));
    assert(document["renderpasses"].IsArray());

    for (auto & rp: document["renderpasses"].GetArray())
    {
        RenderpassConfig renderpass_config{};
        renderpass_config.init(rp);
        renderpass_configs.push_back(renderpass_config);
    }

    assert(document.HasMember("shaders"));
    assert(document["shaders"].IsArray());

    for (auto & s: document["shaders"].GetArray())
    {
        assert(s.IsString());
        shader_names.push_back(s.GetString());
    }

    assert(document.HasMember("uniform_layouts"));
    assert(document["uniform_layouts"].IsArray());

    for (auto & ul: document["uniform_layouts"].GetArray())
    {
        UniformLayout layout{};
        layout.init(ul);
        uniform_layouts.push_back(layout);
    }

    assert(document.HasMember("push_constants"));
    assert(document["push_constants"].IsArray());

    for (auto & pc: document["push_constants"].GetArray())
    {
        VkPushConstantRange push_constant = initVkPushConstantRange(pc);
        push_constants.push_back(push_constant);
    }

    assert(document.HasMember("vertex_bindings"));
    assert(document["vertex_bindings"].IsArray());

    for (auto & vb: document["vertex_bindings"].GetArray())
    {
        VkVertexInputBindingDescription vertex_binding = initVkVertexInputBindingDescription(vb);
        vertex_bindings.push_back(vertex_binding);
    }

    assert(document.HasMember("vertex_attributes"));
    assert(document["vertex_attributes"].IsArray());

    for (auto & va: document["vertex_attributes"].GetArray())
    {
        VkVertexInputAttributeDescription vertex_attribute = initVkVertexInputAttributeDescription(
            va);
        vertex_attributes.push_back(vertex_attribute);
    }

    assert(document.HasMember("pipelines"));
    assert(document["pipelines"].IsArray());

    for (auto & p: document["pipelines"].GetArray())
    {
        PipelineConfig pipeline_config{};
        pipeline_config.init(p);
        pipeline_configs.push_back(pipeline_config);
    }
}

size_t MappedBuffer::copy(size_t size, void const * src_data)
{
    auto * dest_data = static_cast<void *>(static_cast<char *>(data) + offset);

    // assert there's enough space left in the buffers
    assert(size <= memory_size - offset);

    // copy the data over
    memcpy(dest_data, src_data, size);

    auto prev_offset = offset;
    offset += size;
    return prev_offset;
}

void MappedBuffer::reset()
{
    offset = 0;
}

//
//  DRAW COMMANDS
//

void draw(void const * data)
{
    Draw const * realdata = reinterpret_cast<Draw const *>(data);

    vkCmdPushConstants(realdata->commandbuffer,
                       realdata->pipeline_layout,
                       VK_SHADER_STAGE_VERTEX_BIT,
                       0,
                       sizeof(glm::mat4),
                       glm::value_ptr(realdata->transform));

    vkCmdBindVertexBuffers(
        realdata->commandbuffer, 0, 1, &realdata->vertexbuffer, &realdata->vertexbuffer_offset);
    vkCmdBindIndexBuffer(realdata->commandbuffer,
                         realdata->indexbuffer,
                         realdata->indexbuffer_offset,
                         VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(realdata->commandbuffer, realdata->indexbuffer_count, 1, 0, 0, 0);
}

cmd::BackendDispatchFunction const Draw::DISPATCH_FUNCTION = &draw;

void copy(void const * data)
{
    Copy const * copydata = reinterpret_cast<Copy const *>(data);

    auto copyRegion = VkBufferCopy{
        .srcOffset = copydata->srcOffset, .dstOffset = copydata->dstOffset, .size = copydata->size};

    vkCmdCopyBuffer(
        copydata->commandbuffer, copydata->srcBuffer, copydata->dstBuffer, 1, &copyRegion);
}

cmd::BackendDispatchFunction const Copy::DISPATCH_FUNCTION = &copy;

void copyToImage(void const * data)
{
    auto const * copydata = reinterpret_cast<CopyToImage const *>(data);

    auto region = VkBufferImageCopy{.bufferOffset                    = copydata->srcOffset,
                                    .bufferRowLength                 = 0,
                                    .bufferImageHeight               = 0,
                                    .imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                                    .imageSubresource.mipLevel       = 0,
                                    .imageSubresource.baseArrayLayer = 0,
                                    .imageSubresource.layerCount     = 1,
                                    .imageOffset                     = {0, 0, 0},
                                    .imageExtent = {copydata->width, copydata->height, 1}};

    vkCmdCopyBufferToImage(copydata->commandbuffer,
                           copydata->srcBuffer,
                           copydata->dstImage,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1,
                           &region);
}

cmd::BackendDispatchFunction const CopyToImage::DISPATCH_FUNCTION = &copyToImage;

void setImageLayout(void const * data)
{
    auto const * layoutdata = reinterpret_cast<SetImageLayout const *>(data);

    auto barrier = VkImageMemoryBarrier{.sType     = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                        .oldLayout = layoutdata->oldLayout,
                                        .newLayout = layoutdata->newLayout,
                                        .srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED,
                                        .dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED,
                                        .image                           = layoutdata->image,
                                        .subresourceRange.baseMipLevel   = 0,
                                        .subresourceRange.levelCount     = layoutdata->mipLevels,
                                        .subresourceRange.baseArrayLayer = 0,
                                        .subresourceRange.layerCount     = 1,
                                        .subresourceRange.aspectMask     = layoutdata->aspectMask,
                                        .srcAccessMask = layoutdata->srcAccessMask,
                                        .dstAccessMask = layoutdata->dstAccessMask};

    vkCmdPipelineBarrier(layoutdata->commandbuffer,
                         layoutdata->sourceStage,
                         layoutdata->destinationStage,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &barrier);
}

cmd::BackendDispatchFunction const SetImageLayout::DISPATCH_FUNCTION = &setImageLayout;

//
// RENDER DEVICE
//

RenderDevice::RenderDevice(GLFWwindow * window_ptr): window(window_ptr)
{
    checkValidationLayerSupport();
    getRequiredExtensions();
}

bool RenderDevice::init(RenderConfig & render_config)
{
    if (createInstance(render_config.window_name) != VK_SUCCESS)
    {
        return false;
    }

    if (use_validation && createDebugMessenger() != VK_SUCCESS)
    {
        return false;
    }

    if (createSurface() != VK_SUCCESS)
    {
        return false;
    }

    if (!pickPhysicalDevice())
    {
        return false;
    }

    if (createLogicalDevice() != VK_SUCCESS)
    {
        return false;
    }

    vkGetDeviceQueue(logical_device, physical_device_info.present_queue, 0, &present_queue);
    vkGetDeviceQueue(logical_device, physical_device_info.graphics_queue, 0, &graphics_queue);
    vkGetDeviceQueue(logical_device, physical_device_info.transfer_queue, 0, &transfer_queue);

    if (createSwapChain() != VK_SUCCESS)
    {
        return false;
    }

    getSwapChainImages();

    if (createSwapChainImageViews() != VK_SUCCESS)
    {
        return false;
    }

    attachment_configs = std::move(render_config.attachment_configs);
    attachments.resize(attachment_configs.size());
    if (createAttachments() != VK_SUCCESS)
    {
        return false;
    }

    renderpass_configs = std::move(render_config.renderpass_configs);
    renderpasses.resize(renderpass_configs.size());
    framebuffers.resize(renderpass_configs.size());
    if (createRenderPass() != VK_SUCCESS)
    {
        return false;
    }

    uniform_layouts = std::move(render_config.uniform_layouts);
    if (createUniformLayouts() != VK_SUCCESS)
    {
        return false;
    }

    shader_names = std::move(render_config.shader_names);
    shaders.resize(shader_names.size());
    if (createShaders() != VK_SUCCESS)
    {
        return false;
    }

    push_constants    = std::move(render_config.push_constants);
    vertex_bindings   = std::move(render_config.vertex_bindings);
    vertex_attributes = std::move(render_config.vertex_attributes);
    pipeline_configs  = std::move(render_config.pipeline_configs);
    pipelines.resize(pipeline_configs.size());
    if (createGraphicsPipeline() != VK_SUCCESS)
    {
        return false;
    }

    if (createCommandPool() != VK_SUCCESS)
    {
        return false;
    }

    if (createCommandbuffers() != VK_SUCCESS)
    {
        return false;
    }

    if (createSyncObjects() != VK_SUCCESS)
    {
        return false;
    }

    if (createDynamicObjectResources(render_config.dynamic_vertices_count,
                                     render_config.dynamic_indices_count)
        != VK_SUCCESS)
    {
        return false;
    }

    if (createStagingObjectResources(render_config.staging_buffer_size) != VK_SUCCESS)
    {
        return false;
    }

    for (uint32_t i = 0; i < MAX_BUFFERED_RESOURCES; ++i)
    {
        buckets.emplace_back(4);
        transfer_buckets.emplace_back(9);
    }

    return true;
}

void RenderDevice::quit()
{
    vkDeviceWaitIdle(logical_device);

    for (size_t i = 0; i < MAX_BUFFERED_RESOURCES; ++i)
    {
        // DYNAMIC INDEXBUFFER
        vkDestroyBuffer(logical_device, dynamic_mapped_indices[i].buffer, nullptr);
        vkFreeMemory(logical_device, dynamic_mapped_indices[i].memory, nullptr);

        // DYNAMIC VERTEXBUFFER
        vkDestroyBuffer(logical_device, dynamic_mapped_vertices[i].buffer, nullptr);
        vkFreeMemory(logical_device, dynamic_mapped_vertices[i].memory, nullptr);

        // STAGING BUFFER
        vkDestroyBuffer(logical_device, staging_buffer[i].buffer, nullptr);
        vkFreeMemory(logical_device, staging_buffer[i].memory, nullptr);
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(logical_device, render_finished_semaphores[i], nullptr);
        vkDestroySemaphore(logical_device, image_available_semaphores[i], nullptr);
        vkDestroyFence(logical_device, in_flight_fences[i], nullptr);
    }

    for (auto & attachment: attachments)
    {
        destroyTexture(attachment);
    }

    // COMMAND POOL
    vkDestroyCommandPool(logical_device, command_pool, nullptr);

    for (auto & shader: shaders)
    {
        vkDestroyShaderModule(logical_device, shader, nullptr);
    }

    // GRAPHICS PIPELINE
    for (auto & pipeline: pipelines)
    {
        vkDestroyPipeline(logical_device, pipeline.vk_pipeline, nullptr);
        vkDestroyPipelineLayout(logical_device, pipeline.vk_pipeline_layout, nullptr);
    }

    destroyTexture(texture);
    vkDestroySampler(logical_device, sampler, nullptr);

    // DESCRIPTORSET LAYOUT
    for (auto & uniform_layout: uniform_layouts)
    {
        for (auto & uniform_buffer: uniform_layout.uniform_buffers)
        {
            vkDestroyBuffer(logical_device, uniform_buffer.buffer, nullptr);
            vkFreeMemory(logical_device, uniform_buffer.memory, nullptr);
        }

        vkDestroyDescriptorPool(logical_device, uniform_layout.vk_descriptor_pool, nullptr);
        vkDestroyDescriptorSetLayout(
            logical_device, uniform_layout.vk_descriptorset_layout, nullptr);
    }

    for (auto & buffered_framebuffers: framebuffers)
    {
        for (auto & framebuffer: buffered_framebuffers)
        {
            vkDestroyFramebuffer(logical_device, framebuffer, nullptr);
        }
    }

    // RENDER PASS
    for (auto & renderpass: renderpasses)
    {
        vkDestroyRenderPass(logical_device, renderpass, nullptr);
        renderpass = VK_NULL_HANDLE;
    }

    for (size_t i = 0; i < swapchain_image_views.size(); i++)
    {
        vkDestroyImageView(logical_device, swapchain_image_views[i], nullptr);
    }

    if (swapchain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(logical_device, swapchain, nullptr);
    }

    if (logical_device != VK_NULL_HANDLE)
    {
        vkDestroyDevice(logical_device, nullptr);
    }

    if (surface != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(instance, surface, nullptr);
    }

    if (use_validation && debug_messager != VK_NULL_HANDLE)
    {
        cleanupDebugUtilsMessengerEXT(debug_messager, nullptr);
    }

    if (instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(instance, nullptr);
    }
}

void RenderDevice::waitForIdle()
{
    vkDeviceWaitIdle(logical_device);
}

void RenderDevice::drawFrame(uint32_t uniform_count, UniformHandle * p_uniforms)
{
    vkWaitForFences(logical_device,
                    1,
                    &in_flight_fences[currentFrame],
                    VK_TRUE,
                    std::numeric_limits<uint64_t>::max());

    // DRAW OPERATIONS
    auto result = vkAcquireNextImageKHR(logical_device,
                                        swapchain,
                                        std::numeric_limits<uint64_t>::max(),
                                        image_available_semaphores[currentFrame],
                                        VK_NULL_HANDLE,
                                        &currentImage);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        // recreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    // TRANSFER OPERATIONS
    // submit copy operations to the graphics queue

    auto beginInfo = VkCommandBufferBeginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                              .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                              .pInheritanceInfo = nullptr};

    result = vkBeginCommandBuffer(transfer_commandbuffers[currentResource], &beginInfo);
    if (result != VK_SUCCESS)
    {
        return;
    }

    transfer_buckets[currentResource].Submit();

    result = vkEndCommandBuffer(transfer_commandbuffers[currentResource]);
    if (result != VK_SUCCESS)
    {
        return;
    }

    auto submitTransferInfo = VkSubmitInfo{
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &transfer_commandbuffers[currentResource]};

    vkQueueSubmit(graphics_queue, 1, &submitTransferInfo, VK_NULL_HANDLE);

    // the graphics queue will wait to do anything in the color_attachment_output stage
    // until the waitSemaphore is signalled by vkAcquireNextImageKHR
    VkSemaphore          waitSemaphores[] = {image_available_semaphores[currentFrame]};
    VkPipelineStageFlags waitStages[]     = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

    VkSemaphore signalSemaphores[] = {render_finished_semaphores[currentFrame]};

    createCommandbuffer(currentImage, uniform_count, p_uniforms);

    auto submitInfo = VkSubmitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,

                                   .waitSemaphoreCount = 1,
                                   .pWaitSemaphores    = waitSemaphores,
                                   .pWaitDstStageMask  = waitStages,

                                   .commandBufferCount = 1,
                                   .pCommandBuffers    = &commandbuffers[currentResource],

                                   .signalSemaphoreCount = 1,
                                   .pSignalSemaphores    = signalSemaphores};

    vkResetFences(logical_device, 1, &in_flight_fences[currentFrame]);

    if (vkQueueSubmit(graphics_queue, 1, &submitInfo, in_flight_fences[currentFrame]) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkSwapchainKHR swapChains[] = {swapchain};

    auto presentInfo = VkPresentInfoKHR{.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                        .waitSemaphoreCount = 1,
                                        .pWaitSemaphores    = signalSemaphores,

                                        .swapchainCount = 1,
                                        .pSwapchains    = swapChains,
                                        .pImageIndices  = &currentImage,

                                        .pResults = nullptr};

    result = vkQueuePresentKHR(present_queue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebuffer_resized)
    {
        framebuffer_resized = false;
        // recreateSwapChain();
    }
    else if (result != VK_SUCCESS)
    {
        throw std::runtime_error("failed to present swap chain image!");
    }

    currentFrame    = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    currentResource = (currentResource + 1) % MAX_BUFFERED_RESOURCES;

    // reset buffer offsets for copies
    dynamic_mapped_vertices[currentResource].reset();
    dynamic_mapped_indices[currentResource].reset();
    staging_buffer[currentResource].reset();

    buckets[currentResource].Clear();
    transfer_buckets[currentResource].Clear();
}

std::optional<UniformHandle> RenderDevice::newUniform()
{
    auto handle = uniform_layouts[0].newUniform();
    if (!handle.has_value())
    {
        return std::nullopt;
    }

    handle.value().uniform_layout_id = 0;
    return handle;
}

void RenderDevice::updateUniform(UniformHandle handle, glm::mat4 const & data)
{
    UniformLayout layout = uniform_layouts[handle.uniform_layout_id];

    auto opt_uniform = layout.getUniform(handle);

    if (!opt_uniform.has_value())
    {
        throw std::runtime_error("in updateUniform, no uniform returned from getUniform!");
    }

    auto uniform = opt_uniform.value();

    // copy into this uniform buffer slot, don't increase offset (use it later to draw)
    uniform.uniform_buffers[currentFrame].offset = uniform.uniform_buffers[currentFrame].copy(
        sizeof(glm::mat4), static_cast<void const *>(glm::value_ptr(data)));
}

void RenderDevice::dynamicDraw(PipelineHandle    pipeline,
                               glm::mat4 const & transform,
                               uint32_t          vertex_count,
                               Vertex *          vertices,
                               uint32_t          index_count,
                               uint32_t *        indices)
{
    auto & mapped_vertices = dynamic_mapped_vertices[currentResource];
    auto & mapped_indices  = dynamic_mapped_indices[currentResource];

    VkDeviceSize vertex_offset = mapped_vertices.copy(sizeof(Vertex) * vertex_count, vertices);
    VkDeviceSize index_offset  = mapped_indices.copy(sizeof(uint32_t) * index_count, indices);

    draw(pipeline,
         transform,
         mapped_vertices.buffer,
         vertex_offset,
         mapped_indices.buffer,
         index_offset,
         index_count);
}

void RenderDevice::draw(PipelineHandle    pipeline,
                        glm::mat4 const & transform,
                        VkBuffer          vertexbuffer,
                        VkDeviceSize      vertexbuffer_offset,
                        VkBuffer          indexbuffer,
                        VkDeviceSize      indexbuffer_offset,
                        VkDeviceSize      indexbuffer_count)
{
    auto & bucket = buckets[currentResource];

    Draw * command               = bucket.AddCommand<Draw>(0, sizeof(glm::mat4));
    command->commandbuffer       = commandbuffers[currentResource];
    command->pipeline_layout     = pipelines[pipeline].vk_pipeline_layout;
    command->transform           = transform;
    command->vertexbuffer        = vertexbuffer;
    command->vertexbuffer_offset = vertexbuffer_offset;
    command->indexbuffer         = indexbuffer;
    command->indexbuffer_offset  = indexbuffer_offset;
    command->indexbuffer_count   = indexbuffer_count;
}

bool RenderDevice::createVertexbuffer(VkBuffer &       vertexbuffer,
                                      VkDeviceMemory & vertexbuffer_memory,
                                      uint32_t         vertex_count,
                                      Vertex *         vertices)
{
    VkDeviceSize bufferSize = sizeof(Vertex) * vertex_count;

    // copy to staging vertex buffer
    auto &       mapped_vertices       = staging_buffer[currentResource];
    size_t       vertex_data_size      = sizeof(Vertex) * vertex_count;
    VkDeviceSize staging_vertex_offset = mapped_vertices.copy(vertex_data_size, vertices);

    // create
    createBuffer(bufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vertexbuffer,
                 vertexbuffer_memory);

    copyBuffer(mapped_vertices.buffer, staging_vertex_offset, vertexbuffer, 0, bufferSize);

    return true;
}

bool RenderDevice::createIndexbuffer(VkBuffer &       indexbuffer,
                                     VkDeviceMemory & indexbuffer_memory,
                                     uint32_t         index_count,
                                     uint32_t *       indices)
{
    VkDeviceSize bufferSize = sizeof(uint32_t) * index_count;

    auto & mapped_indices = staging_buffer[currentResource];

    size_t index_data_size = sizeof(uint32_t) * index_count;

    VkDeviceSize staging_index_offset = mapped_indices.copy(index_data_size, indices);

    createBuffer(bufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 indexbuffer,
                 indexbuffer_memory);

    copyBuffer(mapped_indices.buffer, staging_index_offset, indexbuffer, 0, bufferSize);

    return true;
}

void RenderDevice::updateDeviceLocalBuffers(VkBuffer   vertexbuffer,
                                            VkBuffer   indexbuffer,
                                            uint32_t   vertex_count,
                                            Vertex *   vertices,
                                            uint32_t   index_count,
                                            uint32_t * indices)
{
    // TODO:
    auto & mapped_vertices = staging_buffer[currentResource];
    auto & mapped_indices  = staging_buffer[currentResource];

    size_t vertex_data_size = sizeof(Vertex) * vertex_count;
    size_t index_data_size  = sizeof(uint32_t) * index_count;

    VkDeviceSize vertex_offset = mapped_vertices.copy(vertex_data_size, vertices);
    VkDeviceSize index_offset  = mapped_indices.copy(index_data_size, indices);

    auto & bucket = transfer_buckets[currentResource];

    Copy * vertex_command         = bucket.AddCommand<Copy>(0, 0);
    vertex_command->commandbuffer = transfer_commandbuffers[currentResource];
    vertex_command->srcBuffer     = mapped_vertices.buffer;
    vertex_command->dstBuffer     = vertexbuffer;
    vertex_command->srcOffset     = vertex_offset;
    vertex_command->dstOffset     = 0;
    vertex_command->size          = vertex_data_size;

    Copy * index_command         = bucket.AddCommand<Copy>(0, 0);
    index_command->commandbuffer = transfer_commandbuffers[currentResource];
    index_command->srcBuffer     = mapped_indices.buffer;
    index_command->dstBuffer     = indexbuffer;
    index_command->srcOffset     = index_offset;
    index_command->dstOffset     = 0;
    index_command->size          = index_data_size;
}

void RenderDevice::destroyDeviceLocalBuffers(VkBuffer       vertexbuffer,
                                             VkDeviceMemory vertex_memory,
                                             VkBuffer       indexbuffer,
                                             VkDeviceMemory index_memory)
{
    // INDEXBUFFER
    vkDestroyBuffer(logical_device, indexbuffer, nullptr);
    vkFreeMemory(logical_device, index_memory, nullptr);

    // VERTEXBUFFER
    vkDestroyBuffer(logical_device, vertexbuffer, nullptr);
    vkFreeMemory(logical_device, vertex_memory, nullptr);
}

void RenderDevice::createTexture(char const * texture_path)
{
    int       texWidth, texHeight, texChannels;
    stbi_uc * pixels = stbi_load(texture_path, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

    VkDeviceSize imageSize = texWidth * texHeight * 4;

    // mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    if (!pixels)
    {
        throw std::runtime_error("failed to load texture image!");
    }

    VkDeviceSize pixel_data_offset = staging_buffer[currentResource].copy(
        static_cast<size_t>(imageSize), pixels);

    stbi_image_free(pixels);

    createImage(texWidth,
                texHeight,
                1,
                VK_SAMPLE_COUNT_1_BIT,
                VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                texture.vk_image,
                texture.vk_image_memory);

    createImageView(texture.vk_image,
                    texture.vk_image_view,
                    VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    1);

    auto & bucket = transfer_buckets[currentResource];

    SetImageLayout * dst_optimal_command  = bucket.AddCommand<SetImageLayout>(0, 0);
    dst_optimal_command->commandbuffer    = transfer_commandbuffers[currentResource];
    dst_optimal_command->srcAccessMask    = 0;
    dst_optimal_command->dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
    dst_optimal_command->oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
    dst_optimal_command->newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    dst_optimal_command->image            = texture.vk_image;
    dst_optimal_command->mipLevels        = 1;
    dst_optimal_command->aspectMask       = VK_IMAGE_ASPECT_COLOR_BIT;
    dst_optimal_command->sourceStage      = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dst_optimal_command->destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

    CopyToImage * copy_command  = bucket.AddCommand<CopyToImage>(0, 0);
    copy_command->commandbuffer = transfer_commandbuffers[currentResource];
    copy_command->srcBuffer     = staging_buffer[currentResource].buffer;
    copy_command->srcOffset     = pixel_data_offset;
    copy_command->dstImage      = texture.vk_image;
    copy_command->width         = static_cast<uint32_t>(texWidth);
    copy_command->height        = static_cast<uint32_t>(texHeight);

    SetImageLayout * shader_optimal_command  = bucket.AddCommand<SetImageLayout>(0, 0);
    shader_optimal_command->commandbuffer    = transfer_commandbuffers[currentResource];
    shader_optimal_command->srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
    shader_optimal_command->dstAccessMask    = VK_ACCESS_SHADER_READ_BIT;
    shader_optimal_command->oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    shader_optimal_command->newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    shader_optimal_command->image            = texture.vk_image;
    shader_optimal_command->mipLevels        = 1;
    shader_optimal_command->aspectMask       = VK_IMAGE_ASPECT_COLOR_BIT;
    shader_optimal_command->sourceStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
    shader_optimal_command->destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

    auto samplerInfo = VkSamplerCreateInfo{.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                           .magFilter    = VK_FILTER_NEAREST,
                                           .minFilter    = VK_FILTER_NEAREST,
                                           .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                           .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                           .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                           .anisotropyEnable = VK_TRUE,
                                           .maxAnisotropy    = 16,
                                           .borderColor      = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
                                           .unnormalizedCoordinates = VK_FALSE,
                                           .compareEnable           = VK_FALSE,
                                           .compareOp               = VK_COMPARE_OP_ALWAYS,
                                           .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                           .mipLodBias = 0.0f,
                                           .minLod     = 0.0f,
                                           .maxLod     = 1};

    if (vkCreateSampler(logical_device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create texture sampler!");
    }

    // create the descriptors
    UniformLayout & uniform_layout = uniform_layouts[1];

    auto allocInfo = VkDescriptorSetAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = uniform_layout.vk_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &uniform_layout.vk_descriptorset_layout};

    auto result = vkAllocateDescriptorSets(logical_device, &allocInfo, &sampler_descriptor);
    if (result != VK_SUCCESS)
    {
        std::cout << "ERROR allocating descriptorsets\n";
        return; // result;
    }

    auto imageInfo = VkDescriptorImageInfo{.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                           .imageView   = texture.vk_image_view,
                                           .sampler     = sampler};

    auto write_info = VkWriteDescriptorSet{
        .sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet           = sampler_descriptor,
        .dstBinding       = 1,
        .dstArrayElement  = 0,
        .descriptorType   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount  = 1,
        .pBufferInfo      = nullptr,
        .pImageInfo       = &imageInfo,
        .pTexelBufferView = nullptr};

    vkUpdateDescriptorSets(logical_device, 1, &write_info, 0, nullptr);
}

void RenderDevice::destroyTexture(Texture & texture)
{
    vkDestroyImageView(logical_device, texture.vk_image_view, nullptr);
    vkDestroyImage(logical_device, texture.vk_image, nullptr);
    vkFreeMemory(logical_device, texture.vk_image_memory, nullptr);
}

void RenderDevice::checkValidationLayerSupport()
{
    uint32_t layerCount{0};
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);

    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char * layerName: validation_layers)
    {
        bool layerFound = false;

        for (const auto & layerProperties: availableLayers)
        {
            if (strcmp(layerName, layerProperties.layerName) == 0)
            {
                layerFound = true;
                break;
            }
        }

        if (!layerFound)
        {
            validation_supported = false;
            break;
        }
    }

    validation_supported = true;
}

void RenderDevice::getRequiredExtensions()
{
    uint32_t      glfwExtensionCount = 0;
    const char ** glfwExtensions{nullptr};
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    auto it = required_extensions.begin();
    required_extensions.insert(it, glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (use_validation)
    {
        required_extensions.push_back(
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME); // literally "VK_EXT_debug_utils" extension name
    }
}

// INSTANCE
VkResult RenderDevice::createInstance(char const * window_name)
{
    if (use_validation && !validation_supported)
    {
        // maybe don't return? Just set use_validation to false and log?
        use_validation = false;
        // throw std::runtime_error("validation layers requested, but not available!");
    }

    // app info struct
    auto appInfo = VkApplicationInfo{.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                     .pApplicationName   = window_name,
                                     .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                                     .pEngineName        = "jed",
                                     .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
                                     .apiVersion         = VK_API_VERSION_1_0};

    // create info struct
    auto createInfo = VkInstanceCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo        = &appInfo,
        .enabledExtensionCount   = static_cast<uint32_t>(required_extensions.size()),
        .ppEnabledExtensionNames = required_extensions.data(),
        .enabledLayerCount       = 0};

    // validation layers
    if (use_validation)
    {
        createInfo.enabledLayerCount   = static_cast<uint32_t>(validation_layers.size());
        createInfo.ppEnabledLayerNames = validation_layers.data();
    }

    // create instance
    return vkCreateInstance(&createInfo, nullptr, &instance);

    /*
    // supported extensions
    uint32_t extensionCount{0};
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> extensions{extensionCount};
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    std::cout << "available extensions:" << std::endl;

    for (const auto & extension: extensions)
    {
        std::cout << "\t" << extension.extensionName << std::endl;
    }
    */
}

// VALIDATION LAYER DEBUG MESSAGER
VKAPI_ATTR VkBool32 VKAPI_CALL
                    RenderDevice::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
                            VkDebugUtilsMessageTypeFlagsEXT              messageType,
                            const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData,
                            void *                                       pUserData)
{
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

VkResult RenderDevice::createDebugMessenger()
{
    auto createInfo = VkDebugUtilsMessengerCreateInfoEXT{
        .sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                           | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                           | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
        .pUserData       = nullptr // Optional
    };

    return createDebugUtilsMessengerEXT(&createInfo, nullptr, &debug_messager);
}

VkResult RenderDevice::createDebugUtilsMessengerEXT(
    const VkDebugUtilsMessengerCreateInfoEXT * pCreateInfo,
    const VkAllocationCallbacks *              pAllocator,
    VkDebugUtilsMessengerEXT *                 pDebugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void RenderDevice::cleanupDebugUtilsMessengerEXT(VkDebugUtilsMessengerEXT      debugMessenger,
                                                 const VkAllocationCallbacks * pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        func(instance, debugMessenger, pAllocator);
    }
}

// SURFACE
VkResult RenderDevice::createSurface()
{
    return glfwCreateWindowSurface(instance, window, nullptr, &surface);
}

bool RenderDevice::PhysicalDeviceInfo::queues_complete() const
{
    return present_queue != -1 && graphics_queue != -1 && transfer_queue != -1;
}

bool RenderDevice::PhysicalDeviceInfo::swapchain_adequate() const
{
    return !formats.empty() && !presentModes.empty();
}

bool RenderDevice::pickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0)
    {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices{deviceCount};
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (auto const & device: devices)
    {
        PhysicalDeviceInfo device_info;

        if (isDeviceSuitable(device, device_info))
        {
            physical_device      = device;
            physical_device_info = device_info;
            getMaxUsableSampleCount();
            break;
        }
    }

    if (physical_device == VK_NULL_HANDLE)
    {
        return false;
    }
    return true;
}

bool RenderDevice::isDeviceSuitable(VkPhysicalDevice device, PhysicalDeviceInfo & device_info)
{
    /*
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);


    return deviceProperties.deviceType ==
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && deviceFeatures.geometryShader;
    */

    findQueueFamilies(device, device_info);

    if (!device_info.queues_complete())
    {
        return false;
    }

    checkDeviceExtensionSupport(device, device_info);

    if (!device_info.has_required_extensions)
    {
        return false;
    }

    querySwapChainSupport(device, device_info);

    if (!device_info.swapchain_adequate())
    {
        return false;
    }

    vkGetPhysicalDeviceFeatures(device, &device_info.features);

    return device_info.features.samplerAnisotropy == VK_TRUE;
}

void RenderDevice::findQueueFamilies(VkPhysicalDevice device, PhysicalDeviceInfo & device_info)
{
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    device_info.queue_family_properties.resize(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(
        device, &queueFamilyCount, device_info.queue_family_properties.data());

    int i = 0;
    for (auto const & queueFamily: device_info.queue_family_properties)
    {
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

        if (device_info.present_queue == -1 && queueFamily.queueCount > 0 && presentSupport)
        {
            device_info.present_queue = i;
        }

        if (device_info.graphics_queue == -1 && queueFamily.queueCount > 0
            && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            device_info.graphics_queue = i;
        }

        if (queueFamily.queueCount > 0 && (i != device_info.graphics_queue)
            && (queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT
                || queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT))
        {
            device_info.transfer_queue = i;
        }

        if (device_info.queues_complete())
        {
            break;
        }

        i++;
    }

    if (device_info.transfer_queue == -1)
    {
        device_info.transfer_queue = device_info.graphics_queue;
    }
}

bool RenderDevice::checkDeviceExtensionSupport(VkPhysicalDevice     device,
                                               PhysicalDeviceInfo & device_info)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    device_info.available_extensions.resize(extensionCount);
    vkEnumerateDeviceExtensionProperties(
        device, nullptr, &extensionCount, device_info.available_extensions.data());

    std::set<std::string> requiredExtensions(required_device_extensions.begin(),
                                             required_device_extensions.end());

    for (const auto & extension: device_info.available_extensions)
    {
        requiredExtensions.erase(extension.extensionName);
    }

    return device_info.has_required_extensions = requiredExtensions.empty();
}

void RenderDevice::querySwapChainSupport(VkPhysicalDevice device, PhysicalDeviceInfo & device_info)
{
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &device_info.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

    if (formatCount != 0)
    {
        device_info.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(
            device, surface, &formatCount, device_info.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

    if (presentModeCount != 0)
    {
        device_info.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            device, surface, &presentModeCount, device_info.presentModes.data());
    }
}

void RenderDevice::getMaxUsableSampleCount()
{
    vkGetPhysicalDeviceProperties(physical_device, &physical_device_info.properties);

    VkSampleCountFlags counts = std::min(
        physical_device_info.properties.limits.framebufferColorSampleCounts,
        physical_device_info.properties.limits.framebufferDepthSampleCounts);
    if (counts & VK_SAMPLE_COUNT_64_BIT)
    {
        physical_device_info.msaa_samples = VK_SAMPLE_COUNT_64_BIT;
        return;
    }
    if (counts & VK_SAMPLE_COUNT_32_BIT)
    {
        physical_device_info.msaa_samples = VK_SAMPLE_COUNT_32_BIT;
        return;
    }
    if (counts & VK_SAMPLE_COUNT_16_BIT)
    {
        physical_device_info.msaa_samples = VK_SAMPLE_COUNT_16_BIT;
        return;
    }
    if (counts & VK_SAMPLE_COUNT_8_BIT)
    {
        physical_device_info.msaa_samples = VK_SAMPLE_COUNT_8_BIT;
        return;
    }
    if (counts & VK_SAMPLE_COUNT_4_BIT)
    {
        physical_device_info.msaa_samples = VK_SAMPLE_COUNT_4_BIT;
        return;
    }
    if (counts & VK_SAMPLE_COUNT_2_BIT)
    {
        physical_device_info.msaa_samples = VK_SAMPLE_COUNT_2_BIT;
        return;
    }

    physical_device_info.msaa_samples = VK_SAMPLE_COUNT_1_BIT;
}

VkResult RenderDevice::createLogicalDevice()
{
    // create queue info for graphics and present queues
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t>                   uniqueQueueFamilies = {
        static_cast<uint32_t>(physical_device_info.present_queue),
        static_cast<uint32_t>(physical_device_info.graphics_queue),
        static_cast<uint32_t>(physical_device_info.transfer_queue)};

    float queuePriority = 1.0f;
    for (uint32_t queueFamily: uniqueQueueFamilies)
    {
        auto queueCreateInfo = VkDeviceQueueCreateInfo{
            .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queueFamily,
            .queueCount       = 1,
            .pQueuePriorities = &queuePriority};

        queueCreateInfos.push_back(queueCreateInfo);
    }

    // ensure physical device supports this
    auto deviceFeatures = VkPhysicalDeviceFeatures{.samplerAnisotropy = VK_TRUE,
                                                   .sampleRateShading = VK_TRUE};

    // create logical device info
    auto createInfo = VkDeviceCreateInfo{
        .sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pQueueCreateInfos    = queueCreateInfos.data(),
        .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
        .pEnabledFeatures     = &deviceFeatures,

        // extensions
        .enabledExtensionCount   = static_cast<uint32_t>(required_device_extensions.size()),
        .ppEnabledExtensionNames = required_device_extensions.data(),

        // layers
        .enabledLayerCount = 0};

    // layers
    if (use_validation)
    {
        createInfo.enabledLayerCount   = static_cast<uint32_t>(validation_layers.size());
        createInfo.ppEnabledLayerNames = validation_layers.data();
    }

    return vkCreateDevice(physical_device, &createInfo, nullptr, &logical_device);
}

VkResult RenderDevice::createSwapChain()
{
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(physical_device_info.formats);
    VkPresentModeKHR   presentMode   = chooseSwapPresentMode(physical_device_info.presentModes);
    VkExtent2D         extent        = chooseSwapExtent(physical_device_info.capabilities);

    swapchain_image_format = surfaceFormat.format;
    swapchain_extent       = extent;
    depth_format           = findDepthFormat();

    // imagecount is greater than min image count and less than or equal to maximage count
    uint32_t imageCount = physical_device_info.capabilities.minImageCount + 1;
    if (physical_device_info.capabilities.maxImageCount > 0
        && imageCount > physical_device_info.capabilities.maxImageCount)
    {
        imageCount = physical_device_info.capabilities.maxImageCount;
    }

    auto createInfo = VkSwapchainCreateInfoKHR{
        .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface          = surface,
        .minImageCount    = imageCount,
        .imageFormat      = surfaceFormat.format,
        .imageColorSpace  = surfaceFormat.colorSpace,
        .imageExtent      = extent,
        .imageArrayLayers = 1,
        .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .preTransform     = physical_device_info.capabilities.currentTransform,
        .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode      = presentMode,
        .clipped          = VK_TRUE,
        .oldSwapchain     = VK_NULL_HANDLE};

    uint32_t queueFamilyIndices[] = {static_cast<uint32_t>(physical_device_info.graphics_queue),
                                     static_cast<uint32_t>(physical_device_info.present_queue)};

    // if there are two queues, enable concurrent access
    // since graphics queue will draw to the swap chain and present queue will present the image
    if (physical_device_info.graphics_queue != physical_device_info.present_queue)
    {
        createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices   = queueFamilyIndices; // queues with concurrent access
    }
    else
    {
        createInfo.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;       // Optional
        createInfo.pQueueFamilyIndices   = nullptr; // Optional
    }

    return vkCreateSwapchainKHR(logical_device, &createInfo, nullptr, &swapchain);
}

VkSurfaceFormatKHR RenderDevice::chooseSwapSurfaceFormat(
    std::vector<VkSurfaceFormatKHR> const & availableFormats)
{
    // surface has no preferred format so we can choose whatever we want
    if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
    {
        return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    }

    // try and find the desired format
    for (const auto & availableFormat: availableFormats)
    {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM
            && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return availableFormat;
        }
    }

    // just return the first since we couldn't find what we wanted
    // could instead rank available formats, but this is probably fine
    return availableFormats[0];
}

VkPresentModeKHR RenderDevice::chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR> & availablePresentModes)
{
    VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

    for (const auto & availablePresentMode: availablePresentModes)
    {
        // Best present mode
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return availablePresentMode;
        }
        // I don't know why we would want this more than FIFO..
        else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
        {
            bestMode = availablePresentMode;
        }
    }

    // should be available on all platforms
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D RenderDevice::chooseSwapExtent(VkSurfaceCapabilitiesKHR const & capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return capabilities.currentExtent;
    }
    else
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

        actualExtent.width = std::max(
            capabilities.minImageExtent.width,
            std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(
            capabilities.minImageExtent.height,
            std::min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}

void RenderDevice::getSwapChainImages()
{
    vkGetSwapchainImagesKHR(logical_device, swapchain, &swapchain_image_count, nullptr);
    swapchain_images.resize(swapchain_image_count);
    vkGetSwapchainImagesKHR(
        logical_device, swapchain, &swapchain_image_count, swapchain_images.data());
}

VkResult RenderDevice::createSwapChainImageViews()
{
    swapchain_image_views.resize(swapchain_images.size());

    for (size_t i = 0; i < swapchain_images.size(); i++)
    {
        VkResult imageViewResult = createImageView(swapchain_images[i],
                                                   swapchain_image_views[i],
                                                   swapchain_image_format,
                                                   VK_IMAGE_ASPECT_COLOR_BIT,
                                                   1);

        if (imageViewResult != VK_SUCCESS)
        {
            return imageViewResult;
        }
    }

    return VK_SUCCESS;
}

VkResult RenderDevice::createImageView(VkImage            image,
                                       VkImageView &      image_view,
                                       VkFormat           format,
                                       VkImageAspectFlags aspectFlags,
                                       uint32_t           mipLevels)
{
    auto viewInfo = VkImageViewCreateInfo{.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                          .image    = image,
                                          .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                          .format   = format,
                                          .subresourceRange.aspectMask     = aspectFlags,
                                          .subresourceRange.baseMipLevel   = 0,
                                          .subresourceRange.levelCount     = mipLevels,
                                          .subresourceRange.baseArrayLayer = 0,
                                          .subresourceRange.layerCount     = 1};

    return vkCreateImageView(logical_device, &viewInfo, nullptr, &image_view);
}

// RENDER PASS & ATTACHMENT DESCRIPTIONS
VkResult RenderDevice::createRenderPass()
{
    for (size_t i = 0; i < renderpasses.size(); ++i)
    {
        auto & renderpass_config = renderpass_configs[i];
        auto & renderpass        = renderpasses[i];

        for (size_t i = 0; i < renderpass_config.descriptions.size(); ++i)
        {
            auto &       description       = renderpass_config.descriptions[i];
            auto         attachment_handle = renderpass_config.framebuffer_config.attachments[i];
            auto const & attachment_config = attachment_configs[attachment_handle];

            if (attachment_config.format == Format::USE_COLOR)
            {
                description.format = swapchain_image_format;
            }
            else if (attachment_config.format == Format::USE_DEPTH)
            {
                description.format = depth_format;
            }

            if (attachment_config.multisampled)
            {
                description.samples = physical_device_info.msaa_samples;
            }
            else
            {
                description.samples = VK_SAMPLE_COUNT_1_BIT;
            }
        }

        std::vector<VkSubpassDescription> subpasses;
        subpasses.reserve(renderpass_config.subpasses.size());

        for (auto & subpass_info: renderpass_config.subpasses)
        {
            subpasses.push_back(VkSubpassDescription{
                .pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS,
                .colorAttachmentCount = static_cast<uint32_t>(
                    subpass_info.color_attachments.size()),
                .pColorAttachments       = subpass_info.color_attachments.data(),
                .pDepthStencilAttachment = &subpass_info.depth_stencil_attachment,
                .pResolveAttachments     = &subpass_info.color_resolve_attachment});
        }

        auto renderPassInfo = VkRenderPassCreateInfo{
            .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = static_cast<uint32_t>(renderpass_config.descriptions.size()),
            .pAttachments    = renderpass_config.descriptions.data(),

            .subpassCount = static_cast<uint32_t>(subpasses.size()),
            .pSubpasses   = subpasses.data(),

            .dependencyCount = static_cast<uint32_t>(renderpass_config.subpass_dependencies.size()),
            .pDependencies   = renderpass_config.subpass_dependencies.data()};

        auto result = vkCreateRenderPass(logical_device, &renderPassInfo, nullptr, &renderpass);
        if (result != VK_SUCCESS)
        {
            return result;
        }

        createFramebuffer(renderpass_config, renderpass, framebuffers[i]);
    }

    return VK_SUCCESS;
}

// FRAMEBUFFER
VkResult RenderDevice::createFramebuffer(RenderpassConfig const &     config,
                                         VkRenderPass const &         renderpass,
                                         std::vector<VkFramebuffer> & framebuffers)
{
    auto const & framebuffer_config = config.framebuffer_config;
    framebuffers.resize(swapchain_image_count);

    for (size_t i = 0; i < swapchain_image_count; ++i)
    {
        auto & framebuffer = framebuffers[i];

        auto fb_attachments = std::vector<VkImageView>{};

        for (auto attachment_handle: framebuffer_config.attachments)
        {
            auto const & attachment_config = attachment_configs[attachment_handle];

            if (attachment_config.is_swapchain_image)
            {
                fb_attachments.push_back(swapchain_image_views[i]);
            }
            else
            {
                fb_attachments.push_back(attachments[attachment_handle].vk_image_view);
            }
        }

        auto framebufferInfo = VkFramebufferCreateInfo{
            .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass      = renderpass,
            .attachmentCount = static_cast<uint32_t>(fb_attachments.size()),
            .pAttachments    = fb_attachments.data(),
            .width           = swapchain_extent.width,
            .height          = swapchain_extent.height,
            .layers          = 1};

        auto result = vkCreateFramebuffer(logical_device, &framebufferInfo, nullptr, &framebuffer);
        if (result != VK_SUCCESS)
        {
            return result;
        }
    }

    return VK_SUCCESS;
}

VkFormat RenderDevice::findDepthFormat()
{
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

VkFormat RenderDevice::findSupportedFormat(const std::vector<VkFormat> & candidates,
                                           VkImageTiling                 tiling,
                                           VkFormatFeatureFlags          features)
{
    for (VkFormat format: candidates)
    {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
        {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL
                 && (props.optimalTilingFeatures & features) == features)
        {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}

VkResult RenderDevice::createUniformLayouts()
{
    for (auto & uniform_layout: uniform_layouts)
    {
        uniform_layout.uniform_size
            = physical_device_info.properties.limits.maxUniformBufferRange
              / physical_device_info.properties.limits.minUniformBufferOffsetAlignment;

        auto layoutInfo = VkDescriptorSetLayoutCreateInfo{
            .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings    = &uniform_layout.binding};

        auto result = vkCreateDescriptorSetLayout(
            logical_device, &layoutInfo, nullptr, &uniform_layout.vk_descriptorset_layout);
        if (result != VK_SUCCESS)
        {
            return result;
        }

        auto poolsize = VkDescriptorPoolSize{
            .type            = uniform_layout.binding.descriptorType,
            .descriptorCount = static_cast<uint32_t>(MAX_BUFFERED_RESOURCES)};

        auto poolInfo = VkDescriptorPoolCreateInfo{
            .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .poolSizeCount = 1,
            .pPoolSizes    = &poolsize,
            .maxSets       = static_cast<uint32_t>(MAX_BUFFERED_RESOURCES)};

        result = vkCreateDescriptorPool(
            logical_device, &poolInfo, nullptr, &uniform_layout.vk_descriptor_pool);
        if (result != VK_SUCCESS)
        {
            return result;
        }

        // TODO REMOVE, UNIFORM LAYOUT SHOULD BE ABLE TO HANDLE THIS
        if (uniform_layout.binding.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
        {
            continue;
        }

        // create the uniforms
        VkDeviceSize bufferSize = sizeof(glm::mat4);

        auto & uniforms = uniform_layout.uniform_buffers;

        uniforms.resize(MAX_BUFFERED_RESOURCES);

        for (size_t i = 0; i < MAX_BUFFERED_RESOURCES; ++i)
        {
            uniforms[i].memory_size = bufferSize;

            createBuffer(uniforms[i].memory_size,
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniforms[i].buffer,
                         uniforms[i].memory);

            vkMapMemory(logical_device,
                        uniforms[i].memory,
                        0,
                        uniforms[i].memory_size,
                        0,
                        &uniforms[i].data);
        }

        // create the descriptors
        std::vector<VkDescriptorSetLayout> layouts{static_cast<size_t>(MAX_BUFFERED_RESOURCES),
                                                   uniform_layout.vk_descriptorset_layout};

        auto allocInfo = VkDescriptorSetAllocateInfo{
            .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool     = uniform_layout.vk_descriptor_pool,
            .descriptorSetCount = static_cast<uint32_t>(MAX_BUFFERED_RESOURCES),
            .pSetLayouts        = layouts.data()};

        auto & descriptorsets = uniform_layout.vk_descriptorsets;

        descriptorsets.resize(MAX_BUFFERED_RESOURCES);

        result = vkAllocateDescriptorSets(logical_device, &allocInfo, descriptorsets.data());
        if (result != VK_SUCCESS)
        {
            return result;
        }

        for (size_t i = 0; i < MAX_BUFFERED_RESOURCES; ++i)
        {
            auto bufferInfo = VkDescriptorBufferInfo{
                .buffer = uniforms[i].buffer, .offset = 0, .range = sizeof(glm::mat4)};

            auto descriptorWrite = VkWriteDescriptorSet{
                .sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet           = descriptorsets[i],
                .dstBinding       = 0,
                .dstArrayElement  = 0,
                .descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .descriptorCount  = 1,
                .pBufferInfo      = &bufferInfo,
                .pImageInfo       = nullptr,
                .pTexelBufferView = nullptr};

            vkUpdateDescriptorSets(logical_device, 1, &descriptorWrite, 0, nullptr);
        }
    }

    return VK_SUCCESS;
}

// SHADERS
VkResult RenderDevice::createShaders()
{
    VkResult result;

    for (size_t i = 0; i < shaders.size(); ++i)
    {
        auto & shader = shaders[i];

        auto shaderCode = readFile(shader_names[i]);

        result = createShaderModule(shaderCode, shader);
        if (result != VK_SUCCESS)
        {
            return result;
        }
    }

    return result;
}

VkResult RenderDevice::createShaderModule(std::vector<char> const & code,
                                          VkShaderModule &          shaderModule)
{
    auto createInfo = VkShaderModuleCreateInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode    = reinterpret_cast<const uint32_t *>(code.data())};

    return vkCreateShaderModule(logical_device, &createInfo, nullptr, &shaderModule);
}

// GRAPHICS PIPELINE
VkResult RenderDevice::createGraphicsPipeline()
{
    for (size_t i = 0; i < pipelines.size(); ++i)
    {
        auto & pipeline        = pipelines[i];
        auto & pipeline_config = pipeline_configs[i];

        auto vertShaderStageInfo = VkPipelineShaderStageCreateInfo{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = shaders[pipeline_config.vertex_shader],
            .pName  = "main"};

        auto fragShaderStageInfo = VkPipelineShaderStageCreateInfo{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = shaders[pipeline_config.fragment_shader],
            .pName  = "main"};

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindings = std::vector<VkVertexInputBindingDescription>{};
        for (auto const & binding: pipeline_config.vertex_bindings)
        {
            bindings.push_back(vertex_bindings[binding]);
        }

        auto attributes = std::vector<VkVertexInputAttributeDescription>{};
        for (auto const & attribute: pipeline_config.vertex_attributes)
        {
            attributes.push_back(vertex_attributes[attribute]);
        }

        auto vertexInputInfo = VkPipelineVertexInputStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount   = static_cast<uint32_t>(bindings.size()),
            .pVertexBindingDescriptions      = bindings.data(),
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
            .pVertexAttributeDescriptions    = attributes.data()};

        auto inputAssembly = VkPipelineInputAssemblyStateCreateInfo{
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE};

        auto viewport = VkViewport{.x        = 0.0f,
                                   .y        = 0.0f,
                                   .width    = (float)swapchain_extent.width,
                                   .height   = (float)swapchain_extent.height,
                                   .minDepth = 0.0f,
                                   .maxDepth = 1.0f};

        auto scissor = VkRect2D{.offset = {0, 0}, .extent = swapchain_extent};

        auto viewportState = VkPipelineViewportStateCreateInfo{
            .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports    = &viewport,
            .scissorCount  = 1,
            .pScissors     = &scissor};

        auto rasterizer = VkPipelineRasterizationStateCreateInfo{
            .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable        = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode             = VK_POLYGON_MODE_FILL,
            .lineWidth               = 1.0f,
            .cullMode                = VK_CULL_MODE_NONE,
            .frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE,

            .depthBiasEnable         = VK_FALSE,
            .depthBiasConstantFactor = 0.0f, // Optional
            .depthBiasClamp          = 0.0f, // Optional
            .depthBiasSlopeFactor    = 0.0f  // Optional
        };

        auto multisampling = VkPipelineMultisampleStateCreateInfo{
            .sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .sampleShadingEnable   = VK_TRUE,
            .rasterizationSamples  = physical_device_info.msaa_samples,
            .minSampleShading      = 0.2f,     // Optional
            .pSampleMask           = nullptr,  // Optional
            .alphaToCoverageEnable = VK_FALSE, // Optional
            .alphaToOneEnable      = VK_FALSE  // Optional
        };

        auto depthStencil = VkPipelineDepthStencilStateCreateInfo{
            .sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable       = VK_TRUE,
            .depthWriteEnable      = VK_TRUE,
            .depthCompareOp        = VK_COMPARE_OP_LESS,
            .depthBoundsTestEnable = VK_FALSE,
            .minDepthBounds        = 0.0f,
            .maxDepthBounds        = 1.0f,
            .stencilTestEnable     = VK_FALSE,
            .front                 = {},
            .back                  = {}};

        auto colorBlendAttachment = VkPipelineColorBlendAttachmentState{
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                              | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            .blendEnable         = VK_FALSE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp        = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp        = VK_BLEND_OP_ADD};

        auto colorBlending = VkPipelineColorBlendStateCreateInfo{
            .sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable     = VK_FALSE,
            .logicOp           = VK_LOGIC_OP_COPY, // Optional
            .attachmentCount   = 1,
            .pAttachments      = &colorBlendAttachment,
            .blendConstants[0] = 0.0f, // Optional
            .blendConstants[1] = 0.0f, // Optional
            .blendConstants[2] = 0.0f, // Optional
            .blendConstants[3] = 0.0f  // Optional
        };

        /*
        VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_LINE_WIDTH};

        auto dynamicState = VkPipelineDynamicStateCreateInfo{
            .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = 2,
            .pDynamicStates    = dynamicStates};
        */

        auto pushConstantRanges = std::vector<VkPushConstantRange>{};
        for (auto const & push_constant: pipeline_config.push_constants)
        {
            pushConstantRanges.push_back(push_constants[push_constant]);
        }

        std::vector<VkDescriptorSetLayout> layouts;

        for (auto & layout_handle: pipeline_config.uniform_layouts)
        {
            layouts.push_back(uniform_layouts[layout_handle].vk_descriptorset_layout);
        }

        auto pipelineLayoutInfo = VkPipelineLayoutCreateInfo{
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts            = layouts.data(),
            .pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size()),
            .pPushConstantRanges    = pushConstantRanges.data()};

        auto result = vkCreatePipelineLayout(
            logical_device, &pipelineLayoutInfo, nullptr, &pipeline.vk_pipeline_layout);
        if (result != VK_SUCCESS)
        {
            return result;
        }

        auto pipelineInfo = VkGraphicsPipelineCreateInfo{
            .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount          = 2,
            .pStages             = shaderStages,
            .pVertexInputState   = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState      = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState   = &multisampling,
            .pDepthStencilState  = &depthStencil,
            .pColorBlendState    = &colorBlending,
            .pDynamicState       = nullptr, // Optional
            .layout              = pipeline.vk_pipeline_layout,
            .renderPass          = renderpasses[pipeline_config.renderpass],
            .subpass             = pipeline_config.subpass,
            .basePipelineHandle  = VK_NULL_HANDLE, // Optional
            .basePipelineIndex   = -1              // Optional
        };

        result = vkCreateGraphicsPipelines(
            logical_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline.vk_pipeline);

        if (result != VK_SUCCESS)
        {
            return result;
        }
    }

    return VK_SUCCESS;
}

// COMMAND POOL
VkResult RenderDevice::createCommandPool()
{
    auto poolInfo = VkCommandPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = static_cast<uint32_t>(physical_device_info.graphics_queue),
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};

    return vkCreateCommandPool(logical_device, &poolInfo, nullptr, &command_pool);
}

VkResult RenderDevice::createAttachments()
{
    for (uint32_t i = 0; i < attachment_configs.size(); ++i)
    {
        auto &       attachment        = attachments[i];
        auto const & attachment_config = attachment_configs[i];

        if (attachment_config.is_swapchain_image)
        {
            continue;
        }

        VkFormat           format;
        VkImageUsageFlags  usage;
        VkImageAspectFlags aspect;
        VkImageLayout      final_layout;

        if (attachment_config.format == Format::USE_COLOR)
        {
            format       = swapchain_image_format;
            usage        = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            aspect       = VK_IMAGE_ASPECT_COLOR_BIT;
            final_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        else if (attachment_config.format == Format::USE_DEPTH)
        {
            format       = depth_format;
            usage        = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            aspect       = VK_IMAGE_ASPECT_DEPTH_BIT;
            final_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }

        VkSampleCountFlagBits samples;

        if (attachment_config.multisampled)
        {
            samples = physical_device_info.msaa_samples;
        }
        else
        {
            samples = VK_SAMPLE_COUNT_1_BIT;
        }

        auto result = createImage(swapchain_extent.width,
                                  swapchain_extent.height,
                                  1,
                                  samples,
                                  format,
                                  VK_IMAGE_TILING_OPTIMAL,
                                  VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | usage,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                  attachment.vk_image,
                                  attachment.vk_image_memory);

        if (result != VK_SUCCESS)
        {
            return result;
        }

        result = createImageView(attachment.vk_image, attachment.vk_image_view, format, aspect, 1);

        if (result != VK_SUCCESS)
        {
            return result;
        }
    }

    return VK_SUCCESS;
}

VkResult RenderDevice::createImage(uint32_t              width,
                                   uint32_t              height,
                                   uint32_t              mipLevels,
                                   VkSampleCountFlagBits numSamples,
                                   VkFormat              format,
                                   VkImageTiling         tiling,
                                   VkImageUsageFlags     usage,
                                   VkMemoryPropertyFlags properties,
                                   VkImage &             image,
                                   VkDeviceMemory &      imageMemory)
{
    auto imageInfo = VkImageCreateInfo{.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                       .imageType     = VK_IMAGE_TYPE_2D,
                                       .extent.width  = width,
                                       .extent.height = height,
                                       .extent.depth  = 1,
                                       .mipLevels     = mipLevels,
                                       .arrayLayers   = 1,
                                       .format        = format,
                                       .tiling        = tiling,
                                       .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                       .usage         = usage,
                                       .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
                                       .samples       = numSamples};

    auto result = vkCreateImage(logical_device, &imageInfo, nullptr, &image);

    if (result != VK_SUCCESS)
    {
        return result;
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(logical_device, image, &memRequirements);

    auto allocInfo = VkMemoryAllocateInfo{
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = memRequirements.size,
        .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};

    result = vkAllocateMemory(logical_device, &allocInfo, nullptr, &imageMemory);

    if (result != VK_SUCCESS)
    {
        return result;
    }

    vkBindImageMemory(logical_device, image, imageMemory, 0);

    return VK_SUCCESS;
}

uint32_t RenderDevice::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if (typeFilter & (1 << i)
            && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void RenderDevice::transitionImageLayout(VkImage       image,
                                         VkFormat      format,
                                         VkImageLayout oldLayout,
                                         VkImageLayout newLayout,
                                         uint32_t      mipLevels)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    auto barrier = VkImageMemoryBarrier{.sType     = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                        .oldLayout = oldLayout,
                                        .newLayout = newLayout,
                                        .srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED,
                                        .dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED,
                                        .image                           = image,
                                        .subresourceRange.baseMipLevel   = 0,
                                        .subresourceRange.levelCount     = mipLevels,
                                        .subresourceRange.baseArrayLayer = 0,
                                        .subresourceRange.layerCount     = 1,
                                        .srcAccessMask                   = 0,
                                        .dstAccessMask                   = 0};

    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

        if (hasStencilComponent(format))
        {
            barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
    }
    else
    {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage      = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
             && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
             && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                                | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        sourceStage      = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
             && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        sourceStage      = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    else
    {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(commandBuffer);
}

bool RenderDevice::hasStencilComponent(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

VkCommandBuffer RenderDevice::beginSingleTimeCommands()
{
    auto allocInfo = VkCommandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandPool        = command_pool,
        .commandBufferCount = 1};

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(logical_device, &allocInfo, &commandBuffer);

    auto beginInfo = VkCommandBufferBeginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                              .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void RenderDevice::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
    vkEndCommandBuffer(commandBuffer);

    auto submitInfo = VkSubmitInfo{.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                   .commandBufferCount = 1,
                                   .pCommandBuffers    = &commandBuffer};

    vkQueueSubmit(graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphics_queue);

    vkFreeCommandBuffers(logical_device, command_pool, 1, &commandBuffer);
}

VkResult RenderDevice::createBuffer(VkDeviceSize          size,
                                    VkBufferUsageFlags    usage,
                                    VkMemoryPropertyFlags properties,
                                    VkBuffer &            buffer,
                                    VkDeviceMemory &      bufferMemory)
{
    auto bufferInfo = VkBufferCreateInfo{.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                         .size        = size,
                                         .usage       = usage,
                                         .sharingMode = VK_SHARING_MODE_EXCLUSIVE};

    auto result = vkCreateBuffer(logical_device, &bufferInfo, nullptr, &buffer);
    if (result != VK_SUCCESS)
    {
        return result;
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(logical_device, buffer, &memRequirements);

    auto allocInfo = VkMemoryAllocateInfo{
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = memRequirements.size,
        .memoryTypeIndex = findMemoryType(
            memRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};

    result = vkAllocateMemory(logical_device, &allocInfo, nullptr, &bufferMemory);
    if (result != VK_SUCCESS)
    {
        return result;
    }

    return vkBindBufferMemory(logical_device, buffer, bufferMemory, 0);
}

void RenderDevice::copyBuffer(VkBuffer     srcBuffer,
                              VkDeviceSize srcOffset,
                              VkBuffer     dstBuffer,
                              VkDeviceSize dstOffset,
                              VkDeviceSize size)
{
    auto & bucket = transfer_buckets[currentResource];

    Copy * vertex_command         = bucket.AddCommand<Copy>(0, 0);
    vertex_command->commandbuffer = transfer_commandbuffers[currentResource];
    vertex_command->srcBuffer     = srcBuffer;
    vertex_command->dstBuffer     = dstBuffer;
    vertex_command->srcOffset     = srcOffset;
    vertex_command->dstOffset     = dstOffset;
    vertex_command->size          = size;
};

// COMMANDBUFFERS
VkResult RenderDevice::createCommandbuffers()
{
    commandbuffers.resize(MAX_BUFFERED_RESOURCES);

    auto allocInfo = VkCommandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = command_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = (uint32_t)commandbuffers.size()};

    auto result = vkAllocateCommandBuffers(logical_device, &allocInfo, commandbuffers.data());
    if (result != VK_SUCCESS)
    {
        return result;
    }

    transfer_commandbuffers.resize(MAX_BUFFERED_RESOURCES);

    allocInfo = VkCommandBufferAllocateInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                            .commandPool        = command_pool,
                                            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                            .commandBufferCount = (uint32_t)commandbuffers.size()};

    return vkAllocateCommandBuffers(logical_device, &allocInfo, transfer_commandbuffers.data());
}

// COMMANDBUFFER
VkResult RenderDevice::createCommandbuffer(uint32_t        image_index,
                                           uint32_t        uniform_count,
                                           UniformHandle * p_uniforms)
{
    auto & mapped_vertices = dynamic_mapped_vertices[currentResource];
    auto & mapped_indices  = dynamic_mapped_indices[currentResource];

    auto beginInfo = VkCommandBufferBeginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                              .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                              .pInheritanceInfo = nullptr};

    auto result = vkBeginCommandBuffer(commandbuffers[currentResource], &beginInfo);
    if (result != VK_SUCCESS)
    {
        return result;
    }

    // memory barrier for copy commands
    auto barrier = VkMemoryBarrier{.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                   .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                                   .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT};

    vkCmdPipelineBarrier(commandbuffers[currentResource],
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                         0,
                         1,
                         &barrier,
                         0,
                         nullptr,
                         0,
                         nullptr);

    auto clearValues = std::array<VkClearValue, 2>{VkClearValue{.color = {0.0f, 0.0f, 0.0f, 1.0f}},
                                                   VkClearValue{.depthStencil = {1.0f, 0}}};

    auto renderPassInfo = VkRenderPassBeginInfo{
        .sType      = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = renderpasses[0],
        .framebuffer
        = framebuffers[0][image_index], // framebuffers[resource_index][0].vk_framebuffer,
        .renderArea.offset = {0, 0},
        .renderArea.extent = swapchain_extent,
        .clearValueCount   = static_cast<uint32_t>(clearValues.size()),
        .pClearValues      = clearValues.data()};

    vkCmdBeginRenderPass(
        commandbuffers[currentResource], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(
        commandbuffers[currentResource], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[0].vk_pipeline);

    std::vector<VkDescriptorSet> descriptorsets;
    std::vector<uint32_t>        dynamic_offsets;

    for (size_t i = 0; i < uniform_count; ++i)
    {
        auto uniform_handle = p_uniforms[i];
        auto opt_uniform    = uniform_layouts[uniform_handle.uniform_layout_id].getUniform(
            uniform_handle);
        auto const & uniform = opt_uniform.value();

        descriptorsets.push_back(uniform.vk_descriptorset);
        dynamic_offsets.push_back(uniform.uniform_buffers[currentResource].offset);
    }
    descriptorsets.push_back(sampler_descriptor);

    vkCmdBindDescriptorSets(commandbuffers[currentResource],
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelines[0].vk_pipeline_layout,
                            0,
                            static_cast<uint32_t>(descriptorsets.size()),
                            descriptorsets.data(),
                            static_cast<uint32_t>(dynamic_offsets.size()),
                            dynamic_offsets.data());

    buckets[currentResource].Submit();

    vkCmdEndRenderPass(commandbuffers[currentResource]);

    result = vkEndCommandBuffer(commandbuffers[currentResource]);
    if (result != VK_SUCCESS)
    {
        return result;
    }

    return VK_SUCCESS;
}

// SYNC OBJECTS
VkResult RenderDevice::createSyncObjects()
{
    image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
    render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
    in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);

    auto semaphoreInfo = VkSemaphoreCreateInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

    auto fenceInfo = VkFenceCreateInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                       .flags = VK_FENCE_CREATE_SIGNALED_BIT};

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (vkCreateSemaphore(
                logical_device, &semaphoreInfo, nullptr, &image_available_semaphores[i])
                != VK_SUCCESS
            || vkCreateSemaphore(
                   logical_device, &semaphoreInfo, nullptr, &render_finished_semaphores[i])
                   != VK_SUCCESS
            || vkCreateFence(logical_device, &fenceInfo, nullptr, &in_flight_fences[i])
                   != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }

    return VK_SUCCESS;
}

VkResult RenderDevice::createDynamicObjectResources(size_t dynamic_vertices_count,
                                                    size_t dynamic_indices_count)
{
    dynamic_mapped_vertices.resize(MAX_BUFFERED_RESOURCES);
    dynamic_mapped_indices.resize(MAX_BUFFERED_RESOURCES);

    for (uint32_t i = 0; i < MAX_BUFFERED_RESOURCES; ++i)
    {
        dynamic_mapped_vertices[i].memory_size = sizeof(Vertex) * dynamic_vertices_count;
        dynamic_mapped_indices[i].memory_size  = sizeof(uint32_t) * dynamic_indices_count;

        createBuffer(dynamic_mapped_vertices[i].memory_size,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     dynamic_mapped_vertices[i].buffer,
                     dynamic_mapped_vertices[i].memory);

        vkMapMemory(logical_device,
                    dynamic_mapped_vertices[i].memory,
                    0,
                    dynamic_mapped_vertices[i].memory_size,
                    0,
                    &dynamic_mapped_vertices[i].data);
        // vkUnmapMemory(logical_device, stagingBufferMemory);

        createBuffer(dynamic_mapped_indices[i].memory_size,
                     VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     dynamic_mapped_indices[i].buffer,
                     dynamic_mapped_indices[i].memory);

        vkMapMemory(logical_device,
                    dynamic_mapped_indices[i].memory,
                    0,
                    dynamic_mapped_indices[i].memory_size,
                    0,
                    &dynamic_mapped_indices[i].data);
        // vkUnmapMemory(logical_device, stagingBufferMemory);
    }

    return VK_SUCCESS;
}

VkResult RenderDevice::createStagingObjectResources(size_t staging_buffer_size)
{
    staging_buffer.resize(MAX_BUFFERED_RESOURCES);

    for (uint32_t i = 0; i < MAX_BUFFERED_RESOURCES; ++i)
    {
        staging_buffer[i].memory_size = staging_buffer_size;

        createBuffer(staging_buffer[i].memory_size,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging_buffer[i].buffer,
                     staging_buffer[i].memory);

        vkMapMemory(logical_device,
                    staging_buffer[i].memory,
                    0,
                    staging_buffer[i].memory_size,
                    0,
                    &staging_buffer[i].data);
    }

    return VK_SUCCESS;
}

}; // namespace gfx

#endif