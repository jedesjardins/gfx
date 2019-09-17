#ifndef JED_GFX_RENDER_DEVICE_HPP
#define JED_GFX_RENDER_DEVICE_HPP

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <array>
#include <set>
#include <fstream>
#include <variant>
#include <optional>
#include <unordered_map>

#include "cmd/cmd.hpp"

#include "rapidjson/document.h"

#include "stb_image.h"

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

std::vector<char> readFile(std::string const & filename);

namespace gfx
{
enum class ErrorCode
{
    NONE,
    VULKAN_ERROR,
    API_ERROR,
    JSON_ERROR
};

char const * error_string(VkResult error_code);

//
//  HANDLES
//

using CommandbufferHandle   = size_t;
using RenderpassHandle      = size_t;
using SubpassHandle         = size_t;
using AttachmentHandle      = size_t;
using FramebufferHandle     = size_t;
using UniformLayoutHandle   = size_t;
using PushConstantHandle    = size_t;
using VertexBindingHandle   = size_t;
using VertexAttributeHandle = size_t;
using ShaderHandle          = size_t;
using PipelineHandle        = size_t;

// this should be replaced with a struct with a generation field
using BufferHandle  = size_t;
using TextureHandle = size_t;

struct UniformHandle
{
    uint64_t uniform_layout_id : 32;
    uint64_t uniform_id : 32;
};

class Memory
{
public:
    template <typename T>
    ErrorCode allocateAndBind(VkPhysicalDevice      physical_device,
                              VkDevice              logical_device,
                              VkMemoryPropertyFlags properties,
                              T                     object_handle)
    {
        VkMemoryRequirements requirements;
        getMemoryRequirements(logical_device, object_handle, requirements);

        auto opt_memory_type = findMemoryType(
            physical_device, requirements.memoryTypeBits, properties);

        if (!opt_memory_type)
        {
            return ErrorCode::VULKAN_ERROR;
        }

        auto allocInfo = VkMemoryAllocateInfo{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                              .allocationSize  = requirements.size,
                                              .memoryTypeIndex = opt_memory_type.value()};

        VK_CHECK_RESULT(vkAllocateMemory(logical_device, &allocInfo, nullptr, &vk_memory),
                        "Unable to allocate VkDeviceMemory");

        return bindMemory(logical_device, object_handle);
    }

    ErrorCode map(VkDevice logical_device, VkDeviceSize offset, VkDeviceSize size, void ** data);

    void destroy(VkDevice logical_device);

    VkDeviceMemory memory_handle();

protected:
    VkDeviceMemory vk_memory{VK_NULL_HANDLE};

private:
    std::optional<uint32_t> findMemoryType(VkPhysicalDevice      physical_device,
                                           uint32_t              typeFilter,
                                           VkMemoryPropertyFlags properties);

    void getMemoryRequirements(VkDevice               logical_device,
                               VkBuffer               buffer,
                               VkMemoryRequirements & requirements);

    void getMemoryRequirements(VkDevice               logical_device,
                               VkImage                image,
                               VkMemoryRequirements & requirements);

    ErrorCode bindMemory(VkDevice logical_device, VkBuffer buffer);

    ErrorCode bindMemory(VkDevice logical_device, VkImage image);
};

class Image: public Memory
{
public:
    ErrorCode create(VkPhysicalDevice      physical_device,
                     VkDevice              logical_device,
                     uint32_t              width,
                     uint32_t              height,
                     uint32_t              mipLevels,
                     VkSampleCountFlagBits numSamples,
                     VkFormat              format,
                     VkImageTiling         tiling,
                     VkImageUsageFlags     usage,
                     VkMemoryPropertyFlags properties,
                     VkImageAspectFlags    aspectFlags);

    void destroy(VkDevice logical_device);

    VkImageView view_handle();

    VkImage image_handle();

protected:
    VkImage     vk_image{VK_NULL_HANDLE};
    VkImageView vk_image_view{VK_NULL_HANDLE};
};

class Sampler: public Image
{
public:
    ErrorCode create(VkPhysicalDevice      physical_device,
                     VkDevice              logical_device,
                     uint32_t              width,
                     uint32_t              height,
                     uint32_t              mipLevels,
                     VkSampleCountFlagBits numSamples,
                     VkFormat              format,
                     VkImageTiling         tiling,
                     VkImageUsageFlags     usage,
                     VkMemoryPropertyFlags properties,
                     VkImageAspectFlags    aspectFlags);

    void destroy(VkDevice logical_device);

    VkSampler sampler_handle();

private:
    VkSampler vk_sampler{VK_NULL_HANDLE};
};

class Buffer: public Memory
{
public:
    ErrorCode create(VkPhysicalDevice      physical_device,
                     VkDevice              logical_device,
                     VkDeviceSize          size,
                     VkBufferUsageFlags    usage,
                     VkMemoryPropertyFlags properties);

    void destroy(VkDevice logical_device);

    VkBuffer buffer_handle();

protected:
    VkBuffer vk_buffer{VK_NULL_HANDLE};
};

class MappedBuffer
{
public:
    MappedBuffer(VkDevice logical_device, Buffer buffer, VkDeviceSize size);

    size_t copy(size_t size, void const * src_data);

    void reset();

    VkBuffer buffer_handle();

    VkDeviceSize offset{0};

private:
    VkDeviceSize memory_size{0};
    VkBuffer     vk_buffer{VK_NULL_HANDLE};
    void *       data{nullptr};
};

struct IndexAllocator
{
    int32_t              next_index;
    int32_t              last_index;
    std::vector<int32_t> indices;

    void init(size_t number_of_indices);

    int32_t acquire();

    void release(int32_t released_index);
};

struct DynamicBufferUniform
{
    size_t       descriptor_set;
    VkDeviceSize offset;
};

struct SamplerCollection
{
    std::vector<VkDescriptorSet> descriptor_sets;
    IndexAllocator               free_uniform_slots;

    std::optional<UniformHandle>   createUniform(VkDevice &  logical_device,
                                                 VkImageView view,
                                                 VkSampler   sampler);
    std::optional<VkDescriptorSet> getUniform(UniformHandle handle);
    std::optional<VkDeviceSize>    getDynamicOffset(UniformHandle handle);
    void                           destroyUniform(UniformHandle handle);
    void                           destroy(VkDevice & logical_device);
};

struct DynamicBufferCollection
{
    // following map one to one
    std::vector<VkDescriptorSet>      descriptor_sets;
    std::vector<MappedBuffer>         uniform_buffers;
    std::vector<DynamicBufferUniform> uniforms;
    IndexAllocator                    free_uniform_buffer_slots;
    IndexAllocator                    free_uniform_slots;

    std::optional<UniformHandle> createUniform(VkDeviceSize size, void * data_ptr);

    void updateUniform(UniformHandle handle, VkDeviceSize size, void * data_ptr);

    std::optional<VkDescriptorSet> getUniform(UniformHandle handle);

    std::optional<VkDeviceSize> getDynamicOffset(UniformHandle handle);

    void destroyUniform(UniformHandle handle);

    void destroy(VkDevice & logical_device);
};

using UniformVariant = std::variant<DynamicBufferCollection, SamplerCollection>;

//
//  CONFIGURATION STRUCTURES
//

enum class Format
{
    USE_DEPTH,
    USE_COLOR
};

struct AttachmentConfig
{
    Format            format;
    VkImageUsageFlags usage;
    bool              multisampled;
    bool              is_swapchain_image;

    void init(rapidjson::Value & document);

    friend bool operator==(AttachmentConfig const & lhs, AttachmentConfig const & rhs);
    friend bool operator!=(AttachmentConfig const & lhs, AttachmentConfig const & rhs);
};

struct SubpassInfo
{
    std::vector<VkAttachmentReference> color_attachments;

    VkAttachmentReference color_resolve_attachment;
    VkAttachmentReference depth_stencil_attachment;

    void init(rapidjson::Value &                        document,
              std::unordered_map<std::string, size_t> & attachment_indices);

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

    void init(rapidjson::Value & document);

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

    size_t max_draw_calls;

    void init(rapidjson::Value & document, std::unordered_map<std::string, std::string> const &);
};

struct Pipeline
{
    VkPipeline       vk_pipeline;
    VkPipelineLayout vk_pipeline_layout;
};

struct UniformConfig
{
    size_t                       max_uniform_count;
    VkDescriptorSetLayoutBinding layout_binding;

    void init(rapidjson::Value & document);
};

struct RenderConfig
{
    char const * config_filename;

    char const * window_name;

    size_t dynamic_vertex_buffer_size;

    size_t dynamic_index_buffer_size;

    size_t staging_buffer_size;

    size_t max_transfer_count;

    size_t max_delete_count;

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

//
//  COMMANDS
//

struct Draw
{
    static cmd::BackendDispatchFunction const DISPATCH_FUNCTION;

    VkCommandBuffer   commandbuffer;
    VkPipelineLayout  pipeline_layout;
    VkBuffer          vertexbuffer;
    VkDeviceSize      vertexbuffer_offset;
    VkBuffer          indexbuffer;
    VkDeviceSize      indexbuffer_offset;
    VkDeviceSize      indexbuffer_count;
    VkDeviceSize      push_constant_size;
    void *            push_constant_data;
    VkDeviceSize      descriptor_set_count;
    VkDescriptorSet * descriptor_sets;
    VkDeviceSize      dynamic_offset_count;
    uint32_t *        dynamic_offsets;
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

struct DeleteBuffers
{
    static cmd::BackendDispatchFunction const DISPATCH_FUNCTION;

    VkDevice         logical_device;
    size_t           buffer_count;
    VkBuffer *       buffers;
    VkDeviceMemory * memories;
};

static_assert(std::is_pod<DeleteBuffers>::value == true, "DeleteBuffers must be a POD.");

void deleteBuffers(void const * data);

struct DeleteTextures
{
    static cmd::BackendDispatchFunction const DISPATCH_FUNCTION;

    VkDevice         logical_device;
    size_t           texture_count;
    VkSampler *      samplers;
    VkImageView *    views;
    VkImage *        images;
    VkDeviceMemory * memories;
};

static_assert(std::is_pod<DeleteTextures>::value == true, "DeleteTextures must be a POD.");

void deleteTextures(void const * data);

struct DeleteUniforms
{
    static cmd::BackendDispatchFunction const DISPATCH_FUNCTION;

    std::vector<UniformVariant> * uniform_collections;
    size_t                        uniform_count;
    UniformHandle *               uniform_handles;
};

static_assert(std::is_pod<DeleteUniforms>::value == true, "DeleteUniforms must be a POD.");

void deleteUniforms(void const * data);

//
// PHYSICAL DEVICE
//

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

//
//  NEW CLASSES
//

namespace module
{
/*
 * This class will hold device specific variables.
 *
 * Instance, Device, Logical Device, Queues, etc.
 */
struct Device
{
public:
    GLFWwindow * window{nullptr};

    VkInstance instance{VK_NULL_HANDLE};

    VkDebugUtilsMessengerEXT debug_messager{VK_NULL_HANDLE};

    VkSurfaceKHR surface{VK_NULL_HANDLE};

    VkPhysicalDevice   physical_device{VK_NULL_HANDLE};
    PhysicalDeviceInfo physical_device_info;

    VkDevice logical_device{VK_NULL_HANDLE};

    VkSwapchainKHR           swapchain{VK_NULL_HANDLE};
    uint32_t                 swapchain_image_count{0};
    std::vector<VkImage>     swapchain_images;
    std::vector<VkImageView> swapchain_image_views;

    VkFormat   swapchain_image_format;
    VkFormat   depth_format;
    VkExtent2D swapchain_extent;

    bool                            use_validation{true};
    bool                            validation_supported{false};
    const std::vector<char const *> validation_layers{"VK_LAYER_KHRONOS_validation"};

    std::vector<char const *>       required_extensions{};
    const std::vector<const char *> required_device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    explicit Device(GLFWwindow * window_ptr);

    bool init(RenderConfig & render_config);
    void quit();

private:
    void checkValidationLayerSupport();

    void getRequiredExtensions();

    // INSTANCE
    ErrorCode createInstance(char const * window_name);

    // VALIDATION LAYER DEBUG MESSAGER
    static VKAPI_ATTR VkBool32 VKAPI_CALL
                               debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
                                             VkDebugUtilsMessageTypeFlagsEXT              messageType,
                                             const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData,
                                             void *                                       pUserData);

    ErrorCode createDebugMessenger();

    ErrorCode createDebugUtilsMessengerEXT(const VkDebugUtilsMessengerCreateInfoEXT * pCreateInfo,
                                           const VkAllocationCallbacks *              pAllocator,
                                           VkDebugUtilsMessengerEXT * pDebugMessenger);

    void cleanupDebugUtilsMessengerEXT(VkDebugUtilsMessengerEXT      debugMessenger,
                                       const VkAllocationCallbacks * pAllocator);

    // SURFACE
    ErrorCode createSurface();

    bool pickPhysicalDevice();

    bool isDeviceSuitable(VkPhysicalDevice device, PhysicalDeviceInfo & device_info);

    void findQueueFamilies(VkPhysicalDevice device, PhysicalDeviceInfo & device_info);

    bool checkDeviceExtensionSupport(VkPhysicalDevice device, PhysicalDeviceInfo & device_info);

    void querySwapChainSupport(VkPhysicalDevice device, PhysicalDeviceInfo & device_info);

    void getMaxUsableSampleCount();

    // LOGICAL DEVICE
    ErrorCode createLogicalDevice();

    // SWAPCHAIN
    ErrorCode createSwapChain();

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(
        std::vector<VkSurfaceFormatKHR> const & availableFormats);

    VkPresentModeKHR chooseSwapPresentMode(
        const std::vector<VkPresentModeKHR> & availablePresentModes);

    VkExtent2D chooseSwapExtent(VkSurfaceCapabilitiesKHR const & capabilities);

    void getSwapChainImages();

    ErrorCode createSwapChainImageViews();

    std::optional<VkFormat> findDepthFormat();

    std::optional<VkFormat> findSupportedFormat(const std::vector<VkFormat> & candidates,
                                                VkImageTiling                 tiling,
                                                VkFormatFeatureFlags          features);
}; // struct Device

/*
 * Holds semaphores, fences, current frame/resource indices etc.
 */
struct FrameResources
{
public:
    std::vector<VkSemaphore> image_available_semaphores;
    std::vector<VkSemaphore> render_finished_semaphores;
    std::vector<VkFence>     in_flight_fences;

    int32_t const MAX_FRAMES_IN_FLIGHT{2};
    int32_t const MAX_BUFFERED_RESOURCES{MAX_FRAMES_IN_FLIGHT + 1};
    uint32_t      currentImage{0};
    size_t        currentFrame{0};
    uint32_t      currentResource{0};

    bool init(RenderConfig & render_config, VkDevice device);
    void quit(VkDevice device);
}; // struct FrameResources

struct ImageResources
{
public:
    bool init(RenderConfig & render_config, Device & device);
    void quit(Device & device);

    std::optional<TextureHandle> create_texture(VkPhysicalDevice      physical_device,
                                                VkDevice              logical_device,
                                                uint32_t              width,
                                                uint32_t              height,
                                                uint32_t              mipLevels,
                                                VkSampleCountFlagBits numSamples,
                                                VkFormat              format,
                                                VkImageTiling         tiling,
                                                VkImageUsageFlags     usage,
                                                VkMemoryPropertyFlags properties,
                                                VkImageAspectFlags    aspectFlags);

    std::optional<Sampler> get_texture(TextureHandle handle);

    void delete_texture(TextureHandle handle);

    std::optional<AttachmentConfig> get_config(AttachmentHandle);

    std::optional<TextureHandle> get_texture_handle(AttachmentHandle);

    std::optional<AttachmentHandle> get_attachment_handle(std::string const & name);

private:
    std::unordered_map<std::string, AttachmentHandle> attachment_handles;
    std::vector<AttachmentConfig>                     attachment_configs;
    std::vector<TextureHandle>                        attachments;
    TextureHandle                                     next_sampler_handle{0};
    std::unordered_map<TextureHandle, Sampler>        samplers;

    ErrorCode createAttachments(Device & device);
}; // struct ImageResources

struct RenderPassResources
{
public:
    std::vector<RenderpassHandle> renderpass_order;

    std::unordered_map<RenderpassHandle, std::vector<std::vector<PipelineHandle>>>
        per_renderpass_subpass_pipelines;

    std::unordered_map<std::string, RenderpassHandle> render_pass_handles;
    std::vector<RenderpassConfig>                     render_pass_configs;
    std::vector<VkRenderPass>                         render_passes;
    std::vector<std::vector<VkFramebuffer>>           framebuffers;
    std::vector<std::vector<VkClearValue>>            clear_values;

    bool init(RenderConfig & render_config, Device & device, ImageResources & image_resources);
    void quit(Device & device);

private:
    ErrorCode createRenderPasses(Device & device, ImageResources & image_resources);

    // FRAMEBUFFER
    ErrorCode createFramebuffer(Device &                     device,
                                ImageResources &             image_resources,
                                RenderpassConfig const &     config,
                                VkRenderPass const &         render_pass,
                                std::vector<VkFramebuffer> & framebuffers);
}; // struct RenderPassResources

struct BufferResources
{
public:
    std::vector<MappedBuffer> dynamic_mapped_vertices;
    std::vector<MappedBuffer> dynamic_mapped_indices;

    std::vector<MappedBuffer> staging_buffer;

    bool init(RenderConfig & render_config, Device & device, FrameResources & frames);
    void quit(Device & device);

    std::optional<BufferHandle> create_buffer(Device &              render_device,
                                              VkDeviceSize          size,
                                              VkBufferUsageFlags    usage,
                                              VkMemoryPropertyFlags properties);

    std::optional<Buffer> get_buffer(BufferHandle handle);

    void delete_buffer(Device & render_device, BufferHandle handle);

private:
    BufferHandle next_buffer_handle{0};
    // todo: this isn't thread safe
    std::unordered_map<BufferHandle, Buffer> buffers;

    ErrorCode createDynamicObjectResources(Device &         device,
                                           FrameResources & frames,
                                           size_t           dynamic_vertices_count,
                                           size_t           dynamic_indices_count);

    ErrorCode createStagingObjectResources(Device &         device,
                                           FrameResources & frames,
                                           size_t           staging_buffer_size);
}; // struct BufferResources

/*
 * Manages Uniforms
 */
struct UniformResources
{
public:
    std::unordered_map<std::string, UniformLayoutHandle> uniform_layout_handles;

    std::vector<VkDescriptorSetLayoutBinding> uniform_layout_infos;
    std::vector<VkDescriptorSetLayout>        uniform_layouts;
    std::vector<VkDescriptorPool>             pools;
    std::vector<size_t>                       uniform_counts;
    std::vector<UniformVariant>               uniform_collections;

    bool init(RenderConfig & render_config, Device & device, BufferResources & buffers);
    void quit(Device & device);

private:
    ErrorCode createUniformLayouts(Device & device, BufferResources & buffers);
}; // struct UniformResources

struct PipelineResources
{
public:
    std::unordered_map<std::string, ShaderHandle> shader_handles;
    std::vector<std::string>                      shader_files;
    std::vector<VkShaderModule>                   shaders;

    std::unordered_map<std::string, PushConstantHandle> push_constant_handles;
    std::vector<VkPushConstantRange>                    push_constants;

    std::unordered_map<std::string, VertexBindingHandle> vertex_binding_handles;
    std::vector<VkVertexInputBindingDescription>         vertex_bindings;

    std::unordered_map<std::string, VertexAttributeHandle> vertex_attribute_handles;
    std::vector<VkVertexInputAttributeDescription>         vertex_attributes;

    std::unordered_map<std::string, PipelineHandle> pipeline_handles;
    std::vector<PipelineConfig>                     pipeline_configs;
    std::vector<Pipeline>                           pipelines;
    std::vector<cmd::CommandBucket<int>>            draw_buckets;

    bool init(RenderConfig &        render_config,
              Device &              device,
              RenderPassResources & render_passes,
              UniformResources &    uniforms);
    void quit(Device & device);

private:
    ErrorCode createShaderModule(Device &                  device,
                                 std::vector<char> const & code,
                                 VkShaderModule &          shaderModule);

    ErrorCode createShaders(Device & device);

    ErrorCode createGraphicsPipeline(Device &              device,
                                     RenderPassResources & render_passes,
                                     UniformResources &    uniforms);
}; // struct PipelineResources

/*
 * Holds all commands/command buckets
 */
struct CommandResources
{
public:
    VkQueue graphics_queue{VK_NULL_HANDLE};
    VkQueue transfer_queue{VK_NULL_HANDLE};
    VkQueue present_queue{VK_NULL_HANDLE};

    std::vector<VkCommandBuffer> draw_commandbuffers;
    std::vector<VkCommandBuffer> transfer_commandbuffers;

    std::vector<cmd::CommandBucket<int>> transfer_buckets;
    std::vector<cmd::CommandBucket<int>> delete_buckets;

    // pools belong to queue types, need one per queue (for when we use the transfer and graphics
    // queue) eventually have one pool per thread per
    VkCommandPool command_pool{VK_NULL_HANDLE};

    bool init(RenderConfig & render_config, Device & device);
    void quit(Device & device);

private:
    void getQueues(Device & device);

    ErrorCode createCommandPool(Device & device);

    ErrorCode createCommandbuffers(Device & device);
}; // struct CommandResources

}; // namespace module

/*
 *
 * This is basically the RenderDevice, but will break all functionality out into smaller classes
 *
 * The public API will just wrap these internal classes
 *
 */
class Renderer
{
public:
    explicit Renderer(GLFWwindow * window_ptr);

    bool init(RenderConfig & render_config);

    void quit();

    void wait_for_idle();

    bool submit_frame();

    ErrorCode draw(PipelineHandle  pipeline,
                   size_t          vertices_size,
                   void *          vertices,
                   uint32_t        index_count,
                   uint32_t *      indices,
                   size_t          push_constant_size,
                   void *          push_constant_data,
                   size_t          uniform_count,
                   UniformHandle * p_uniforms);

    ErrorCode draw(PipelineHandle  pipeline,
                   BufferHandle    vertexbuffer_handle,
                   VkDeviceSize    vertexbuffer_offset,
                   BufferHandle    indexbuffer_handle,
                   VkDeviceSize    indexbuffer_offset,
                   VkDeviceSize    indexbuffer_count,
                   size_t          push_constant_size,
                   void *          push_constant_data,
                   size_t          uniform_count,
                   UniformHandle * p_uniforms);

    std::optional<UniformLayoutHandle> get_uniform_layout_handle(std::string layout_name);
    std::optional<PipelineHandle>      get_pipeline_handle(std::string pipeline_name);

    /*
    // TODO: figure out how to do this better
    template <typename... Args>
    std::optional<UniformHandle> newUniform(UniformLayoutHandle layout_handle, Args &&... args)
    {
        auto & uniform_collection = uniforms.uniform_collections[layout_handle];

        auto opt_uniform_handle = std::visit(
            [&](auto && collection) -> std::optional<UniformHandle> {
                using T = std::decay_t<decltype(collection)>;
                if constexpr (std::is_same_v<T, DynamicBufferCollection>)
                    return collection.createUniform(std::forward<Args>(args)...);
                else
                    return std::nullopt;
            },
            uniform_collection);

        if (opt_uniform_handle)
        {
            opt_uniform_handle.value().uniform_layout_id = layout_handle;
        }

        return opt_uniform_handle;
    }
    */

    std::optional<UniformHandle> new_uniform(UniformLayoutHandle layout_handle,
                                             VkDeviceSize        size,
                                             void *              data_ptr);

    std::optional<UniformHandle> new_uniform(UniformLayoutHandle layout_handle,
                                             TextureHandle       texture_handle);

    template <typename... Args>
    void update_uniform(UniformHandle handle, Args &&... args)
    {
        auto & uniform_collection = uniforms.uniform_collections[handle.uniform_layout_id];

        std::visit(
            [&](auto && collection) {
                using T = std::decay_t<decltype(collection)>;
                if constexpr (std::is_same_v<T, DynamicBufferCollection>)
                {
                    collection.updateUniform(handle, std::forward<Args>(args)...);
                }
            },
            uniform_collection);
    }

    void delete_uniforms(size_t uniform_count, UniformHandle * uniforms);

    std::optional<BufferHandle> create_buffer(VkDeviceSize          size,
                                              VkBufferUsageFlags    usage,
                                              VkMemoryPropertyFlags properties);

    void update_buffer(BufferHandle buffer, VkDeviceSize size, void * data);

    void delete_buffers(size_t buffer_count, BufferHandle * buffers);

    std::optional<TextureHandle> create_texture(char const * texture_path);

    void delete_textures(size_t sampler_count, TextureHandle * sampler_handles);

private:
    ErrorCode make_draw_command(cmd::CommandBucket<int> & bucket,
                                VkPipelineLayout          layout,
                                VkBuffer                  vertex_buffer,
                                VkDeviceSize              vertex_buffer_offset,
                                VkBuffer                  index_buffer,
                                VkDeviceSize              Index_buffer_offset,
                                VkDeviceSize              index_buffer_count,
                                size_t                    push_constant_size,
                                void *                    push_constant_data,
                                size_t                    uniform_count,
                                UniformHandle *           p_uniforms);

    std::optional<VkDescriptorSet> getUniform(UniformHandle handle);

    std::optional<VkDeviceSize> getDynamicOffset(UniformHandle handle);

    ErrorCode createCommandbuffer(uint32_t image_index);

    void copyBuffer(VkBuffer     srcBuffer,
                    VkDeviceSize srcOffset,
                    VkBuffer     dstBuffer,
                    VkDeviceSize dstOffset,
                    VkDeviceSize size);

    module::Device              render_device;
    module::FrameResources      frames;
    module::ImageResources      images;
    module::RenderPassResources render_passes;
    module::UniformResources    uniforms;
    module::PipelineResources   pipelines;
    module::CommandResources    commands;
    module::BufferResources     buffers;

}; // class Renderer
}; // namespace gfx

#endif

#ifdef JED_GFX_IMPLEMENTATION

std::vector<char> readFile(std::string const & filename)
{
    LOG_DEBUG("Reading file {}", filename);
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        LOG_ERROR("Failed to open file {}", filename);
        return {};
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
char const * error_string(VkResult error_code)
{
    switch (error_code)
    {
#define STR(r)   \
    case VK_##r: \
        return #r
        STR(NOT_READY);
        STR(TIMEOUT);
        STR(EVENT_SET);
        STR(EVENT_RESET);
        STR(INCOMPLETE);
        STR(ERROR_OUT_OF_HOST_MEMORY);
        STR(ERROR_OUT_OF_DEVICE_MEMORY);
        STR(ERROR_INITIALIZATION_FAILED);
        STR(ERROR_DEVICE_LOST);
        STR(ERROR_MEMORY_MAP_FAILED);
        STR(ERROR_LAYER_NOT_PRESENT);
        STR(ERROR_EXTENSION_NOT_PRESENT);
        STR(ERROR_FEATURE_NOT_PRESENT);
        STR(ERROR_INCOMPATIBLE_DRIVER);
        STR(ERROR_TOO_MANY_OBJECTS);
        STR(ERROR_FORMAT_NOT_SUPPORTED);
        STR(ERROR_SURFACE_LOST_KHR);
        STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
        STR(SUBOPTIMAL_KHR);
        STR(ERROR_OUT_OF_DATE_KHR);
        STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
        STR(ERROR_VALIDATION_FAILED_EXT);
        STR(ERROR_INVALID_SHADER_NV);
#undef STR
    default:
        return "UNKNOWN_ERROR";
    }
}

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

ErrorCode Memory::map(VkDevice logical_device, VkDeviceSize offset, VkDeviceSize size, void ** data)
{
    VK_CHECK_RESULT(vkMapMemory(logical_device, vk_memory, offset, size, 0, data),
                    "Unable to map VkDeviceMemory");

    return ErrorCode::NONE;
}

void Memory::destroy(VkDevice logical_device)
{
    vkFreeMemory(logical_device, vk_memory, nullptr);
    vk_memory = VK_NULL_HANDLE;
}

VkDeviceMemory Memory::memory_handle()
{
    return vk_memory;
}

std::optional<uint32_t> Memory::findMemoryType(VkPhysicalDevice      physical_device,
                                               uint32_t              typeFilter,
                                               VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
    {
        if (typeFilter & (1 << i)
            && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    LOG_WARN("Couldn't find VkDeviceMemory satisfying VkMemoryPropertyFlags");
    return std::nullopt;
}

void Memory::getMemoryRequirements(VkDevice               logical_device,
                                   VkBuffer               buffer,
                                   VkMemoryRequirements & requirements)
{
    vkGetBufferMemoryRequirements(logical_device, buffer, &requirements);
}

void Memory::getMemoryRequirements(VkDevice               logical_device,
                                   VkImage                image,
                                   VkMemoryRequirements & requirements)
{
    vkGetImageMemoryRequirements(logical_device, image, &requirements);
}

ErrorCode Memory::bindMemory(VkDevice logical_device, VkBuffer buffer)
{
    VK_CHECK_RESULT(vkBindBufferMemory(logical_device, buffer, vk_memory, 0),
                    "Unable to bind VkDeviceMemory to VkBuffer");

    return ErrorCode::NONE;
}

ErrorCode Memory::bindMemory(VkDevice logical_device, VkImage image)
{
    VK_CHECK_RESULT(vkBindImageMemory(logical_device, image, vk_memory, 0),
                    "Unable to bind VkDeviceMemory to VkImage");

    return ErrorCode::NONE;
}

ErrorCode Image::create(VkPhysicalDevice      physical_device,
                        VkDevice              logical_device,
                        uint32_t              width,
                        uint32_t              height,
                        uint32_t              mipLevels,
                        VkSampleCountFlagBits numSamples,
                        VkFormat              format,
                        VkImageTiling         tiling,
                        VkImageUsageFlags     usage,
                        VkMemoryPropertyFlags properties,
                        VkImageAspectFlags    aspectFlags)
{
    LOG_DEBUG("Creating VkImage and VkImageView");
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

    VK_CHECK_RESULT(vkCreateImage(logical_device, &imageInfo, nullptr, &vk_image),
                    "Unable to create VkImage");

    ErrorCode error = allocateAndBind(physical_device, logical_device, properties, vk_image);
    if (error != ErrorCode::NONE)
    {
        LOG_ERROR("Couldn't bind device memory to VkImage {}", static_cast<void *>(vk_image));
        return error;
    }

    auto viewInfo = VkImageViewCreateInfo{.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                          .image    = vk_image,
                                          .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                          .format   = format,
                                          .subresourceRange.aspectMask     = aspectFlags,
                                          .subresourceRange.baseMipLevel   = 0,
                                          .subresourceRange.levelCount     = mipLevels,
                                          .subresourceRange.baseArrayLayer = 0,
                                          .subresourceRange.layerCount     = 1};

    VK_CHECK_RESULT(vkCreateImageView(logical_device, &viewInfo, nullptr, &vk_image_view),
                    "Unable to create VkImageView");

    return ErrorCode::NONE;
}

void Image::destroy(VkDevice logical_device)
{
    vkDestroyImageView(logical_device, vk_image_view, nullptr);
    vk_image_view = VK_NULL_HANDLE;
    vkDestroyImage(logical_device, vk_image, nullptr);
    vk_image = VK_NULL_HANDLE;
    static_cast<Memory &>(*this).destroy(logical_device);
}

VkImageView Image::view_handle()
{
    return vk_image_view;
}

VkImage Image::image_handle()
{
    return vk_image;
}

ErrorCode Sampler::create(VkPhysicalDevice      physical_device,
                          VkDevice              logical_device,
                          uint32_t              width,
                          uint32_t              height,
                          uint32_t              mipLevels,
                          VkSampleCountFlagBits numSamples,
                          VkFormat              format,
                          VkImageTiling         tiling,
                          VkImageUsageFlags     usage,
                          VkMemoryPropertyFlags properties,
                          VkImageAspectFlags    aspectFlags)
{
    LOG_DEBUG("Creating VkSampler");
    auto error = static_cast<Image &>(*this).create(physical_device,
                                                    logical_device,
                                                    width,
                                                    height,
                                                    mipLevels,
                                                    numSamples,
                                                    format,
                                                    tiling,
                                                    usage,
                                                    properties,
                                                    aspectFlags);
    if (error != ErrorCode::NONE)
    {
        return error;
    }

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

    VK_CHECK_RESULT(vkCreateSampler(logical_device, &samplerInfo, nullptr, &vk_sampler),
                    "Unable to create VkSampler");

    return ErrorCode::NONE;
}

void Sampler::destroy(VkDevice logical_device)
{
    vkDestroySampler(logical_device, vk_sampler, nullptr);
    vk_sampler = VK_NULL_HANDLE;
    static_cast<Image &>(*this).destroy(logical_device);
}

VkSampler Sampler::sampler_handle()
{
    return vk_sampler;
}

ErrorCode Buffer::create(VkPhysicalDevice      physical_device,
                         VkDevice              logical_device,
                         VkDeviceSize          size,
                         VkBufferUsageFlags    usage,
                         VkMemoryPropertyFlags properties)
{
    auto bufferInfo = VkBufferCreateInfo{.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                         .size        = size,
                                         .usage       = usage,
                                         .sharingMode = VK_SHARING_MODE_EXCLUSIVE};

    VK_CHECK_RESULT(vkCreateBuffer(logical_device, &bufferInfo, nullptr, &vk_buffer),
                    "Unable to create VkBuffer");

    return allocateAndBind(physical_device, logical_device, properties, vk_buffer);
}

void Buffer::destroy(VkDevice logical_device)
{
    vkDestroyBuffer(logical_device, vk_buffer, nullptr);
    vk_buffer = VK_NULL_HANDLE;
    static_cast<Memory &>(*this).destroy(logical_device);
}

VkBuffer Buffer::buffer_handle()
{
    return vk_buffer;
}

MappedBuffer::MappedBuffer(VkDevice logical_device, Buffer buffer, VkDeviceSize size)
: offset{0}, memory_size{size}, vk_buffer{buffer.buffer_handle()}
{
    if (buffer.map(logical_device, 0, size, &data) != ErrorCode::NONE)
    {
        memory_size = 0;
        vk_buffer   = VK_NULL_HANDLE;
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

VkBuffer MappedBuffer::buffer_handle()
{
    return vk_buffer;
}

void IndexAllocator::init(size_t number_of_indices)
{
    indices.resize(number_of_indices);
    for (size_t i = 0; i < indices.size() - 1; ++i)
    {
        indices[i] = i + 1;
    }

    next_index = 0;
    last_index = indices.size() - 1;
}

int32_t IndexAllocator::acquire()
{
    auto free_index = next_index;

    if (next_index == last_index)
    {
        next_index = -1;
        last_index = -1;
    }
    else
    {
        next_index = indices[next_index];
    }

    return free_index;
}

void IndexAllocator::release(int32_t released_index)
{
    if (last_index == -1)
    {
        next_index = released_index;
        last_index = released_index;
    }
    else
    {
        indices[last_index] = released_index;
        last_index          = released_index;
    }
}

/*
 * Acquires a open slot in the uniform buffer
 * Acquires an available DynamicBufferUniform element
 * Copies the data into the uniform buffer
 * Returns a UniformHandle to the DynamicBuffer Uniform (which holds the index of it's
 * descriptorset, and the offset into it's Uniform Buffer)
 */
std::optional<UniformHandle> DynamicBufferCollection::createUniform(VkDeviceSize size,
                                                                    void *       data_ptr)
{
    // this is the "block" in the Uniform Buffer
    auto uniform_buffer_slot = free_uniform_buffer_slots.acquire();
    if (uniform_buffer_slot < 0)
    {
        return std::nullopt;
    }

    size_t descriptor_index = uniform_buffer_slot / 8; // slots_per_buffer;
    auto & uniform_buffer   = uniform_buffers[descriptor_index];

    // this is the free index in the DynamicBuffer list (offset, descriptorset pair)
    auto uniform_slot = free_uniform_slots.acquire();
    if (uniform_slot < 0)
    {
        return std::nullopt;
    }

    DynamicBufferUniform & uniform = uniforms[uniform_slot];
    uniform.descriptor_set         = descriptor_index;
    uniform.offset                 = (uniform_buffer_slot % 8) * 256;
    uniform_buffer.offset          = uniform.offset;
    uniform_buffer.copy(size, data_ptr);

    return UniformHandle{.uniform_layout_id = 0, .uniform_id = static_cast<uint64_t>(uniform_slot)};
}

/*
 * Releases the uniform buffer slot
 * Acquires a new uniform buffer slot
 * UniformHandle (and DynamicBufferUniform slot) stays the same
 */
void DynamicBufferCollection::updateUniform(UniformHandle handle,
                                            VkDeviceSize  size,
                                            void *        data_ptr)
{
    auto & uniform = uniforms[handle.uniform_id];

    auto old_buffer_slot = (uniform.offset / 256) + (uniform.descriptor_set * 8);
    free_uniform_buffer_slots.release(old_buffer_slot);
    LOG_TRACE("Release {}", old_buffer_slot);

    // get new buffer slot
    auto uniform_buffer_slot = free_uniform_buffer_slots.acquire();
    LOG_TRACE("Acquire {}", uniform_buffer_slot);
    if (uniform_buffer_slot < 0)
    {
        return;
    }

    // get descriptor and uniform_buffer for that slot
    size_t descriptor_index = uniform_buffer_slot / 8; // slots_per_buffer;
    auto & uniform_buffer   = uniform_buffers[descriptor_index];

    uniform.descriptor_set = descriptor_index;
    uniform.offset         = (uniform_buffer_slot % 8) * 256;
    uniform_buffer.offset  = uniform.offset;
    uniform_buffer.copy(size, data_ptr);
}

/*
 * Release the DynamicBufferUniform slot
 * Release the Uniform Buffer "block"
 */
void DynamicBufferCollection::destroyUniform(UniformHandle handle)
{
    DynamicBufferUniform & uniform = uniforms[handle.uniform_id];
    free_uniform_slots.release(handle.uniform_id);

    auto uniform_buffer_slot = (uniform.offset / 256) + (uniform.descriptor_set * 8);
    free_uniform_buffer_slots.release(uniform_buffer_slot);
}

std::optional<VkDescriptorSet> DynamicBufferCollection::getUniform(UniformHandle handle)
{
    if (uniforms.size() > handle.uniform_id)
    {
        return descriptor_sets[uniforms[handle.uniform_id].descriptor_set];
    }
    else
    {
        return std::nullopt;
    }
}

std::optional<VkDeviceSize> DynamicBufferCollection::getDynamicOffset(UniformHandle handle)
{
    return uniforms[handle.uniform_id].offset;
}

void DynamicBufferCollection::destroy(VkDevice & logical_device)
{
    for (auto & mapped_buffer: uniform_buffers)
    {
        // mapped_buffer.destroy(logical_device);
    }
}

std::optional<UniformHandle> SamplerCollection::createUniform(VkDevice &  logical_device,
                                                              VkImageView view,
                                                              VkSampler   sampler)
{
    int64_t descriptor_set_index = free_uniform_slots.acquire();
    if (descriptor_set_index < 0)
    {
        LOG_ERROR("Couldn't acquire a VkDescriptorSet in SamplerCollection", descriptor_set_index);
        return std::nullopt;
    }
    LOG_DEBUG("Acquired descriptor_set_index {} in SamplerCollection", descriptor_set_index);

    auto imageInfo = VkDescriptorImageInfo{.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                           .imageView   = view,
                                           .sampler     = sampler};

    auto descriptorWrite = VkWriteDescriptorSet{
        .sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet           = descriptor_sets[0],
        .dstBinding       = 1,
        .dstArrayElement  = 0,
        .descriptorType   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount  = 1,
        .pBufferInfo      = nullptr,
        .pImageInfo       = &imageInfo,
        .pTexelBufferView = nullptr};

    vkUpdateDescriptorSets(logical_device, 1, &descriptorWrite, 0, nullptr);

    return UniformHandle{.uniform_id = static_cast<uint64_t>(descriptor_set_index)};
}

std::optional<VkDescriptorSet> SamplerCollection::getUniform(UniformHandle handle)
{
    return descriptor_sets[handle.uniform_id];
}

std::optional<VkDeviceSize> SamplerCollection::getDynamicOffset(UniformHandle handle)
{
    return std::nullopt;
}

void SamplerCollection::destroyUniform(UniformHandle)
{}

void SamplerCollection::destroy(VkDevice & logical_device)
{}

//
// CONFIGURATION CODE
//

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
    static std::unordered_map<std::string, VkFormat> formats{
        {"R32G32B32_SFLOAT", VK_FORMAT_R32G32B32_SFLOAT},
    };

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
    reference.attachment = attachment_indices[document["attachment_name"].GetString()];

    /*
    assert(document.HasMember("attachment_index"));
    assert(document["attachment_index"].IsInt());
    reference.attachment = document["attachment_index"].GetInt();
    */

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

    usage = static_cast<VkImageUsageFlags>(0);
    assert(document.HasMember("usage"));
    assert(document["usage"].IsArray());
    for (auto & usage_bit_name: document["usage"].GetArray())
    {
        assert(usage_bit_name.IsString());
        usage |= getVkImageUsageFlagBits(usage_bit_name.GetString());
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

void SubpassInfo::init(rapidjson::Value &                        document,
                       std::unordered_map<std::string, size_t> & attachment_indices)
{
    assert(document.IsObject());

    if (document.HasMember("color_attachments"))
    {
        assert(document["color_attachments"].IsArray());
        for (auto & ca: document["color_attachments"].GetArray())
        {
            color_attachments.push_back(initAttachmentReference(ca, attachment_indices));
        }
    }

    if (document.HasMember("resolve_attachment"))
    {
        color_resolve_attachment = initAttachmentReference(document["resolve_attachment"],
                                                           attachment_indices);
    }

    if (document.HasMember("depth_stencil_attachment"))
    {
        depth_stencil_attachment = initAttachmentReference(document["depth_stencil_attachment"],
                                                           attachment_indices);
    }
}

void RenderpassConfig::init(rapidjson::Value & document)
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

        attachments[name]  = attachment_index++;
        descriptions[name] = initAttachmentDescription(ad);

        if (ad.HasMember("clear_value"))
        {
            clear_values[ad["attachment_name"].GetString()] = initClearValue(ad["clear_value"]);
        }
    }

    assert(document.HasMember("subpasses"));
    assert(document["subpasses"].IsArray());

    for (auto & sp: document["subpasses"].GetArray())
    {
        assert(sp.IsObject());
        assert(sp.HasMember("name"));
        assert(sp["name"].IsString());

        SubpassHandle handle = subpasses.size();

        SubpassInfo info{};
        info.init(sp, attachments);

        subpass_handles[sp["name"].GetString()] = handle;
        subpasses.push_back(info);
    }

    assert(document.HasMember("subpass_dependencies"));
    assert(document["subpass_dependencies"].IsArray());

    for (auto & spd: document["subpass_dependencies"].GetArray())
    {
        subpass_dependencies.push_back(initDependency(spd));
    }
}

void PipelineConfig::init(rapidjson::Value &                                   document,
                          std::unordered_map<std::string, std::string> const & shader_names)
{
    assert(document.IsObject());

    assert(document.HasMember("vertex_shader_name"));
    assert(document["vertex_shader_name"].IsString());
    vertex_shader_name = document["vertex_shader_name"].GetString();

    assert(document.HasMember("fragment_shader_name"));
    assert(document["fragment_shader_name"].IsString());
    fragment_shader_name = document["fragment_shader_name"].GetString();

    assert(document.HasMember("vertex_bindings"));
    assert(document["vertex_bindings"].IsArray());
    for (auto const & vbi: document["vertex_bindings"].GetArray())
    {
        assert(vbi.IsString());
        vertex_binding_names.push_back(vbi.GetString());
    }

    assert(document.HasMember("vertex_attributes"));
    assert(document["vertex_attributes"].IsArray());
    for (auto const & vai: document["vertex_attributes"].GetArray())
    {
        assert(vai.IsString());
        vertex_attribute_names.push_back(vai.GetString());
    }

    assert(document.HasMember("uniform_layouts"));
    assert(document["uniform_layouts"].IsArray());
    for (auto const & uli: document["uniform_layouts"].GetArray())
    {
        assert(uli.IsString());
        uniform_layout_names.push_back(uli.GetString());
    }

    assert(document.HasMember("push_constants"));
    assert(document["push_constants"].IsArray());
    for (auto const & pci: document["push_constants"].GetArray())
    {
        assert(pci.IsString());
        push_constant_names.push_back(pci.GetString());
    }

    assert(document.HasMember("renderpass"));
    assert(document["renderpass"].IsString());
    renderpass = document["renderpass"].GetString();

    assert(document.HasMember("subpass"));
    assert(document["subpass"].IsString());
    subpass = document["subpass"].GetString();

    assert(document.HasMember("max_drawn_objects"));
    assert(document["max_drawn_objects"].IsUint());
    max_draw_calls = document["max_drawn_objects"].GetUint();
}

void UniformConfig::init(rapidjson::Value & document)
{
    assert(document.IsObject());
    assert(document.HasMember("max_count"));
    assert(document["max_count"].IsUint());

    max_uniform_count = document["max_count"].GetUint();
    assert(max_uniform_count != 0);

    layout_binding = initVkDescriptorSetLayoutBinding(document);
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

    assert(document.HasMember("dynamic_vertex_buffer_size"));
    assert(document["dynamic_vertex_buffer_size"].IsNumber());
    assert(document["dynamic_vertex_buffer_size"].IsInt());
    dynamic_vertex_buffer_size = document["dynamic_vertex_buffer_size"].GetInt();

    assert(document.HasMember("dynamic_index_buffer_size"));
    assert(document["dynamic_index_buffer_size"].IsNumber());
    assert(document["dynamic_index_buffer_size"].IsInt());
    dynamic_index_buffer_size = document["dynamic_index_buffer_size"].GetInt();

    assert(document.HasMember("staging_buffer_size"));
    assert(document["staging_buffer_size"].IsNumber());
    assert(document["staging_buffer_size"].IsInt());
    staging_buffer_size = document["staging_buffer_size"].GetInt();

    assert(document.HasMember("max_updated_objects"));
    assert(document["max_updated_objects"].IsUint());
    max_transfer_count = document["max_updated_objects"].GetUint();

    assert(document.HasMember("max_deleted_objects"));
    assert(document["max_deleted_objects"].IsUint());
    max_delete_count = document["max_deleted_objects"].GetUint();

    assert(document.HasMember("max_deleted_objects"));

    assert(document.HasMember("attachments"));
    assert(document["attachments"].IsArray());

    for (auto & a: document["attachments"].GetArray())
    {
        assert(a.IsObject());
        assert(a.HasMember("name"));
        assert(a["name"].IsString());

        attachment_configs[a["name"].GetString()].init(a);
    }

    assert(document.HasMember("renderpasses"));
    assert(document["renderpasses"].IsArray());

    for (auto & rp: document["renderpasses"].GetArray())
    {
        assert(rp.IsObject());
        assert(rp.HasMember("name"));
        assert(rp["name"].IsString());

        renderpass_configs[rp["name"].GetString()].init(rp);
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

        // uniform_layout_infos[ul["name"].GetString()] = initVkDescriptorSetLayoutBinding(ul);

        uniform_configs[ul["name"].GetString()].init(ul);
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

        vertex_attributes[va["name"].GetString()] = initVkVertexInputAttributeDescription(va);
    }

    assert(document.HasMember("pipelines"));
    assert(document["pipelines"].IsArray());

    for (auto & p: document["pipelines"].GetArray())
    {
        assert(p.IsObject());
        assert(p.HasMember("name"));
        assert(p["name"].IsString());

        pipeline_configs[p["name"].GetString()].init(p, shader_names);
    }
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
                       realdata->push_constant_size,
                       realdata->push_constant_data);

    vkCmdBindDescriptorSets(realdata->commandbuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            realdata->pipeline_layout,
                            0,
                            realdata->descriptor_set_count,
                            realdata->descriptor_sets,
                            realdata->dynamic_offset_count,
                            realdata->dynamic_offsets);

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

void deleteBuffers(void const * data)
{
    LOG_TRACE("Entering deleteBuffers");
    auto const * delete_data = reinterpret_cast<DeleteBuffers const *>(data);

    for (size_t i = 0; i < delete_data->buffer_count; ++i)
    {
        LOG_DEBUG("Deleting buffer {} {}",
                  static_cast<void *>(delete_data->buffers[i]),
                  static_cast<void *>(delete_data->memories[i]));

        vkDestroyBuffer(delete_data->logical_device, delete_data->buffers[i], nullptr);
        vkFreeMemory(delete_data->logical_device, delete_data->memories[i], nullptr);
    }

    LOG_TRACE("Exiting deleteBuffers");
}

cmd::BackendDispatchFunction const DeleteBuffers::DISPATCH_FUNCTION = &deleteBuffers;

void deleteTextures(void const * data)
{
    LOG_TRACE("Entering deleteTextures");
    auto const * delete_data = reinterpret_cast<DeleteTextures const *>(data);

    for (size_t i = 0; i < delete_data->texture_count; ++i)
    {
        LOG_DEBUG("Deleting texture {} {} {} {}",
                  static_cast<void *>(delete_data->samplers[i]),
                  static_cast<void *>(delete_data->views[i]),
                  static_cast<void *>(delete_data->images[i]),
                  static_cast<void *>(delete_data->memories[i]));
        vkDestroySampler(delete_data->logical_device, delete_data->samplers[i], nullptr);
        vkDestroyImageView(delete_data->logical_device, delete_data->views[i], nullptr);
        vkDestroyImage(delete_data->logical_device, delete_data->images[i], nullptr);
        vkFreeMemory(delete_data->logical_device, delete_data->memories[i], nullptr);
    }
    LOG_TRACE("Exiting deleteTextures");
}

cmd::BackendDispatchFunction const DeleteTextures::DISPATCH_FUNCTION = &deleteTextures;

void deleteUniforms(void const * data)
{
    LOG_TRACE("Entering deleteUniforms");
    auto const * delete_data = reinterpret_cast<DeleteUniforms const *>(data);

    for (size_t i = 0; i < delete_data->uniform_count; ++i)
    {
        auto   uniform_handle = delete_data->uniform_handles[i];
        auto & uniform_collection
            = (*delete_data->uniform_collections)[uniform_handle.uniform_layout_id];

        LOG_DEBUG("Deleting Uniform {} {}",
                  delete_data->uniform_handles[i].uniform_layout_id,
                  delete_data->uniform_handles[i].uniform_id);

        // delete the uniforms somehow here...
        std::visit(
            [uniform_handle](auto && collection) { collection.destroyUniform(uniform_handle); },
            uniform_collection);
    }
    LOG_TRACE("Exiting deleteUniforms");
}

cmd::BackendDispatchFunction const DeleteUniforms::DISPATCH_FUNCTION = &deleteUniforms;

//
// PHYSICALDEVICEINFO
//

bool PhysicalDeviceInfo::queues_complete() const
{
    return present_queue != -1 && graphics_queue != -1 && transfer_queue != -1;
}

bool PhysicalDeviceInfo::swapchain_adequate() const
{
    return !formats.empty() && !presentModes.empty();
}

namespace module
{
Device::Device(GLFWwindow * window_ptr): window{window_ptr}
{}

bool Device::init(RenderConfig & render_config)
{
    // clang-format off
    #ifndef NDEBUG
    checkValidationLayerSupport();
    #endif
    // clang-format on
    getRequiredExtensions();

    if (createInstance(render_config.window_name) != ErrorCode::NONE)
    {
        return false;
    }

    if (use_validation && createDebugMessenger() != ErrorCode::NONE)
    {
        return false;
    }

    if (createSurface() != ErrorCode::NONE)
    {
        return false;
    }

    if (!pickPhysicalDevice())
    {
        return false;
    }

    if (createLogicalDevice() != ErrorCode::NONE)
    {
        return false;
    }

    if (createSwapChain() != ErrorCode::NONE)
    {
        return false;
    }

    getSwapChainImages();

    if (createSwapChainImageViews() != ErrorCode::NONE)
    {
        return false;
    }

    return true;
}

void Device::quit()
{
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

void Device::checkValidationLayerSupport()
{
    uint32_t layerCount{0};
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);

    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char * layerName: validation_layers)
    {
        if (std::any_of(availableLayers.cbegin(),
                        availableLayers.cend(),
                        [layerName](VkLayerProperties const & supported_layer) {
                            return strcmp(layerName, supported_layer.layerName) == 0;
                        }))
        {
            break;
        }

        validation_supported = false;
        LOG_DEBUG("Vulkan validation layer {} not supported");
        break;
    }

    LOG_DEBUG("All required Vulkan validation layers are supported");
    validation_supported = true;
}

void Device::getRequiredExtensions()
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
ErrorCode Device::createInstance(char const * window_name)
{
    // clang-format off
    #ifdef NDEBUG
    use_validation = false;
    #else
    if (!validation_supported)
    {
        use_validation = false;
    }
    #endif
    // clang-format on

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
    VK_CHECK_RESULT(vkCreateInstance(&createInfo, nullptr, &instance),
                    "Unable to create VkInstance");

    return ErrorCode::NONE;
}

// VALIDATION LAYER DEBUG MESSAGER
VKAPI_ATTR VkBool32 VKAPI_CALL
                    Device::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
                      VkDebugUtilsMessageTypeFlagsEXT              messageType,
                      const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData,
                      void *                                       pUserData)
{
    if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
    {
        LOG_DEBUG("vulkan validation: {}", pCallbackData->pMessage);
    }
    else if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
    {
        LOG_INFO("vulkan validation: {}", pCallbackData->pMessage);
    }
    else if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        LOG_WARN("vulkan validation: {}", pCallbackData->pMessage);
    }
    else if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        LOG_ERROR("vulkan validation: {}", pCallbackData->pMessage);
    }

    return VK_FALSE;
}

ErrorCode Device::createDebugMessenger()
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

ErrorCode Device::createDebugUtilsMessengerEXT(
    const VkDebugUtilsMessengerCreateInfoEXT * pCreateInfo,
    const VkAllocationCallbacks *              pAllocator,
    VkDebugUtilsMessengerEXT *                 pDebugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        VK_CHECK_RESULT(func(instance, pCreateInfo, pAllocator, pDebugMessenger),
                        "Unable to create VkDebugUtilsMessengerEXT");

        return ErrorCode::NONE;
    }
    else
    {
        LOG_WARN("Vulkan DebugUtilsMessengerEXT extension is not present");
    }

    return ErrorCode::NONE;
}

void Device::cleanupDebugUtilsMessengerEXT(VkDebugUtilsMessengerEXT      debugMessenger,
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
ErrorCode Device::createSurface()
{
    VK_CHECK_RESULT(glfwCreateWindowSurface(instance, window, nullptr, &surface),
                    "Unable to create VkSurfaceKHR");

    return ErrorCode::NONE;
}

bool Device::pickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0)
    {
        LOG_ERROR("Failed to find GPUs with Vulkan support!");
        return false;
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

bool Device::isDeviceSuitable(VkPhysicalDevice device, PhysicalDeviceInfo & device_info)
{
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

void Device::findQueueFamilies(VkPhysicalDevice device, PhysicalDeviceInfo & device_info)
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

bool Device::checkDeviceExtensionSupport(VkPhysicalDevice device, PhysicalDeviceInfo & device_info)
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

void Device::querySwapChainSupport(VkPhysicalDevice device, PhysicalDeviceInfo & device_info)
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

void Device::getMaxUsableSampleCount()
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

// LOGICAL DEVICE
ErrorCode Device::createLogicalDevice()
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

    VK_CHECK_RESULT(vkCreateDevice(physical_device, &createInfo, nullptr, &logical_device),
                    "Unable to create VkDevice");

    return ErrorCode::NONE;
}

// SWAPCHAIN
ErrorCode Device::createSwapChain()
{
    VkSurfaceFormatKHR surfaceFormat    = chooseSwapSurfaceFormat(physical_device_info.formats);
    VkPresentModeKHR   presentMode      = chooseSwapPresentMode(physical_device_info.presentModes);
    VkExtent2D         extent           = chooseSwapExtent(physical_device_info.capabilities);
    auto               opt_depth_format = findDepthFormat();

    swapchain_image_format = surfaceFormat.format;
    swapchain_extent       = extent;
    if (!opt_depth_format)
    {
        LOG_ERROR("Unable to find proper depth format");
        return ErrorCode::VULKAN_ERROR;
    }
    depth_format = opt_depth_format.value();

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

    VK_CHECK_RESULT(vkCreateSwapchainKHR(logical_device, &createInfo, nullptr, &swapchain),
                    "Unable to create the VkSwapchainKHR");

    return ErrorCode::NONE;
}

VkSurfaceFormatKHR Device::chooseSwapSurfaceFormat(
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

VkPresentModeKHR Device::chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR> & availablePresentModes)
{
    if (std::any_of(availablePresentModes.cbegin(),
                    availablePresentModes.cend(),
                    [](VkPresentModeKHR available_mode) {
                        return available_mode == VK_PRESENT_MODE_MAILBOX_KHR;
                    }))
    {
        return VK_PRESENT_MODE_MAILBOX_KHR;
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D Device::chooseSwapExtent(VkSurfaceCapabilitiesKHR const & capabilities)
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

void Device::getSwapChainImages()
{
    vkGetSwapchainImagesKHR(logical_device, swapchain, &swapchain_image_count, nullptr);
    assert(swapchain_image_count > 0);
    swapchain_images.resize(swapchain_image_count);
    vkGetSwapchainImagesKHR(
        logical_device, swapchain, &swapchain_image_count, swapchain_images.data());
}

ErrorCode Device::createSwapChainImageViews()
{
    swapchain_image_views.resize(swapchain_images.size());

    for (size_t i = 0; i < swapchain_images.size(); i++)
    {
        auto viewInfo                            = VkImageViewCreateInfo{};
        viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image                           = swapchain_images[i];
        viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format                          = swapchain_image_format;
        viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel   = 0;
        viewInfo.subresourceRange.levelCount     = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount     = 1;

        VK_CHECK_RESULT(
            vkCreateImageView(logical_device, &viewInfo, nullptr, &swapchain_image_views[i]),
            "Unable to create VkImageView for Swapchain image");
    }

    return ErrorCode::NONE;
}

std::optional<VkFormat> Device::findDepthFormat()
{
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

std::optional<VkFormat> Device::findSupportedFormat(const std::vector<VkFormat> & candidates,
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

    return std::nullopt;
}

bool FrameResources::init(RenderConfig & render_config, VkDevice device)
{
    image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
    render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
    in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);

    auto semaphoreInfo = VkSemaphoreCreateInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

    auto fenceInfo = VkFenceCreateInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                       .flags = VK_FENCE_CREATE_SIGNALED_BIT};

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &image_available_semaphores[i])
                != VK_SUCCESS
            || vkCreateSemaphore(device, &semaphoreInfo, nullptr, &render_finished_semaphores[i])
                   != VK_SUCCESS
            || vkCreateFence(device, &fenceInfo, nullptr, &in_flight_fences[i]) != VK_SUCCESS)
        {
            return false;
        }
    }

    return true;
}

void FrameResources::quit(VkDevice device)
{
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
        vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
        vkDestroyFence(device, in_flight_fences[i], nullptr);
    }
}

bool ImageResources::init(RenderConfig & render_config, Device & device)
{
    // attachment_configs = std::move(render_config.attachment_configs);
    // attachment_handles.resize(attachment_configs.size());

    for (auto & config_iter: render_config.attachment_configs)
    {
        AttachmentHandle handle = attachment_configs.size();
        attachment_configs.push_back(config_iter.second);
        attachment_handles[config_iter.first] = handle;

        LOG_DEBUG("Added Attachment Handle {} for Layout {}", handle, config_iter.first);
    }

    attachments.resize(attachment_configs.size());

    return createAttachments(device) == ErrorCode::NONE;
}

void ImageResources::quit(Device & device)
{
    attachment_handles.clear();

    for (auto & sampler_iter: samplers)
    {
        sampler_iter.second.destroy(device.logical_device);
    }

    samplers.clear();
}

std::optional<TextureHandle> ImageResources::create_texture(VkPhysicalDevice      physical_device,
                                                            VkDevice              logical_device,
                                                            uint32_t              width,
                                                            uint32_t              height,
                                                            uint32_t              mipLevels,
                                                            VkSampleCountFlagBits numSamples,
                                                            VkFormat              format,
                                                            VkImageTiling         tiling,
                                                            VkImageUsageFlags     usage,
                                                            VkMemoryPropertyFlags properties,
                                                            VkImageAspectFlags    aspectFlags)
{
    TextureHandle handle = next_sampler_handle++;

    Sampler & sampler = samplers[handle];

    if (sampler.create(physical_device,
                       logical_device,
                       width,
                       height,
                       mipLevels,
                       numSamples,
                       format,
                       tiling,
                       usage,
                       properties,
                       aspectFlags)
        != ErrorCode::NONE)
    {
        LOG_DEBUG("Failed Sampler creation: {} {} {} {}",
                  static_cast<void *>(sampler.sampler_handle()),
                  static_cast<void *>(sampler.view_handle()),
                  static_cast<void *>(sampler.image_handle()),
                  static_cast<void *>(sampler.memory_handle()));

        sampler.destroy(logical_device);

        delete_texture(handle);

        return std::nullopt;
    }

    return handle;
}

std::optional<Sampler> ImageResources::get_texture(TextureHandle handle)
{
    auto sampler_iter = samplers.find(handle);

    if (sampler_iter != samplers.end())
    {
        return sampler_iter->second;
    }

    return std::nullopt;
}

void ImageResources::delete_texture(TextureHandle handle)
{
    auto sampler_iter = samplers.find(handle);

    if (sampler_iter != samplers.end())
    {
        samplers.erase(sampler_iter);
    }
}

std::optional<AttachmentConfig> ImageResources::get_config(AttachmentHandle handle)
{
    if (handle >= attachment_configs.size())
    {
        return std::nullopt;
    }

    return attachment_configs[handle];
}

std::optional<TextureHandle> ImageResources::get_texture_handle(AttachmentHandle handle)
{
    if (handle >= attachments.size())
    {
        return std::nullopt;
    }

    return attachments[handle];
}

std::optional<AttachmentHandle> ImageResources::get_attachment_handle(std::string const & name)
{
    auto iter = attachment_handles.find(name);
    if (iter == attachment_handles.end())
    {
        LOG_DEBUG("Didn't find AttachmentHandle for Attachment {}", name);
        return std::nullopt;
    }

    return iter->second;
}

ErrorCode ImageResources::createAttachments(Device & device)
{
    for (size_t i = 0; i < attachment_configs.size(); ++i)
    {
        auto const & attachment_config = attachment_configs[i];

        if (attachment_config.is_swapchain_image)
        {
            continue;
        }

        VkFormat           format;
        VkImageUsageFlags  usage = attachment_config.usage;
        VkImageAspectFlags aspect;
        VkImageLayout      final_layout;

        if (attachment_config.format == Format::USE_COLOR)
        {
            format       = device.swapchain_image_format;
            usage        = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            aspect       = VK_IMAGE_ASPECT_COLOR_BIT;
            final_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        else if (attachment_config.format == Format::USE_DEPTH)
        {
            format       = device.depth_format;
            usage        = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            aspect       = VK_IMAGE_ASPECT_DEPTH_BIT;
            final_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }

        VkSampleCountFlagBits samples;

        if (attachment_config.multisampled)
        {
            samples = device.physical_device_info.msaa_samples;
        }
        else
        {
            samples = VK_SAMPLE_COUNT_1_BIT;
        }

        auto opt_handle = create_texture(device.physical_device,
                                         device.logical_device,
                                         device.swapchain_extent.width,
                                         device.swapchain_extent.height,
                                         1,
                                         samples,
                                         format,
                                         VK_IMAGE_TILING_OPTIMAL,
                                         VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | usage,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                         aspect);

        if (!opt_handle)
        {
            return ErrorCode::API_ERROR;
        }

        attachments[i] = opt_handle.value();
    }

    return ErrorCode::NONE;
}

bool RenderPassResources::init(RenderConfig &   render_config,
                               Device &         device,
                               ImageResources & image_resources)
{
    for (auto & iter: render_config.renderpass_configs)
    {
        RenderpassHandle handle         = render_pass_configs.size();
        render_pass_handles[iter.first] = render_pass_configs.size();
        render_pass_configs.push_back(iter.second);

        LOG_DEBUG("Added Renderpass Handle {} for Renderpass {}", handle, iter.first);
    }

    for (auto & iter: render_config.renderpass_order)
    {
        LOG_DEBUG("Added Renderpass {} to draw ordering", iter);
        auto rp_handle = render_pass_handles[iter];
        auto sp_count  = render_pass_configs[rp_handle].subpasses.size();

        renderpass_order.push_back(rp_handle);
        per_renderpass_subpass_pipelines[rp_handle].resize(sp_count);
    }

    render_passes.resize(render_pass_configs.size());
    framebuffers.resize(render_pass_configs.size());
    clear_values.resize(render_pass_configs.size());

    return createRenderPasses(device, image_resources) == ErrorCode::NONE;
}

void RenderPassResources::quit(Device & device)
{
    for (auto & buffered_framebuffers: framebuffers)
    {
        for (auto & framebuffer: buffered_framebuffers)
        {
            vkDestroyFramebuffer(device.logical_device, framebuffer, nullptr);
        }
    }

    for (auto & render_pass: render_passes)
    {
        vkDestroyRenderPass(device.logical_device, render_pass, nullptr);
    }
}

ErrorCode RenderPassResources::createRenderPasses(Device & device, ImageResources & image_resources)
{
    for (size_t rp_i = 0; rp_i < render_passes.size(); ++rp_i)
    {
        auto & render_pass_config = render_pass_configs[rp_i];
        auto & render_pass        = render_passes[rp_i];

        auto & clear_value_list = clear_values[rp_i];
        clear_value_list.reserve(render_pass_config.clear_values.size());

        std::vector<std::string>             sorted_names{render_pass_config.descriptions.size()};
        std::vector<VkAttachmentDescription> sorted_descriptions{
            render_pass_config.descriptions.size()};

        for (auto & iter: render_pass_config.descriptions)
        {
            auto & attachment_name  = iter.first;
            auto & description      = iter.second;
            auto   attachment_index = render_pass_config.attachments[attachment_name];

            auto opt_attachment_handle = image_resources.get_attachment_handle(attachment_name);
            assert(opt_attachment_handle.has_value());
            auto attachment_handle = opt_attachment_handle.value();

            auto opt_attachment_config = image_resources.get_config(attachment_handle);
            assert(opt_attachment_config.has_value());
            auto attachment_config = opt_attachment_config.value();

            if (attachment_config.format == Format::USE_COLOR)
            {
                description.format = device.swapchain_image_format;
            }
            else if (attachment_config.format == Format::USE_DEPTH)
            {
                description.format = device.depth_format;
            }

            if (attachment_config.multisampled)
            {
                description.samples = device.physical_device_info.msaa_samples;
            }
            else
            {
                description.samples = VK_SAMPLE_COUNT_1_BIT;
            }

            sorted_descriptions[attachment_index] = description;
            sorted_names[attachment_index]        = attachment_name;
        }

        for (auto & name: sorted_names)
        {
            auto name_clear_value = render_pass_config.clear_values.find(name);
            if (name_clear_value != render_pass_config.clear_values.end())
            {
                clear_value_list.push_back(render_pass_config.clear_values[name]);
                LOG_DEBUG("Found clear value for attachment {}", name);
            }
        }

        std::vector<VkSubpassDescription> subpasses;
        subpasses.reserve(render_pass_config.subpasses.size());

        for (auto & subpass_info: render_pass_config.subpasses)
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
            .attachmentCount = static_cast<uint32_t>(render_pass_config.descriptions.size()),
            .pAttachments    = sorted_descriptions.data(),

            .subpassCount = static_cast<uint32_t>(subpasses.size()),
            .pSubpasses   = subpasses.data(),

            .dependencyCount = static_cast<uint32_t>(
                render_pass_config.subpass_dependencies.size()),
            .pDependencies = render_pass_config.subpass_dependencies.data()};

        VK_CHECK_RESULT(
            vkCreateRenderPass(device.logical_device, &renderPassInfo, nullptr, &render_pass),
            "Unable to create VkRenderPass");

        auto error = createFramebuffer(
            device, image_resources, render_pass_config, render_pass, framebuffers[rp_i]);
        if (error != ErrorCode::NONE)
        {
            return error;
        }
    }

    return ErrorCode::NONE;
}

// FRAMEBUFFER
ErrorCode RenderPassResources::createFramebuffer(Device &                     device,
                                                 ImageResources &             image_resources,
                                                 RenderpassConfig const &     config,
                                                 VkRenderPass const &         render_pass,
                                                 std::vector<VkFramebuffer> & framebuffers)
{
    framebuffers.resize(device.swapchain_image_count);

    for (size_t i = 0; i < device.swapchain_image_count; ++i)
    {
        auto & framebuffer = framebuffers[i];

        auto fb_attachments = std::vector<VkImageView>{config.attachments.size()};

        for (auto iter: config.attachments)
        {
            auto attachment_name  = iter.first;
            auto attachment_index = iter.second;

            auto attachment_handle = image_resources.get_attachment_handle(attachment_name).value();
            auto opt_attachment_config = image_resources.get_config(attachment_handle);
            if (!opt_attachment_config)
            {
                return ErrorCode::JSON_ERROR;
            }

            auto attachment_config = opt_attachment_config.value();

            if (attachment_config.is_swapchain_image)
            {
                fb_attachments[attachment_index] = device.swapchain_image_views[i];
            }
            else
            {
                auto opt_attachment_handle = image_resources.get_texture_handle(attachment_handle);
                if (!opt_attachment_handle)
                {
                    LOG_ERROR("Couldn't get texture_handle for AttachmentHandle");
                    return ErrorCode::JSON_ERROR;
                }

                auto opt_texture_handle = image_resources.get_texture(
                    opt_attachment_handle.value());
                if (!opt_texture_handle)
                {
                    LOG_ERROR("Unable to get TextureHandle for attachment to VkFramebuffer");
                    return ErrorCode::API_ERROR;
                }

                fb_attachments[attachment_index] = opt_texture_handle.value().view_handle();
            }
        }

        auto framebufferInfo = VkFramebufferCreateInfo{
            .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass      = render_pass,
            .attachmentCount = static_cast<uint32_t>(fb_attachments.size()),
            .pAttachments    = fb_attachments.data(),
            .width           = device.swapchain_extent.width,
            .height          = device.swapchain_extent.height,
            .layers          = 1};

        VK_CHECK_RESULT(
            vkCreateFramebuffer(device.logical_device, &framebufferInfo, nullptr, &framebuffer),
            "Unable to create VkFramebuffer");
    }

    return ErrorCode::NONE;
}

bool UniformResources::init(RenderConfig &    render_config,
                            Device &          device,
                            BufferResources & buffers)
{
    uniform_layout_infos.reserve(render_config.uniform_configs.size());
    uniform_counts.reserve(render_config.uniform_configs.size());

    for (auto iter: render_config.uniform_configs)
    {
        UniformLayoutHandle handle = uniform_layout_infos.size();

        uniform_layout_handles[iter.first] = handle;
        uniform_layout_infos.push_back(iter.second.layout_binding);
        uniform_counts.push_back(iter.second.max_uniform_count);

        LOG_DEBUG("Added Uniform Layout Handle {} for Layout {}", handle, iter.first);
    }

    uniform_layouts.resize(uniform_layout_infos.size());
    pools.resize(uniform_layout_infos.size());
    uniform_collections.resize(uniform_layout_infos.size());

    return createUniformLayouts(device, buffers) == ErrorCode::NONE;
}

void UniformResources::quit(Device & device)
{
    for (auto & uniform_layout: uniform_layouts)
    {
        vkDestroyDescriptorSetLayout(device.logical_device, uniform_layout, nullptr);
    }

    for (auto & pool: pools)
    {
        vkDestroyDescriptorPool(device.logical_device, pool, nullptr);
    }

    for (auto & collection: uniform_collections)
    {
        std::visit([&](auto && collection) { collection.destroy(device.logical_device); },
                   collection);
    }
}

ErrorCode UniformResources::createUniformLayouts(Device & device, BufferResources & buffers)
{
    for (size_t ul_i = 0; ul_i < uniform_layout_infos.size(); ++ul_i)
    {
        auto const & uniform_layout_info = uniform_layout_infos[ul_i];
        auto &       uniform_layout      = uniform_layouts[ul_i];
        auto &       pool                = pools[ul_i];
        auto &       uniform_collection  = uniform_collections[ul_i];

        auto layoutInfo = VkDescriptorSetLayoutCreateInfo{
            .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings    = &uniform_layout_info};

        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
                            device.logical_device, &layoutInfo, nullptr, &uniform_layout),
                        "Unable to create VkDescriptorSetLayout");

        if (uniform_layout_info.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
            || uniform_layout_info.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
        {
            const size_t uniform_count              = uniform_counts[ul_i];
            const size_t uniforms_per_descriptorset = 8;
            const size_t descriptor_count = ((uniform_count - 1) / uniforms_per_descriptorset) + 1;
            const size_t uniform_block_size = 256;

            LOG_DEBUG("descriptor_count for dynamic buffer uniform is {}", descriptor_count);

            auto poolsize = VkDescriptorPoolSize{
                .type            = uniform_layout_info.descriptorType,
                .descriptorCount = static_cast<uint32_t>(descriptor_count)};

            auto poolInfo = VkDescriptorPoolCreateInfo{
                .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .poolSizeCount = 1,
                .pPoolSizes    = &poolsize,
                .maxSets       = static_cast<uint32_t>(descriptor_count)};

            VK_CHECK_RESULT(
                vkCreateDescriptorPool(device.logical_device, &poolInfo, nullptr, &pool),
                "Unable to create VkDescriptorPool");

            std::vector<MappedBuffer> uniform_buffers; //{descriptor_count};

            VkDeviceSize memory_size = uniforms_per_descriptorset * uniform_block_size;

            for (size_t i = 0; i < descriptor_count; ++i)
            {
                LOG_DEBUG("Creating buffer for uniforms");

                auto opt_uniform_buffer_handle = buffers.create_buffer(
                    device,
                    memory_size,
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

                if (!opt_uniform_buffer_handle)
                {
                    LOG_ERROR("create_buffer returned nullopt when creating a uniform buffer");
                    return ErrorCode::API_ERROR;
                }

                auto opt_uniform_buffer = buffers.get_buffer(opt_uniform_buffer_handle.value());

                if (!opt_uniform_buffer)
                {
                    LOG_ERROR("get_buffer returned nullopt when getting a uniform buffer");
                    return ErrorCode::API_ERROR;
                }

                uniform_buffers.emplace_back(
                    device.logical_device, opt_uniform_buffer.value(), memory_size);
            }

            // ALLOCATE DESCRIPTORSETS GUY

            std::vector<VkDescriptorSet> descriptor_sets;

            descriptor_sets.resize(descriptor_count);

            std::vector<VkDescriptorSetLayout> layouts{descriptor_sets.size(), uniform_layout};

            auto allocInfo = VkDescriptorSetAllocateInfo{
                .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool     = pool,
                .descriptorSetCount = static_cast<uint32_t>(descriptor_sets.size()),
                .pSetLayouts        = layouts.data()};

            VK_CHECK_RESULT(
                vkAllocateDescriptorSets(device.logical_device, &allocInfo, descriptor_sets.data()),
                "Unable to allocate VkDescriptorSets");

            for (size_t ds_i = 0; ds_i < descriptor_sets.size(); ++ds_i)
            {
                auto & uniform_buffer = uniform_buffers[ds_i];

                auto bufferInfo = VkDescriptorBufferInfo{
                    .buffer = uniform_buffer.buffer_handle(),
                    .offset = 0,
                    .range  = device.physical_device_info.properties.limits
                                 .minUniformBufferOffsetAlignment};

                auto descriptorWrite = VkWriteDescriptorSet{
                    .sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet           = descriptor_sets[ds_i],
                    .dstBinding       = 0,
                    .dstArrayElement  = 0,
                    .descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    .descriptorCount  = 1,
                    .pBufferInfo      = &bufferInfo,
                    .pImageInfo       = nullptr,
                    .pTexelBufferView = nullptr};

                vkUpdateDescriptorSets(device.logical_device, 1, &descriptorWrite, 0, nullptr);
            }

            uniform_collection = DynamicBufferCollection{
                .descriptor_sets           = std::move(descriptor_sets),
                .uniform_buffers           = std::move(uniform_buffers),
                .uniforms                  = std::vector<DynamicBufferUniform>{descriptor_count
                                                              * uniforms_per_descriptorset},
                .free_uniform_buffer_slots = IndexAllocator(),
                .free_uniform_slots        = IndexAllocator()};

            std::get<DynamicBufferCollection>(uniform_collection)
                .free_uniform_buffer_slots.init(descriptor_count * uniforms_per_descriptorset);

            std::get<DynamicBufferCollection>(uniform_collection)
                .free_uniform_slots.init(descriptor_count * uniforms_per_descriptorset);
        }
        else if (uniform_layout_info.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
        {
            const size_t descriptor_count = uniform_counts[ul_i];

            LOG_DEBUG("descriptor_count for sampler uniform is {}", descriptor_count);

            auto poolsize = VkDescriptorPoolSize{
                .type            = uniform_layout_info.descriptorType,
                .descriptorCount = static_cast<uint32_t>(descriptor_count)};

            auto poolInfo = VkDescriptorPoolCreateInfo{
                .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .poolSizeCount = 1,
                .pPoolSizes    = &poolsize,
                .maxSets       = static_cast<uint32_t>(descriptor_count)};

            VK_CHECK_RESULT(
                vkCreateDescriptorPool(device.logical_device, &poolInfo, nullptr, &pool),
                "Unable to create VkDescriptorPool");

            std::vector<VkDescriptorSet> descriptor_sets;

            descriptor_sets.resize(uniform_counts[ul_i]);

            std::vector<VkDescriptorSetLayout> layouts{descriptor_sets.size(), uniform_layout};

            auto allocInfo = VkDescriptorSetAllocateInfo{
                .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool     = pool,
                .descriptorSetCount = static_cast<uint32_t>(descriptor_sets.size()),
                .pSetLayouts        = layouts.data()};

            VK_CHECK_RESULT(
                vkAllocateDescriptorSets(device.logical_device, &allocInfo, descriptor_sets.data()),
                "Unable to allocate VkDescriptorSets");

            uniform_collection = SamplerCollection{.descriptor_sets    = descriptor_sets,
                                                   .free_uniform_slots = IndexAllocator()};

            std::get<SamplerCollection>(uniform_collection)
                .free_uniform_slots.init(descriptor_sets.size());
        }
    }

    return ErrorCode::NONE;
}

bool PipelineResources::init(RenderConfig &        render_config,
                             Device &              device,
                             RenderPassResources & render_passes,
                             UniformResources &    uniforms)
{
    shader_files.reserve(render_config.shader_names.size());
    shader_files.reserve(render_config.shader_names.size());
    for (auto & iter: render_config.shader_names)
    {
        auto shader_handle = shader_files.size();

        shader_handles[iter.first] = shader_handle;
        shader_files.push_back(iter.second);
        shaders.push_back(VK_NULL_HANDLE);

        LOG_DEBUG("Added Shader Handle {} for shader {}, file: {}",
                  shader_handle,
                  iter.first,
                  iter.second);
    }

    if (createShaders(device) != ErrorCode::NONE)
    {
        return false;
    }

    push_constants.reserve(render_config.push_constants.size());
    for (auto & iter: render_config.push_constants)
    {
        auto push_constant_handle         = push_constants.size();
        push_constant_handles[iter.first] = push_constant_handle;
        push_constants.push_back(iter.second);

        LOG_DEBUG("Added Push Constant Handle {} for Binding {}", push_constant_handle, iter.first);
    }

    // vertex bindings
    vertex_bindings.reserve(render_config.vertex_bindings.size());
    for (auto & iter: render_config.vertex_bindings)
    {
        auto binding_handle                = vertex_bindings.size();
        vertex_binding_handles[iter.first] = binding_handle;
        vertex_bindings.push_back(iter.second);

        LOG_DEBUG("Added Vertex Binding Handle {} for Binding {}", binding_handle, iter.first);
    }

    // vertex attributes
    vertex_attributes.reserve(render_config.vertex_attributes.size());
    for (auto & iter: render_config.vertex_attributes)
    {
        auto attribute_handle                = vertex_attributes.size();
        vertex_attribute_handles[iter.first] = attribute_handle;
        vertex_attributes.push_back(iter.second);

        LOG_DEBUG(
            "Added Vertex Attribute Handle {} for Attribute {}", attribute_handle, iter.first);
    }

    pipeline_configs.reserve(render_config.pipeline_configs.size());
    draw_buckets.reserve(render_config.pipeline_configs.size());
    for (auto & iter: render_config.pipeline_configs)
    {
        auto pipeline_handle         = pipeline_configs.size();
        pipeline_handles[iter.first] = pipeline_handle;
        pipeline_configs.push_back(iter.second);
        draw_buckets.emplace_back(iter.second.max_draw_calls);

        LOG_DEBUG("Added Pipeline Handle {} for Pipeline {}", pipeline_handle, iter.first);
    }
    pipelines.resize(pipeline_configs.size());

    return createGraphicsPipeline(device, render_passes, uniforms) == ErrorCode::NONE;
}

void PipelineResources::quit(Device & device)
{
    for (auto & shader: shaders)
    {
        vkDestroyShaderModule(device.logical_device, shader, nullptr);
    }

    for (auto & pipeline: pipelines)
    {
        vkDestroyPipeline(device.logical_device, pipeline.vk_pipeline, nullptr);
        vkDestroyPipelineLayout(device.logical_device, pipeline.vk_pipeline_layout, nullptr);
    }
}

ErrorCode PipelineResources::createShaderModule(Device &                  device,
                                                std::vector<char> const & code,
                                                VkShaderModule &          shaderModule)
{
    auto createInfo = VkShaderModuleCreateInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode    = reinterpret_cast<const uint32_t *>(code.data())};

    VK_CHECK_RESULT(
        vkCreateShaderModule(device.logical_device, &createInfo, nullptr, &shaderModule),
        "Unable to create VkShaderModule");

    return ErrorCode::NONE;
}

ErrorCode PipelineResources::createShaders(Device & device)
{
    for (size_t i = 0; i < shaders.size(); ++i)
    {
        auto & shader = shaders[i];

        auto shaderCode = readFile(shader_files[i]);

        auto error = createShaderModule(device, shaderCode, shader);

        if (error != ErrorCode::NONE)
        {
            return error;
        }
    }

    return ErrorCode::NONE;
}

ErrorCode PipelineResources::createGraphicsPipeline(Device &              device,
                                                    RenderPassResources & render_passes,
                                                    UniformResources &    uniforms)
{
    for (size_t i = 0; i < pipelines.size(); ++i)
    {
        PipelineHandle pipeline_handle = i;
        auto &         pipeline        = pipelines[pipeline_handle];
        auto &         pipeline_config = pipeline_configs[pipeline_handle];

        LOG_DEBUG("Pipeline {} uses fragment shader {}, handle {}",
                  i,
                  pipeline_config.fragment_shader_name,
                  shader_handles[pipeline_config.fragment_shader_name]);
        auto vertShaderStageInfo = VkPipelineShaderStageCreateInfo{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = shaders[shader_handles[pipeline_config.vertex_shader_name]],
            .pName  = "main"};

        auto fragShaderStageInfo = VkPipelineShaderStageCreateInfo{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = shaders[shader_handles[pipeline_config.fragment_shader_name]],
            .pName  = "main"};

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindings = std::vector<VkVertexInputBindingDescription>{};
        for (auto const & binding_name: pipeline_config.vertex_binding_names)
        {
            bindings.push_back(vertex_bindings[vertex_binding_handles[binding_name]]);
        }

        auto attributes = std::vector<VkVertexInputAttributeDescription>{};
        for (auto const & attribute_name: pipeline_config.vertex_attribute_names)
        {
            attributes.push_back(vertex_attributes[vertex_attribute_handles[attribute_name]]);
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
                                   .width    = (float)device.swapchain_extent.width,
                                   .height   = (float)device.swapchain_extent.height,
                                   .minDepth = 0.0f,
                                   .maxDepth = 1.0f};

        auto scissor = VkRect2D{.offset = {0, 0}, .extent = device.swapchain_extent};

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
            .rasterizationSamples  = device.physical_device_info.msaa_samples,
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
        for (auto const & push_constant_name: pipeline_config.push_constant_names)
        {
            auto handle = push_constant_handles[push_constant_name];

            pushConstantRanges.push_back(push_constants[handle]);
        }

        std::vector<VkDescriptorSetLayout> layouts;

        for (auto & layout_name: pipeline_config.uniform_layout_names)
        {
            layouts.push_back(
                uniforms.uniform_layouts[uniforms.uniform_layout_handles[layout_name]]);
        }

        auto pipelineLayoutInfo = VkPipelineLayoutCreateInfo{
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts            = layouts.data(),
            .pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size()),
            .pPushConstantRanges    = pushConstantRanges.data()};

        VK_CHECK_RESULT(
            vkCreatePipelineLayout(
                device.logical_device, &pipelineLayoutInfo, nullptr, &pipeline.vk_pipeline_layout),
            "Unable to create VkPipelineLayout");

        auto   render_pass_handle = render_passes.render_pass_handles[pipeline_config.renderpass];
        auto & render_pass_config = render_passes.render_pass_configs[render_pass_handle];
        auto   subpass_handle     = render_pass_config.subpass_handles[pipeline_config.subpass];

        // push this pipeline handle into the map of commandbuckets

        LOG_DEBUG("Adding Pipeline {} to Renderpass {} at Subpass {}",
                  pipeline_handle,
                  render_pass_handle,
                  subpass_handle);

        render_passes.per_renderpass_subpass_pipelines[render_pass_handle][subpass_handle]
            .push_back(pipeline_handle);

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
            .renderPass          = render_passes.render_passes[render_pass_handle],
            .subpass             = static_cast<uint32_t>(subpass_handle),
            .basePipelineHandle  = VK_NULL_HANDLE, // Optional
            .basePipelineIndex   = -1              // Optional
        };

        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device.logical_device,
                                                  VK_NULL_HANDLE,
                                                  1,
                                                  &pipelineInfo,
                                                  nullptr,
                                                  &pipeline.vk_pipeline),
                        "Unable to create VkPipeline");
    }

    return ErrorCode::NONE;
}

bool CommandResources::init(RenderConfig & render_config, Device & device)
{
    int32_t const MAX_BUFFERED_RESOURCES = 3;

    size_t max_transfer_count = render_config.max_transfer_count;
    size_t max_delete_count   = render_config.max_delete_count;

    for (uint32_t i = 0; i < MAX_BUFFERED_RESOURCES; ++i)
    {
        transfer_buckets.emplace_back(max_transfer_count);
        delete_buckets.emplace_back(max_delete_count);
    }

    getQueues(device);

    if (createCommandPool(device) != ErrorCode::NONE)
    {
        return false;
    }

    return createCommandbuffers(device) == ErrorCode::NONE;
}

void CommandResources::quit(Device & device)
{
    for (uint32_t i = 0; i < delete_buckets.size(); ++i)
    {
        delete_buckets[i].Submit();
        delete_buckets[i].Clear();
        transfer_buckets[i].Clear();
    }

    vkDestroyCommandPool(device.logical_device, command_pool, nullptr);
}

void CommandResources::getQueues(Device & device)
{
    vkGetDeviceQueue(
        device.logical_device, device.physical_device_info.present_queue, 0, &present_queue);
    vkGetDeviceQueue(
        device.logical_device, device.physical_device_info.graphics_queue, 0, &graphics_queue);
    vkGetDeviceQueue(
        device.logical_device, device.physical_device_info.transfer_queue, 0, &transfer_queue);
}

ErrorCode CommandResources::createCommandPool(Device & device)
{
    auto poolInfo = VkCommandPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = static_cast<uint32_t>(device.physical_device_info.graphics_queue),
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};

    VK_CHECK_RESULT(vkCreateCommandPool(device.logical_device, &poolInfo, nullptr, &command_pool),
                    "Unable to create VkCommandPool");

    return ErrorCode::NONE;
}

ErrorCode CommandResources::createCommandbuffers(Device & device)
{
    int32_t const MAX_BUFFERED_RESOURCES = 3;

    draw_commandbuffers.resize(MAX_BUFFERED_RESOURCES);

    auto allocInfo = VkCommandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = command_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = (uint32_t)draw_commandbuffers.size()};

    VK_CHECK_RESULT(
        vkAllocateCommandBuffers(device.logical_device, &allocInfo, draw_commandbuffers.data()),
        "Unable to allocate VkCommandBuffer");

    transfer_commandbuffers.resize(MAX_BUFFERED_RESOURCES);

    allocInfo = VkCommandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = command_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = (uint32_t)transfer_commandbuffers.size()};

    VK_CHECK_RESULT(
        vkAllocateCommandBuffers(device.logical_device, &allocInfo, transfer_commandbuffers.data()),
        "Unable to allocate VkCommandBuffer");

    return ErrorCode::NONE;
}

bool BufferResources::init(RenderConfig & render_config, Device & device, FrameResources & frames)
{
    if (createDynamicObjectResources(device,
                                     frames,
                                     render_config.dynamic_vertex_buffer_size,
                                     render_config.dynamic_index_buffer_size)
        != ErrorCode::NONE)
    {
        return false;
    }

    if (createStagingObjectResources(device, frames, render_config.staging_buffer_size)
        != ErrorCode::NONE)
    {
        return false;
    }

    return true;
}

void BufferResources::quit(Device & device)
{
    for (auto & buffer_iter: buffers)
    {
        buffer_iter.second.destroy(device.logical_device);
    }

    buffers.clear();
}

ErrorCode BufferResources::createDynamicObjectResources(Device &         device,
                                                        FrameResources & frames,
                                                        size_t           dynamic_vertex_buffer_size,
                                                        size_t           dynamic_index_buffer_size)
{
    dynamic_mapped_vertices.reserve(frames.MAX_BUFFERED_RESOURCES);
    dynamic_mapped_indices.reserve(frames.MAX_BUFFERED_RESOURCES);

    VkDeviceSize vertices_memory_size = dynamic_vertex_buffer_size;
    VkDeviceSize indices_memory_size  = dynamic_index_buffer_size;

    for (uint32_t i = 0; i < frames.MAX_BUFFERED_RESOURCES; ++i)
    {
        LOG_DEBUG("Creating mapped vertices buffer for resource frame {}", i);

        auto opt_vertex_buffer_handle = create_buffer(
            device,
            vertices_memory_size,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!opt_vertex_buffer_handle)
        {
            LOG_ERROR("create_buffer returned nullopt when creating a vertex buffer for dynamic "
                      "vertex buffer");
            return ErrorCode::API_ERROR;
        }

        auto opt_vertex_buffer = get_buffer(opt_vertex_buffer_handle.value());

        if (!opt_vertex_buffer)
        {
            LOG_ERROR("get_buffer returned nullopt when getting a vertex buffer for dynamic vertex "
                      "buffer");
            return ErrorCode::API_ERROR;
        }

        dynamic_mapped_vertices.emplace_back(
            device.logical_device, opt_vertex_buffer.value(), vertices_memory_size);

        LOG_DEBUG("Creating mapped vertices buffer for resource frame {}", i);

        auto opt_index_buffer_handle = create_buffer(
            device,
            indices_memory_size,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (!opt_index_buffer_handle)
        {
            LOG_ERROR("create_buffer returned nullopt when creating an index buffer for dynamic "
                      "index buffer");
            return ErrorCode::API_ERROR;
        }

        auto opt_index_buffer = get_buffer(opt_index_buffer_handle.value());

        if (!opt_index_buffer)
        {
            LOG_ERROR("get_buffer returned nullopt when getting an index buffer for dynamic index "
                      "buffer");
            return ErrorCode::API_ERROR;
        }

        dynamic_mapped_indices.emplace_back(
            device.logical_device, opt_index_buffer.value(), indices_memory_size);
    }

    return ErrorCode::NONE;
}

ErrorCode BufferResources::createStagingObjectResources(Device &         device,
                                                        FrameResources & frames,
                                                        size_t           staging_buffer_size)
{
    staging_buffer.reserve(frames.MAX_BUFFERED_RESOURCES);

    for (uint32_t i = 0; i < frames.MAX_BUFFERED_RESOURCES; ++i)
    {
        LOG_DEBUG("Creating staged buffer for resource {}", i);

        auto opt_staging_buffer_handle = create_buffer(
            device,
            staging_buffer_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!opt_staging_buffer_handle)
        {
            LOG_ERROR("create_buffer returned nullopt when creating a staging buffer");
            return ErrorCode::API_ERROR;
        }

        auto opt_staging_buffer = get_buffer(opt_staging_buffer_handle.value());

        if (!opt_staging_buffer)
        {
            LOG_ERROR("get_buffer returned nullopt when getting a staging buffer");
            return ErrorCode::API_ERROR;
        }

        staging_buffer.emplace_back(
            device.logical_device, opt_staging_buffer.value(), staging_buffer_size);
    }

    return ErrorCode::NONE;
}

std::optional<BufferHandle> BufferResources::create_buffer(Device &              render_device,
                                                           VkDeviceSize          size,
                                                           VkBufferUsageFlags    usage,
                                                           VkMemoryPropertyFlags properties)
{
    BufferHandle handle = next_buffer_handle++;

    Buffer & buffer = buffers[handle];

    if (buffer.create(
            render_device.physical_device, render_device.logical_device, size, usage, properties)
        != ErrorCode::NONE)
    {
        LOG_ERROR("Couldn't create buffer, cleaning up..");
        buffer.destroy(render_device.logical_device);
        return std::nullopt;
    }

    LOG_DEBUG("Created buffer {} {}",
              static_cast<void *>(buffer.buffer_handle()),
              static_cast<void *>(buffer.memory_handle()));

    return handle;
}

std::optional<Buffer> BufferResources::get_buffer(BufferHandle handle)
{
    auto buffer_iter = buffers.find(handle);

    if (buffer_iter != buffers.end())
    {
        return buffer_iter->second;
    }

    return std::nullopt;
}

void BufferResources::delete_buffer(Device & render_device, BufferHandle handle)
{
    auto buffer_iter = buffers.find(handle);

    if (buffer_iter != buffers.end())
    {
        // buffer_iter->second.destroy(render_device.logical_device);
        buffers.erase(buffer_iter);
    }
}

}; // namespace module

Renderer::Renderer(GLFWwindow * window_ptr): render_device{window_ptr}
{}

bool Renderer::init(RenderConfig & render_config)
{
    LOG_INFO("Initializing Renderer");

    if (!render_device.init(render_config))
    {
        LOG_ERROR("Failed to initialize RenderDevice in Renderer");
        return false;
    }

    if (!frames.init(render_config, render_device.logical_device))
    {
        LOG_ERROR("Failed to initialize FrameResources in Renderer");
        return false;
    }

    if (!images.init(render_config, render_device))
    {
        LOG_ERROR("Failed to initialize ImageResources in Renderer");
        return false;
    }

    if (!render_passes.init(render_config, render_device, images))
    {
        LOG_ERROR("Failed to initialize RenderPassResources in Renderer");
        return false;
    }

    if (!buffers.init(render_config, render_device, frames))
    {
        LOG_ERROR("Failed to initialize BufferResources in Renderer");
        return false;
    }

    if (!uniforms.init(render_config, render_device, buffers))
    {
        LOG_ERROR("Failed to initialize UniformResources in Renderer");
        return false;
    }

    if (!pipelines.init(render_config, render_device, render_passes, uniforms))
    {
        LOG_ERROR("Failed to initialize PipelineResources in Renderer");
        return false;
    }

    if (!commands.init(render_config, render_device))
    {
        LOG_ERROR("Failed to initialize CommandResources in Renderer");
        return false;
    }

    return true;
}

void Renderer::quit()
{
    LOG_INFO("Quitting Renderer");
    commands.quit(render_device);
    pipelines.quit(render_device);
    uniforms.quit(render_device);
    buffers.quit(render_device);
    render_passes.quit(render_device);
    images.quit(render_device);
    frames.quit(render_device.logical_device);
    render_device.quit();
}

void Renderer::wait_for_idle()
{
    LOG_INFO("Waiting for Graphics Card to become Idle");
    vkDeviceWaitIdle(render_device.logical_device);
}

bool Renderer::submit_frame()
{
    LOG_DEBUG("Drawing frame");
    vkWaitForFences(render_device.logical_device,
                    1,
                    &frames.in_flight_fences[frames.currentFrame],
                    VK_TRUE,
                    std::numeric_limits<uint64_t>::max());

    // DRAW OPERATIONS
    auto result = vkAcquireNextImageKHR(render_device.logical_device,
                                        render_device.swapchain,
                                        std::numeric_limits<uint64_t>::max(),
                                        frames.image_available_semaphores[frames.currentFrame],
                                        VK_NULL_HANDLE,
                                        &frames.currentImage);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        LOG_DEBUG("Swapchain is out of date");
        return false;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        LOG_ERROR("Failed to acquire swap chain image!");
        return false;
    }

    // TRANSFER OPERATIONS
    // submit copy operations to the graphics queue

    auto beginInfo = VkCommandBufferBeginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                              .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                              .pInheritanceInfo = nullptr};

    result = vkBeginCommandBuffer(commands.transfer_commandbuffers[frames.currentResource],
                                  &beginInfo);
    if (result != VK_SUCCESS)
    {
        return false;
    }

    commands.transfer_buckets[frames.currentResource].Submit();

    result = vkEndCommandBuffer(commands.transfer_commandbuffers[frames.currentResource]);
    if (result != VK_SUCCESS)
    {
        return false;
    }

    auto submitTransferInfo = VkSubmitInfo{
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &commands.transfer_commandbuffers[frames.currentResource]};

    if (vkQueueSubmit(commands.graphics_queue, 1, &submitTransferInfo, VK_NULL_HANDLE)
        != VK_SUCCESS)
    {
        LOG_ERROR("Failed to submit transfer command buffer!");
        return false;
    }
    // the graphics queue will wait to do anything in the color_attachment_output stage
    // until the waitSemaphore is signalled by vkAcquireNextImageKHR
    VkSemaphore waitSemaphores[]      = {frames.image_available_semaphores[frames.currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

    VkSemaphore signalSemaphores[] = {frames.render_finished_semaphores[frames.currentFrame]};

    createCommandbuffer(frames.currentImage);

    auto submitInfo = VkSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,

        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = waitSemaphores,
        .pWaitDstStageMask  = waitStages,

        .commandBufferCount = 1,
        .pCommandBuffers    = &commands.draw_commandbuffers[frames.currentResource],

        .signalSemaphoreCount = 1,
        .pSignalSemaphores    = signalSemaphores};

    vkResetFences(render_device.logical_device, 1, &frames.in_flight_fences[frames.currentFrame]);

    if (vkQueueSubmit(
            commands.graphics_queue, 1, &submitInfo, frames.in_flight_fences[frames.currentFrame])
        != VK_SUCCESS)
    {
        LOG_ERROR("Failed to submit draw command buffer!");
        return false;
    }

    VkSwapchainKHR swapChains[] = {render_device.swapchain};

    auto presentInfo = VkPresentInfoKHR{.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                        .waitSemaphoreCount = 1,
                                        .pWaitSemaphores    = signalSemaphores,

                                        .swapchainCount = 1,
                                        .pSwapchains    = swapChains,
                                        .pImageIndices  = &frames.currentImage,

                                        .pResults = nullptr};

    result = vkQueuePresentKHR(commands.present_queue, &presentInfo);

    bool framebuffer_resized = false;
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebuffer_resized)
    {
        framebuffer_resized = false;
        LOG_DEBUG("Swapchain is out of date");
        return false;
    }
    else if (result != VK_SUCCESS)
    {
        LOG_ERROR("Failed to present swap chain image!");
        return false;
    }

    frames.currentFrame    = (frames.currentFrame + 1) % frames.MAX_FRAMES_IN_FLIGHT;
    frames.currentResource = (frames.currentResource + 1) % frames.MAX_BUFFERED_RESOURCES;

    // reset buffer offsets for copies
    buffers.dynamic_mapped_vertices[frames.currentResource].reset();
    buffers.dynamic_mapped_indices[frames.currentResource].reset();
    buffers.staging_buffer[frames.currentResource].reset();

    commands.transfer_buckets[frames.currentResource].Clear();
    commands.delete_buckets[frames.currentResource].Submit();

    return true;
}

ErrorCode Renderer::draw(PipelineHandle  pipeline,
                         size_t          vertices_size,
                         void *          vertices,
                         uint32_t        index_buffer_count,
                         uint32_t *      indices,
                         size_t          push_constant_size,
                         void *          push_constant_data,
                         size_t          uniform_count,
                         UniformHandle * p_uniforms)
{
    assert(push_constant_size < 128);
    LOG_DEBUG("Queuing draw call");

    auto & mapped_vertices = buffers.dynamic_mapped_vertices[frames.currentResource];
    auto & mapped_indices  = buffers.dynamic_mapped_indices[frames.currentResource];

    VkDeviceSize vertex_buffer_offset = mapped_vertices.copy(vertices_size, vertices);
    VkDeviceSize index_buffer_offset  = mapped_indices.copy(sizeof(uint32_t) * index_buffer_count,
                                                           indices);

    auto & bucket = pipelines.draw_buckets[pipeline];

    return make_draw_command(bucket,
                             pipelines.pipelines[pipeline].vk_pipeline_layout,
                             mapped_vertices.buffer_handle(),
                             vertex_buffer_offset,
                             mapped_indices.buffer_handle(),
                             index_buffer_offset,
                             index_buffer_count,
                             push_constant_size,
                             push_constant_data,
                             uniform_count,
                             p_uniforms);
}

ErrorCode Renderer::draw(PipelineHandle  pipeline,
                         BufferHandle    vertexbuffer_handle,
                         VkDeviceSize    vertexbuffer_offset,
                         BufferHandle    indexbuffer_handle,
                         VkDeviceSize    indexbuffer_offset,
                         VkDeviceSize    indexbuffer_count,
                         size_t          push_constant_size,
                         void *          push_constant_data,
                         size_t          uniform_count,
                         UniformHandle * p_uniforms)
{
    assert(push_constant_size < 128);
    LOG_DEBUG("Queuing draw call");

    // get bucket
    auto & bucket = pipelines.draw_buckets[pipeline];

    // get buffers
    auto opt_vertex_buffer = buffers.get_buffer(vertexbuffer_handle);

    if (!opt_vertex_buffer)
    {
        LOG_ERROR("Unable to get Vertex Buffer for draw call, ignoring call..");
        return ErrorCode::API_ERROR;
    }

    auto opt_index_buffer = buffers.get_buffer(indexbuffer_handle);

    if (!opt_index_buffer)
    {
        LOG_ERROR("Unable to get Index Buffer for draw call, ignoring call..");
        return ErrorCode::API_ERROR;
    }

    Buffer vertexbuffer = opt_vertex_buffer.value();
    Buffer indexbuffer  = opt_index_buffer.value();

    return make_draw_command(bucket,
                             pipelines.pipelines[pipeline].vk_pipeline_layout,
                             vertexbuffer.buffer_handle(),
                             vertexbuffer_offset,
                             indexbuffer.buffer_handle(),
                             indexbuffer_offset,
                             indexbuffer_count,
                             push_constant_size,
                             push_constant_data,
                             uniform_count,
                             p_uniforms);
}

ErrorCode Renderer::make_draw_command(cmd::CommandBucket<int> & bucket,
                                      VkPipelineLayout          pipeline_layout,
                                      VkBuffer                  vertex_buffer,
                                      VkDeviceSize              vertex_buffer_offset,
                                      VkBuffer                  index_buffer,
                                      VkDeviceSize              index_buffer_offset,
                                      VkDeviceSize              index_buffer_count,
                                      size_t                    push_constant_size,
                                      void *                    push_constant_data,
                                      size_t                    uniform_count,
                                      UniformHandle *           p_uniforms

)
{
    std::vector<VkDescriptorSet> descriptorsets;
    std::vector<uint32_t>        dynamic_offsets;

    descriptorsets.reserve(uniform_count);

    for (size_t i = 0; i < uniform_count; ++i)
    {
        auto uniform_handle = p_uniforms[i];
        auto opt_uniform    = getUniform(uniform_handle);

        if (opt_uniform.has_value())
        {
            descriptorsets.push_back(opt_uniform.value());
        }
        else
        {
            LOG_ERROR("No Descriptor Set returned for Uniform {} {}",
                      uniform_handle.uniform_layout_id,
                      uniform_handle.uniform_id);
            return ErrorCode::API_ERROR;
        }

        auto opt_offset = getDynamicOffset(uniform_handle);

        if (opt_offset.has_value())
        {
            dynamic_offsets.push_back(opt_offset.value());
        }
    }

    size_t descriptor_sets_size = sizeof(VkDescriptorSet) * descriptorsets.size();
    size_t dynamic_offsets_size = sizeof(uint32_t) * dynamic_offsets.size();

    Draw * command = bucket.AddCommand<Draw>(
        0, push_constant_size + descriptor_sets_size + dynamic_offsets_size);

    char * command_memory = cmd::commandPacket::GetAuxiliaryMemory(command);

    memcpy(command_memory, push_constant_data, push_constant_size);
    memcpy(command_memory + push_constant_size, descriptorsets.data(), descriptor_sets_size);
    memcpy(command_memory + push_constant_size + descriptor_sets_size,
           dynamic_offsets.data(),
           dynamic_offsets_size);

    command->commandbuffer        = commands.draw_commandbuffers[frames.currentResource];
    command->pipeline_layout      = pipeline_layout;
    command->vertexbuffer         = vertex_buffer;
    command->vertexbuffer_offset  = vertex_buffer_offset;
    command->indexbuffer          = index_buffer;
    command->indexbuffer_offset   = index_buffer_offset;
    command->indexbuffer_count    = index_buffer_count;
    command->push_constant_size   = push_constant_size;
    command->push_constant_data   = reinterpret_cast<void *>(command_memory);
    command->descriptor_set_count = descriptorsets.size();
    command->descriptor_sets      = reinterpret_cast<VkDescriptorSet *>(command_memory
                                                                   + push_constant_size);
    command->dynamic_offset_count = dynamic_offsets.size();
    command->dynamic_offsets      = reinterpret_cast<uint32_t *>(command_memory + push_constant_size
                                                            + descriptor_sets_size);

    return ErrorCode::NONE;
}

std::optional<UniformLayoutHandle> Renderer::get_uniform_layout_handle(std::string layout_name)
{
    auto handle_iter = uniforms.uniform_layout_handles.find(layout_name);
    if (handle_iter == uniforms.uniform_layout_handles.end())
    {
        return std::nullopt;
    }

    return handle_iter->second;
}

std::optional<PipelineHandle> Renderer::get_pipeline_handle(std::string pipeline_name)
{
    auto handle_iter = pipelines.pipeline_handles.find(pipeline_name);
    if (handle_iter == pipelines.pipeline_handles.end())
    {
        return std::nullopt;
    }

    return handle_iter->second;
}

std::optional<UniformHandle> Renderer::new_uniform(UniformLayoutHandle layout_handle,
                                                   VkDeviceSize        size,
                                                   void *              data_ptr)
{
    LOG_INFO("Creating a new Uniform");
    auto & uniform_collection = uniforms.uniform_collections[layout_handle];

    if (!std::holds_alternative<DynamicBufferCollection>(uniform_collection))
    {
        LOG_WARN("UniformLayout {} doesn't hold Dynamic Uniform Buffer resources", layout_handle);
        return std::nullopt;
    }

    auto & dynamic_buffer_collection = std::get<DynamicBufferCollection>(uniform_collection);

    auto opt_uniform_handle = dynamic_buffer_collection.createUniform(size, data_ptr);

    if (opt_uniform_handle)
    {
        opt_uniform_handle.value().uniform_layout_id = layout_handle;
    }

    return opt_uniform_handle;
}

std::optional<UniformHandle> Renderer::new_uniform(UniformLayoutHandle layout_handle,
                                                   TextureHandle       texture_handle)
{
    LOG_INFO("Creating a new Uniform");
    auto opt_sampler = images.get_texture(texture_handle);

    if (!opt_sampler)
    {
        LOG_ERROR("Unable to get Sampler for new_uniform call, ignoring call");
        return std::nullopt;
    }

    auto sampler = images.get_texture(texture_handle).value();

    auto & uniform_collection = uniforms.uniform_collections[layout_handle];

    if (!std::holds_alternative<SamplerCollection>(uniform_collection))
    {
        LOG_WARN("UniformLayout {} doesn't hold Sampler Uniform resources", layout_handle);
        return std::nullopt;
    }

    auto & sampler_collection = std::get<SamplerCollection>(uniform_collection);

    auto opt_uniform_handle = sampler_collection.createUniform(
        render_device.logical_device, sampler.view_handle(), sampler.sampler_handle());

    if (opt_uniform_handle)
    {
        opt_uniform_handle.value().uniform_layout_id = layout_handle;
    }

    return opt_uniform_handle;
}

std::optional<VkDescriptorSet> Renderer::getUniform(UniformHandle handle)
{
    auto & uniform_collection = uniforms.uniform_collections[handle.uniform_layout_id];

    return std::visit(
        [handle](auto && collection) -> std::optional<VkDescriptorSet> {
            return collection.getUniform(handle);
        },
        uniform_collection);
}

std::optional<VkDeviceSize> Renderer::getDynamicOffset(UniformHandle handle)
{
    auto & uniform_collection = uniforms.uniform_collections[handle.uniform_layout_id];

    return std::visit(
        [handle](auto && collection) -> std::optional<VkDeviceSize> {
            return collection.getDynamicOffset(handle);
        },
        uniform_collection);
}

void Renderer::delete_uniforms(size_t uniform_count, UniformHandle * uniform_handles)
{
    LOG_INFO("Deleting Uniforms");
    auto & bucket = commands.delete_buckets[frames.currentResource];

    DeleteUniforms * delete_command = bucket.AddCommand<DeleteUniforms>(
        0, uniform_count + sizeof(UniformHandle));

    char * command_memory = cmd::commandPacket::GetAuxiliaryMemory(delete_command);

    delete_command->uniform_collections = &uniforms.uniform_collections;
    delete_command->uniform_count       = uniform_count;
    delete_command->uniform_handles     = reinterpret_cast<UniformHandle *>(command_memory);

    memcpy(delete_command->uniform_handles, uniform_handles, uniform_count * sizeof(UniformHandle));
}

std::optional<BufferHandle> Renderer::create_buffer(VkDeviceSize          size,
                                                    VkBufferUsageFlags    usage,
                                                    VkMemoryPropertyFlags properties)
{
    LOG_INFO("Creating Buffer");

    return buffers.create_buffer(render_device, size, usage, properties);
}

void Renderer::update_buffer(BufferHandle buffer_handle, VkDeviceSize size, void * data)
{
    LOG_INFO("Updating Buffer");

    auto opt_buffer = buffers.get_buffer(buffer_handle);

    if (!opt_buffer)
    {
        LOG_ERROR("Unable to get buffer for update_buffer call, ignoring call");
        return;
    }

    Buffer buffer = buffers.get_buffer(buffer_handle).value();

    auto & mapped_buffer = buffers.staging_buffer[frames.currentResource];

    VkDeviceSize data_offset = mapped_buffer.copy(size, data);

    auto & bucket = commands.transfer_buckets[frames.currentResource];

    Copy * vertex_command         = bucket.AddCommand<Copy>(0, 0);
    vertex_command->commandbuffer = commands.transfer_commandbuffers[frames.currentResource];
    vertex_command->srcBuffer     = mapped_buffer.buffer_handle();
    vertex_command->dstBuffer     = buffer.buffer_handle();
    vertex_command->srcOffset     = data_offset;
    vertex_command->dstOffset     = 0;
    vertex_command->size          = size;
}

void Renderer::delete_buffers(size_t buffer_count, BufferHandle * buffer_handles)
{
    LOG_INFO("Deleting Buffers");

    auto & bucket = commands.delete_buckets[frames.currentResource];

    size_t buffer_size = buffer_count * sizeof(VkBuffer);
    size_t memory_size = buffer_count * sizeof(VkDeviceMemory);

    DeleteBuffers * delete_command = bucket.AddCommand<DeleteBuffers>(0, buffer_size + memory_size);

    char *           command_memory = cmd::commandPacket::GetAuxiliaryMemory(delete_command);
    VkBuffer *       buffer_iter    = reinterpret_cast<VkBuffer *>(command_memory);
    VkDeviceMemory * memory_iter = reinterpret_cast<VkDeviceMemory *>(command_memory + buffer_size);

    delete_command->logical_device = render_device.logical_device;
    delete_command->buffer_count   = buffer_count;
    delete_command->buffers        = buffer_iter;
    delete_command->memories       = memory_iter;

    for (size_t i = 0; i < buffer_count; ++i)
    {
        auto opt_buffer = buffers.get_buffer(buffer_handles[i]);
        if (!opt_buffer)
        {
            LOG_ERROR("Unable to get buffer for delete_buffers call, ignoring buffer {}",
                      buffer_handles[i]);

            *(buffer_iter++) = VK_NULL_HANDLE;
            *(memory_iter++) = VK_NULL_HANDLE;

            continue;
        }

        auto buffer = opt_buffer.value();
        buffers.delete_buffer(render_device, buffer_handles[i]);

        LOG_DEBUG("Queuing buffer {} {} for delete",
                  static_cast<void *>(buffer.buffer_handle()),
                  static_cast<void *>(buffer.memory_handle()));

        *(buffer_iter++) = buffer.buffer_handle();
        *(memory_iter++) = buffer.memory_handle();
    }
}

std::optional<TextureHandle> Renderer::create_texture(char const * texture_path)
{
    LOG_INFO("Creating Texture");
    int       texWidth, texHeight, texChannels;
    stbi_uc * pixels = stbi_load(texture_path, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

    VkDeviceSize imageSize = texWidth * texHeight * 4;

    // mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) +
    // 1;

    if (!pixels)
    {
        LOG_ERROR("Failed to load texture image {}", texture_path);
        return std::nullopt;
    }

    VkDeviceSize pixel_data_offset = buffers.staging_buffer[frames.currentResource].copy(
        static_cast<size_t>(imageSize), pixels);

    stbi_image_free(pixels);

    auto opt_texture_handle = images.create_texture(
        render_device.physical_device,
        render_device.logical_device,
        texWidth,
        texHeight,
        1,
        VK_SAMPLE_COUNT_1_BIT,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);

    if (!opt_texture_handle)
    {
        LOG_ERROR("Unable to create texture");
        return std::nullopt;
    }

    TextureHandle texture_handle = opt_texture_handle.value();

    auto opt_texture = images.get_texture(texture_handle);

    if (!opt_texture)
    {
        LOG_ERROR("Unable to get newly created texture");
        return std::nullopt;
    }

    Sampler texture = opt_texture.value();

    auto & bucket = commands.transfer_buckets[frames.currentResource];

    SetImageLayout * dst_optimal_command = bucket.AddCommand<SetImageLayout>(0, 0);
    dst_optimal_command->commandbuffer   = commands.transfer_commandbuffers[frames.currentResource];
    dst_optimal_command->srcAccessMask   = 0;
    dst_optimal_command->dstAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
    dst_optimal_command->oldLayout       = VK_IMAGE_LAYOUT_UNDEFINED;
    dst_optimal_command->newLayout       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    dst_optimal_command->image           = texture.image_handle();
    dst_optimal_command->mipLevels       = 1;
    dst_optimal_command->aspectMask      = VK_IMAGE_ASPECT_COLOR_BIT;
    dst_optimal_command->sourceStage     = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dst_optimal_command->destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

    CopyToImage * copy_command  = bucket.AddCommand<CopyToImage>(0, 0);
    copy_command->commandbuffer = commands.transfer_commandbuffers[frames.currentResource];
    copy_command->srcBuffer     = buffers.staging_buffer[frames.currentResource].buffer_handle();
    copy_command->srcOffset     = pixel_data_offset;
    copy_command->dstImage      = texture.image_handle();
    copy_command->width         = static_cast<uint32_t>(texWidth);
    copy_command->height        = static_cast<uint32_t>(texHeight);

    SetImageLayout * shader_optimal_command = bucket.AddCommand<SetImageLayout>(0, 0);
    shader_optimal_command->commandbuffer
        = commands.transfer_commandbuffers[frames.currentResource];
    shader_optimal_command->srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
    shader_optimal_command->dstAccessMask    = VK_ACCESS_SHADER_READ_BIT;
    shader_optimal_command->oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    shader_optimal_command->newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    shader_optimal_command->image            = texture.image_handle();
    shader_optimal_command->mipLevels        = 1;
    shader_optimal_command->aspectMask       = VK_IMAGE_ASPECT_COLOR_BIT;
    shader_optimal_command->sourceStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
    shader_optimal_command->destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

    return texture_handle;
}

void Renderer::delete_textures(size_t texture_count, TextureHandle * texture_handles)
{
    LOG_INFO("Deleting Textures");

    auto & bucket = commands.delete_buckets[frames.currentResource];

    size_t sampler_offset = 0;
    size_t sampler_size   = texture_count * sizeof(VkSampler);

    size_t view_offset = sampler_offset + sampler_size;
    size_t view_size   = texture_count * sizeof(VkImageView);

    size_t image_offset = view_offset + view_size;
    size_t image_size   = texture_count * sizeof(VkImage);

    size_t memory_offset = image_offset + image_size;
    size_t memory_size   = texture_count * sizeof(VkDeviceMemory);

    DeleteTextures * delete_command = bucket.AddCommand<DeleteTextures>(
        0, memory_offset + memory_size);

    char *           command_memory = cmd::commandPacket::GetAuxiliaryMemory(delete_command);
    VkSampler *      sampler_iter = reinterpret_cast<VkSampler *>(command_memory + sampler_offset);
    VkImageView *    view_iter    = reinterpret_cast<VkImageView *>(command_memory + view_offset);
    VkImage *        image_iter   = reinterpret_cast<VkImage *>(command_memory + image_offset);
    VkDeviceMemory * memory_iter  = reinterpret_cast<VkDeviceMemory *>(command_memory
                                                                      + memory_offset);

    delete_command->logical_device = render_device.logical_device;
    delete_command->texture_count  = texture_count;
    delete_command->samplers       = sampler_iter;
    delete_command->views          = view_iter;
    delete_command->images         = image_iter;
    delete_command->memories       = memory_iter;

    for (size_t i = 0; i < texture_count; ++i)
    {
        auto opt_sampler = images.get_texture(texture_handles[i]);
        if (!opt_sampler)
        {
            LOG_ERROR("Unable to get texture for delete_textures call, ignoring texture {}",
                      texture_handles[i]);

            *(sampler_iter++) = VK_NULL_HANDLE;
            *(view_iter++)    = VK_NULL_HANDLE;
            *(image_iter++)   = VK_NULL_HANDLE;
            *(memory_iter++)  = VK_NULL_HANDLE;

            continue;
        }
        auto sampler = opt_sampler.value();
        images.delete_texture(texture_handles[i]);

        *(sampler_iter++) = sampler.sampler_handle();
        *(view_iter++)    = sampler.view_handle();
        *(image_iter++)   = sampler.image_handle();
        *(memory_iter++)  = sampler.memory_handle();
    }
}

ErrorCode Renderer::createCommandbuffer(uint32_t image_index)
{
    auto & mapped_vertices = buffers.dynamic_mapped_vertices[frames.currentResource];
    auto & mapped_indices  = buffers.dynamic_mapped_indices[frames.currentResource];

    auto commandbuffer = commands.draw_commandbuffers[frames.currentResource];

    auto beginInfo = VkCommandBufferBeginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                              .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                              .pInheritanceInfo = nullptr};

    VK_CHECK_RESULT(
        vkBeginCommandBuffer(commands.draw_commandbuffers[frames.currentResource], &beginInfo),
        "Unable to begin VkCommandBuffer recording");

    // memory barrier for copy commands
    auto barrier = VkMemoryBarrier{.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                   .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                                   .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT};

    vkCmdPipelineBarrier(commandbuffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                         0,
                         1,
                         &barrier,
                         0,
                         nullptr,
                         0,
                         nullptr);

    for (RenderpassHandle const & rp_handle: render_passes.renderpass_order)
    {
        LOG_TRACE("Drawing Renderpass {}", rp_handle);

        auto & clearValues = render_passes.clear_values[rp_handle];

        auto renderPassInfo = VkRenderPassBeginInfo{
            .sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass        = render_passes.render_passes[rp_handle],
            .framebuffer       = render_passes.framebuffers[rp_handle][image_index],
            .renderArea.offset = {0, 0},
            .renderArea.extent = render_device.swapchain_extent,
            .clearValueCount   = static_cast<uint32_t>(clearValues.size()),
            .pClearValues      = clearValues.data()};

        vkCmdBeginRenderPass(commandbuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        auto & subpasses = render_passes.per_renderpass_subpass_pipelines[rp_handle];

        for (SubpassHandle sp_handle = 0; sp_handle < subpasses.size(); ++sp_handle)
        {
            auto & subpass_pipelines = subpasses[sp_handle];

            for (auto pipeline_handle: subpass_pipelines)
            {
                vkCmdBindPipeline(commandbuffer,
                                  VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  pipelines.pipelines[pipeline_handle].vk_pipeline);

                pipelines.draw_buckets[pipeline_handle].Submit();
                pipelines.draw_buckets[pipeline_handle].Clear();
            }
            if (sp_handle != subpasses.size() - 1)
            {
                vkCmdNextSubpass(commandbuffer, VK_SUBPASS_CONTENTS_INLINE);
            }
        }

        vkCmdEndRenderPass(commandbuffer);
    }

    VK_CHECK_RESULT(vkEndCommandBuffer(commandbuffer), "Unable to end VkCommandBuffer recording");

    return ErrorCode::NONE;
}

void Renderer::copyBuffer(VkBuffer     srcBuffer,
                          VkDeviceSize srcOffset,
                          VkBuffer     dstBuffer,
                          VkDeviceSize dstOffset,
                          VkDeviceSize size)
{
    auto & bucket = commands.transfer_buckets[frames.currentResource];

    Copy * vertex_command         = bucket.AddCommand<Copy>(0, 0);
    vertex_command->commandbuffer = commands.transfer_commandbuffers[frames.currentResource];
    vertex_command->srcBuffer     = srcBuffer;
    vertex_command->dstBuffer     = dstBuffer;
    vertex_command->srcOffset     = srcOffset;
    vertex_command->dstOffset     = dstOffset;
    vertex_command->size          = size;
};

}; // namespace gfx

#endif