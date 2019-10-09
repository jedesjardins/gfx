#ifndef JED_GFX_RENDERER_HPP
#define JED_GFX_RENDERER_HPP

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <array>
#include <set>
#include <variant>
#include <optional>
#include <unordered_map>

#include "cmd/cmd.hpp"

#ifdef JED_GFX_USER_CONFIG
#include JED_GFX_USER_CONFIG
#endif

#include "gfx/gfx_helper.hpp"
#include "gfx/render_config.hpp"

namespace gfx
{
const size_t max_calls_per_bucket = COMMANDS_PER_BUCKET;
const size_t descriptors_per_pool = DESCRIPTORS_PER_POOL;

class Memory
{
public:
    template <typename T>
    ErrorCode allocate_and_bind(VkPhysicalDevice      physical_device,
                                VkDevice              logical_device,
                                VkMemoryPropertyFlags properties,
                                T                     object_handle)
    {
        VkMemoryRequirements requirements;
        get_memory_requirements(logical_device, object_handle, requirements);

        auto opt_memory_type = find_memory_type(
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

        return bind_memory(logical_device, object_handle);
    }

    ErrorCode map(VkDevice logical_device, VkDeviceSize offset, VkDeviceSize size, void ** data);

    void destroy(VkDevice logical_device);

    VkDeviceMemory memory_handle();

protected:
    VkDeviceMemory vk_memory{VK_NULL_HANDLE};

private:
    std::optional<uint32_t> find_memory_type(VkPhysicalDevice      physical_device,
                                             uint32_t              typeFilter,
                                             VkMemoryPropertyFlags properties);

    void get_memory_requirements(VkDevice               logical_device,
                                 VkBuffer               buffer,
                                 VkMemoryRequirements & requirements);

    void get_memory_requirements(VkDevice               logical_device,
                                 VkImage                image,
                                 VkMemoryRequirements & requirements);

    ErrorCode bind_memory(VkDevice logical_device, VkBuffer buffer);

    ErrorCode bind_memory(VkDevice logical_device, VkImage image);
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

struct MappedBuffer
{
public:
    size_t copy(size_t size, void const * src_data);

    void reset();

    VkBuffer buffer_handle();

    VkDeviceSize offset{0};
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

    VkSampleCountFlagBits max_msaa_samples{VK_SAMPLE_COUNT_1_BIT};
};

struct BufferWrite
{
    BufferHandle buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
};

struct ImageWrite
{
    TextureHandle texture;
};

struct UniformWrite
{
    uint32_t first_array_element;

    size_t        buffer_write_count;
    BufferWrite * buffer_writes;

    size_t       image_write_count;
    ImageWrite * image_writes;
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
class Device
{
public:
    explicit Device(GLFWwindow * window_ptr);

    bool init(RenderConfig & render_config);
    void quit();

    VkDevice const &              get_logical_device() const;
    VkPhysicalDevice const &      get_physical_device() const;
    VkFormat const &              get_color_format() const;
    VkFormat const &              get_depth_format() const;
    VkExtent2D const &            get_extent() const;
    uint32_t const &              get_image_count() const;
    VkSampleCountFlagBits const & get_max_msaa_samples() const;
    VkImageView const &           get_swapchain_image_view(size_t index) const;
    VkSwapchainKHR const &        get_swapchain() const;
    PhysicalDeviceInfo const &    get_device_info() const;
    GLFWwindow * const            get_window() const;
    VkPresentModeKHR const &      get_present_mode() const;
    VkColorSpaceKHR const &       get_color_space() const;

    bool create_swapchain();
    void destroy_swapchain();

    void update_swapchain_support();

private:
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

    VkPresentModeKHR present_mode;

    VkFormat        swapchain_image_format;
    VkColorSpaceKHR swapchain_color_space;
    VkExtent2D      swapchain_extent;
    VkFormat        depth_format;
    uint32_t        imageCount;

    bool                            use_validation{true};
    bool                            validation_supported{false};
    const std::vector<char const *> validation_layers{"VK_LAYER_KHRONOS_validation"};

    std::vector<char const *>       required_extensions{};
    const std::vector<const char *> required_device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    void check_validation_layer_support();

    void get_required_extensions();

    // INSTANCE
    ErrorCode create_instance(char const * window_name);

    // VALIDATION LAYER DEBUG MESSAGER
    static VKAPI_ATTR VkBool32 VKAPI_CALL
                               debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
                                              VkDebugUtilsMessageTypeFlagsEXT              messageType,
                                              const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData,
                                              void *                                       pUserData);

    ErrorCode create_debug_messenger();

    ErrorCode create_debug_utils_messenger_ext(
        const VkDebugUtilsMessengerCreateInfoEXT * pCreateInfo,
        const VkAllocationCallbacks *              pAllocator,
        VkDebugUtilsMessengerEXT *                 pDebugMessenger);

    void cleanup_debug_utils_messenger_ext(VkDebugUtilsMessengerEXT      debugMessenger,
                                           const VkAllocationCallbacks * pAllocator);

    // SURFACE
    ErrorCode create_surface();

    bool pick_physical_device();

    bool is_device_suitable(VkPhysicalDevice device, PhysicalDeviceInfo & device_info);

    void find_queue_families(VkPhysicalDevice device, PhysicalDeviceInfo & device_info);

    bool check_device_extension_support(VkPhysicalDevice device, PhysicalDeviceInfo & device_info);

    void query_swapchain_support(VkPhysicalDevice device, PhysicalDeviceInfo & device_info);

    void get_max_usable_sample_count();

    // LOGICAL DEVICE
    ErrorCode create_logical_device();

    ErrorCode choose_swapchain_config();

    ErrorCode create_swapchain_khr();

    VkSurfaceFormatKHR choose_swap_surface_format(
        std::vector<VkSurfaceFormatKHR> const & availableFormats);

    VkPresentModeKHR choose_swap_present_mode(
        const std::vector<VkPresentModeKHR> & availablePresentModes);

    VkExtent2D choose_swap_extent(VkSurfaceCapabilitiesKHR const & capabilities);

    void get_swapchain_images();

    ErrorCode create_swapchain_image_views();

    std::optional<VkFormat> find_depth_format();

    std::optional<VkFormat> find_supported_format(const std::vector<VkFormat> & candidates,
                                                  VkImageTiling                 tiling,
                                                  VkFormatFeatureFlags          features);
}; // class Device

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

    bool init(RenderConfig & render_config, VkDevice const & device);
    void quit(VkDevice const & device);
}; // struct FrameResources

class ImageResources
{
public:
    bool init(RenderConfig & render_config, Device const & device);
    void quit(Device const & device);

    ErrorCode recreate_attachments(Device const & device);

    std::optional<TextureHandle> create_texture(VkPhysicalDevice const & physical_device,
                                                VkDevice const &         logical_device,
                                                uint32_t                 width,
                                                uint32_t                 height,
                                                uint32_t                 mipLevels,
                                                VkSampleCountFlagBits    numSamples,
                                                VkFormat                 format,
                                                VkImageTiling            tiling,
                                                VkImageUsageFlags        usage,
                                                VkMemoryPropertyFlags    properties,
                                                VkImageAspectFlags       aspectFlags);

    std::optional<Sampler> get_texture(TextureHandle const & handle) const;

    void delete_texture(TextureHandle const & handle);

    std::optional<AttachmentConfig> get_config(AttachmentHandle const & handle) const;

    std::optional<TextureHandle> get_texture_handle(AttachmentHandle const & handle) const;

    std::optional<AttachmentHandle> get_attachment_handle(std::string const & name) const;

private:
    std::unordered_map<std::string, AttachmentHandle> attachment_handles;
    std::vector<AttachmentConfig>                     attachment_configs;
    std::vector<TextureHandle>                        attachments;
    TextureHandle                                     next_sampler_handle{0};
    std::unordered_map<TextureHandle, Sampler>        samplers;

    ErrorCode create_attachments(Device const & device);

    ErrorCode create_attachment(Device const &           device,
                                AttachmentConfig const & attachment_config,
                                TextureHandle &          attachment);

    void destroy_attachments(Device const & device);
}; // class ImageResources

struct RenderPassResources
{
public:
    std::vector<RenderPassHandle> render_pass_order;

    std::unordered_map<RenderPassHandle, std::vector<std::vector<PipelineHandle>>>
        per_render_pass_subpass_pipelines;

    std::unordered_map<std::string, RenderPassHandle> render_pass_handles;
    std::vector<RenderPassConfig>                     render_pass_configs;
    std::vector<VkRenderPass>                         render_passes;
    std::vector<std::vector<VkFramebuffer>>           framebuffers;
    std::vector<std::vector<VkClearValue>>            clear_values;
    std::vector<VkSampleCountFlagBits>                samples;
    std::vector<VkExtent2D>                           extents;

    bool init(RenderConfig &         render_config,
              Device const &         device,
              ImageResources const & image_resources);
    void quit(Device const & device);

    void recreate_framebuffers(Device const & device, ImageResources const & image_resources);

private:
    ErrorCode createRenderPasses(Device const & device, ImageResources const & image_resources);

    // FRAMEBUFFER
    ErrorCode createFramebuffer(Device const &               device,
                                ImageResources const &       image_resources,
                                RenderPassConfig const &     config,
                                VkRenderPass const &         render_pass,
                                std::vector<VkFramebuffer> & framebuffers,
                                VkExtent2D &                 extent);
}; // struct RenderPassResources

class BufferResources
{
public:
    bool init();
    void quit(Device const & device);

    std::optional<BufferHandle> create_buffer(Device const &        device,
                                              VkDeviceSize          size,
                                              VkBufferUsageFlags    usage,
                                              VkMemoryPropertyFlags properties);

    std::optional<void *> map_buffer(BufferHandle const & handle) const;

    std::optional<Buffer> get_buffer(BufferHandle const & handle) const;

    void delete_buffer(BufferHandle const & handle);

private:
    BufferHandle next_buffer_handle{0};
    // todo: this isn't thread safe
    std::unordered_map<BufferHandle, Buffer> buffers;
    std::unordered_map<BufferHandle, void *> mapped_memory;
}; // class BufferResources

struct UniformSet
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;

    std::vector<VkDescriptorPool> pools;

    // maps a UniformHandle to a DescriptorSet
    std::vector<uint16_t>        generations;
    std::vector<VkDescriptorSet> uniforms;

    IndexAllocator free_descriptor_sets;
};

class UniformResources
{
public:
    std::unordered_map<std::string, UniformSetHandle> uniform_set_handles;

    std::vector<VkDescriptorSetLayout> uniform_layouts;
    std::vector<UniformSet>            uniform_sets;

    bool init(RenderConfig & render_config, Device const & device);
    void quit(Device const & device);

    std::optional<UniformHandle> create_uniform(UniformSetHandle const & set_handle);

    void update_uniform(Device &              device,
                        UniformHandle const & uniform_handle,
                        BufferResources &     buffers,
                        ImageResources &      images,
                        size_t                uniform_write_count,
                        UniformWrite *        uniform_write_infos);

    std::optional<VkDescriptorSet> get_uniform(UniformHandle const & handle);

    void delete_uniform(UniformHandle const & handle);

    std::optional<VkDescriptorSetLayout> get_layout(std::string const & name);

private:
    ErrorCode create_uniform_layout(Device const &                                    device,
                                    VkDescriptorSetLayout &                           layout,
                                    std::vector<VkDescriptorSetLayoutBinding> const & bindings);

    ErrorCode create_uniform_set(Device const &                              device,
                                 VkDescriptorSetLayout const &               layout,
                                 UniformSet &                                uniform_set,
                                 std::vector<VkDescriptorSetLayoutBinding> & bindings);

    ErrorCode create_pool(Device const &                                    device,
                          VkDescriptorPool &                                pool,
                          std::vector<VkDescriptorSetLayoutBinding> const & bindings);

    ErrorCode allocate_descriptor_sets(Device const &                device,
                                       VkDescriptorSetLayout const & layout,
                                       VkDescriptorPool &            pool,
                                       VkDescriptorSet *             descriptor_sets);

    ErrorCode check_valid_set(UniformSetHandle const & set_handle);

    ErrorCode check_valid_uniform(UniformHandle const & uniform_handle);
};

struct Pipeline
{
    VkPipeline         vk_pipeline;
    VkPipelineLayout   vk_pipeline_layout;
    VkShaderStageFlags push_constant_stages;
};

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

    std::unordered_map<std::string, PipelineHandle>   pipeline_handles;
    std::vector<PipelineConfig>                       pipeline_configs;
    std::vector<Pipeline>                             pipelines;
    std::vector<std::vector<cmd::CommandBucket<int>>> draw_buckets;

    bool init(RenderConfig &        render_config,
              Device const &        device,
              RenderPassResources & render_passes,
              UniformResources &    uniforms);
    void quit(Device const & device);

    ErrorCode recreate_pipelines(Device const &        device,
                                 RenderPassResources & render_passes,
                                 UniformResources &    uniforms);

    cmd::CommandBucket<int> & get_draw_bucket(PipelineHandle const & pipeline);

private:
    ErrorCode create_shader_module(Device const &            device,
                                   std::vector<char> const & code,
                                   VkShaderModule &          shaderModule);

    ErrorCode create_shaders(Device const & device, ReadFileFn read_file);

    ErrorCode create_pipelines(Device const &        device,
                               RenderPassResources & render_passes,
                               UniformResources &    uniforms);

    ErrorCode create_pipeline(Device const &         device,
                              RenderPassResources &  render_passes,
                              UniformResources &     uniforms,
                              PipelineHandle         pipeline_handle,
                              Pipeline &             pipeline,
                              PipelineConfig const & pipeline_config);

    void destroy_pipelines(Device const & device);
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

    std::vector<std::vector<cmd::CommandBucket<int>>> transfer_buckets;
    std::vector<std::vector<cmd::CommandBucket<int>>> delete_buckets;

    // pools belong to queue types, need one per queue (for when we use the transfer and graphics
    // queue) eventually have one pool per thread per
    VkCommandPool command_pool{VK_NULL_HANDLE};

    bool init(RenderConfig & render_config, Device const & device);
    void quit(Device const & device);

    cmd::CommandBucket<int> & get_transfer_bucket(uint32_t currentResource);
    cmd::CommandBucket<int> & get_delete_bucket(uint32_t currentResource);

private:
    void get_queues(Device const & device);

    ErrorCode create_command_pool(Device const & device);

    ErrorCode create_command_buffers(Device const & device);
}; // struct CommandResources

}; // namespace module

struct DrawParameters
{
    PipelineHandle  pipeline;
    size_t          vertex_buffer_count;
    BufferHandle *  vertex_buffers;
    VkDeviceSize *  vertex_buffer_offsets;
    BufferHandle    index_buffer;
    size_t          index_buffer_offset;
    size_t          index_count;
    size_t          push_constant_size;
    void *          push_constant_data;
    size_t          uniform_count;
    UniformHandle * uniforms;
    size_t          dynamic_offset_count;
    uint32_t *      dynamic_offsets;
    VkRect2D *      scissor;
    VkViewport *    viewport;
};

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

    size_t max_buffered_resources();

    size_t current_resource_index();

    ErrorCode draw(DrawParameters const & args);

    std::optional<AttachmentHandle> get_attachment_handle(std::string const & attachment_name);
    // std::optional<UniformSetHandle> get_uniform_layout_handle(std::string const & layout_name);
    std::optional<UniformSetHandle> get_uniform_set_handle(std::string const & set_name);
    std::optional<PipelineHandle>   get_pipeline_handle(std::string const & pipeline_name);

    std::optional<BufferHandle> create_buffer(VkDeviceSize          size,
                                              VkBufferUsageFlags    usage,
                                              VkMemoryPropertyFlags properties);

    std::optional<void *> map_buffer(BufferHandle const & buffer_handle);

    void update_buffer(BufferHandle const & buffer, VkDeviceSize size, void * data);

    void delete_buffers(size_t buffer_count, BufferHandle const * buffers);

    std::optional<TextureHandle> create_texture(size_t       width,
                                                size_t       height,
                                                size_t       pixel_size,
                                                void * const pixels);

    std::optional<TextureHandle> get_texture(AttachmentHandle const & attachment);

    void delete_textures(size_t sampler_count, TextureHandle const * texture_handles);

    std::optional<UniformHandle> create_uniform(UniformSetHandle set_handle,
                                                size_t           write_count,
                                                UniformWrite *   write_infos);

    void delete_uniforms(size_t uniform_count, UniformHandle const * uniforms);

private:
    std::optional<VkDescriptorSet> get_uniform(UniformHandle const & handle);

    std::optional<VkDeviceSize> get_dynamic_offset(UniformHandle const & handle);

    ErrorCode create_command_buffer(uint32_t image_index);

    void copy_buffer(VkBuffer     srcBuffer,
                     VkDeviceSize srcOffset,
                     VkBuffer     dstBuffer,
                     VkDeviceSize dstOffset,
                     VkDeviceSize size);

    void change_swapchain();

    module::Device              device;
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

std::optional<uint32_t> Memory::find_memory_type(VkPhysicalDevice      physical_device,
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

void Memory::get_memory_requirements(VkDevice               logical_device,
                                     VkBuffer               buffer,
                                     VkMemoryRequirements & requirements)
{
    vkGetBufferMemoryRequirements(logical_device, buffer, &requirements);
}

void Memory::get_memory_requirements(VkDevice               logical_device,
                                     VkImage                image,
                                     VkMemoryRequirements & requirements)
{
    vkGetImageMemoryRequirements(logical_device, image, &requirements);
}

ErrorCode Memory::bind_memory(VkDevice logical_device, VkBuffer buffer)
{
    VK_CHECK_RESULT(vkBindBufferMemory(logical_device, buffer, vk_memory, 0),
                    "Unable to bind VkDeviceMemory to VkBuffer");

    return ErrorCode::NONE;
}

ErrorCode Memory::bind_memory(VkDevice logical_device, VkImage image)
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

    ErrorCode error = allocate_and_bind(physical_device, logical_device, properties, vk_image);
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

    return allocate_and_bind(physical_device, logical_device, properties, vk_buffer);
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
    for (int32_t i = 0; i < static_cast<int32_t>(indices.size()) - 1; ++i)
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

//
// COMMANDS
//

// DRAW

struct Draw
{
    static cmd::BackendDispatchFunction const DISPATCH_FUNCTION;

    VkCommandBuffer    commandbuffer;
    VkPipelineLayout * pipeline_layout;

    size_t         vertex_buffer_count;
    VkDeviceSize * vertex_buffer_offsets;
    VkBuffer *     vertex_buffers;

    size_t   index_count;
    size_t   index_buffer_offset;
    VkBuffer index_buffer;

    size_t             push_constant_size;
    void *             push_constant_data;
    VkShaderStageFlags push_constant_flags;

    size_t            descriptor_set_count;
    VkDescriptorSet * descriptor_sets;
    size_t            dynamic_offset_count;
    uint32_t *        dynamic_offsets;

    VkViewport * viewport;

    VkRect2D * scissor;
};

static_assert(std::is_pod<Draw>::value == true, "Draw must be a POD.");

void draw(void const * data)
{
    Draw const * realdata = reinterpret_cast<Draw const *>(data);

    if (realdata->scissor != nullptr)
    {
        vkCmdSetScissor(realdata->commandbuffer, 0, 1, realdata->scissor);
    }

    if (realdata->viewport != nullptr)
    {
        vkCmdSetViewport(realdata->commandbuffer, 0, 1, realdata->viewport);
    }

    if (realdata->push_constant_size != 0)
    {
        vkCmdPushConstants(realdata->commandbuffer,
                           *realdata->pipeline_layout,
                           realdata->push_constant_flags,
                           0,
                           realdata->push_constant_size,
                           realdata->push_constant_data);
    }

    if (realdata->descriptor_set_count != 0)
    {
        vkCmdBindDescriptorSets(realdata->commandbuffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                *realdata->pipeline_layout,
                                0,
                                realdata->descriptor_set_count,
                                realdata->descriptor_sets,
                                realdata->dynamic_offset_count,
                                realdata->dynamic_offsets);
    }

    vkCmdBindVertexBuffers(realdata->commandbuffer,
                           0,
                           realdata->vertex_buffer_count,
                           realdata->vertex_buffers,
                           realdata->vertex_buffer_offsets);

    vkCmdBindIndexBuffer(realdata->commandbuffer,
                         realdata->index_buffer,
                         realdata->index_buffer_offset * sizeof(uint32_t),
                         VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(realdata->commandbuffer, realdata->index_count, 1, 0, 0, 0);
}

cmd::BackendDispatchFunction const Draw::DISPATCH_FUNCTION = &draw;

// COPY BUFFERS

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

void copy(void const * data)
{
    Copy const * copydata = reinterpret_cast<Copy const *>(data);

    auto copyRegion = VkBufferCopy{
        .srcOffset = copydata->srcOffset, .dstOffset = copydata->dstOffset, .size = copydata->size};

    vkCmdCopyBuffer(
        copydata->commandbuffer, copydata->srcBuffer, copydata->dstBuffer, 1, &copyRegion);
}

cmd::BackendDispatchFunction const Copy::DISPATCH_FUNCTION = &copy;

// COPY TO IMAGES

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

// TRANSITION IMAGE LAYOUT

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

// DELETE BUFFERS

struct DeleteBuffers
{
    static cmd::BackendDispatchFunction const DISPATCH_FUNCTION;

    VkDevice         logical_device;
    size_t           buffer_count;
    VkBuffer *       buffers;
    VkDeviceMemory * memories;
};

static_assert(std::is_pod<DeleteBuffers>::value == true, "DeleteBuffers must be a POD.");

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

// DELETE TEXTURES

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

// DELETE UNIFORMS

struct DeleteUniforms
{
    static cmd::BackendDispatchFunction const DISPATCH_FUNCTION;

    module::UniformResources * uniform_resources;

    size_t          uniform_count;
    UniformHandle * uniform_handles;
};

static_assert(std::is_pod<DeleteUniforms>::value == true, "DeleteUniforms must be a POD.");

void deleteUniforms(void const * data)
{
    LOG_TRACE("Entering deleteUniforms");
    auto const * delete_data = reinterpret_cast<DeleteUniforms const *>(data);

    for (size_t i = 0; i < delete_data->uniform_count; ++i)
    {
        LOG_DEBUG("Deleting Uniform {} {}",
                  delete_data->uniform_handles[i].set,
                  delete_data->uniform_handles[i].uniform);

        delete_data->uniform_resources->delete_uniform(delete_data->uniform_handles[i]);
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
    check_validation_layer_support();
    #endif
    // clang-format on
    get_required_extensions();

    if (create_instance(render_config.window_name) != ErrorCode::NONE)
    {
        return false;
    }

    if (use_validation && create_debug_messenger() != ErrorCode::NONE)
    {
        return false;
    }

    if (create_surface() != ErrorCode::NONE)
    {
        return false;
    }

    if (!pick_physical_device())
    {
        return false;
    }

    if (create_logical_device() != ErrorCode::NONE)
    {
        return false;
    }

    return create_swapchain();
}

void Device::quit()
{
    destroy_swapchain();

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
        cleanup_debug_utils_messenger_ext(debug_messager, nullptr);
    }

    if (instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(instance, nullptr);
    }
}

VkDevice const & Device::get_logical_device() const
{
    return logical_device;
}

VkPhysicalDevice const & Device::get_physical_device() const
{
    return physical_device;
}

VkFormat const & Device::get_color_format() const
{
    return swapchain_image_format;
}

VkFormat const & Device::get_depth_format() const
{
    return depth_format;
}

VkExtent2D const & Device::get_extent() const
{
    return swapchain_extent;
}

uint32_t const & Device::get_image_count() const
{
    return swapchain_image_count;
}

VkSampleCountFlagBits const & Device::get_max_msaa_samples() const
{
    return physical_device_info.max_msaa_samples;
}

VkImageView const & Device::get_swapchain_image_view(size_t index) const
{
    return swapchain_image_views[index];
}

VkSwapchainKHR const & Device::get_swapchain() const
{
    return swapchain;
}

PhysicalDeviceInfo const & Device::get_device_info() const
{
    return physical_device_info;
}

GLFWwindow * const Device::get_window() const
{
    return window;
}

VkPresentModeKHR const & Device::get_present_mode() const
{
    return present_mode;
}

VkColorSpaceKHR const & Device::get_color_space() const
{
    return swapchain_color_space;
}

bool Device::create_swapchain()
{
    if (choose_swapchain_config() != ErrorCode::NONE)
    {
        return false;
    }

    if (create_swapchain_khr() != ErrorCode::NONE)
    {
        return false;
    }

    get_swapchain_images();

    if (create_swapchain_image_views() != ErrorCode::NONE)
    {
        return false;
    }

    return true;
}

void Device::destroy_swapchain()
{
    for (VkImageView & image_view: swapchain_image_views)
    {
        vkDestroyImageView(logical_device, image_view, nullptr);
    }

    if (swapchain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(logical_device, swapchain, nullptr);
    }
}

void Device::update_swapchain_support()
{
    query_swapchain_support(physical_device, physical_device_info);
}

void Device::check_validation_layer_support()
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

void Device::get_required_extensions()
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
ErrorCode Device::create_instance(char const * window_name)
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
                    Device::debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
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

ErrorCode Device::create_debug_messenger()
{
    auto createInfo = VkDebugUtilsMessengerCreateInfoEXT{
        .sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                           | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                           | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debug_callback,
        .pUserData       = nullptr // Optional
    };

    return create_debug_utils_messenger_ext(&createInfo, nullptr, &debug_messager);
}

ErrorCode Device::create_debug_utils_messenger_ext(
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

void Device::cleanup_debug_utils_messenger_ext(VkDebugUtilsMessengerEXT      debugMessenger,
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
ErrorCode Device::create_surface()
{
    VK_CHECK_RESULT(glfwCreateWindowSurface(instance, window, nullptr, &surface),
                    "Unable to create VkSurfaceKHR");

    return ErrorCode::NONE;
}

bool Device::pick_physical_device()
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

        if (is_device_suitable(device, device_info))
        {
            physical_device      = device;
            physical_device_info = device_info;
            get_max_usable_sample_count();
            break;
        }
    }

    if (physical_device == VK_NULL_HANDLE)
    {
        return false;
    }
    return true;
}

bool Device::is_device_suitable(VkPhysicalDevice device, PhysicalDeviceInfo & device_info)
{
    find_queue_families(device, device_info);

    if (!device_info.queues_complete())
    {
        return false;
    }

    check_device_extension_support(device, device_info);

    if (!device_info.has_required_extensions)
    {
        return false;
    }

    query_swapchain_support(device, device_info);

    if (!device_info.swapchain_adequate())
    {
        return false;
    }

    vkGetPhysicalDeviceFeatures(device, &device_info.features);

    return device_info.features.samplerAnisotropy == VK_TRUE;
}

void Device::find_queue_families(VkPhysicalDevice device, PhysicalDeviceInfo & device_info)
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

bool Device::check_device_extension_support(VkPhysicalDevice     device,
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

void Device::query_swapchain_support(VkPhysicalDevice device, PhysicalDeviceInfo & device_info)
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

void Device::get_max_usable_sample_count()
{
    vkGetPhysicalDeviceProperties(physical_device, &physical_device_info.properties);

    VkSampleCountFlags counts = std::min(
        physical_device_info.properties.limits.framebufferColorSampleCounts,
        physical_device_info.properties.limits.framebufferDepthSampleCounts);
    if (counts & VK_SAMPLE_COUNT_64_BIT)
    {
        physical_device_info.max_msaa_samples = VK_SAMPLE_COUNT_64_BIT;
        return;
    }
    if (counts & VK_SAMPLE_COUNT_32_BIT)
    {
        physical_device_info.max_msaa_samples = VK_SAMPLE_COUNT_32_BIT;
        return;
    }
    if (counts & VK_SAMPLE_COUNT_16_BIT)
    {
        physical_device_info.max_msaa_samples = VK_SAMPLE_COUNT_16_BIT;
        return;
    }
    if (counts & VK_SAMPLE_COUNT_8_BIT)
    {
        physical_device_info.max_msaa_samples = VK_SAMPLE_COUNT_8_BIT;
        return;
    }
    if (counts & VK_SAMPLE_COUNT_4_BIT)
    {
        physical_device_info.max_msaa_samples = VK_SAMPLE_COUNT_4_BIT;
        return;
    }
    if (counts & VK_SAMPLE_COUNT_2_BIT)
    {
        physical_device_info.max_msaa_samples = VK_SAMPLE_COUNT_2_BIT;
        return;
    }

    physical_device_info.max_msaa_samples = VK_SAMPLE_COUNT_1_BIT;
}

// LOGICAL DEVICE
ErrorCode Device::create_logical_device()
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
    auto deviceFeatures = VkPhysicalDeviceFeatures{
        .samplerAnisotropy = VK_TRUE, .sampleRateShading = VK_TRUE, .fillModeNonSolid = VK_TRUE};

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

ErrorCode Device::choose_swapchain_config()
{
    present_mode = choose_swap_present_mode(physical_device_info.presentModes);

    VkSurfaceFormatKHR surface_format = choose_swap_surface_format(physical_device_info.formats);
    swapchain_image_format            = surface_format.format;
    swapchain_color_space             = surface_format.colorSpace;
    swapchain_extent                  = choose_swap_extent(physical_device_info.capabilities);
    auto opt_depth_format             = find_depth_format();
    if (!opt_depth_format)
    {
        LOG_ERROR("Couldn't find a depth format");
        return ErrorCode::VULKAN_ERROR;
    }
    depth_format = opt_depth_format.value();

    // imagecount is greater than min image count and less than or equal to maximage count
    imageCount = physical_device_info.capabilities.minImageCount + 1;
    if (physical_device_info.capabilities.maxImageCount > 0
        && imageCount > physical_device_info.capabilities.maxImageCount)
    {
        imageCount = physical_device_info.capabilities.maxImageCount;
    }

    return ErrorCode::NONE;
}

// SWAPCHAIN
ErrorCode Device::create_swapchain_khr()
{
    auto createInfo = VkSwapchainCreateInfoKHR{
        .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface          = surface,
        .minImageCount    = imageCount,
        .imageFormat      = swapchain_image_format,
        .imageColorSpace  = swapchain_color_space,
        .imageExtent      = swapchain_extent,
        .imageArrayLayers = 1,
        .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .preTransform     = physical_device_info.capabilities.currentTransform,
        .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode      = present_mode,
        .clipped          = VK_TRUE,
        .oldSwapchain     = VK_NULL_HANDLE};

    // if there are two queues, enable concurrent access
    // since graphics queue will draw to the swap chain and present queue will present the image
    if (physical_device_info.graphics_queue != physical_device_info.present_queue)
    {
        uint32_t queueFamilyIndices[] = {static_cast<uint32_t>(physical_device_info.graphics_queue),
                                         static_cast<uint32_t>(physical_device_info.present_queue)};

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

VkSurfaceFormatKHR Device::choose_swap_surface_format(
    std::vector<VkSurfaceFormatKHR> const & availableFormats)
{
    // surface has no preferred format so we can choose whatever we want
    if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
    {
        return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    }

    auto result = std::find_if(
        availableFormats.begin(), availableFormats.end(), [](auto const & format) {
            return format.format == VK_FORMAT_B8G8R8A8_UNORM
                   && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        });

    if (result != availableFormats.end())
    {
        return *result;
    }
    else
    {
        return availableFormats[0];
    }
}

VkPresentModeKHR Device::choose_swap_present_mode(
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

VkExtent2D Device::choose_swap_extent(VkSurfaceCapabilitiesKHR const & capabilities)
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

        LOG_INFO("ActualExtent {} {}", width, height);

        actualExtent.width = std::max(
            capabilities.minImageExtent.width,
            std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(
            capabilities.minImageExtent.height,
            std::min(capabilities.maxImageExtent.height, actualExtent.height));

        LOG_INFO("ActualExtent {} {}", width, height);

        return actualExtent;
    }
}

void Device::get_swapchain_images()
{
    vkGetSwapchainImagesKHR(logical_device, swapchain, &swapchain_image_count, nullptr);
    assert(swapchain_image_count == imageCount);
    assert(swapchain_image_count > 0);
    swapchain_images.resize(swapchain_image_count);
    vkGetSwapchainImagesKHR(
        logical_device, swapchain, &swapchain_image_count, swapchain_images.data());
}

ErrorCode Device::create_swapchain_image_views()
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

std::optional<VkFormat> Device::find_depth_format()
{
    return find_supported_format(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

std::optional<VkFormat> Device::find_supported_format(const std::vector<VkFormat> & candidates,
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

bool FrameResources::init(RenderConfig & render_config, VkDevice const & device)
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

void FrameResources::quit(VkDevice const & device)
{
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
        vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
        vkDestroyFence(device, in_flight_fences[i], nullptr);
    }
}

bool ImageResources::init(RenderConfig & render_config, Device const & device)
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

    return create_attachments(device) == ErrorCode::NONE;
}

void ImageResources::quit(Device const & device)
{
    attachment_handles.clear();

    for (auto & sampler_iter: samplers)
    {
        sampler_iter.second.destroy(device.get_logical_device());
    }

    samplers.clear();
}

ErrorCode ImageResources::recreate_attachments(Device const & device)
{
    destroy_attachments(device);
    return create_attachments(device);
}

ErrorCode ImageResources::create_attachments(Device const & device)
{
    for (size_t i = 0; i < attachment_configs.size(); ++i)
    {
        auto const & attachment_config = attachment_configs[i];

        auto error = create_attachment(device, attachment_config, attachments[i]);

        if (error != ErrorCode::NONE)
        {
            return error;
        }
    }

    return ErrorCode::NONE;
}

ErrorCode ImageResources::create_attachment(Device const &           device,
                                            AttachmentConfig const & attachment_config,
                                            TextureHandle &          attachment)
{
    if (attachment_config.is_swapchain_image)
    {
        return ErrorCode::NONE;
    }

    VkFormat           format;
    VkImageUsageFlags  usage = attachment_config.usage;
    VkImageAspectFlags aspect;
    VkImageLayout      final_layout;
    VkExtent2D         extent;

    if (attachment_config.use_swapchain_size)
    {
        extent = device.get_extent();
    }
    else
    {
        extent = attachment_config.extent;
    }

    if (attachment_config.format == Format::USE_COLOR)
    {
        format = device.get_color_format();
        usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        // final_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // subpass dependencies handle
        // this
    }
    else if (attachment_config.format == Format::USE_DEPTH)
    {
        format = device.get_depth_format();
        usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        // final_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // subpass dependencies
        // handle this
    }

    auto samples = std::min(attachment_config.multisamples, device.get_max_msaa_samples());

    auto opt_handle = create_texture(device.get_physical_device(),
                                     device.get_logical_device(),
                                     extent.width,
                                     extent.height,
                                     1,
                                     samples,
                                     format,
                                     VK_IMAGE_TILING_OPTIMAL,
                                     usage,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                     aspect);

    if (!opt_handle)
    {
        return ErrorCode::API_ERROR;
    }

    attachment = opt_handle.value();

    return ErrorCode::NONE;
}

void ImageResources::destroy_attachments(Device const & device)
{
    for (TextureHandle attachment: attachments)
    {
        samplers[attachment].destroy(device.get_logical_device());
    }
}

std::optional<TextureHandle> ImageResources::create_texture(
    VkPhysicalDevice const & physical_device,
    VkDevice const &         logical_device,
    uint32_t                 width,
    uint32_t                 height,
    uint32_t                 mipLevels,
    VkSampleCountFlagBits    numSamples,
    VkFormat                 format,
    VkImageTiling            tiling,
    VkImageUsageFlags        usage,
    VkMemoryPropertyFlags    properties,
    VkImageAspectFlags       aspectFlags)
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

std::optional<Sampler> ImageResources::get_texture(TextureHandle const & handle) const
{
    auto sampler_iter = samplers.find(handle);

    if (sampler_iter != samplers.end())
    {
        return sampler_iter->second;
    }

    return std::nullopt;
}

void ImageResources::delete_texture(TextureHandle const & handle)
{
    auto sampler_iter = samplers.find(handle);

    if (sampler_iter != samplers.end())
    {
        samplers.erase(sampler_iter);
    }
}

std::optional<AttachmentConfig> ImageResources::get_config(AttachmentHandle const & handle) const
{
    if (handle >= attachment_configs.size())
    {
        return std::nullopt;
    }

    return attachment_configs[handle];
}

std::optional<TextureHandle> ImageResources::get_texture_handle(
    AttachmentHandle const & handle) const
{
    if (handle >= attachments.size())
    {
        return std::nullopt;
    }

    return attachments[handle];
}

std::optional<AttachmentHandle> ImageResources::get_attachment_handle(
    std::string const & name) const
{
    auto iter = attachment_handles.find(name);
    if (iter == attachment_handles.end())
    {
        LOG_DEBUG("Didn't find AttachmentHandle for Attachment {}", name);
        return std::nullopt;
    }

    return iter->second;
}

bool RenderPassResources::init(RenderConfig &         render_config,
                               Device const &         device,
                               ImageResources const & image_resources)
{
    for (auto & iter: render_config.render_pass_configs)
    {
        RenderPassHandle handle         = render_pass_configs.size();
        render_pass_handles[iter.first] = render_pass_configs.size();
        render_pass_configs.push_back(iter.second);

        LOG_DEBUG("Added Render Pass Handle {} for Render Pass {}", handle, iter.first);
    }

    for (auto & iter: render_config.render_pass_order)
    {
        LOG_DEBUG("Added Render Pass {} to draw ordering", iter);
        auto rp_handle = render_pass_handles[iter];
        auto sp_count  = render_pass_configs[rp_handle].subpasses.size();

        render_pass_order.push_back(rp_handle);
        per_render_pass_subpass_pipelines[rp_handle].resize(sp_count);
    }

    render_passes.resize(render_pass_configs.size());
    framebuffers.resize(render_pass_configs.size());
    clear_values.resize(render_pass_configs.size());
    samples.resize(render_pass_configs.size());
    extents.resize(render_pass_configs.size());

    return createRenderPasses(device, image_resources) == ErrorCode::NONE;
}

void RenderPassResources::quit(Device const & device)
{
    for (auto & buffered_framebuffers: framebuffers)
    {
        for (auto & framebuffer: buffered_framebuffers)
        {
            vkDestroyFramebuffer(device.get_logical_device(), framebuffer, nullptr);
        }
    }

    for (auto & render_pass: render_passes)
    {
        vkDestroyRenderPass(device.get_logical_device(), render_pass, nullptr);
    }
}

void RenderPassResources::recreate_framebuffers(Device const &         device,
                                                ImageResources const & image_resources)
{
    for (size_t fb_i = 0; fb_i < framebuffers.size(); ++fb_i)
    {
        auto & render_pass_config = render_pass_configs[fb_i];
        auto & render_pass        = render_passes[fb_i];
        auto & extent             = extents[fb_i];

        for (auto & framebuffer: framebuffers[fb_i])
        {
            vkDestroyFramebuffer(device.get_logical_device(), framebuffer, nullptr);
        }

        createFramebuffer(
            device, image_resources, render_pass_config, render_pass, framebuffers[fb_i], extent);
    }
}

ErrorCode RenderPassResources::createRenderPasses(Device const &         device,
                                                  ImageResources const & image_resources)
{
    for (size_t rp_i = 0; rp_i < render_passes.size(); ++rp_i)
    {
        auto & render_pass_config = render_pass_configs[rp_i];
        auto & render_pass        = render_passes[rp_i];
        auto & extent             = extents[rp_i];

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
                description.format = device.get_color_format();
            }
            else if (attachment_config.format == Format::USE_DEPTH)
            {
                description.format = device.get_depth_format();
            }

            description.samples = std::min(attachment_config.multisamples,
                                           device.get_max_msaa_samples());

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
            subpasses.push_back(
                VkSubpassDescription{.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     .colorAttachmentCount = static_cast<uint32_t>(
                                         subpass_info.color_attachments.size()),
                                     .pColorAttachments = subpass_info.color_attachments.data(),
                                     .pDepthStencilAttachment = nullptr});

            if (subpass_info.depth_stencil_attachment)
            {
                subpasses.back().pDepthStencilAttachment
                    = &subpass_info.depth_stencil_attachment.value();
            }

            if (subpass_info.color_resolve_attachments.size() > 0)
            {
                subpasses.back().pResolveAttachments
                    = subpass_info.color_resolve_attachments.data();
            }

            if (subpass_info.preserve_attachments.size() > 0)
            {
                subpasses.back().preserveAttachmentCount = subpass_info.preserve_attachments.size();
                subpasses.back().pPreserveAttachments    = subpass_info.preserve_attachments.data();
            }

            if (subpass_info.input_attachments.size() > 0)
            {
                subpasses.back().inputAttachmentCount = subpass_info.input_attachments.size();
                subpasses.back().pInputAttachments    = subpass_info.input_attachments.data();
            }
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
            vkCreateRenderPass(device.get_logical_device(), &renderPassInfo, nullptr, &render_pass),
            "Unable to create VkRenderPass");

        auto error = createFramebuffer(
            device, image_resources, render_pass_config, render_pass, framebuffers[rp_i], extent);
        if (error != ErrorCode::NONE)
        {
            return error;
        }
    }

    return ErrorCode::NONE;
}

// FRAMEBUFFER
ErrorCode RenderPassResources::createFramebuffer(Device const &               device,
                                                 ImageResources const &       image_resources,
                                                 RenderPassConfig const &     config,
                                                 VkRenderPass const &         render_pass,
                                                 std::vector<VkFramebuffer> & framebuffers,
                                                 VkExtent2D &                 extent)
{
    framebuffers.resize(device.get_image_count());

    for (size_t i = 0; i < device.get_image_count(); ++i)
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
                fb_attachments[attachment_index] = device.get_swapchain_image_view(i);
                extent                           = device.get_extent();
            }
            else
            {
                if (attachment_config.use_swapchain_size)
                {
                    extent = device.get_extent();
                }
                else
                {
                    extent = attachment_config.extent;
                }

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
            .width           = extent.width,
            .height          = extent.height,
            .layers          = 1};

        VK_CHECK_RESULT(vkCreateFramebuffer(
                            device.get_logical_device(), &framebufferInfo, nullptr, &framebuffer),
                        "Unable to create VkFramebuffer");
    }

    return ErrorCode::NONE;
}

bool UniformResources::init(RenderConfig & render_config, Device const & device)
{
    uniform_layouts.reserve(render_config.uniform_sets.size());
    uniform_sets.reserve(render_config.uniform_sets.size());

    for (auto const & iter: render_config.uniform_sets)
    {
        auto & name          = iter.first;
        auto & binding_names = iter.second;

        VkDescriptorSetLayout uniform_layout;

        // Get all vkDescriptorSetLayoutBindings for this layout
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (auto const & binding_name: binding_names)
        {
            auto const & binding_iter = render_config.uniform_bindings.find(binding_name);
            if (binding_iter == render_config.uniform_bindings.end())
            {
                LOG_ERROR(
                    "Couldn't find Uniform Binding {} for Uniform Set {}", binding_name, name);
                return false;
            }

            bindings.push_back(binding_iter->second);
        }

        // create layout
        UniformSetHandle handle   = uniform_layouts.size();
        uniform_set_handles[name] = handle;

        if (create_uniform_layout(device, uniform_layout, bindings) != ErrorCode::NONE)
        {
            LOG_DEBUG("Unable to create VkDescriptorSetLayout for set {}", name);
            return false;
        }

        uniform_layouts.push_back(uniform_layout);

        // create a pool

        uniform_sets.emplace_back();
        create_uniform_set(device, uniform_layouts.back(), uniform_sets.back(), bindings);
    }

    return true;
}

void UniformResources::quit(Device const & device)
{
    for (auto & uniform_layout: uniform_layouts)
    {
        vkDestroyDescriptorSetLayout(device.get_logical_device(), uniform_layout, nullptr);
    }

    for (auto & uniform_set: uniform_sets)
    {
        for (auto & pool: uniform_set.pools)
        {
            vkDestroyDescriptorPool(device.get_logical_device(), pool, nullptr);
        }

        uniform_set.pools.clear();
        uniform_set.uniforms.clear();
        uniform_set.free_descriptor_sets.init(0);
    }
}

std::optional<UniformHandle> UniformResources::create_uniform(UniformSetHandle const & set_handle)
{
    if (check_valid_set(set_handle) != ErrorCode::NONE)
    {
        LOG_DEBUG("UniformHandle passed to create_uniform was not valid");
        return std::nullopt;
    }

    UniformSet & uniform_set = uniform_sets[set_handle];

    auto descriptor_index = uniform_set.free_descriptor_sets.acquire();
    if (descriptor_index < 0)
    {
        LOG_WARN("Need to allocate more descriptor sets");
        return std::nullopt;
    }

    auto generation = ++uniform_set.generations[descriptor_index];

    return UniformHandle{.set        = set_handle,
                         .generation = generation,
                         .uniform    = static_cast<uint16_t>(descriptor_index)};
}

void UniformResources::update_uniform(Device &              device,
                                      UniformHandle const & uniform_handle,
                                      BufferResources &     buffers,
                                      ImageResources &      images,
                                      size_t                uniform_write_count,
                                      UniformWrite *        uniform_write_infos)
{
    check_valid_set(uniform_handle.set);
    check_valid_uniform(uniform_handle);

    assert(uniform_write_count != 0 && uniform_write_infos != nullptr);

    auto uniform_set = uniform_sets[uniform_handle.set];

    assert(uniform_write_count == uniform_set.bindings.size());

    std::vector<VkWriteDescriptorSet>   writes;
    std::vector<VkDescriptorBufferInfo> buffer_writes;
    std::vector<VkDescriptorImageInfo>  image_writes;

    auto descriptor_set = uniform_set.uniforms[uniform_handle.uniform];

    for (size_t write_info_idx = 0; write_info_idx < uniform_write_count; write_info_idx++)
    {
        auto & write_info = uniform_write_infos[write_info_idx];
        auto & binding    = uniform_set.bindings[write_info_idx];

        assert(write_info.buffer_write_count == 0 || write_info.image_write_count == 0);
        assert(write_info.buffer_write_count == binding.descriptorCount
               || write_info.image_write_count == binding.descriptorCount);

        if (write_info.buffer_write_count != 0)
        {
            size_t write_offset = buffer_writes.size();

            for (size_t buffer_write_idx = 0; buffer_write_idx < write_info.buffer_write_count;
                 buffer_write_idx++)
            {
                auto & buffer_write = write_info.buffer_writes[buffer_write_idx];

                auto opt_buffer = buffers.get_buffer(buffer_write.buffer);
                if (!opt_buffer)
                {
                    LOG_ERROR("Tried to write to Uniform Set with invalid buffer");
                    return;
                }

                VkDescriptorBufferInfo buffer_info{};
                buffer_info.buffer = opt_buffer.value().buffer_handle();
                buffer_info.offset = buffer_write.offset;
                buffer_info.range  = buffer_write.size;

                buffer_writes.push_back(buffer_info);
            }

            VkWriteDescriptorSet descriptor_write{};
            descriptor_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write.dstSet          = descriptor_set;
            descriptor_write.dstArrayElement = 0;
            descriptor_write.dstBinding      = binding.binding;
            descriptor_write.descriptorCount = binding.descriptorCount;
            descriptor_write.descriptorType  = binding.descriptorType;
            descriptor_write.pBufferInfo     = &buffer_writes[write_offset];

            writes.push_back(descriptor_write);
        }
        else if (write_info.image_write_count != 0)
        {
            size_t write_offset = image_writes.size();

            for (size_t image_write_idx = 0; image_write_idx < write_info.image_write_count;
                 image_write_idx++)
            {
                auto & image_write = write_info.image_writes[image_write_idx];

                auto opt_sampler = images.get_texture(image_write.texture);
                if (!opt_sampler)
                {
                    LOG_ERROR("Tried to write to Uniform Set with invalid texture");
                    return;
                }

                VkDescriptorImageInfo image_info{};
                image_info.sampler     = opt_sampler.value().sampler_handle();
                image_info.imageView   = opt_sampler.value().view_handle();
                image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_writes.push_back(image_info);
            }

            VkWriteDescriptorSet descriptor_write{};
            descriptor_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write.dstSet          = descriptor_set;
            descriptor_write.dstArrayElement = 0;
            descriptor_write.dstBinding      = binding.binding;
            descriptor_write.descriptorCount = binding.descriptorCount;
            descriptor_write.descriptorType  = binding.descriptorType;
            descriptor_write.pImageInfo      = &image_writes[write_offset];

            writes.push_back(descriptor_write);
        }
    }

    vkUpdateDescriptorSets(device.get_logical_device(), writes.size(), writes.data(), 0, nullptr);
}

std::optional<VkDescriptorSet> UniformResources::get_uniform(UniformHandle const & handle)
{
    if (check_valid_set(handle.set) != ErrorCode::NONE
        || check_valid_uniform(handle) != ErrorCode::NONE)
    {
        return std::nullopt;
    }

    return uniform_sets[handle.set].uniforms[handle.uniform];
}

void UniformResources::delete_uniform(UniformHandle const & handle)
{
    if (check_valid_set(handle.set) != ErrorCode::NONE
        || check_valid_uniform(handle) != ErrorCode::NONE)
    {
        return;
    }

    auto & uniform_set = uniform_sets[handle.set];

    uniform_set.free_descriptor_sets.release(handle.uniform);
}

std::optional<VkDescriptorSetLayout> UniformResources::get_layout(std::string const & name)
{
    auto iter = uniform_set_handles.find(name);
    if (iter == uniform_set_handles.end())
    {
        return std::nullopt;
    }

    return uniform_layouts[iter->second];
}

ErrorCode UniformResources::create_uniform_layout(
    Device const &                                    device,
    VkDescriptorSetLayout &                           layout,
    std::vector<VkDescriptorSetLayoutBinding> const & bindings)
{
    auto layoutInfo = VkDescriptorSetLayoutCreateInfo{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings    = bindings.data()};

    VK_CHECK_RESULT(
        vkCreateDescriptorSetLayout(device.get_logical_device(), &layoutInfo, nullptr, &layout),
        "Unable to create VkDescriptorSetLayout");

    return ErrorCode::NONE;
}

ErrorCode UniformResources::create_uniform_set(Device const &                device,
                                               VkDescriptorSetLayout const & layout,
                                               UniformSet &                  uniform_set,
                                               std::vector<VkDescriptorSetLayoutBinding> & bindings)
{
    uniform_set.bindings = std::move(bindings);

    uniform_set.pools.push_back(VK_NULL_HANDLE);

    // create pool
    auto error = create_pool(device, uniform_set.pools.back(), uniform_set.bindings);
    if (error != ErrorCode::NONE)
    {
        return error;
    }

    // allocate descriptor sets
    uniform_set.uniforms.resize(descriptors_per_pool);
    error = allocate_descriptor_sets(
        device, layout, uniform_set.pools.back(), uniform_set.uniforms.data());
    if (error != ErrorCode::NONE)
    {
        return error;
    }

    uniform_set.free_descriptor_sets.init(descriptors_per_pool);
    uniform_set.generations.resize(descriptors_per_pool, 0);

    return ErrorCode::NONE;
}

ErrorCode UniformResources::create_pool(Device const &                                    device,
                                        VkDescriptorPool &                                pool,
                                        std::vector<VkDescriptorSetLayoutBinding> const & bindings)
{
    std::vector<VkDescriptorPoolSize> pool_sizes{bindings.size()};

    for (size_t i = 0; i < pool_sizes.size(); ++i)
    {
        pool_sizes[i].type            = bindings[i].descriptorType;
        pool_sizes[i].descriptorCount = descriptors_per_pool;
    }

    auto poolInfo = VkDescriptorPoolCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
        .pPoolSizes    = pool_sizes.data(),
        .maxSets       = descriptors_per_pool};

    VK_CHECK_RESULT(vkCreateDescriptorPool(device.get_logical_device(), &poolInfo, nullptr, &pool),
                    "Unable to create VkDescriptorPool");

    return ErrorCode::NONE;
}

ErrorCode UniformResources::allocate_descriptor_sets(Device const &                device,
                                                     VkDescriptorSetLayout const & layout,
                                                     VkDescriptorPool &            pool,
                                                     VkDescriptorSet *             descriptor_sets)
{
    std::vector<VkDescriptorSetLayout> layouts{descriptors_per_pool, layout};

    auto allocInfo = VkDescriptorSetAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = pool,
        .descriptorSetCount = descriptors_per_pool,
        .pSetLayouts        = layouts.data()};

    VK_CHECK_RESULT(
        vkAllocateDescriptorSets(device.get_logical_device(), &allocInfo, descriptor_sets),
        "Unable to allocate VkDescriptorSets");

    return ErrorCode::NONE;
}

ErrorCode UniformResources::check_valid_set(UniformSetHandle const & set_handle)
{
    assert(set_handle < uniform_sets.size());
    if (set_handle > uniform_sets.size())
    {
        LOG_ERROR("UniformSetHandle was not valid");
        return ErrorCode::API_ERROR;
    }
    return ErrorCode::NONE;
}

ErrorCode UniformResources::check_valid_uniform(UniformHandle const & uniform_handle)
{
    auto & uniform_set = uniform_sets[uniform_handle.set];

    assert(uniform_handle.uniform < uniform_set.uniforms.size()
           && uniform_handle.generation == uniform_set.generations[uniform_handle.uniform]);

    if (uniform_handle.uniform > uniform_set.uniforms.size()
        || uniform_handle.generation != uniform_set.generations[uniform_handle.uniform])
    {
        LOG_ERROR("UniformHandle was not valid");
        return ErrorCode::API_ERROR;
    }

    return ErrorCode::NONE;
}

bool PipelineResources::init(RenderConfig &        render_config,
                             Device const &        device,
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

    if (create_shaders(device, render_config.read_file) != ErrorCode::NONE)
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
        draw_buckets.push_back({}); // push an empty vector<commandbucket>
        draw_buckets.back().emplace_back(max_calls_per_bucket);

        LOG_DEBUG("Added Pipeline Handle {} for Pipeline {}", pipeline_handle, iter.first);
    }
    pipelines.resize(pipeline_configs.size());

    return create_pipelines(device, render_passes, uniforms) == ErrorCode::NONE;
}

void PipelineResources::quit(Device const & device)
{
    for (auto & shader: shaders)
    {
        vkDestroyShaderModule(device.get_logical_device(), shader, nullptr);
    }

    destroy_pipelines(device);
}

void PipelineResources::destroy_pipelines(Device const & device)
{
    for (auto & pipeline: pipelines)
    {
        vkDestroyPipeline(device.get_logical_device(), pipeline.vk_pipeline, nullptr);
        vkDestroyPipelineLayout(device.get_logical_device(), pipeline.vk_pipeline_layout, nullptr);
    }
}

ErrorCode PipelineResources::create_shader_module(Device const &            device,
                                                  std::vector<char> const & code,
                                                  VkShaderModule &          shaderModule)
{
    auto createInfo = VkShaderModuleCreateInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode    = reinterpret_cast<const uint32_t *>(code.data())};

    VK_CHECK_RESULT(
        vkCreateShaderModule(device.get_logical_device(), &createInfo, nullptr, &shaderModule),
        "Unable to create VkShaderModule");

    return ErrorCode::NONE;
}

ErrorCode PipelineResources::create_shaders(Device const & device, ReadFileFn read_file)
{
    for (size_t i = 0; i < shaders.size(); ++i)
    {
        auto & shader = shaders[i];

        std::vector<char> shader_code;

        read_file(shader_files[i].c_str(), shader_code);

        // auto shaderCode = readFile(shader_files[i]);

        auto error = create_shader_module(device, shader_code, shader);

        if (error != ErrorCode::NONE)
        {
            return error;
        }
    }

    return ErrorCode::NONE;
}

ErrorCode PipelineResources::recreate_pipelines(Device const &        device,
                                                RenderPassResources & render_passes,
                                                UniformResources &    uniforms)
{
    destroy_pipelines(device);
    return create_pipelines(device, render_passes, uniforms);
}

ErrorCode PipelineResources::create_pipelines(Device const &        device,
                                              RenderPassResources & render_passes,
                                              UniformResources &    uniforms)
{
    for (size_t i = 0; i < pipelines.size(); ++i)
    {
        PipelineHandle pipeline_handle = i;
        auto &         pipeline        = pipelines[pipeline_handle];
        auto &         pipeline_config = pipeline_configs[pipeline_handle];

        create_pipeline(
            device, render_passes, uniforms, pipeline_handle, pipeline, pipeline_config);
    }

    return ErrorCode::NONE;
}

ErrorCode PipelineResources::create_pipeline(Device const &         device,
                                             RenderPassResources &  render_passes,
                                             UniformResources &     uniforms,
                                             PipelineHandle         pipeline_handle,
                                             Pipeline &             pipeline,
                                             PipelineConfig const & pipeline_config)
{
    auto   render_pass_handle = render_passes.render_pass_handles[pipeline_config.render_pass];
    auto & render_pass_config = render_passes.render_pass_configs[render_pass_handle];
    auto   subpass_handle     = render_pass_config.subpass_handles[pipeline_config.subpass];
    auto const & subpass_info = render_pass_config.subpasses[subpass_handle];
    auto const & extent       = render_passes.extents[render_pass_handle];

    LOG_DEBUG("Pipeline {} uses fragment shader {}, handle {}",
              pipeline_handle,
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
        .sType                         = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size()),
        .pVertexBindingDescriptions    = bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
        .pVertexAttributeDescriptions    = attributes.data()};

    auto inputAssembly = VkPipelineInputAssemblyStateCreateInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE};

    auto viewport = VkViewport{.x        = 0.0f,
                               .y        = 0.0f,
                               .width    = static_cast<float>(extent.width),
                               .height   = static_cast<float>(extent.height),
                               .minDepth = 0.0f,
                               .maxDepth = 1.0f};

    auto scissor = VkRect2D{.offset = {0, 0}, .extent = device.get_extent()};

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

    auto samples = std::min(subpass_info.multisamples, device.get_max_msaa_samples());

    auto multisampling = VkPipelineMultisampleStateCreateInfo{
        .sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .sampleShadingEnable   = VK_TRUE,
        .rasterizationSamples  = samples,
        .minSampleShading      = 0.2f,     // Optional
        .pSampleMask           = nullptr,  // Optional
        .alphaToCoverageEnable = VK_FALSE, // Optional
        .alphaToOneEnable      = VK_FALSE  // Optional
    };

    auto depthStencil = VkPipelineDepthStencilStateCreateInfo{
        .sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable       = pipeline_config.tests_depth,
        .depthWriteEnable      = VK_TRUE,
        .depthCompareOp        = VK_COMPARE_OP_LESS_OR_EQUAL,
        .depthBoundsTestEnable = VK_FALSE,
        .minDepthBounds        = 0.0f,
        .maxDepthBounds        = 1.0f,
        .stencilTestEnable     = VK_FALSE,
        .front                 = {},
        .back                  = {}};

    auto colorBlendAttachment = VkPipelineColorBlendAttachmentState{
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                          | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        .blendEnable         = pipeline_config.blendable,
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

    auto dynamicState = VkPipelineDynamicStateCreateInfo{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(pipeline_config.dynamic_state.size()),
        .pDynamicStates    = pipeline_config.dynamic_state.data()};

    auto pushConstantRanges = std::vector<VkPushConstantRange>{};
    for (auto const & push_constant_name: pipeline_config.push_constant_names)
    {
        auto handle = push_constant_handles[push_constant_name];

        pipeline.push_constant_stages = pipeline.push_constant_stages
                                        | push_constants[handle].stageFlags;

        pushConstantRanges.push_back(push_constants[handle]);
    }

    std::vector<VkDescriptorSetLayout> layouts;

    for (auto & layout_name: pipeline_config.uniform_layout_names)
    {
        auto opt_layout = uniforms.get_layout(layout_name);
        if (!opt_layout)
        {
            LOG_ERROR("Couldn't get Layout {}", layout_name);
            return ErrorCode::JSON_ERROR;
        }

        layouts.push_back(opt_layout.value());
    }

    auto pipelineLayoutInfo = VkPipelineLayoutCreateInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts            = layouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size()),
        .pPushConstantRanges    = pushConstantRanges.data()};

    VK_CHECK_RESULT(vkCreatePipelineLayout(device.get_logical_device(),
                                           &pipelineLayoutInfo,
                                           nullptr,
                                           &pipeline.vk_pipeline_layout),
                    "Unable to create VkPipelineLayout");

    // push this pipeline handle into the map of commandbuckets

    LOG_DEBUG("Adding Pipeline {} to Render Pass {} at Sub Pass {}",
              pipeline_handle,
              render_pass_handle,
              subpass_handle);

    render_passes.per_render_pass_subpass_pipelines[render_pass_handle][subpass_handle].push_back(
        pipeline_handle);

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
        .pDynamicState       = &dynamicState, // Optional
        .layout              = pipeline.vk_pipeline_layout,
        .renderPass          = render_passes.render_passes[render_pass_handle],
        .subpass             = static_cast<uint32_t>(subpass_handle),
        .basePipelineHandle  = VK_NULL_HANDLE, // Optional
        .basePipelineIndex   = -1              // Optional
    };

    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device.get_logical_device(),
                                              VK_NULL_HANDLE,
                                              1,
                                              &pipelineInfo,
                                              nullptr,
                                              &pipeline.vk_pipeline),
                    "Unable to create VkPipeline");

    return ErrorCode::NONE;
}

cmd::CommandBucket<int> & PipelineResources::get_draw_bucket(PipelineHandle const & pipeline)
{
    auto & pipeline_draw_buckets = draw_buckets[pipeline];

    for (auto & draw_bucket: pipeline_draw_buckets)
    {
        if (draw_bucket.size() < draw_bucket.capacity())
        {
            return draw_bucket;
        }
    }

    LOG_DEBUG("Creating a new bucket for pipeline {} with capacity {}",
              pipeline,
              max_calls_per_bucket * (pipeline_draw_buckets.size() + 1));

    pipeline_draw_buckets.emplace_back(max_calls_per_bucket * (pipeline_draw_buckets.size() + 1));
    return pipeline_draw_buckets.back();
}

bool CommandResources::init(RenderConfig & render_config, Device const & device)
{
    int32_t const MAX_BUFFERED_RESOURCES = 3;

    for (uint32_t i = 0; i < MAX_BUFFERED_RESOURCES; ++i)
    {
        transfer_buckets.push_back({});
        transfer_buckets.back().emplace_back(max_calls_per_bucket);
        delete_buckets.push_back({});
        delete_buckets.back().emplace_back(max_calls_per_bucket);
    }

    get_queues(device);

    if (create_command_pool(device) != ErrorCode::NONE)
    {
        return false;
    }

    return create_command_buffers(device) == ErrorCode::NONE;
}

void CommandResources::quit(Device const & device)
{
    for (auto & frame_delete_buckets: delete_buckets)
    {
        for (auto & delete_bucket: frame_delete_buckets)
        {
            delete_bucket.Submit();
            delete_bucket.Clear();
        }
    }

    transfer_buckets.clear();

    vkDestroyCommandPool(device.get_logical_device(), command_pool, nullptr);
}

cmd::CommandBucket<int> & CommandResources::get_transfer_bucket(uint32_t currentResource)
{
    auto & resource_transfer_buckets = transfer_buckets[currentResource];

    for (auto & transfer_bucket: resource_transfer_buckets)
    {
        if (transfer_bucket.size() < transfer_bucket.capacity())
        {
            return transfer_bucket;
        }
    }

    LOG_DEBUG("Creating a new transfer bucket for resource frame {} with capacity {}",
              currentResource,
              max_calls_per_bucket * (resource_transfer_buckets.size() + 1));

    resource_transfer_buckets.emplace_back(max_calls_per_bucket
                                           * (resource_transfer_buckets.size() + 1));
    return resource_transfer_buckets.back();
}

cmd::CommandBucket<int> & CommandResources::get_delete_bucket(uint32_t currentResource)
{
    auto & resource_delete_buckets = delete_buckets[currentResource];

    for (auto & delete_bucket: resource_delete_buckets)
    {
        if (delete_bucket.size() < delete_bucket.capacity())
        {
            return delete_bucket;
        }
    }

    LOG_DEBUG("Creating a new delete bucket for resource frame {} with capacity {}",
              currentResource,
              max_calls_per_bucket * (resource_delete_buckets.size() + 1));

    resource_delete_buckets.emplace_back(max_calls_per_bucket
                                         * (resource_delete_buckets.size() + 1));
    return resource_delete_buckets.back();
}

void CommandResources::get_queues(Device const & device)
{
    vkGetDeviceQueue(
        device.get_logical_device(), device.get_device_info().present_queue, 0, &present_queue);
    vkGetDeviceQueue(
        device.get_logical_device(), device.get_device_info().graphics_queue, 0, &graphics_queue);
    vkGetDeviceQueue(
        device.get_logical_device(), device.get_device_info().transfer_queue, 0, &transfer_queue);
}

ErrorCode CommandResources::create_command_pool(Device const & device)
{
    auto poolInfo = VkCommandPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = static_cast<uint32_t>(device.get_device_info().graphics_queue),
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};

    VK_CHECK_RESULT(
        vkCreateCommandPool(device.get_logical_device(), &poolInfo, nullptr, &command_pool),
        "Unable to create VkCommandPool");

    return ErrorCode::NONE;
}

ErrorCode CommandResources::create_command_buffers(Device const & device)
{
    int32_t const MAX_BUFFERED_RESOURCES = 3;

    draw_commandbuffers.resize(MAX_BUFFERED_RESOURCES);

    auto allocInfo = VkCommandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = command_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = (uint32_t)draw_commandbuffers.size()};

    VK_CHECK_RESULT(vkAllocateCommandBuffers(
                        device.get_logical_device(), &allocInfo, draw_commandbuffers.data()),
                    "Unable to allocate VkCommandBuffer");

    transfer_commandbuffers.resize(MAX_BUFFERED_RESOURCES);

    allocInfo = VkCommandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = command_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = (uint32_t)transfer_commandbuffers.size()};

    VK_CHECK_RESULT(vkAllocateCommandBuffers(
                        device.get_logical_device(), &allocInfo, transfer_commandbuffers.data()),
                    "Unable to allocate VkCommandBuffer");

    return ErrorCode::NONE;
}

bool BufferResources::init()
{
    return true;
}

void BufferResources::quit(Device const & device)
{
    for (auto & buffer_iter: buffers)
    {
        buffer_iter.second.destroy(device.get_logical_device());
    }

    buffers.clear();
}

std::optional<BufferHandle> BufferResources::create_buffer(Device const &        device,
                                                           VkDeviceSize          size,
                                                           VkBufferUsageFlags    usage,
                                                           VkMemoryPropertyFlags properties)
{
    BufferHandle handle = next_buffer_handle++;

    Buffer & buffer = buffers[handle];

    if (buffer.create(
            device.get_physical_device(), device.get_logical_device(), size, usage, properties)
        != ErrorCode::NONE)
    {
        LOG_ERROR("Couldn't create buffer, cleaning up..");
        buffer.destroy(device.get_logical_device());
        return std::nullopt;
    }

    LOG_DEBUG("Created buffer {} {}",
              static_cast<void *>(buffer.buffer_handle()),
              static_cast<void *>(buffer.memory_handle()));

    if (properties & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
    {
        buffer.map(device.get_logical_device(), 0, size, &mapped_memory[handle]);
    }

    return handle;
}

std::optional<void *> BufferResources::map_buffer(BufferHandle const & handle) const
{
    auto mapped_memory_iter = mapped_memory.find(handle);

    if (mapped_memory_iter != mapped_memory.end())
    {
        return mapped_memory_iter->second;
    }

    return std::nullopt;
}

std::optional<Buffer> BufferResources::get_buffer(BufferHandle const & handle) const
{
    auto buffer_iter = buffers.find(handle);

    if (buffer_iter != buffers.end())
    {
        return buffer_iter->second;
    }

    return std::nullopt;
}

void BufferResources::delete_buffer(BufferHandle const & handle)
{
    auto buffer_iter = buffers.find(handle);

    if (buffer_iter != buffers.end())
    {
        buffers.erase(buffer_iter);
    }

    auto mapped_memory_iter = mapped_memory.find(handle);

    if (mapped_memory_iter != mapped_memory.end())
    {
        mapped_memory.erase(mapped_memory_iter);
    }
}

}; // namespace module

Renderer::Renderer(GLFWwindow * window_ptr): device{window_ptr}
{}

bool Renderer::init(RenderConfig & render_config)
{
    LOG_INFO("Initializing Renderer");

    if (!device.init(render_config))
    {
        LOG_ERROR("Failed to initialize RenderDevice in Renderer");
        return false;
    }

    if (!frames.init(render_config, device.get_logical_device()))
    {
        LOG_ERROR("Failed to initialize FrameResources in Renderer");
        return false;
    }

    if (!images.init(render_config, device))
    {
        LOG_ERROR("Failed to initialize ImageResources in Renderer");
        return false;
    }

    if (!render_passes.init(render_config, device, images))
    {
        LOG_ERROR("Failed to initialize RenderPassResources in Renderer");
        return false;
    }

    if (!buffers.init())
    {
        LOG_ERROR("Failed to initialize BufferResources in Renderer");
        return false;
    }

    if (!uniforms.init(render_config, device))
    {
        LOG_ERROR("Failed to initialize UniformResources in Renderer");
        return false;
    }

    if (!pipelines.init(render_config, device, render_passes, uniforms))
    {
        LOG_ERROR("Failed to initialize PipelineResources in Renderer");
        return false;
    }

    if (!commands.init(render_config, device))
    {
        LOG_ERROR("Failed to initialize CommandResources in Renderer");
        return false;
    }

    return true;
}

void Renderer::quit()
{
    LOG_INFO("Quitting Renderer");
    commands.quit(device);
    pipelines.quit(device);
    uniforms.quit(device);
    buffers.quit(device);
    render_passes.quit(device);
    images.quit(device);
    frames.quit(device.get_logical_device());
    device.quit();
}

void Renderer::wait_for_idle()
{
    LOG_INFO("Waiting for Graphics Card to become Idle");
    vkDeviceWaitIdle(device.get_logical_device());
}

bool Renderer::submit_frame()
{
    LOG_DEBUG("Drawing frame");
    vkWaitForFences(device.get_logical_device(),
                    1,
                    &frames.in_flight_fences[frames.currentFrame],
                    VK_TRUE,
                    std::numeric_limits<uint64_t>::max());

    // DRAW OPERATIONS
    auto result = vkAcquireNextImageKHR(device.get_logical_device(),
                                        device.get_swapchain(),
                                        std::numeric_limits<uint64_t>::max(),
                                        frames.image_available_semaphores[frames.currentFrame],
                                        VK_NULL_HANDLE,
                                        &frames.currentImage);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        LOG_DEBUG("Swapchain is out of date, found in vkAcquireNextImageKHR");
        change_swapchain();

        vkAcquireNextImageKHR(device.get_logical_device(),
                              device.get_swapchain(),
                              std::numeric_limits<uint64_t>::max(),
                              frames.image_available_semaphores[frames.currentFrame],
                              VK_NULL_HANDLE,
                              &frames.currentImage);
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

    for (auto & transfer_bucket: commands.transfer_buckets[frames.currentResource])
    {
        transfer_bucket.Submit();
        transfer_bucket.Clear();
    }

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

    create_command_buffer(frames.currentImage);

    auto submitInfo = VkSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,

        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = waitSemaphores,
        .pWaitDstStageMask  = waitStages,

        .commandBufferCount = 1,
        .pCommandBuffers    = &commands.draw_commandbuffers[frames.currentResource],

        .signalSemaphoreCount = 1,
        .pSignalSemaphores    = signalSemaphores};

    vkResetFences(device.get_logical_device(), 1, &frames.in_flight_fences[frames.currentFrame]);

    if (vkQueueSubmit(
            commands.graphics_queue, 1, &submitInfo, frames.in_flight_fences[frames.currentFrame])
        != VK_SUCCESS)
    {
        LOG_ERROR("Failed to submit draw command buffer!");
        return false;
    }

    VkSwapchainKHR swapChains[] = {device.get_swapchain()};

    auto presentInfo = VkPresentInfoKHR{.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                        .waitSemaphoreCount = 1,
                                        .pWaitSemaphores    = signalSemaphores,

                                        .swapchainCount = 1,
                                        .pSwapchains    = swapChains,
                                        .pImageIndices  = &frames.currentImage,

                                        .pResults = nullptr};

    result = vkQueuePresentKHR(commands.present_queue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
    {
        LOG_DEBUG("Swapchain is out of date, found in vkQueuePresentKHR");
        return false;
    }
    else if (result != VK_SUCCESS)
    {
        LOG_ERROR("Failed to present swap chain image!");
        return false;
    }

    frames.currentFrame    = (frames.currentFrame + 1) % frames.MAX_FRAMES_IN_FLIGHT;
    frames.currentResource = (frames.currentResource + 1) % frames.MAX_BUFFERED_RESOURCES;

    for (auto & delete_bucket: commands.delete_buckets[frames.currentResource])
    {
        delete_bucket.Submit();
        delete_bucket.Clear();
    }

    return true;
}

size_t Renderer::max_buffered_resources()
{
    return frames.MAX_BUFFERED_RESOURCES;
}

size_t Renderer::current_resource_index()
{
    return frames.currentResource;
}

ErrorCode Renderer::draw(DrawParameters const & args)
{
    // get bucket
    auto & bucket = pipelines.get_draw_bucket(args.pipeline);

    // get vertex_buffers
    std::vector<VkBuffer> vk_buffers;
    vk_buffers.reserve(args.vertex_buffer_count);
    for (size_t i = 0; i < args.vertex_buffer_count; ++i)
    {
        auto opt_vertex_buffer = buffers.get_buffer(args.vertex_buffers[i]);

        if (!opt_vertex_buffer)
        {
            LOG_ERROR("Unable to get Vertex Buffer for draw call, ignoring call..");
            return ErrorCode::API_ERROR;
        }

        vk_buffers.push_back(opt_vertex_buffer.value().buffer_handle());
    }

    // get index buffers
    auto opt_index_buffer = buffers.get_buffer(args.index_buffer);

    if (!opt_index_buffer)
    {
        LOG_ERROR("Unable to get Index Buffer for draw call, ignoring call..");
        return ErrorCode::API_ERROR;
    }

    // get descriptor sets
    std::vector<VkDescriptorSet> descriptorsets;
    descriptorsets.reserve(args.uniform_count);
    for (size_t i = 0; i < args.uniform_count; ++i)
    {
        auto uniform_handle = args.uniforms[i];
        auto opt_uniform    = get_uniform(uniform_handle);

        if (opt_uniform.has_value())
        {
            descriptorsets.push_back(opt_uniform.value());
        }
        else
        {
            LOG_ERROR("No Descriptor Set returned for Uniform {} {}",
                      uniform_handle.set,
                      uniform_handle.uniform);
            return ErrorCode::API_ERROR;
        }
    }

    size_t vk_vertex_buffers_offset = 0;
    size_t vk_vertex_buffers_size   = vk_buffers.size() * sizeof(VkBuffer);

    size_t vk_vertex_buffer_offsets_offset = vk_vertex_buffers_offset + vk_vertex_buffers_size;
    size_t vk_vertex_buffer_offsets_size   = vk_buffers.size() * sizeof(VkDeviceSize);

    size_t vk_descriptorsets_offset = vk_vertex_buffer_offsets_offset
                                      + vk_vertex_buffer_offsets_size;
    size_t vk_descriptorsets_size = descriptorsets.size() * sizeof(VkDescriptorSet);

    size_t dynamic_offsets_offset = vk_descriptorsets_offset + vk_descriptorsets_size;
    size_t dynamic_offsets_size   = args.dynamic_offset_count * sizeof(uint32_t);

    size_t push_constant_offset = dynamic_offsets_offset + dynamic_offsets_size;

    size_t scissor_offset = push_constant_offset + args.push_constant_size;
    size_t scissor_size   = 0;
    if (args.scissor)
    {
        scissor_size = sizeof(VkRect2D);
    }

    size_t viewport_offset = scissor_offset + scissor_size;
    size_t viewport_size   = 0;
    if (args.viewport)
    {
        viewport_size = sizeof(VkViewport);
    }

    Draw * command = bucket.AddCommand<Draw>(
        0,
        vk_vertex_buffers_size + vk_vertex_buffer_offsets_size + vk_descriptorsets_size
            + dynamic_offsets_size + args.push_constant_size + scissor_size + viewport_size);

    assert(command != nullptr);

    char * command_memory = cmd::commandPacket::GetAuxiliaryMemory(command);

    command->commandbuffer   = commands.draw_commandbuffers[frames.currentResource];
    command->pipeline_layout = &pipelines.pipelines[args.pipeline].vk_pipeline_layout;

    // vertex buffers and offsets
    command->vertex_buffer_count   = args.vertex_buffer_count;
    command->vertex_buffer_offsets = reinterpret_cast<VkDeviceSize *>(
        command_memory + vk_vertex_buffer_offsets_offset);
    command->vertex_buffers = reinterpret_cast<VkBuffer *>(command_memory
                                                           + vk_vertex_buffers_offset);

    memcpy(command->vertex_buffers, vk_buffers.data(), vk_vertex_buffers_size);
    memcpy(
        command->vertex_buffer_offsets, args.vertex_buffer_offsets, vk_vertex_buffer_offsets_size);

    // index buffer, offset, and count
    command->index_count         = args.index_count;
    command->index_buffer_offset = args.index_buffer_offset;
    command->index_buffer        = opt_index_buffer.value().buffer_handle();

    // push_constant size and data
    command->push_constant_size  = args.push_constant_size;
    command->push_constant_data  = reinterpret_cast<void *>(command_memory + push_constant_offset);
    command->push_constant_flags = pipelines.pipelines[args.pipeline].push_constant_stages;
    memcpy(command->push_constant_data, args.push_constant_data, args.push_constant_size);

    // descriptor sets and dynamic offsets
    command->descriptor_set_count = args.uniform_count;
    command->descriptor_sets      = reinterpret_cast<VkDescriptorSet *>(command_memory
                                                                   + vk_descriptorsets_offset);
    command->dynamic_offset_count = args.dynamic_offset_count;
    command->dynamic_offsets      = reinterpret_cast<uint32_t *>(command_memory
                                                            + dynamic_offsets_offset);

    memcpy(command->descriptor_sets, descriptorsets.data(), vk_descriptorsets_size);
    memcpy(command->dynamic_offsets, args.dynamic_offsets, dynamic_offsets_size);

    // scissor
    if (args.scissor)
    {
        command->scissor = reinterpret_cast<VkRect2D *>(command_memory + scissor_offset);
        memcpy(command->scissor, args.scissor, scissor_size);
    }
    else
    {
        command->scissor = nullptr;
    }

    // viewport
    if (args.viewport)
    {
        command->viewport = reinterpret_cast<VkViewport *>(command_memory + viewport_offset);
        memcpy(command->viewport, args.viewport, viewport_size);
    }
    else
    {
        command->viewport = nullptr;
    }

    return ErrorCode::NONE;
}

std::optional<AttachmentHandle> Renderer::get_attachment_handle(std::string const & attachment_name)
{
    return images.get_attachment_handle(attachment_name);
}

std::optional<UniformSetHandle> Renderer::get_uniform_set_handle(std::string const & set_name)
{
    auto handle_iter = uniforms.uniform_set_handles.find(set_name);
    if (handle_iter == uniforms.uniform_set_handles.end())
    {
        return std::nullopt;
    }

    return handle_iter->second;
}

std::optional<PipelineHandle> Renderer::get_pipeline_handle(std::string const & pipeline_name)
{
    auto handle_iter = pipelines.pipeline_handles.find(pipeline_name);
    if (handle_iter == pipelines.pipeline_handles.end())
    {
        return std::nullopt;
    }

    return handle_iter->second;
}

std::optional<BufferHandle> Renderer::create_buffer(VkDeviceSize          size,
                                                    VkBufferUsageFlags    usage,
                                                    VkMemoryPropertyFlags properties)
{
    LOG_INFO("Creating Buffer");

    return buffers.create_buffer(device, size, usage, properties);
}

std::optional<void *> Renderer::map_buffer(BufferHandle const & buffer_handle)
{
    LOG_INFO("Mapping Buffer");

    return buffers.map_buffer(buffer_handle);
}

void Renderer::update_buffer(BufferHandle const & buffer_handle, VkDeviceSize size, void * data)
{
    LOG_INFO("Updating Buffer");

    auto opt_buffer = buffers.get_buffer(buffer_handle);

    if (!opt_buffer)
    {
        LOG_ERROR("Unable to get buffer for update_buffer call, ignoring call");
        return;
    }

    Buffer buffer = buffers.get_buffer(buffer_handle).value();

    auto opt_mapped_buffer_handle = buffers.create_buffer(
        device,
        size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (!opt_mapped_buffer_handle)
    {
        LOG_ERROR("Couldn't create a staging buffer for updating device local buffer {}",
                  buffer_handle);
    }

    auto opt_mapped_buffer_ptr = buffers.map_buffer(opt_mapped_buffer_handle.value());

    if (!opt_mapped_buffer_ptr)
    {
        LOG_ERROR("Couldn't map a staging buffer for uploading texture");
    }

    memcpy(opt_mapped_buffer_ptr.value(), data, size);

    auto opt_mapped_buffer = buffers.get_buffer(opt_mapped_buffer_handle.value());

    if (!opt_mapped_buffer)
    {
        LOG_ERROR("Couldn't get the staging buffer for updating device local buffer {}",
                  buffer_handle);
    }

    auto & bucket = commands.get_transfer_bucket(frames.currentResource);

    Copy * vertex_command         = bucket.AddCommand<Copy>(0, 0);
    vertex_command->commandbuffer = commands.transfer_commandbuffers[frames.currentResource];
    vertex_command->srcBuffer     = opt_mapped_buffer.value().buffer_handle();
    vertex_command->dstBuffer     = buffer.buffer_handle();
    vertex_command->srcOffset     = 0;
    vertex_command->dstOffset     = 0;
    vertex_command->size          = size;

    BufferHandle destroy_buffer = opt_mapped_buffer_handle.value();
    delete_buffers(1, &destroy_buffer);
}

void Renderer::delete_buffers(size_t buffer_count, BufferHandle const * buffer_handles)
{
    LOG_INFO("Deleting Buffers");

    auto & bucket = commands.get_delete_bucket(frames.currentResource);

    size_t buffer_size = buffer_count * sizeof(VkBuffer);
    size_t memory_size = buffer_count * sizeof(VkDeviceMemory);

    DeleteBuffers * delete_command = bucket.AddCommand<DeleteBuffers>(0, buffer_size + memory_size);

    char *           command_memory = cmd::commandPacket::GetAuxiliaryMemory(delete_command);
    VkBuffer *       buffer_iter    = reinterpret_cast<VkBuffer *>(command_memory);
    VkDeviceMemory * memory_iter = reinterpret_cast<VkDeviceMemory *>(command_memory + buffer_size);

    delete_command->logical_device = device.get_logical_device();
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
        buffers.delete_buffer(buffer_handles[i]);

        LOG_DEBUG("Queuing buffer {} {} for delete",
                  static_cast<void *>(buffer.buffer_handle()),
                  static_cast<void *>(buffer.memory_handle()));

        *(buffer_iter++) = buffer.buffer_handle();
        *(memory_iter++) = buffer.memory_handle();
    }
}

std::optional<TextureHandle> Renderer::create_texture(size_t       width,
                                                      size_t       height,
                                                      size_t       pixel_size,
                                                      void * const pixels)
{
    LOG_INFO("Creating Texture");

    size_t imageSize = width * height * pixel_size;

    auto opt_mapped_buffer_handle = buffers.create_buffer(
        device,
        imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (!opt_mapped_buffer_handle)
    {
        LOG_ERROR("Couldn't create a staging buffer for uploading texture");
        return std::nullopt;
    }

    auto opt_mapped_buffer_ptr = buffers.map_buffer(opt_mapped_buffer_handle.value());

    if (!opt_mapped_buffer_ptr)
    {
        LOG_ERROR("Couldn't map a staging buffer for uploading texture");
        return std::nullopt;
    }

    memcpy(opt_mapped_buffer_ptr.value(), pixels, imageSize);

    auto opt_mapped_buffer = buffers.get_buffer(opt_mapped_buffer_handle.value());

    if (!opt_mapped_buffer)
    {
        LOG_ERROR("Couldn't get the staging buffer for uploading texture");
        return std::nullopt;
    }

    auto opt_texture_handle = images.create_texture(
        device.get_physical_device(),
        device.get_logical_device(),
        width,
        height,
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

    auto & bucket = commands.get_transfer_bucket(frames.currentResource);

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
    copy_command->srcBuffer     = opt_mapped_buffer.value().buffer_handle();
    copy_command->srcOffset     = 0;
    copy_command->dstImage      = texture.image_handle();
    copy_command->width         = static_cast<uint32_t>(width);
    copy_command->height        = static_cast<uint32_t>(height);

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

    BufferHandle destroy_buffer = opt_mapped_buffer_handle.value();
    delete_buffers(1, &destroy_buffer);

    return texture_handle;
}

std::optional<TextureHandle> Renderer::get_texture(AttachmentHandle const & attachment)
{
    return images.get_texture_handle(attachment);
}

void Renderer::delete_textures(size_t texture_count, TextureHandle const * texture_handles)
{
    LOG_INFO("Deleting Textures");

    auto & bucket = commands.get_delete_bucket(frames.currentResource);

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

    delete_command->logical_device = device.get_logical_device();
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

std::optional<UniformHandle> Renderer::create_uniform(UniformSetHandle set_handle,
                                                      size_t           write_count,
                                                      UniformWrite *   write_infos)
{
    auto opt_uniform = uniforms.create_uniform(set_handle);
    if (!opt_uniform)
    {
        return std::nullopt;
    }

    uniforms.update_uniform(device, opt_uniform.value(), buffers, images, write_count, write_infos);

    return opt_uniform;
}

void Renderer::delete_uniforms(size_t uniform_count, UniformHandle const * uniform_handles)
{
    LOG_INFO("Deleting Uniform");

    auto & bucket = commands.get_delete_bucket(frames.currentResource);

    DeleteUniforms * delete_command = bucket.AddCommand<DeleteUniforms>(
        0, uniform_count * sizeof(UniformHandle));

    delete_command->uniform_resources = &uniforms;
    delete_command->uniform_count     = uniform_count;
    delete_command->uniform_handles   = reinterpret_cast<UniformHandle *>(
        cmd::commandPacket::GetAuxiliaryMemory(delete_command));

    memcpy(delete_command->uniform_handles, uniform_handles, uniform_count * sizeof(UniformHandle));
}

std::optional<VkDescriptorSet> Renderer::get_uniform(UniformHandle const & handle)
{
    return uniforms.get_uniform(handle);
}

void delete_uniforms(size_t uniform_count, UniformHandle const * uniforms)
{}

ErrorCode Renderer::create_command_buffer(uint32_t image_index)
{
    auto commandbuffer = commands.draw_commandbuffers[frames.currentResource];

    auto beginInfo = VkCommandBufferBeginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                              .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                              .pInheritanceInfo = nullptr};

    VK_CHECK_RESULT(
        vkBeginCommandBuffer(commands.draw_commandbuffers[frames.currentResource], &beginInfo),
        "Unable to begin VkCommandBuffer recording");

    for (RenderPassHandle const & rp_handle: render_passes.render_pass_order)
    {
        LOG_TRACE("Drawing Render Pass {}", rp_handle);

        auto & clearValues = render_passes.clear_values[rp_handle];

        auto renderPassInfo = VkRenderPassBeginInfo{
            .sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass        = render_passes.render_passes[rp_handle],
            .framebuffer       = render_passes.framebuffers[rp_handle][image_index],
            .renderArea.offset = {0, 0},
            .renderArea.extent = render_passes.extents[rp_handle],
            .clearValueCount   = static_cast<uint32_t>(clearValues.size()),
            .pClearValues      = clearValues.data()};

        vkCmdBeginRenderPass(commandbuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        auto & subpasses = render_passes.per_render_pass_subpass_pipelines[rp_handle];

        for (SubpassHandle sp_handle = 0; sp_handle < subpasses.size(); ++sp_handle)
        {
            auto & subpass_pipelines = subpasses[sp_handle];

            for (auto pipeline_handle: subpass_pipelines)
            {
                vkCmdBindPipeline(commandbuffer,
                                  VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  pipelines.pipelines[pipeline_handle].vk_pipeline);

                for (auto & draw_bucket: pipelines.draw_buckets[pipeline_handle])
                {
                    draw_bucket.Submit();
                    draw_bucket.Clear();
                }
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

void Renderer::copy_buffer(VkBuffer     srcBuffer,
                           VkDeviceSize srcOffset,
                           VkBuffer     dstBuffer,
                           VkDeviceSize dstOffset,
                           VkDeviceSize size)
{
    auto & bucket = commands.get_transfer_bucket(frames.currentResource);

    Copy * vertex_command         = bucket.AddCommand<Copy>(0, 0);
    vertex_command->commandbuffer = commands.transfer_commandbuffers[frames.currentResource];
    vertex_command->srcBuffer     = srcBuffer;
    vertex_command->dstBuffer     = dstBuffer;
    vertex_command->srcOffset     = srcOffset;
    vertex_command->dstOffset     = dstOffset;
    vertex_command->size          = size;
};

/*
 * This should handle anything from resize, to changing the presentMode from FIFO (VSync) to
 * Immediate/Mailbox (not VSync).
 *
 */
void Renderer::change_swapchain()
{
    int width = 0, height = 0;
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(device.get_window(), &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(device.get_logical_device());

    VkPresentModeKHR last_present_mode           = device.get_present_mode();
    VkFormat         last_swapchain_image_format = device.get_color_format();
    VkColorSpaceKHR  last_swapchain_color_space  = device.get_color_space();
    VkExtent2D       last_swapchain_extent       = device.get_extent();
    VkFormat         last_depth_format           = device.get_depth_format();
    uint32_t         last_imageCount             = device.get_image_count();

    device.destroy_swapchain();
    device.update_swapchain_support();
    device.create_swapchain();

    assert(last_present_mode == device.get_present_mode());
    assert(last_swapchain_image_format == device.get_color_format());
    assert(last_swapchain_color_space == device.get_color_space());
    assert(last_depth_format == device.get_depth_format());
    assert(last_imageCount == device.get_image_count());

    if (last_swapchain_extent != device.get_extent())
    {
        LOG_DEBUG("Extent doesn't match");
    }

    images.recreate_attachments(device);

    render_passes.recreate_framebuffers(device, images);
}

}; // namespace gfx

#endif