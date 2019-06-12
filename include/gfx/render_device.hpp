#ifndef JED_GFX_RENDER_DEVICE_HPP
#define JED_GFX_RENDER_DEVICE_HPP

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vulkan/vulkan.h>
#include <glfw/glfw3.h>

#include <vector>
#include <array>
#include <set>
#include <iostream>
#include <fstream>
#include <variant>

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 color;

    // binding description is what bound vertex binding it's coming from
    static VkVertexInputBindingDescription getBindingDescription()
    {
        auto bindingDescription = VkVertexInputBindingDescription{
            .binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};

        return bindingDescription;
    }

    // attribute description is info about attribute and from which binding it comes from
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
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
};

enum class ObjectType
{
    NONE,
    STATIC,  // never updated
    DYNAMIC, // updated infrequently through staging buffer
    STREAMED // updated frequently with host visible/coherent buffer
};

struct StaticVertexData // can be edited with a
{
    VkBuffer       vertexbuffer;
    VkDeviceMemory vertexbuffer_memory;
    VkDeviceSize   vertexbuffer_offset;

    VkBuffer       indexbuffer;
    VkDeviceMemory indexbuffer_memory;
    size_t         indexbuffer_offset;
    size_t         indexbuffer_size;
};

struct StreamedVertexData
{
    size_t     vertex_count;
    Vertex *   vertices;
    size_t     index_count;
    uint32_t * indices;
};

struct Object
{
    ObjectType type{ObjectType::NONE};
    glm::mat4  transform{1.0f};
    union
    {
        StaticVertexData   s_vertex_data;
        StreamedVertexData d_vertex_data;
    };
};

namespace gfx
{
enum class Format
{
    USE_DEPTH,
    USE_COLOR
};

struct AttachmentInfo
{
    VkAttachmentDescription description;
    Format                  format;
    bool                    use_samples;
};

using AttachmentInfoHandle = size_t;

struct MappedBuffer
{
    void *         data;
    VkBuffer       buffer;
    VkDeviceMemory memory;
    VkDeviceSize   memory_size;
    size_t         offset;
};

struct DynamicUniformBuffer
{
    VkDescriptorSet           vk_descriptorset;
    std::vector<MappedBuffer> uniform_buffers;
};

using AttachmentHandle = int32_t; // -1 signifies the swapchain images
struct Attachment
{
    Format format;
    bool   use_samples;

    VkDeviceMemory vk_image_memory{VK_NULL_HANDLE};
    VkImage        vk_image{VK_NULL_HANDLE};
    VkImageView    vk_image_view{VK_NULL_HANDLE};
};

struct SubpassInfo
{
    std::vector<VkAttachmentReference> color_attachments;
    VkAttachmentReference              color_resolve_attachment;
    VkAttachmentReference              depth_stencil_attachment;
};

using CommandbufferHandle = size_t;

using RenderpassHandle = size_t;
struct Renderpass
{
    std::vector<AttachmentInfoHandle> attachments;

    std::vector<SubpassInfo> subpasses; // attachment references for creating subpass description

    std::vector<VkSubpassDependency> subpass_dependencies;

    std::vector<CommandbufferHandle> subpass_commandbuffers;

    VkRenderPass vk_renderpass{VK_NULL_HANDLE};
};

using FramebufferHandle = size_t;
struct Framebuffer
{
    RenderpassHandle              renderpass;
    std::vector<AttachmentHandle> attachments;
    uint32_t                      width;
    uint32_t                      height;
    uint32_t                      depth;
    VkFramebuffer                 vk_framebuffer;
};

using UniformLayoutHandle = size_t;
struct UniformLayout
{
    VkDescriptorSetLayoutBinding        binding;
    VkDescriptorSetLayout               vk_descriptorset_layout{VK_NULL_HANDLE};
    VkDescriptorPool                    vk_descriptor_pool{VK_NULL_HANDLE};
    std::variant<int, std::vector<int>> uniforms;
};

using PushConstantHandle    = size_t;
using VertexBindingHandle   = size_t;
using VertexAttributeHandle = size_t;

struct Shader
{
    std::string shader_name;

    VkShaderModule vk_shader_module{VK_NULL_HANDLE};
};

using ShaderHandle = size_t;

struct Pipeline
{
    ShaderHandle vertex_shader;
    ShaderHandle fragment_shader;

    // vertex binding stuff
    std::vector<VertexBindingHandle>   vertex_bindings;
    std::vector<VertexAttributeHandle> vertex_attributes;

    // uniform layouts
    std::vector<UniformLayoutHandle> uniform_layouts;
    // push constants
    std::vector<PushConstantHandle> push_constants;

    RenderpassHandle renderpass;
    size_t           subpass;

    VkPipeline       vk_pipeline;
    VkPipelineLayout vk_pipeline_layout;
};

using PipelineHandle = size_t;

class RenderDevice
{
public:
    RenderDevice(GLFWwindow * window_ptr): window(window_ptr)
    {
        checkValidationLayerSupport();
        getRequiredExtensions();
    }

    bool init(char const * window_name, size_t dynamic_vertices_count, size_t dynamic_indices_count)
    {
        if (createInstance(window_name) != VK_SUCCESS)
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

        if (createRenderPass() != VK_SUCCESS)
        {
            return false;
        }

        if (createUniformLayouts() != VK_SUCCESS)
        {
            return false;
        }

        createUniformBuffers();

        if (createDescriptorSets() != VK_SUCCESS)
        {
            return false;
        }

        if (createShaders() != VK_SUCCESS)
        {
            return false;
        }

        if (createGraphicsPipeline() != VK_SUCCESS)
        {
            return false;
        }

        if (createCommandPool() != VK_SUCCESS)
        {
            return false;
        }

        if (createAttachments() != VK_SUCCESS)
        {
            return false;
        }

        if (createFramebuffers() != VK_SUCCESS)
        {
            return false;
        }

        if (createCommandbuffers() != VK_SUCCESS)
        {
            return false;
        }

        if (createSingleTimeUseBuffers() != VK_SUCCESS)
        {
            return false;
        }

        if (createSyncObjects() != VK_SUCCESS)
        {
            return false;
        }

        if (createDynamicObjectResources(dynamic_vertices_count, dynamic_indices_count)
            != VK_SUCCESS)
        {
            return false;
        }

        if (createStagingObjectResources(dynamic_vertices_count, dynamic_indices_count)
            != VK_SUCCESS)
        {
            return false;
        }

        return true;
    }

    void quit()
    {
        vkDeviceWaitIdle(logical_device);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vkDestroyBuffer(logical_device, mapped_uniforms[i].buffer, nullptr);
            vkFreeMemory(logical_device, mapped_uniforms[i].memory, nullptr);
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            // DYNAMIC INDEXBUFFER
            vkDestroyBuffer(logical_device, dynamic_mapped_indices[i].buffer, nullptr);
            vkFreeMemory(logical_device, dynamic_mapped_indices[i].memory, nullptr);

            // DYNAMIC VERTEXBUFFER
            vkDestroyBuffer(logical_device, dynamic_mapped_vertices[i].buffer, nullptr);
            vkFreeMemory(logical_device, dynamic_mapped_vertices[i].memory, nullptr);

            // STAGING INDEXBUFFER
            vkDestroyBuffer(logical_device, staging_mapped_indices[i].buffer, nullptr);
            vkFreeMemory(logical_device, staging_mapped_indices[i].memory, nullptr);

            // STAGING VERTEXBUFFER
            vkDestroyBuffer(logical_device, staging_mapped_vertices[i].buffer, nullptr);
            vkFreeMemory(logical_device, staging_mapped_vertices[i].memory, nullptr);
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkDestroySemaphore(logical_device, render_finished_semaphores[i], nullptr);
            vkDestroySemaphore(logical_device, image_available_semaphores[i], nullptr);
            vkDestroyFence(logical_device, in_flight_fences[i], nullptr);
        }

        for (auto & attachment: attachments)
        {
            vkDestroyImageView(logical_device, attachment.vk_image_view, nullptr);
            vkDestroyImage(logical_device, attachment.vk_image, nullptr);
            vkFreeMemory(logical_device, attachment.vk_image_memory, nullptr);
        }

        // FRAMEBUFFERS
        for (auto & framebuffer_list: framebuffers)
        {
            for (auto & framebuffer: framebuffer_list)
            {
                vkDestroyFramebuffer(logical_device, framebuffer.vk_framebuffer, nullptr);
            }
        }

        // COMMAND POOL
        vkDestroyCommandPool(logical_device, command_pool, nullptr);

        for (auto & shader: shaders)
        {
            vkDestroyShaderModule(logical_device, shader.vk_shader_module, nullptr);
        }

        // GRAPHICS PIPELINE
        for (auto & pipeline: pipelines)
        {
            vkDestroyPipeline(logical_device, pipeline.vk_pipeline, nullptr);
            vkDestroyPipelineLayout(logical_device, pipeline.vk_pipeline_layout, nullptr);
        }

        // DESCRIPTORSET LAYOUT
        for (auto & uniform_layout: uniform_layouts)
        {
            vkDestroyDescriptorPool(logical_device, uniform_layout.vk_descriptor_pool, nullptr);
            vkDestroyDescriptorSetLayout(
                logical_device, uniform_layout.vk_descriptorset_layout, nullptr);
        }

        // RENDER PASS

        for (auto & renderpass: renderpasses)
        {
            vkDestroyRenderPass(logical_device, renderpass.vk_renderpass, nullptr);
            renderpass.vk_renderpass = VK_NULL_HANDLE;
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

    void waitForIdle()
    {
        vkDeviceWaitIdle(logical_device);
    }

    void startFrame()
    {
        vkWaitForFences(logical_device,
                        1,
                        &in_flight_fences[currentFrame],
                        VK_TRUE,
                        std::numeric_limits<uint64_t>::max());

        // clear copy command buffers
        if (!one_time_use_buffers[currentFrame].empty())
        {
            vkFreeCommandBuffers(logical_device,
                                 command_pool,
                                 one_time_use_buffers[currentFrame].size(),
                                 one_time_use_buffers[currentFrame].data());
        }

        one_time_use_buffers[currentFrame].clear();

        // reset buffer offsets for copies
        dynamic_mapped_vertices[currentFrame].offset = 0;
        dynamic_mapped_indices[currentFrame].offset  = 0;
        staging_mapped_vertices[currentFrame].offset = 0;
        staging_mapped_indices[currentFrame].offset  = 0;

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
    }

    void drawFrame(uint32_t object_count, Object * p_objects)
    {
        // TRANSFER OPERATIONS
        // submit copy operations to the graphics queue

        if (one_time_use_buffers[currentFrame].size())
        {
            auto submitCopyInfo = VkSubmitInfo{
                .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = static_cast<uint32_t>(
                    one_time_use_buffers[currentFrame].size()),
                .pCommandBuffers = one_time_use_buffers[currentFrame].data()};

            vkQueueSubmit(graphics_queue, 1, &submitCopyInfo, VK_NULL_HANDLE);
        }

        // the graphics queue will wait to do anything in the color_attachment_output stage
        // until the waitSemaphore is signalled by vkAcquireNextImageKHR
        VkSemaphore          waitSemaphores[] = {image_available_semaphores[currentFrame]};
        VkPipelineStageFlags waitStages[]     = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

        VkSemaphore signalSemaphores[] = {render_finished_semaphores[currentFrame]};

        createCommandbuffer(currentImage, object_count, p_objects);

        auto submitInfo = VkSubmitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,

                                       .waitSemaphoreCount = 1,
                                       .pWaitSemaphores    = waitSemaphores,
                                       .pWaitDstStageMask  = waitStages,

                                       .commandBufferCount = 1,
                                       .pCommandBuffers    = &commandbuffers[currentFrame],

                                       .signalSemaphoreCount = 1,
                                       .pSignalSemaphores    = signalSemaphores};

        vkResetFences(logical_device, 1, &in_flight_fences[currentFrame]);

        if (vkQueueSubmit(graphics_queue, 1, &submitInfo, in_flight_fences[currentFrame])
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkSwapchainKHR swapChains[] = {swapchain};

        auto presentInfo = VkPresentInfoKHR{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                            .waitSemaphoreCount = 1,
                                            .pWaitSemaphores    = signalSemaphores,

                                            .swapchainCount = 1,
                                            .pSwapchains    = swapChains,
                                            .pImageIndices  = &currentImage,

                                            .pResults = nullptr};

        auto result = vkQueuePresentKHR(present_queue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR
            || framebuffer_resized)
        {
            framebuffer_resized = false;
            // recreateSwapChain();
        }
        else if (result != VK_SUCCESS)
        {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

        // clear next frames single time use buffers
    }

    bool createStaticObject(Object &   object,
                            uint32_t   vertex_count,
                            Vertex *   vertices,
                            uint32_t   index_count,
                            uint32_t * indices)

    {
        object.type          = ObjectType::STATIC;
        object.s_vertex_data = StaticVertexData{.vertexbuffer        = VK_NULL_HANDLE,
                                                .vertexbuffer_memory = VK_NULL_HANDLE,
                                                .vertexbuffer_offset = 0,
                                                .indexbuffer         = VK_NULL_HANDLE,
                                                .indexbuffer_memory  = VK_NULL_HANDLE,
                                                .indexbuffer_offset  = 0,
                                                .indexbuffer_size    = index_count};

        if (createVertexbuffer(object.s_vertex_data.vertexbuffer,
                               object.s_vertex_data.vertexbuffer_memory,
                               vertex_count,
                               vertices)
            != VK_SUCCESS)
        {
            return false;
        }

        if (createIndexbuffer(object.s_vertex_data.indexbuffer,
                              object.s_vertex_data.indexbuffer_memory,
                              index_count,
                              indices)
            != VK_SUCCESS)
        {
            return false;
        }

        return true;
    }

    void updateStaticObject(Object &   object,
                            uint32_t   vertex_count,
                            Vertex *   vertices,
                            uint32_t   index_count,
                            uint32_t * indices)
    {
        auto & mapped_vertices = staging_mapped_vertices[currentFrame];
        auto & mapped_indices  = staging_mapped_indices[currentFrame];

        // mapped_*_data is the pointer to the next valid location to fill
        auto mapped_vertex_data = static_cast<void *>(static_cast<char *>(mapped_vertices.data)
                                                      + mapped_vertices.offset);
        auto mapped_index_data  = static_cast<void *>(static_cast<char *>(mapped_indices.data)
                                                     + mapped_indices.offset);

        // assert there's enough space left in the buffers
        assert(sizeof(Vertex) * vertex_count
               <= mapped_vertices.memory_size - mapped_vertices.offset);
        assert(sizeof(uint32_t) * index_count
               <= mapped_indices.memory_size - mapped_indices.offset);

        // copy the data over to the staging buffer
        memcpy(mapped_vertex_data, vertices, sizeof(Vertex) * vertex_count);
        memcpy(mapped_index_data, indices, sizeof(uint32_t) * index_count);

        VkDeviceSize vertex_offset = mapped_vertices.offset;

        auto allocInfo = VkCommandBufferAllocateInfo{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandPool        = command_pool,
            .commandBufferCount = 1};

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(logical_device, &allocInfo, &commandBuffer);

        auto beginInfo = VkCommandBufferBeginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        auto copyRegion = VkBufferCopy{.srcOffset = mapped_vertices.offset,
                                       .dstOffset = 0,
                                       .size      = sizeof(Vertex) * vertex_count};

        vkCmdCopyBuffer(commandBuffer,
                        mapped_vertices.buffer,
                        object.s_vertex_data.vertexbuffer,
                        1,
                        &copyRegion);

        copyRegion = VkBufferCopy{.srcOffset = mapped_indices.offset,
                                  .dstOffset = 0,
                                  .size      = sizeof(uint32_t) * index_count};

        vkCmdCopyBuffer(
            commandBuffer, mapped_indices.buffer, object.s_vertex_data.indexbuffer, 1, &copyRegion);

        vkEndCommandBuffer(commandBuffer);

        one_time_use_buffers[currentFrame].push_back(commandBuffer);

        mapped_vertices.offset += sizeof(Vertex) * vertex_count;
        mapped_indices.offset += sizeof(uint32_t) * index_count;
    }

    void destroyStaticObject(Object & object)
    {
        // INDEXBUFFER
        vkDestroyBuffer(logical_device, object.s_vertex_data.indexbuffer, nullptr);
        vkFreeMemory(logical_device, object.s_vertex_data.indexbuffer_memory, nullptr);

        // VERTEXBUFFER
        vkDestroyBuffer(logical_device, object.s_vertex_data.vertexbuffer, nullptr);
        vkFreeMemory(logical_device, object.s_vertex_data.vertexbuffer_memory, nullptr);
    }

    void updateUniformBuffer(glm::mat4 const & data)
    {
        memcpy(mapped_uniforms[currentFrame].data, glm::value_ptr(data), sizeof(glm::mat4));
    }

private:
    void checkValidationLayerSupport()
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

    void getRequiredExtensions()
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
    VkResult createInstance(char const * window_name)
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
    static VKAPI_ATTR VkBool32 VKAPI_CALL
                               debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
                                             VkDebugUtilsMessageTypeFlagsEXT              messageType,
                                             const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData,
                                             void *                                       pUserData)
    {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }

    VkResult createDebugMessenger()
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

    VkResult createDebugUtilsMessengerEXT(const VkDebugUtilsMessengerCreateInfoEXT * pCreateInfo,
                                          const VkAllocationCallbacks *              pAllocator,
                                          VkDebugUtilsMessengerEXT * pDebugMessenger)
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

    void cleanupDebugUtilsMessengerEXT(VkDebugUtilsMessengerEXT      debugMessenger,
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
    VkResult createSurface()
    {
        return glfwCreateWindowSurface(instance, window, nullptr, &surface);
    }

    // PHYSICAL DEVICE
    struct PhysicalDeviceInfo
    {
        std::vector<VkQueueFamilyProperties> queue_family_properties{};
        int32_t                              present_queue{-1};
        int32_t                              graphics_queue{-1};
        int32_t                              transfer_queue{-1};

        bool queues_complete() const
        {
            return present_queue != -1 && graphics_queue != -1 && transfer_queue != -1;
        }

        std::vector<VkExtensionProperties> available_extensions{};

        bool has_required_extensions{false};

        VkSurfaceCapabilitiesKHR        capabilities{};
        std::vector<VkSurfaceFormatKHR> formats{};
        std::vector<VkPresentModeKHR>   presentModes{};

        bool swapchain_adequate() const
        {
            return !formats.empty() && !presentModes.empty();
        }

        VkPhysicalDeviceFeatures features{};

        VkPhysicalDeviceProperties properties{};

        VkSampleCountFlagBits msaa_samples{VK_SAMPLE_COUNT_1_BIT};
    };

    bool pickPhysicalDevice()
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

    bool isDeviceSuitable(VkPhysicalDevice device, PhysicalDeviceInfo & device_info)
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

    void findQueueFamilies(VkPhysicalDevice device, PhysicalDeviceInfo & device_info)
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

    bool checkDeviceExtensionSupport(VkPhysicalDevice device, PhysicalDeviceInfo & device_info)
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

    void querySwapChainSupport(VkPhysicalDevice device, PhysicalDeviceInfo & device_info)
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

    void getMaxUsableSampleCount()
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
    VkResult createLogicalDevice()
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

    // SWAPCHAIN
    VkResult createSwapChain()
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

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(
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

    VkPresentModeKHR chooseSwapPresentMode(
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

    VkExtent2D chooseSwapExtent(VkSurfaceCapabilitiesKHR const & capabilities)
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

    // SWAPCHAIN IMAGES
    void getSwapChainImages()
    {
        vkGetSwapchainImagesKHR(logical_device, swapchain, &swapchain_image_count, nullptr);
        swapchain_images.resize(swapchain_image_count);
        vkGetSwapchainImagesKHR(
            logical_device, swapchain, &swapchain_image_count, swapchain_images.data());
    }

    VkResult createSwapChainImageViews()
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

    VkResult createImageView(VkImage            image,
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
    VkResult createRenderPass()
    {
        for (uint32_t r_i = 0; r_i < renderpasses.size(); ++r_i)
        {
            // attachment descriptions
            auto attachments = std::vector<VkAttachmentDescription>(
                renderpasses[r_i].attachments.size());

            for (uint32_t a_i = 0; a_i < attachments.size(); ++a_i)
            {
                auto const & attachment_info = attachment_infos[renderpasses[r_i].attachments[a_i]];

                attachments[a_i] = attachment_info.description;

                switch (attachment_info.format)
                {
                case Format::USE_COLOR:
                    attachments[a_i].format = swapchain_image_format;
                    break;
                case Format::USE_DEPTH:
                    attachments[a_i].format = depth_format;
                    break;
                }

                if (attachment_info.use_samples)
                {
                    attachments[a_i].samples = physical_device_info.msaa_samples;
                }
                else
                {
                    attachments[a_i].samples = VK_SAMPLE_COUNT_1_BIT;
                }
            }

            // subpass descriptions
            auto subpasses = std::vector<VkSubpassDescription>(renderpasses[r_i].subpasses.size());

            for (uint32_t s_i = 0; s_i < subpasses.size(); ++s_i)
            {
                subpasses[s_i] = VkSubpassDescription{
                    .pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS,
                    .colorAttachmentCount = static_cast<uint32_t>(
                        renderpasses[r_i].subpasses[s_i].color_attachments.size()),
                    .pColorAttachments = renderpasses[r_i].subpasses[s_i].color_attachments.data(),
                    .pDepthStencilAttachment
                    = &renderpasses[r_i].subpasses[s_i].depth_stencil_attachment,
                    .pResolveAttachments
                    = &renderpasses[r_i].subpasses[s_i].color_resolve_attachment};
            }

            auto const & dependencies = renderpasses[r_i].subpass_dependencies;

            auto renderPassInfo = VkRenderPassCreateInfo{
                .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                .attachmentCount = static_cast<uint32_t>(attachments.size()),
                .pAttachments    = attachments.data(),

                .subpassCount = static_cast<uint32_t>(subpasses.size()),
                .pSubpasses   = subpasses.data(),

                .dependencyCount = static_cast<uint32_t>(dependencies.size()),
                .pDependencies   = dependencies.data()};

            auto result = vkCreateRenderPass(
                logical_device, &renderPassInfo, nullptr, &renderpasses[r_i].vk_renderpass);
            if (result != VK_SUCCESS)
            {
                return result;
            }
        }

        return VK_SUCCESS;
    }

    VkFormat findDepthFormat()
    {
        return findSupportedFormat(
            {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    VkFormat findSupportedFormat(const std::vector<VkFormat> & candidates,
                                 VkImageTiling                 tiling,
                                 VkFormatFeatureFlags          features)
    {
        for (VkFormat format: candidates)
        {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR
                && (props.linearTilingFeatures & features) == features)
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

    VkResult createUniformLayouts()
    {
        for (auto & uniform_layout: uniform_layouts)
        {
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
                .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)};

            auto poolInfo = VkDescriptorPoolCreateInfo{
                .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .poolSizeCount = 1,
                .pPoolSizes    = &poolsize,
                .maxSets       = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)};

            result = vkCreateDescriptorPool(
                logical_device, &poolInfo, nullptr, &uniform_layout.vk_descriptor_pool);
            if (result != VK_SUCCESS)
            {
                return result;
            }
        }

        return VK_SUCCESS;
    }

    void createUniformBuffers()
    {
        VkDeviceSize bufferSize = sizeof(glm::mat4);

        mapped_uniforms.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            mapped_uniforms[i].memory_size = bufferSize;

            createBuffer(mapped_uniforms[i].memory_size,
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         mapped_uniforms[i].buffer,
                         mapped_uniforms[i].memory);

            vkMapMemory(logical_device,
                        mapped_uniforms[i].memory,
                        0,
                        mapped_uniforms[i].memory_size,
                        0,
                        &mapped_uniforms[i].data);
        }
    }

    VkResult createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts{static_cast<size_t>(MAX_FRAMES_IN_FLIGHT),
                                                   uniform_layouts[0].vk_descriptorset_layout};

        auto allocInfo = VkDescriptorSetAllocateInfo{
            .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool     = uniform_layouts[0].vk_descriptor_pool,
            .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            .pSetLayouts        = layouts.data()};

        descriptorsets.resize(MAX_FRAMES_IN_FLIGHT);

        auto result = vkAllocateDescriptorSets(logical_device, &allocInfo, descriptorsets.data());
        if (result != VK_SUCCESS)
        {
            return result;
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            auto bufferInfo = VkDescriptorBufferInfo{
                .buffer = mapped_uniforms[i].buffer, .offset = 0, .range = sizeof(glm::mat4)};

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

        return VK_SUCCESS;
    }

    // SHADERS
    VkResult createShaders()
    {
        VkResult result;

        for (auto & shader: shaders)
        {
            auto shaderCode = readFile(shader.shader_name);

            result = createShaderModule(shaderCode, shader.vk_shader_module);
            if (result != VK_SUCCESS)
            {
                return result;
            }
        }

        return result;
    }

    VkResult createShaderModule(std::vector<char> const & code, VkShaderModule & shaderModule)
    {
        auto createInfo = VkShaderModuleCreateInfo{
            .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code.size(),
            .pCode    = reinterpret_cast<const uint32_t *>(code.data())};

        return vkCreateShaderModule(logical_device, &createInfo, nullptr, &shaderModule);
    }

    // GRAPHICS PIPELINE
    VkResult createGraphicsPipeline()
    {
        for (auto & pipeline: pipelines)
        {
            auto vertShaderStageInfo = VkPipelineShaderStageCreateInfo{
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = VK_SHADER_STAGE_VERTEX_BIT,
                .module = shaders[pipeline.vertex_shader].vk_shader_module,
                .pName  = "main"};

            auto fragShaderStageInfo = VkPipelineShaderStageCreateInfo{
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = shaders[pipeline.fragment_shader].vk_shader_module,
                .pName  = "main"};

            VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                              fragShaderStageInfo};

            auto bindings = std::vector<VkVertexInputBindingDescription>{};
            for (auto const & binding: pipeline.vertex_bindings)
            {
                bindings.push_back(vertex_bindings[binding]);
            }

            auto attributes = std::vector<VkVertexInputAttributeDescription>{};
            for (auto const & attribute: pipeline.vertex_attributes)
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
                .sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
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
                .sType            = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .depthClampEnable = VK_FALSE,
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
            for (auto const & push_constant: pipeline.push_constants)
            {
                pushConstantRanges.push_back(push_constants[push_constant]);
            }

            auto pipelineLayoutInfo = VkPipelineLayoutCreateInfo{
                .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount         = 1,
                .pSetLayouts            = &uniform_layouts[0].vk_descriptorset_layout,
                .pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size()),
                .pPushConstantRanges    = pushConstantRanges.data()};

            auto result = vkCreatePipelineLayout(
                logical_device, &pipelineLayoutInfo, nullptr, &pipeline.vk_pipeline_layout);
            if (result != VK_SUCCESS)
            {
                return result;
            }

            VkRenderPass renderpass = renderpasses[pipeline.renderpass].vk_renderpass;
            uint32_t     subpass    = pipeline.subpass;

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
                .renderPass          = renderpass, // render_pass,
                .subpass             = subpass,
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

    // COMMAND POOL
    VkResult createCommandPool()
    {
        auto poolInfo = VkCommandPoolCreateInfo{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = static_cast<uint32_t>(physical_device_info.graphics_queue),
            .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};

        return vkCreateCommandPool(logical_device, &poolInfo, nullptr, &command_pool);
    }

    VkResult createAttachments()
    {
        for (uint32_t i = 0; i < attachments.size(); ++i)
        {
            auto & attachment = attachments[i];

            VkFormat           format;
            VkImageUsageFlags  usage;
            VkImageAspectFlags aspect;
            VkImageLayout      final_layout;

            if (attachment.format == Format::USE_COLOR)
            {
                format       = swapchain_image_format;
                usage        = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
                aspect       = VK_IMAGE_ASPECT_COLOR_BIT;
                final_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            }
            else if (attachment.format == Format::USE_DEPTH)
            {
                format       = depth_format;
                usage        = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
                aspect       = VK_IMAGE_ASPECT_DEPTH_BIT;
                final_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            }

            VkSampleCountFlagBits samples;

            if (attachment.use_samples)
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

            result = createImageView(
                attachment.vk_image, attachment.vk_image_view, format, aspect, 1);

            if (result != VK_SUCCESS)
            {
                return result;
            }

            transitionImageLayout(
                attachment.vk_image, format, VK_IMAGE_LAYOUT_UNDEFINED, final_layout, 1);
        }

        return VK_SUCCESS;
    }

    VkResult createImage(uint32_t              width,
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

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
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

    void transitionImageLayout(VkImage       image,
                               VkFormat      format,
                               VkImageLayout oldLayout,
                               VkImageLayout newLayout,
                               uint32_t      mipLevels)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        auto barrier = VkImageMemoryBarrier{.sType     = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                            .oldLayout = oldLayout,
                                            .newLayout = newLayout,
                                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .image               = image,
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

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
            && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
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

    bool hasStencilComponent(VkFormat format)
    {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    VkCommandBuffer beginSingleTimeCommands()
    {
        auto allocInfo = VkCommandBufferAllocateInfo{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandPool        = command_pool,
            .commandBufferCount = 1};

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(logical_device, &allocInfo, &commandBuffer);

        auto beginInfo = VkCommandBufferBeginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer)
    {
        vkEndCommandBuffer(commandBuffer);

        auto submitInfo = VkSubmitInfo{.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                       .commandBufferCount = 1,
                                       .pCommandBuffers    = &commandBuffer};

        vkQueueSubmit(graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphics_queue);

        vkFreeCommandBuffers(logical_device, command_pool, 1, &commandBuffer);
    }

    // FRAMEBUFFER
    VkResult createFramebuffers()
    {
        framebuffers.resize(swapchain_image_count);

        for (size_t i = 0; i < swapchain_image_count; ++i)
        {
            for (auto & framebuffer: framebuffers[i])
            {
                auto fb_attachments = std::vector<VkImageView>{};

                for (auto attachment_id: framebuffer.attachments)
                {
                    if (attachment_id == -1)
                    {
                        fb_attachments.push_back(swapchain_image_views[i]);
                    }
                    else
                    {
                        fb_attachments.push_back(attachments[attachment_id].vk_image_view);
                    }
                }

                auto framebufferInfo = VkFramebufferCreateInfo{
                    .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                    .renderPass      = renderpasses[framebuffer.renderpass].vk_renderpass,
                    .attachmentCount = static_cast<uint32_t>(fb_attachments.size()),
                    .pAttachments    = fb_attachments.data(),
                    .width           = swapchain_extent.width,
                    .height          = swapchain_extent.height,
                    .layers          = 1};

                auto result = vkCreateFramebuffer(
                    logical_device, &framebufferInfo, nullptr, &framebuffer.vk_framebuffer);
                if (result != VK_SUCCESS)
                {
                    return result;
                }
            }
        }

        return VK_SUCCESS;
    }

    // VERTEXBUFFER
    VkResult createVertexbuffer(VkBuffer &       vertexbuffer,
                                VkDeviceMemory & vertexbuffer_memory,
                                uint32_t         vertex_count,
                                Vertex *         vertices)
    {
        VkDeviceSize bufferSize = sizeof(Vertex) * vertex_count;

        VkBuffer       stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer,
                     stagingBufferMemory);

        void * data;
        vkMapMemory(logical_device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices, (size_t)bufferSize);
        vkUnmapMemory(logical_device, stagingBufferMemory);

        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     vertexbuffer,
                     vertexbuffer_memory);

        copyBuffer(stagingBuffer, vertexbuffer, bufferSize);

        vkDestroyBuffer(logical_device, stagingBuffer, nullptr);
        vkFreeMemory(logical_device, stagingBufferMemory, nullptr);

        return VK_SUCCESS;
    }

    VkResult createBuffer(VkDeviceSize          size,
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

        vkBindBufferMemory(logical_device, buffer, bufferMemory, 0);

        return VK_SUCCESS;
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
    {
        auto commandBuffer = beginSingleTimeCommands();

        auto copyRegion = VkBufferCopy{.srcOffset = 0, // Optional
                                       .dstOffset = 0, // Optional
                                       .size      = size};

        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    };

    // INDEXBUFFER
    VkResult createIndexbuffer(VkBuffer &       indexbuffer,
                               VkDeviceMemory & indexbuffer_memory,
                               uint32_t         index_count,
                               uint32_t *       indices)
    {
        VkDeviceSize bufferSize = sizeof(uint32_t) * index_count;

        VkBuffer       stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer,
                     stagingBufferMemory);

        void * data;
        vkMapMemory(logical_device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices, (size_t)bufferSize);
        vkUnmapMemory(logical_device, stagingBufferMemory);

        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     indexbuffer,
                     indexbuffer_memory);

        copyBuffer(stagingBuffer, indexbuffer, bufferSize);

        vkDestroyBuffer(logical_device, stagingBuffer, nullptr);
        vkFreeMemory(logical_device, stagingBufferMemory, nullptr);

        return VK_SUCCESS;
    }

    // COMMANDBUFFERS
    // TODO: rewrite to return VkResult
    VkResult createCommandbuffers()
    {
        commandbuffers.resize(MAX_FRAMES_IN_FLIGHT);

        auto allocInfo = VkCommandBufferAllocateInfo{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool        = command_pool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = (uint32_t)commandbuffers.size()};

        auto result = vkAllocateCommandBuffers(logical_device, &allocInfo, commandbuffers.data());

        return result;
    }

    VkResult createSingleTimeUseBuffers()
    {
        one_time_use_buffers.resize(MAX_FRAMES_IN_FLIGHT);
        return VK_SUCCESS;
    }

    // COMMANDBUFFER
    VkResult createCommandbuffer(uint32_t resource_index, uint32_t object_count, Object * p_objects)
    {
        auto & mapped_vertices = dynamic_mapped_vertices[currentFrame];
        auto & mapped_indices  = dynamic_mapped_indices[currentFrame];

        auto beginInfo = VkCommandBufferBeginInfo{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr};

        auto result = vkBeginCommandBuffer(commandbuffers[currentFrame], &beginInfo);
        if (result != VK_SUCCESS)
        {
            return result;
        }

        // memory barrier for copy commands
        auto barrier = VkMemoryBarrier{.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                       .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                                       .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT};

        vkCmdPipelineBarrier(commandbuffers[currentFrame],
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                             0,
                             1,
                             &barrier,
                             0,
                             nullptr,
                             0,
                             nullptr);

        VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};

        auto clearValues = std::array<VkClearValue, 2>{
            VkClearValue{.color = {0.0f, 0.0f, 0.0f, 1.0f}},
            VkClearValue{.depthStencil = {1.0f, 0}}};

        auto renderPassInfo = VkRenderPassBeginInfo{
            .sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass        = renderpasses[0].vk_renderpass,
            .framebuffer       = framebuffers[resource_index][0].vk_framebuffer,
            .renderArea.offset = {0, 0},
            .renderArea.extent = swapchain_extent,
            .clearValueCount   = static_cast<uint32_t>(clearValues.size()),
            .pClearValues      = clearValues.data()};

        vkCmdBeginRenderPass(
            commandbuffers[currentFrame], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandbuffers[currentFrame],
                          VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipelines[0].vk_pipeline);

        uint32_t dynamic_offset{0};

        vkCmdBindDescriptorSets(commandbuffers[currentFrame],
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelines[0].vk_pipeline_layout,
                                0,
                                1,
                                &descriptorsets[currentFrame],
                                1,
                                &dynamic_offset);

        for (uint32_t object_index = 0; object_index < object_count; ++object_index)
        {
            auto & object = p_objects[object_index];

            vkCmdPushConstants(commandbuffers[currentFrame],
                               pipelines[0].vk_pipeline_layout,
                               VK_SHADER_STAGE_VERTEX_BIT,
                               0,
                               sizeof(glm::mat4),
                               glm::value_ptr(object.transform));

            if (object.type == ObjectType::STATIC)
            {
                vkCmdBindVertexBuffers(commandbuffers[currentFrame],
                                       0,
                                       1,
                                       &object.s_vertex_data.vertexbuffer,
                                       &object.s_vertex_data.vertexbuffer_offset);
                vkCmdBindIndexBuffer(commandbuffers[currentFrame],
                                     object.s_vertex_data.indexbuffer,
                                     object.s_vertex_data.indexbuffer_offset,
                                     VK_INDEX_TYPE_UINT32);

                vkCmdDrawIndexed(commandbuffers[currentFrame],
                                 object.s_vertex_data.indexbuffer_size,
                                 1,
                                 0,
                                 0,
                                 0);
            }
            else if (object.type == ObjectType::STREAMED)
            {
                // mapped_*_data is the pointer to the next valid location to fill
                auto mapped_vertex_data = static_cast<void *>(
                    static_cast<char *>(mapped_vertices.data) + mapped_vertices.offset);
                auto mapped_index_data = static_cast<void *>(
                    static_cast<char *>(mapped_indices.data) + mapped_indices.offset);

                // assert there's enough space left in the buffers
                assert(sizeof(Vertex) * object.d_vertex_data.vertex_count
                       <= mapped_vertices.memory_size - mapped_vertices.offset);
                assert(sizeof(uint32_t) * object.d_vertex_data.index_count
                       <= mapped_indices.memory_size - mapped_indices.offset);

                // copy the data over
                memcpy(mapped_vertex_data,
                       object.d_vertex_data.vertices,
                       sizeof(Vertex) * object.d_vertex_data.vertex_count);
                memcpy(mapped_index_data,
                       object.d_vertex_data.indices,
                       sizeof(uint32_t) * object.d_vertex_data.index_count);

                VkDeviceSize vertex_offset{mapped_vertices.offset};

                vkCmdBindVertexBuffers(
                    commandbuffers[currentFrame], 0, 1, &mapped_vertices.buffer, &vertex_offset);

                vkCmdBindIndexBuffer(commandbuffers[currentFrame],
                                     mapped_indices.buffer,
                                     mapped_indices.offset,
                                     VK_INDEX_TYPE_UINT32);

                vkCmdDrawIndexed(
                    commandbuffers[currentFrame], object.d_vertex_data.index_count, 1, 0, 0, 0);

                mapped_vertices.offset += sizeof(Vertex) * object.d_vertex_data.vertex_count;
                mapped_indices.offset += sizeof(uint32_t) * object.d_vertex_data.index_count;
            }
        }

        vkCmdEndRenderPass(commandbuffers[currentFrame]);

        result = vkEndCommandBuffer(commandbuffers[currentFrame]);
        if (result != VK_SUCCESS)
        {
            return result;
        }

        return VK_SUCCESS;
    }

    // SYNC OBJECTS
    VkResult createSyncObjects()
    {
        image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);

        auto semaphoreInfo = VkSemaphoreCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

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

    VkResult createDynamicObjectResources(size_t dynamic_vertices_count,
                                          size_t dynamic_indices_count)
    {
        dynamic_mapped_vertices.resize(MAX_FRAMES_IN_FLIGHT);
        dynamic_mapped_indices.resize(MAX_FRAMES_IN_FLIGHT);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
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

    VkResult createStagingObjectResources(size_t dynamic_vertices_count,
                                          size_t dynamic_indices_count)
    {
        staging_mapped_vertices.resize(MAX_FRAMES_IN_FLIGHT);
        staging_mapped_indices.resize(MAX_FRAMES_IN_FLIGHT);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            staging_mapped_vertices[i].memory_size = sizeof(Vertex) * dynamic_vertices_count;
            staging_mapped_indices[i].memory_size  = sizeof(uint32_t) * dynamic_indices_count;

            createBuffer(staging_mapped_vertices[i].memory_size,
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         staging_mapped_vertices[i].buffer,
                         staging_mapped_vertices[i].memory);

            vkMapMemory(logical_device,
                        staging_mapped_vertices[i].memory,
                        0,
                        staging_mapped_vertices[i].memory_size,
                        0,
                        &staging_mapped_vertices[i].data);
            // vkUnmapMemory(logical_device, stagingBufferMemory);

            createBuffer(staging_mapped_indices[i].memory_size,
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         staging_mapped_indices[i].buffer,
                         staging_mapped_indices[i].memory);

            vkMapMemory(logical_device,
                        staging_mapped_indices[i].memory,
                        0,
                        staging_mapped_indices[i].memory_size,
                        0,
                        &staging_mapped_indices[i].data);
            // vkUnmapMemory(logical_device, stagingBufferMemory);
        }

        return VK_SUCCESS;
    }

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

    std::vector<VkDescriptorSet> descriptorsets;

    VkCommandPool command_pool;

    std::vector<VkCommandBuffer> commandbuffers;

    std::vector<VkSemaphore> image_available_semaphores;
    std::vector<VkSemaphore> render_finished_semaphores;
    std::vector<VkFence>     in_flight_fences;

    uint32_t      currentImage{0};
    size_t        currentFrame{0};
    int32_t const MAX_FRAMES_IN_FLIGHT{2};

    bool framebuffer_resized;

    std::vector<std::vector<VkCommandBuffer>> one_time_use_buffers;

    std::vector<MappedBuffer> dynamic_mapped_vertices;
    std::vector<MappedBuffer> dynamic_mapped_indices;

    std::vector<MappedBuffer> staging_mapped_vertices;
    std::vector<MappedBuffer> staging_mapped_indices;

    std::vector<MappedBuffer> mapped_uniforms;

    std::vector<AttachmentInfo> attachment_infos
    {
        AttachmentInfo{
            .format = Format::USE_COLOR,
            .use_samples = true,
            .description = VkAttachmentDescription{
                           //.format  = swapchain_image_format,
                           //.samples        = physical_device_info.msaa_samples,
                           .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
                           .storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE, // not needed for storing, result is stored in color resolve
                           .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                           .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                           .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
                           .finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}},
        AttachmentInfo{
            .format = Format::USE_DEPTH,
            .use_samples = true,
            .description = VkAttachmentDescription{
                            //.format         = swapchain_image_format,
                            //.samples        = VK_SAMPLE_COUNT_1_BIT,
                            .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
                            .storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                            .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                            .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
                            .finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}},
        AttachmentInfo{
            .format = Format::USE_COLOR,
            .use_samples = false,
            .description = VkAttachmentDescription{
                            //.format       = depth_format,
                            //.samples        = physical_device_info.msaa_samples,
                            .loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                            .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
                            .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                            .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
                            .finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR}}
    };

    std::vector<Renderpass> renderpasses{Renderpass{
        .attachments          = {0, 1, 2},
        .subpasses            = {SubpassInfo{
            .color_attachments = {VkAttachmentReference{0,
                                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}},
            .color_resolve_attachment
            = VkAttachmentReference{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            .depth_stencil_attachment
            = VkAttachmentReference{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}}},
        .subpass_dependencies = {
            VkSubpassDependency{.srcSubpass    = VK_SUBPASS_EXTERNAL,
                                .dstSubpass    = 0,
                                .srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .srcAccessMask = 0,
                                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                                 | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT}}}};

    // need a way of handling the swapchain images
    // attachments that are used after this frame need to be double buffered etc (i.e. like
    // swapchain) if Store op is DONT_CARE, it doesn't need to be buffered
    std::vector<Attachment> attachments{
        Attachment{.format = Format::USE_COLOR, .use_samples = true},
        Attachment{.format = Format::USE_DEPTH, .use_samples = true}};

    std::vector<std::vector<Framebuffer>> framebuffers{
        std::vector<Framebuffer>{Framebuffer{.renderpass = 0, .attachments = {0, 1, -1}}},
        std::vector<Framebuffer>{Framebuffer{.renderpass = 0, .attachments = {0, 1, -1}}},
        std::vector<Framebuffer>{Framebuffer{.renderpass = 0, .attachments = {0, 1, -1}}}};

    std::vector<UniformLayout> uniform_layouts{
        UniformLayout{.binding = {.binding            = 0,
                                  .descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                  .descriptorCount    = 1,
                                  .stageFlags         = VK_SHADER_STAGE_VERTEX_BIT,
                                  .pImmutableSamplers = nullptr}}};

    std::vector<VkPushConstantRange> push_constants{
        VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                            .offset     = 0,
                            .size       = sizeof(glm::mat4)}};

    std::vector<VkVertexInputBindingDescription> vertex_bindings{
        VkVertexInputBindingDescription{.binding   = 0,
                                        .stride    = sizeof(Vertex),
                                        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX}};

    std::vector<VkVertexInputAttributeDescription> vertex_attributes = {
        VkVertexInputAttributeDescription{.binding  = 0,
                                          .location = 0,
                                          .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                          .offset   = offsetof(Vertex, pos)},
        VkVertexInputAttributeDescription{.binding  = 0,
                                          .location = 1,
                                          .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                          .offset   = offsetof(Vertex, color)}};

    std::vector<Shader> shaders{Shader{.shader_name = "shaders/vert.spv"},
                                Shader{.shader_name = "shaders/frag.spv"}};

    std::vector<Pipeline> pipelines{Pipeline{.vertex_shader     = 0,
                                             .fragment_shader   = 1,
                                             .vertex_bindings   = {0},
                                             .vertex_attributes = {0, 1},
                                             .uniform_layouts   = {},
                                             .push_constants    = {0},
                                             .renderpass        = 0,
                                             .subpass           = 0}};

}; // class RenderDevice
}; // namespace gfx

#endif