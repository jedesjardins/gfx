# The Renderer class

The Renderer class, in the gfx namespace, is the main interface used to do graphics operations on. It requires a GLFWwindow * to create at the moment, as it relies on GLFW to manage the window. It is configured with [the RenderConfig class](docs/render_config.md), to define what resources it can use, and how it draws to the screen. 

## API

### Setup and Teardown

`explicit gfx::Renderer::Renderer(GLFWwindow * window_ptr);`

Creates a Renderer object. The Renderer must still be initialized after calling the constructor.

Parameters:
    * `GLFWwindow * window_ptr` - The Renderer will draw to the window designated by this window pointer. The window must have already been created with `glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);` to show that it will use vulkan, not opengl.

---
`bool gfx::Renderer::init(RenderConfig & render_config);`

Initializes the Renderer with the given RenderConfig. This puts the Renderer in a valid state to use.

Parameters:
    * `RenderConfig & render_config` - The configuration that this Renderer will use. The RenderConfig must have already been initialized before it can be used to initialize the Renderer.

Return Value
    * `bool` - signifies the success or failure of the operation. If this operation fails, one can look at the generated logs for more verbose output of what went wrong.

---
<a name="quit"> `void gfx::Renderer::quit();`

Destroys all the graphics resources managed by the Renderer. The Renderer must be idle when quit is called, which can be ensured with a call to [wait_for_idle](#wait_for_idle). If the Renderer is not idle then this function can cause errors.

---
<a name="wait_for_idle"></a> `void gfx::Renderer::wait_for_idle();`

Waits until all of the Renderers current operations have completed on the graphics card. This operation must be run before calling [quit](#quit)

---

### Drawing

```
ErrorCode gfx::Renderer::draw(PipelineHandle  pipeline,
                              size_t          vertices_size,
                              void *          vertices,
                              uint32_t        index_count,
                              uint32_t *      indices,
                              size_t          push_constant_size,
                              void *          push_constant_data,
                              size_t          uniform_count,
                              UniformHandle * p_uniforms);
```

This function queues a draw command to the given pipeline with the given vertices, indices, uniforms, and push constant data.

It returns an ErrorCode::NONE if it executes successfully.

Parameters:
    * PipelineHandle  pipeline - the pipeline this command will draw in. The PipelineHandle must have been gotten with [get_pipeline_handle](#get_pipeline_handle) for the call to be valid.
    * size_t          vertices_size - the size of the data in vertices
    * void *          vertices - a pointer to an array of vertex data to be drawn from. The format of the vertices must match that which the pipeline was created with to be valid
    * uint32_t        index_count - the count of indices in indices
    * uint32_t *      indices - a pointer to an array of indices into the vertex data to be drawn
    * size_t          push_constant_size - the size of the push_constant_data
    * void *          push_constant_data - the push constant data. The format of the push consant data
    * size_t          uniform_count - the number of uniforms
    * UniformHandle * p_uniforms - a pointer to an array of UniformHandles

Return Value
    * [ErrorCode](#ErrorCode)

---
```
ErrorCode gfx::Renderer::draw(PipelineHandle  pipeline,
                              BufferHandle    vertexbuffer_handle,
                              VkDeviceSize    vertexbuffer_offset,
                              BufferHandle    indexbuffer_handle,
                              VkDeviceSize    indexbuffer_offset,
                              VkDeviceSize    indexbuffer_count,
                              size_t          push_constant_size,
                              void *          push_constant_data,
                              size_t          uniform_count,
                              UniformHandle * p_uniforms);
```

This function queues a draw command to the given pipeline with the vertice buffer, indice buffer, uniforms, and push constant data.

Parameters:
    * PipelineHandle  pipeline - the pipeline this command will draw in. The PipelineHandle must have been gotten with [get_pipeline_handle](#get_pipeline_handle) for the call to be valid.
    * BufferHandle    vertexbuffer_handle - a handle to an existing vertex buffer to be drawn from. The format of the vertices must match that which the pipeline was created with to be valid
    * VkDeviceSize    vertexbuffer_offset - the offset into the array buffer to be drawn from.
    * BufferHandle    indexbuffer_handle - a handle to an existing index buffer to be drawn from.
    * VkDeviceSize    indexbuffer_offset - the offset into the array buffer to be drawn from.
    * VkDeviceSize    index_count - the count of indices in indices
    * size_t          push_constant_size - the size of the push_constant_data
    * void *          push_constant_data - the push constant data. The format of the push consant data
    * size_t          uniform_count - the number of uniforms
    * UniformHandle * p_uniforms - a pointer to an array of UniformHandles

Return Value
    * [ErrorCode](#ErrorCode) - The function returns ErrorCode::NONE if it executes successfully. If this operation fails, one can look at the generated logs for more verbose output of what went wrong.

---
`bool gfx::Renderer::submit_frame();`

Submits all queued commands since the last call to submit_frame, or Renderer creation, to be executed on the graphics card. 

Return Value
    * `bool` - signifies the success or failure of the operation. If this operation fails, one can look at the generated logs for more verbose output of what went wrong.

---

### Getting existing graphics Resources


`std::optional<UniformLayoutHandle> gfx::Renderer::get_uniform_layout_handle(std::string layout_name);`

This function gets the handle of a UniformLayout, specified by the layout_name. The layout_name corresponds to the name of the layout in the configuration file.

Parameters:
    * std::string layout_name - the name of the uniform layout to get the handle of

Return Value
    * [ErrorCode](#ErrorCode) - The function returns ErrorCode::NONE if it executes successfully. If this operation fails, one can look at the generated logs for more verbose output of what went wrong.

---
`std::optional<PipelineHandle> gfx::Renderer::get_pipeline_handle(std::string pipeline_name)`

This function gets the handle of a Pipeline, specified by the pipeline_name. The pipeline_name corresponds to the name of the pipeline in the configuration file.

Parameters:
    * std::string pipeline_name - the name of the pipeline layout in the configuration whose handle will be returned.

Return Value
    * std::optional<PipelineHandle> - The function returns std::nullopt if it encounters an error. If this operation fails, one can look at the generated logs for more verbose output of what went wrong.

---

### Creating, Updating, and Deleting graphics resources
<a name="new_uniform"></a>
```
std::optional<UniformHandle> gfx::Renderer::new_uniform(UniformLayoutHandle layout_handle,
                                                        VkDeviceSize        size,
                                                        void *              data_ptr);
```

This function creates a new uniform in the layout of the given layout_handle, then copies the data in data_ptr into it. The layout handle must have been configured with descriptor type UNIFORM_BUFFER_DYNAMIC. The size of the data must be less than 256 bytes.

Parameters:
    * UniformLayoutHandle layout_handle - the handle of a uniform layout, this should reference one of the layouts in the configuration, and can be gotten using [get_uniform_layout_handle](#get_uniform_layout_handle)
    * VkDeviceSize        size - the size of the data to initialize the uniform with
    * void *              data_ptr - the data to initialize the uniform with

Return Value
    * std::optional<UniformHandle> - The function returns std::nullopt if it encounters an error. If this operation fails, one can look at the generated logs for more verbose output of what went wrong.

---
```
std::optional<UniformHandle> gfx::Renderer::new_uniform(UniformLayoutHandle layout_handle,
                                                        TextureHandle       texture_handle);
```

This function creates a new uniform in the layout of the given layout_handle, then copies the data in data_ptr into it. The layout handle must have been configured with descriptor type COMBINED_IMAGE_SAMPLER. The texture_handle must refer to a texture that was already created with this Renderer using [create_texture](#create_texture).

Parameters:
    * UniformLayoutHandle layout_handle - the handle of a uniform layout, this should reference one of the layouts in the configuration, and can be gotten using [get_uniform_layout_handle](#get_uniform_layout_handle)
    * TextureHandle       texture_handle - the handle this uniform will refer to.

Return Value
    * std::optional<UniformHandle> - The function returns std::nullopt if it encounters an error. If this operation fails, one can look at the generated logs for more verbose output of what went wrong.

---
```
template <typename... Args>
void gfx::Renderer::update_uniform(UniformHandle handle, Args &&... args);
```

This function updates the Uniform for this UniformHandle. At the moment, args must be a `VkDeviceSize`, and `void*`, as this function can only update uniforms with the layout UNIFORM_BUFFER_DYNAMIC. The handle must refer to a uniform that was already created with this Renderer using [new_uniform](#new_uniform).

Parameters:
    * UniformHandle handle - the handle of a uniform created with layout UNIFORM_BUFFER_DYNAMIC
    * Args... - must be a `VkDeviceSize`, and `void*`

---
`void gfx::Renderer::delete_uniforms(size_t uniform_count, UniformHandle * uniforms);`

This function deletes all the uniforms pointed to by `uniforms` when it is safe to do so.

Parameters:
    * size_t uniform_count - the number of uniforms in the uniform array
    * UniformHandle * uniforms - a pointer to an array of UniformHandles

---
```
std::optional<BufferHandle> gfx::Renderer::create_buffer(VkDeviceSize          size,
                                                         VkBufferUsageFlags    usage,
                                                         VkMemoryPropertyFlags properties);
```

This function creates a buffer for loading data into.

Parameters:
    * VkDeviceSize          size - this size of the buffer
    * VkBufferUsageFlags    usage - how it will be used
    * VkMemoryPropertyFlags properties - the buffers memory properties.

---
`void update_buffer(BufferHandle buffer, VkDeviceSize size, void * data);`

updates a previously created buffer.

Parameters:
    * BufferHandle buffer - the buffer to update
    * VkDeviceSize size - the size of the data to update it with
    * void * data - the data to update it with

---
`void delete_buffers(size_t buffer_count, BufferHandle * buffers);`

deletes all the buffers in the array `buffers` when it is safe to do so.

Parameters:
    * size_t buffer_count - the number of buffers in the array
    * BufferHandle * buffers - a pointer to an array of buffers

---
`std::optional<TextureHandle> create_texture(char const * texture_path);`

creates a texture with the file at the path texture_path

Parameters:
    * char const * texture_path - the path of the image file

---
`void delete_textures(size_t texture_count, TextureHandle * textures);`

deletes all the textures in the array `textures` when it is safe to do so.

Parameters:
    * size_t texture_count - the number of textures in the array
    * TextureHandle * textures - a pointer to an array of textures

---
