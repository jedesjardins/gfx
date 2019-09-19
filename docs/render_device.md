# The Renderer class

The Renderer class, in the gfx namespace, is the main interface used to do graphics operations on. It requires a GLFWwindow * to create at the moment, as it relies on GLFW to manage the window. It is configured with [the RenderConfig class](docs/render_config.md), to define what resources it can use, and how it draws to the screen. 

## API

#### `explicit gfx::Renderer::Renderer(GLFWwindow * window_ptr);`

Creates a Renderer object. The Renderer must still be initialized after calling the constructor.

Parameters:
	* `GLFWwindow * window_ptr` - The Renderer will draw to the window designated by this window pointer. The window must have already been created with `glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);` to show that it will use vulkan, not opengl.

---
#### `bool gfx::Renderer::init(RenderConfig & render_config);`

Initializes the Renderer with the given RenderConfig. This puts the Renderer in a valid state to use.

Parameters:
	* `RenderConfig & render_config` - The configuration that this Renderer will use. The RenderConfig must have already been initialized before it can be used to initialize the Renderer.

Return Value
	* `bool` - signifies the success or failure of the operation. If this operation fails, one can look at the generated logs

---
#### <a name="quit"> `void gfx::Renderer::quit();`

Destroys all the graphics resources managed by the Renderer. The Renderer must be idle when quit is called, which can be ensured with a call to [wait_for_idle](#wait_for_idle). If the Renderer is not idle then this function can cause errors.

---
#### <a name="wait_for_idle"></a> `void gfx::Renderer::wait_for_idle();`

Waits until all of the Renderers current operations have completed on the graphics card. This operation must be run before calling [quit](#quit)

---
#### `bool gfx::Renderer::submit_frame();`

Submits all queued actions since the last call to submit_frame, or Renderer creation, to be executed on the graphics card. 

Return Value
	* `bool` - signifies the success or failure of the operation. If this operation fails, one can look at the generated logs

---
#### ```ErrorCode gfx::Renderer::draw(PipelineHandle  pipeline,
                                    size_t          vertices_size,
                                    void *          vertices,
                                    uint32_t        index_count,
                                    uint32_t *      indices,
                                    size_t          push_constant_size,
                                    void *          push_constant_data,
                                    size_t          uniform_count,
                                    UniformHandle * p_uniforms);
```

Draw

---

