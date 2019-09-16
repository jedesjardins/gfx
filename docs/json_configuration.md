# JSON Configuration file

To initialize the Vulkan Renderer you need a JSON configuration file that outlines how you will use it.

## Fields

* window_name
	* type - string
	* usage - the name of the window

* dynamic_vertex_buffer_size
	* type - uint
	* usage - the size of the vertex buffer used to upload dynamic vertices every frame

* dynamic_index_buffer_size
	* type - uint
	* usage - the size of the index buffer used to upload dynamic indices every frame

* staging_buffer_size
	* type - uint
	* usage - this size of the staging buffer used to upload data to Device Local buffers, Textures, etc.

* max_updated_objects
	* type - uint
	* usage - the maximum number of device memory updates per frame

* max_deleted_objects
	* type - uint
	* usage - the maximum size of device updates per frame


* attachments
	* type - array of attachment objects
	* usage - used to describe the number and type of attachments created at configuration time
	* attachment object
		* name
			* type - string
			* usage - the name of this attachment, used to referenced this attachment from other configuration objects
		* format
			* type - string
			* usage - either "depth" or "color", used to declare if this is a color or depth/stencil attachment
		* multisampled
			* type - boolean
			* usage - says if this is a multisampled attachment
		* screen_image (optional)
			* type - boolean
			* usage - says if this attachment represents the screen

* renderpasses
	* type - array of renderpass objects
	* usage - used to describe the number and layout of renderpasses created at configuration time
	* renderpass object
		* name
			* type - string
			* usage - the name of this renderpass, used to referenced this renderpass from other configuration objects
		* framebuffer
			* type - framebuffer object
			* usage - the framebuffer that the enclosing renderpass object draws to
			* framebuffer object
				* attachments
					* type - array of attachment description objects
					* usage - names and usages of attachments
					* attachment description object
						* attachment name

		* subpasses
		* subpass_dependencies