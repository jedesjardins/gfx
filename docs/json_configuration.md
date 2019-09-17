# JSON Configuration file

To initialize the Vulkan Renderer you need a JSON configuration file that outlines how you will use it.

## Top level fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |
| window_name | string | the name of the window | Y |
| dynamic_vertex_buffer_size | uint | the size of the vertex buffer used to upload dynamic vertices every frame | Y |
| dynamic_vertex_buffer_size | uint | the size of the vertex buffer used to upload dynamic vertices every frame | Y |
| staging_buffer_size| uint | the size of the staging buffer used to upload data to Device Local buffers, Textures, etc. | Y |
| max_updated_objects | uint | the maximum number of device memory updates per frame | Y |
| max_deleted_objects | uint | the maximum number of object delete calls per frame | Y |
| attachments | array of (Attachment objects)[#Attachment] | describes the number and type of attachments | Y |
| renderpasses | array of Renderpass objects | describes the number and layout of renderpasses | Y |
| renderpass_order | array of strings | order of renderpasses to execute, where each string is the name of the renderpass | Y |
| shaders | array of Shader objects | describes all shaders, vertex and fragment, that are used | Y |
| uniform_layouts | array of Uniform Layout objects | describes all uniform types that are used | Y |
| push_constants | array of Push Constant objects | describes all push constants that are used | Y |
| vertex_bindings | array of Vertex Binding objects | describes all vertex bindings that are used | Y |
| vertex_attributes | array of Vertex Attribute objects | describes all vertex attributes that are used | Y |
| pipelines | array of Pipeline objects | describes all pipelines that are used | Y |

### <a name="Attachment"></a> Attachment object fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |
| name | string | name to reference the attachment by | Y |
| format | string | either "color" or "depth" | Y |
| multisampled | boolean | whether or not this attachment is multisampled | N |
| screen_image | boolean | if this attachment refers to screen memory | N |

###	Renderpass object fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |


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