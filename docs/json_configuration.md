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
| attachments | array of [Attachment objects](#Attachment) | describes the number and type of attachments | Y |
| renderpasses | array of [Renderpass objects](#Renderpass) | describes the number and layout of renderpasses | Y |
| renderpass_order | array of strings | order of renderpasses to execute, where each string is the name of the renderpass | Y |
| shaders | array of [Shader objects](#Shader) | describes all shaders, vertex and fragment, that are used | Y |
| uniform_layouts | array of [Uniform Layout objects](#Uniform_Layout) | describes all uniform types that are used | Y |
| push_constants | array of [Push Constant objects](#Push_Constant) | describes all push constants that are used | Y |
| vertex_bindings | array of [Vertex Binding objects](#Vertex_Binding) | describes all vertex bindings that are used | Y |
| vertex_attributes | array of [Vertex Attribute objects](#Vertex_Attribute) | describes all vertex attributes that are used | Y |
| pipelines | array of [Pipeline objects](#Pipeline) | describes all pipelines that are used | Y |

### <a name="Attachment"></a> Attachment object fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |
| name | string | name to reference the attachment by | Y |
| format | string | either "color" or "depth" | Y |
| multisampled | boolean | whether or not this attachment is multisampled | N |
| screen_image | boolean | if this attachment refers to screen memory | N |

###	<a name="Renderpass"></a> Renderpass object fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |
| name | string | name to reference the Renderpass by | Y |
| framebuffer | array of [Attachment Description objects](#Attachment_Description) | the framebuffer object this Renderpass uses | Y |
| subpasses | Array of [Subpass objects](#Subpass_Description) | the subpasses in this renderpass | Y |
| subpass_dependencies | Array of [Subpass Dependency objects](#Subpass_Dependencies) | the inter subpass dependencies | Y |

###    <a name="Attachment_Description"></a> Attachment Description object fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |
| attachment_name | string | | Y |
| load_op | string | | Y |
| store_op | string | | Y |
| initial_layout | string | | Y |
| final_layout | string | | Y |
| clear value | [Clear Value object](#Clear_Value) | | required if load_op is clear |

###	<a name="Subpass_Description"></a> Subpass object fields

###	<a name="Subpass_Dependencies"></a> Subpass Dependency object fields

###	<a name="Shader"></a> Shader object fields

###	<a name="Uniform_Layout"></a> Uniform Layout object fields

###	<a name="Push_Constant"></a> Push Constant object fields

###	<a name="Vertex_Binding"></a> Vertex Binding object fields

###	<a name="Vertex_Attribute"></a> Vertex Attribute object fields

###	<a name="Pipeline"></a> Pipeline object fields



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