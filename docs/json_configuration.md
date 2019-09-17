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
| attachment_name | string | name of the attachment | Y |
| load_op | string | the [load operation](#Load_Op) for this attachment | Y |
| store_op | string | the [store operation](#Store_Op) for this attachment | Y |
| initial_layout | string | the initial [Image Layout](#Image_Layout) of this attachment | Y |
| final_layout | string | the [Image Layout](#Image_Layout) this attachment will be in at the end of the renderpass | Y |
| clear value | [Clear Value object](#Clear_Value) | | required if load_op is clear |

#### <a name="Load_Op"></a> Attachment Load Operations

These operations describe how an attachment is loaded at the beginning of a Renderpass

| Value | Description |
| ----- | ----------- |
| "LOAD" | use if you want to keep the contents of the attachment from a previous renderpass |
| "CLEAR" | use if you want to clear the contents of the attachment before the renderpass starts |
| "DONT_CARE" | use if the contents don't matter, i.e. you will entirely overwrite the contents without using them |

#### <a name="Store_Op"></a> Attachment Store Operations

These operations describe how an attachment is stored at the end of a Renderpass

| Value | Description |
| ----- | ----------- |
| "STORE" | use if you want to use the contents of the attachment in a later renderpass |
| "DONT_CARE" | use if the contents are only used in this renderpass |

#### <a name="Image_Layout"></a> Image Layouts

| Value | Description |
| ----- | ----------- |
| "UNDEFINED" | Can only be used as an initial layout. When transitioning from this layout, contents are not preserved |
| "PREINITIALIZED" | Can only be used as an initial layout. When transitioning from this layout, contents are preserved |
| "GENERAL" | Supports all types of usage but is suboptimal |
| "COLOR_ATTACHMENT_OPTIMAL" | Attachments in this layout can only be used as a color or resolve attachment, the format of the attachment must also have been "color" |
| "DEPTH_STENCIL_ATTACHMENT_OPTIMAL" | Attachments in this layout can only be used as a depth/stencil attachment, the format of the attachment must also have been "depth" |
| "DEPTH_STENCIL_READ_ONLY_OPTIMAL" | Attachments in this layout can only be used as a read only depth/stencil attachment or as a read only image in a shader, the format of the attachment must also have been "depth" |
| "SHADER_READ_ONLY_OPTIMAL" | Attachments in this format can only be used as a read only image in a shader |
| "TRANSFER_SRC_OPTIMAL" | Attachments in this format can only be used as the source for a transfer operation |
| "TRANSFER_DST_OPTIMAL" | Attachments in this format can only be used as the destination in a transfer operation |
| "DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL" | Attachments in this layout can only be used as a depth/stencil attachment where the depth aspect is read only, or as a read only image in a shader, the format of the attachment must also have been "depth" |
| "DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL" | Attachments in this layout can only be used as a depth/stencil attachment where the stencil aspect is read only, or as a read only image in a shader, the format of the attachment must also have been "depth" |
| "PRESENT_SRC_KHR" | Attachments in this layout are used to present to the screen. Attachments to be used in this layout must have been created with their "is_swapchain_image" field set to true |
| "SHARED_PRESENT_KHR" | Used for Shared Presentable images |

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