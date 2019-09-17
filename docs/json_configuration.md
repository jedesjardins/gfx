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
| usage | array of [Attachment Usages](#Attachment_Usage) | Lists all possible usages of this attachment | Y |
| multisampled | boolean | whether or not this attachment is multisampled | N |
| is_swapchain_image | boolean | if this attachment refers to the screen swapchain  | N |


#### <a name="Attachment_Usage"></a> Attachment Usage values

| Value | Description |
| ----- | ----------- |


###	<a name="Renderpass"></a> Renderpass object fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |
| name | string | name to reference the Renderpass by | Y |
| framebuffer | array of [Attachment Description objects](#Attachment_Description) | the framebuffer object this Renderpass uses | Y |
| subpasses | Array of [Subpass objects](#Subpass_Description) | the subpasses in this renderpass | Y |
| subpass_dependencies | Array of [Subpass Dependency objects](#Subpass_Dependencies) | the inter subpass dependencies | Y |


### <a name="Attachment_Description"></a> Attachment Description object fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |
| attachment_name | string | name of the attachment | Y |
| load_op | string | the [load operation](#Load_Op) for this attachment | Y |
| store_op | string | the [store operation](#Store_Op) for this attachment | Y |
| initial_layout | string | the initial [Image Layout](#Image_Layout) of this attachment | Y |
| final_layout | string | the [Image Layout](#Image_Layout) this attachment will be in at the end of the renderpass | Y |
| clear value | [Clear Value object](#Clear_Value) | this is the value that the attachment is cleared to | required if load_op is CLEAR |


#### <a name="Load_Op"></a> Attachment Load Operation values

These operations describe how an attachment is loaded at the beginning of a Renderpass

| Value | Description |
| ----- | ----------- |
| "LOAD" | use if you want to keep the contents of the attachment from a previous renderpass |
| "CLEAR" | use if you want to clear the contents of the attachment before the renderpass starts |
| "DONT_CARE" | use if the contents don't matter, i.e. you will entirely overwrite the contents without using them |


#### <a name="Store_Op"></a> Attachment Store Operation values

These operations describe how an attachment is stored at the end of a Renderpass

| Value | Description |
| ----- | ----------- |
| "STORE" | use if you want to use the contents of the attachment in a later renderpass |
| "DONT_CARE" | use if the contents are only used in this renderpass |


#### <a name="Image_Layout"></a> Image Layout values

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


#### <a name="Clear_Value"></a> Clear Values

| Value | Description |
| ----- | ----------- |
| color | array with 4 number elements. The values should be between 0 and 1. The elements correspond to the Red, Green, Blue, and Alpha conponents |
| depth | array with 2 number elements. The first value is what the depth buffer is cleared to and should be between -1 and 1, the second is what the stencil buffer is cleared to and should be positive.


###	<a name="Subpass_Description"></a> Subpass object fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |
| name | string | the name of this subpass | Y |
| color_attachments | array of [Subpass Attachment objects](#Subpass_Attachment) | all the color attachments for this subpass | Y |
| resolve_attachment | [Subpass Attachment](#Subpass_Attachment) | the multisample resolve attachment for this subpass | Y |
| depth_stencil_attachment | [Subpass Attachment](#Subpass_Attachment) | the depth/stencil attachment for this subpass | Y |


###	<a name="Subpass_Attachment"></a> Subpass Attachment fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |
| attachment_name | string | name of the subpass | Y |
| layout | [Image Layout](#Image_Layout) | the layout the attachment will be in for this subpass | Y |


###	<a name="Subpass_Dependencies"></a> Subpass Dependency object fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |
| src_subpass | string | the name of the first subpass in the dependency or EXTERNAL_SUBPASS to refer to before the first subpass in the Renderpass | N |
| src_stage_mask | array of [Pipeline Stages](#Pipeline_Stage) | the stage in the src_subpass to wait on | Y |
| src_access_mask | array of [Access Flags](#Access_Flags) | the type of access to wait on in the src_subpass and src_stage | N |
| dst_subpass | string | the name of the second subpass in the dependency. If empty, it defaults to after the last subpass in the Renderpass | N |
| dst_stage_mask | array of [Pipeline Stages](#Pipeline_Stage) | the stage in the dst_subpass that is dependant | Y |
| dst_access_mask | array of [Access Flags](#Access_Flags) | | N |


#### <a name="Pipeline_Stage"></a> Pipeline Stage values

| Value | Description |
| ----- | ----------- |


#### <a name="Access_Flags"></a> Access Flag values

| Value | Description |
| ----- | ----------- |


###	<a name="Shader"></a> Shader object fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |
| name | string | the name of this shader | Y |
| file | string | the path of the file for this shader, the file must be a SPIRV shader | Y |

###	<a name="Uniform_Layout"></a> Uniform Layout object fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |
| name | string | the name of this uniform layout | Y |
| binding | uint | which slot this uniform will be bound to | Y |
| descriptor_type | [Uniform Type](#Uniform_Type) | the type of this uniform | Y |
| descriptor_count | uint | must be 1 for now | Y |
| stage | array of [Shader Stage](#Shader_Stage) | the shader stages where this Uniform is used | Y |
| max_count | uint | the maximum number of uniforms that can be created for this Layout | Y |


#### <a name="Uniform_Type"></a> Uniform Type values

| Value | Description |
| ----- | ----------- |


#### <a name="Shader_Stage"></a> Shader Stage values

| Value | Description |
| ----- | ----------- |


###	<a name="Push_Constant"></a> Push Constant object fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |
| name | string | the name of this push constant | Y |
| stage | array of [Shader Stage](#Shader_Stage) | the shader stages where this Uniform is used | Y |
| offset | uint | the offset into the push constant block of memory | Y |
| size | uint | the size of the push constant memory | Y |

###	<a name="Vertex_Binding"></a> Vertex Binding object fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |
| name | string | the name of this vertex binding | Y |
| binding_slot | uint | which slot this vertex type will be bound to | Y |
| stride | uint | the size of each vertex | Y |
| input_rate | an [Input Rate](#Input_Rate) | the rate at which this vertex binding is consumed | Y |


#### <a name="Input_Rate"></a> Input Rate values

| Value | Description |
| ----- | ----------- |


###	<a name="Vertex_Attribute"></a> Vertex Attribute object fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |
| name | string | the name of this vertex attribute | Y |
| vertex_binding_name | string | which vertex binding this vertex attribute belongs to | Y |
| location | uint | the location this attribute is made available in the shader | Y |
| format | an [Format value](#Format) | the rate at which this vertex binding is consumed | Y |
| offset | uint | the offset into the vertex where this attribute starts | Y |


#### <a name="Format"></a> Input Rate values

| Value | Description |
| ----- | ----------- |


###	<a name="Pipeline"></a> Pipeline object fields

| Field Name | Type | Description | Mandatory |
| ---------- | ---- | ----------- | --------- |

