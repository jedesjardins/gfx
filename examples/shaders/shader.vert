#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(push_constant) uniform PushConstants {
    layout(offset = 0) mat4 transform;
} constants;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = ubo.view * constants.transform * vec4(inPosition, 1.0);
    fragColor = inColor;
}