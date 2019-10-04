#version 450

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputColor;

void main() {
    outColor = subpassLoad(inputColor).rgba;
}