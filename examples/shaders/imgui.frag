#version 450 core

layout(set=0, binding=0) uniform sampler2D sTexture;

layout(location = 0) in struct {
    vec4 Color;
    vec2 UV;
} In;

layout(location = 0) out vec4 fColor;

void main()
{
    fColor = In.Color.rgba * texture(sTexture, In.UV.st);
}
