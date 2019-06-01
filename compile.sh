mkdir shaders
glslc -fshader-stage=vert ../examples/shaders/shader.vert -o shaders/vert.spv
glslc -fshader-stage=frag ../examples/shaders/shader.frag -o shaders/frag.spv