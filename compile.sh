mkdir shaders
glslc -fshader-stage=vert ../examples/shaders/shader.vert -o shaders/vert.spv
glslc -fshader-stage=frag ../examples/shaders/shader.frag -o shaders/frag.spv

glslc -fshader-stage=frag ../examples/shaders/simple.frag -o shaders/simple-frag.spv

glslc -fshader-stage=frag ../examples/shaders/square.frag -o shaders/square_frag.spv
glslc -fshader-stage=vert ../examples/shaders/square.vert -o shaders/square_vert.spv