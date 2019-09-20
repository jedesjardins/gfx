mkdir ../build/shaders
glslc -fshader-stage=vert shaders/shader.vert -o ../build/shaders/vert.spv
glslc -fshader-stage=frag shaders/shader.frag -o ../build/shaders/frag.spv

glslc -fshader-stage=frag shaders/simple.frag -o ../build/shaders/simple-frag.spv

glslc -fshader-stage=frag shaders/square.frag -o ../build/shaders/square_frag.spv
glslc -fshader-stage=vert shaders/square.vert -o ../build/shaders/square_vert.spv