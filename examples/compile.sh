mkdir ../build/shaders
glslc -fshader-stage=vert shaders/shader.vert -o ../build/shaders/vert.spv
glslc -fshader-stage=frag shaders/shader.frag -o ../build/shaders/frag.spv

glslc -fshader-stage=frag shaders/simple.frag -o ../build/shaders/simple-frag.spv

glslc -fshader-stage=frag shaders/square.frag -o ../build/shaders/square_frag.spv
glslc -fshader-stage=vert shaders/square.vert -o ../build/shaders/square_vert.spv

glslc -fshader-stage=frag shaders/vb.frag -o ../build/shaders/vb_frag.spv
glslc -fshader-stage=vert shaders/vb.vert -o ../build/shaders/vb_vert.spv

glslc -fshader-stage=vert shaders/pos.vert -o ../build/shaders/pos_vert.spv
glslc -fshader-stage=frag shaders/white_pos.frag -o ../build/shaders/white_pos_frag.spv

glslc -fshader-stage=vert shaders/pos_uv_texture.vert -o ../build/shaders/pos_uv_texture_vert.spv
glslc -fshader-stage=frag shaders/uv_texture.frag -o ../build/shaders/uv_texture_frag.spv