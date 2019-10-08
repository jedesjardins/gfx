#ifndef GFX_EXAMPLES_COMMON_HPP_
#define GFX_EXAMPLES_COMMON_HPP_

#include "gfx/renderer.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

gfx::ErrorCode readFile(char const * file_name, std::vector<char> & buffer);

struct BufferUniform
{
    gfx::UniformHandle uniform_handle;
    gfx::BufferHandle  buffer_handle;
};

BufferUniform make_matrix_uniform(gfx::Renderer &     renderer,
                                  std::string const & set_name,
                                  glm::mat4 &         view_matrix);

struct TextureUniform
{
    gfx::UniformHandle uniform_handle;
    gfx::TextureHandle texture_handle;
};

TextureUniform make_texture_uniform(gfx::Renderer &            renderer,
                                    std::string const &        set_name,
                                    gfx::TextureHandle const & texture_handle);

#endif