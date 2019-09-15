
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#define JED_LOG_IMPLEMENTATION
#include "log/logger.hpp"
#undef JED_LOG_IMPLEMENTATION

#define JED_GFX_IMPLEMENTATION
#include "gfx/render_device.hpp"
#undef JED_GFX_IMPLEMENTATION

#define JED_CMD_IMPLEMENTATION
#include "cmd/cmd.hpp"
#undef JED_CMD_IMPLEMENTATION

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <utility>
#include <algorithm>

// clang-format off
std::vector<glm::vec3> vertices{{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}};

std::vector<glm::vec3> number_vertices{
    {0.f, 0.f, 0.f}, // row 0
    {1.f, 0.f, 0.f}, //
    {2.f, 0.f, 0.f}, //
    {3.f, 0.f, 0.f}, //
    {0.f, 1.f, 0.f}, // row 1
    {1.f, 1.f, 0.f}, //
    {2.f, 1.f, 0.f}, //
    {3.f, 1.f, 0.f}, //
    {0.f, 2.f, 0.f}, // row 2
    {1.f, 2.f, 0.f}, //
    {2.f, 2.f, 0.f}, //
    {3.f, 2.f, 0.f}, //
    {0.f, 3.f, 0.f}, // row 3
    {1.f, 3.f, 0.f}, //
    {2.f, 3.f, 0.f}, //
    {3.f, 3.f, 0.f}, //
    {0.f, 4.f, 0.f}, // row 4
    {1.f, 4.f, 0.f}, //
    {2.f, 4.f, 0.f}, //
    {3.f, 4.f, 0.f}, //
    {0.f, 5.f, 0.f}, // row 5
    {1.f, 5.f, 0.f}, //
    {2.f, 5.f, 0.f}, //
    {3.f, 5.f, 0.f} //
};

std::vector<uint32_t> zero_indices{
    0, 3, 7, 0, 7, 4,    // top
    0, 1, 21, 0, 21, 20, // left
    2, 3, 23, 2, 23, 22, // right
    16, 19, 23, 16, 23, 20 // bottom
};

std::vector<uint32_t> one_indices{
    1, 2, 22, 1, 22, 21
};

std::vector<uint32_t> two_indices{
    0, 3, 7, 0, 7, 4,    // top
    6, 7, 11, 6, 11, 10, // top right square
    8, 11, 15, 8, 15, 12, // middle
    12, 13, 17, 12, 17, 16, // bottom left square
    16, 19, 23, 16, 23, 20 // bottom
};

std::vector<uint32_t> three_indices{
    0, 3, 7, 0, 7, 4,    // top
    6, 7, 11, 6, 11, 10, // top right square
    8, 11, 15, 8, 15, 12, // middle
    14, 15, 19, 14, 19, 18, // bottom right square
    16, 19, 23, 16, 23, 20 // bottom
};

std::vector<uint32_t> four_indices{
    0, 1, 13, 0, 13, 12, // left fork
    8, 11, 15, 8, 15, 12, // middle
    2, 3, 23, 2, 23, 22 // right
};

std::vector<uint32_t> five_indices{
    0, 3, 7, 0, 7, 4,    // top
    4, 5, 9, 4, 9, 8, // top left square
    8, 11, 15, 8, 15, 12, // middle
    14, 15, 19, 14, 19, 18, // bottom right square
    16, 19, 23, 16, 23, 20 // bottom
};

std::vector<uint32_t> six_indices{
    0, 3, 7, 0, 7, 4,    // top
    4, 5, 9, 4, 9, 8,   // top left square
    8, 11, 15, 8, 15, 12, // middle
    12, 13, 17, 12, 17, 16, // bottom left square
    14, 15, 19, 14, 19, 18, // bottom right square
    16, 19, 23, 16, 23, 20 // bottom
};

std::vector<uint32_t> seven_indices{
    0, 3, 7, 0, 7, 4,    // top
    2, 3, 23, 2, 23, 22 // right
};

std::vector<uint32_t> indices{0, 1, 2, 0, 2, 3};

// clang-format on

gfx::BufferHandle make_vertex_buffer(gfx::Renderer &          render_device,
                                     std::vector<glm::vec3> & vertices)
{
    auto handle = render_device
                      .create_buffer(
                          vertices.size() * sizeof(glm::vec3),
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                      .value();

    render_device.update_buffer(handle, vertices.size() * sizeof(glm::vec3), vertices.data());

    return handle;
}

gfx::BufferHandle make_index_buffer(gfx::Renderer & render_device, std::vector<uint32_t> & indices)
{
    auto handle = render_device
                      .create_buffer(
                          indices.size() * sizeof(uint32_t),
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                      .value();

    render_device.update_buffer(handle, indices.size() * sizeof(uint32_t), indices.data());

    return handle;
}

class Object
{
public:
    gfx::BufferHandle vertex_buffer;
    gfx::BufferHandle index_buffer;
    size_t            index_count;

    glm::vec2 pos;
    glm::vec2 dim;

    glm::mat4 transform;

    Object(gfx::BufferHandle vertex_handle,
           gfx::BufferHandle index_handle,
           size_t            indices_count,
           glm::vec2 const & i_pos,
           glm::vec2 const & i_dim)
    : vertex_buffer{vertex_handle},
      index_buffer{index_handle},
      index_count{indices_count},
      pos{i_pos},
      dim{i_dim},
      transform{1.f}
    {
        transform = glm::scale(glm::translate(glm::mat4{1.f}, glm::vec3{pos.x, pos.y, 0.f}),
                               glm::vec3{dim.x, dim.y, 1.f});
    }

    std::pair<float, float> get_y_dim() const
    {
        return {pos.y, pos.y + dim.y};
    }

    std::pair<float, float> get_x_dim() const
    {
        return {pos.x, pos.x + dim.x};
    }

    void draw(gfx::Renderer &     render_device,
              gfx::PipelineHandle pipeline_handle,
              gfx::UniformHandle  view_handle)
    {
        render_device.draw(pipeline_handle,
                           vertex_buffer,
                           0,
                           index_buffer,
                           0,
                           index_count,
                           sizeof(glm::mat4),
                           glm::value_ptr(transform),
                           1,
                           &view_handle);
    }

    void draw(gfx::Renderer &     render_device,
              gfx::PipelineHandle pipeline_handle,
              gfx::UniformHandle  view_handle,
              glm::mat4 &         n_transform)
    {
        render_device.draw(pipeline_handle,
                           vertex_buffer,
                           0,
                           index_buffer,
                           0,
                           index_count,
                           sizeof(glm::mat4),
                           glm::value_ptr(n_transform),
                           1,
                           &view_handle);
    }
};

struct Ball: public Object
{
public:
    glm::vec2 vel;

    Ball(gfx::BufferHandle vertex_handle,
         gfx::BufferHandle index_handle,
         uint32_t          indices_count,
         glm::vec2 const & i_pos,
         glm::vec2 const & i_dim,
         glm::vec2 const & i_vel)
    : Object{vertex_handle, index_handle, indices_count, i_pos, i_dim}, vel{i_vel}
    {}
};

float overlap(std::pair<float, float> const & dim_1, std::pair<float, float> const & dim_2)
{
    if (dim_1.first < dim_2.second && dim_1.first > dim_2.first)
    {
        return dim_2.second - dim_1.first;
    }

    if (dim_1.second < dim_2.second && dim_1.second > dim_2.first)
    {
        return dim_2.first - dim_1.second;
    }

    return 0.f;
}

std::pair<float, float> SAT(Object const & paddle, Ball const & ball)
{
    auto paddle_x_axis = paddle.get_x_dim();
    auto paddle_y_axis = paddle.get_y_dim();

    auto ball_x_axis = ball.get_x_dim();
    auto ball_y_axis = ball.get_y_dim();

    auto overlap_x = overlap(ball_x_axis, paddle_x_axis);
    auto overlap_y = overlap(ball_y_axis, paddle_y_axis);

    return {overlap_x, overlap_y};
}

bool move_up{false};
bool move_down{false};
bool enter{false};

enum class States
{
    WAIT,
    PLAY,
    LOSE_POINT,
    LOSE
};

States wait_to_start_step(Object &   paddle_1,
                          Object &   paddle_2,
                          Ball &     ball,
                          uint32_t & player_score,
                          uint32_t & ai_score)
{
    if (enter)
    {
        LOG_INFO("Moving to Play state");
        return States::PLAY;
    }

    return States::WAIT;
}

States lose_point_step(Object &   paddle_1,
                       Object &   paddle_2,
                       Ball &     ball,
                       uint32_t & player_score,
                       uint32_t & ai_score)
{
    if (enter)
    {
        ball.pos = {2.f, 19.5f};
        ball.vel = {.2f, .2f};

        ball.transform = glm::scale(
            glm::translate(glm::mat4{1.f}, glm::vec3{ball.pos.x, ball.pos.y, 0.f}),
            glm::vec3{ball.dim.x, ball.dim.y, 1.f});

        paddle_1.pos = {1.f, 17.5f};

        paddle_2.pos = {58.f, 17.5f};

        paddle_1.transform = glm::scale(
            glm::translate(glm::mat4{1.f}, glm::vec3{paddle_1.pos.x, paddle_1.pos.y, 0.f}),
            glm::vec3{paddle_1.dim.x, paddle_1.dim.y, 1.f});

        paddle_2.transform = glm::scale(
            glm::translate(glm::mat4{1.f}, glm::vec3{paddle_2.pos.x, paddle_2.pos.y, 0.f}),
            glm::vec3{paddle_2.dim.x, paddle_2.dim.y, 1.f});
        return States::WAIT;
    }

    return States::LOSE_POINT;
}

States game_step(Object &   paddle_1,
                 Object &   paddle_2,
                 Ball &     ball,
                 uint32_t & player_score,
                 uint32_t & ai_score)
{
    ball.pos += ball.vel;

    // hits wall
    if (ball.pos.y > 39.f)
    {
        ball.pos.y = 39.f;
        ball.vel.y *= -1;
    }

    // hits wall
    if (ball.pos.y < 0.f)
    {
        ball.pos.y = 0.f;
        ball.vel.y *= -1;
    }

    // hits paddle 2
    auto overlap_pair = SAT(paddle_2, ball);

    if (overlap_pair.first != 0.f && overlap_pair.second != 0.f)
    {
        LOG_INFO("Collision with AI Paddle. x: {}, y: {}", overlap_pair.first, overlap_pair.second);

        if (std::abs(overlap_pair.first) < std::abs(overlap_pair.second))
        {
            LOG_INFO("Overlap was smaller on x axis");
            ball.pos.x = paddle_2.pos.x - 1.f;
            ball.vel.x *= -1;
            ball.vel.y += ((ball.pos.y + .5f) - (paddle_2.pos.y + 2.5f)) / 5;
        }
        else
        {
            LOG_INFO("Overlap was smaller on y axis");
            ball.pos.y += overlap_pair.second;
            ball.vel.y *= -1;
        }
    }

    overlap_pair = SAT(paddle_1, ball);
    if (overlap_pair.first != 0.f && overlap_pair.second != 0.f)
    {
        LOG_INFO(
            "Collision with Player Paddle. x: {}, y: {}", overlap_pair.first, overlap_pair.second);

        if (std::abs(overlap_pair.first) < std::abs(overlap_pair.second))
        {
            LOG_INFO("Overlap was smaller on x axis");
            ball.pos.x = paddle_1.pos.x + 1.f;
            ball.vel.x *= -1;
            ball.vel.y += ((ball.pos.y + .5f) - (paddle_1.pos.y + 2.5f)) / 5;
        }
        else
        {
            LOG_INFO("Overlap was smaller on y axis");
            ball.pos.y += overlap_pair.second;
            ball.vel.y *= -1;
        }
    }

    if (move_up)
    {
        paddle_1.pos.y -= .5f;

        if (paddle_1.pos.y < 0.f)
        {
           paddle_1.pos.y = 0.f; 
        }
    }

    if (move_down)
    {
        paddle_1.pos.y += .5f;

        if (paddle_1.pos.y + paddle_1.dim.y > 40.f)
        {
            paddle_1.pos.y = 40.f - paddle_1.dim.y;
        }
    }

    if (paddle_2.pos.y + 1 > ball.pos.y + 1.f)
    {
        paddle_2.pos.y -= std::min(.5f, (paddle_2.pos.y + 1) - (ball.pos.y + 1.f));

        if (paddle_2.pos.y < 0.f)
        {
           paddle_2.pos.y = 0.f; 
        }
    }

    if (paddle_2.pos.y + 4.f < ball.pos.y)
    {
        paddle_2.pos.y += std::min(.5f, (ball.pos.y) - (paddle_2.pos.y + 4.f));

        if (paddle_2.pos.y + paddle_2.dim.y > 40.f)
        {
            paddle_2.pos.y = 40.f - paddle_2.dim.y;
        }
    }

    paddle_1.transform = glm::scale(
        glm::translate(glm::mat4{1.f}, glm::vec3{paddle_1.pos.x, paddle_1.pos.y, 0.f}),
        glm::vec3{paddle_1.dim.x, paddle_1.dim.y, 1.f});

    paddle_2.transform = glm::scale(
        glm::translate(glm::mat4{1.f}, glm::vec3{paddle_2.pos.x, paddle_2.pos.y, 0.f}),
        glm::vec3{paddle_2.dim.x, paddle_2.dim.y, 1.f});

    ball.transform = glm::scale(
        glm::translate(glm::mat4{1.f}, glm::vec3{ball.pos.x, ball.pos.y, 0.f}),
        glm::vec3{ball.dim.x, ball.dim.y, 1.f});

    bool reset = false;

    if (ball.pos.x < 0.f)
    {
        ai_score++;
        reset = true;
    }

    if (ball.pos.x > 59.f)
    {
        player_score++;
        reset = true;
    }

    if (reset)
    {
        if (ai_score == 7 || player_score == 7)
        {
            return States::LOSE;
        }
        else
        {
            return States::LOSE_POINT;
        }
    }

    return States::PLAY;
}

States lose_step(Object &   paddle_1,
                 Object &   paddle_2,
                 Ball &     ball,
                 uint32_t & player_score,
                 uint32_t & ai_score)
{
    if (enter)
    {
        LOG_INFO("Moving to Wait state");

        ball.pos = {2.f, 19.5f};
        ball.vel = {.2f, .2f};

        ball.transform = glm::scale(
            glm::translate(glm::mat4{1.f}, glm::vec3{ball.pos.x, ball.pos.y, 0.f}),
            glm::vec3{ball.dim.x, ball.dim.y, 1.f});

        paddle_1.pos = {1.f, 17.5f};

        paddle_2.pos = {58.f, 17.5f};

        paddle_1.transform = glm::scale(
            glm::translate(glm::mat4{1.f}, glm::vec3{paddle_1.pos.x, paddle_1.pos.y, 0.f}),
            glm::vec3{paddle_1.dim.x, paddle_1.dim.y, 1.f});

        paddle_2.transform = glm::scale(
            glm::translate(glm::mat4{1.f}, glm::vec3{paddle_2.pos.x, paddle_2.pos.y, 0.f}),
            glm::vec3{paddle_2.dim.x, paddle_2.dim.y, 1.f});

        player_score = 0;
        ai_score     = 0;

        return States::WAIT;
    }

    return States::LOSE;
}

void key_callback(GLFWwindow * window, int key, int scancode, int action, int mods)
{
    switch (key)
    {
    case GLFW_KEY_UP:
        move_up = action != GLFW_RELEASE;
        break;
    case GLFW_KEY_DOWN:
        move_down = action != GLFW_RELEASE;
        break;
    case GLFW_KEY_ENTER:
        enter = action != GLFW_RELEASE;
        break;
    }
}

int main()
{
    get_console_sink()->set_level(spdlog::level::info);
    get_file_sink()->set_level(spdlog::level::trace);
    get_logger()->set_level(spdlog::level::debug);

    LOG_INFO("Starting Pong");

    if (glfwInit() == GLFW_FALSE)
    {
        LOG_ERROR("GLFW didn't initialize correctly");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    auto window = glfwCreateWindow(600, 400, "Pong", nullptr, nullptr);

    glfwSetKeyCallback(window, key_callback);

    auto render_device = gfx::Renderer{window};

    auto render_config = gfx::RenderConfig{.config_filename = "../examples/pong_config.json"};

    render_config.init();

    if (!render_device.init(render_config))
    {
        LOG_ERROR("Couldn't initialize the Renderer");
        return 0;
    }

    auto opt_layout_handle = render_device.get_uniform_layout_handle("ul_camera_matrix");

    glm::mat4 view = glm::ortho(0.0f, 60.f, 0.0f, 40.f);

    gfx::UniformHandle view_handle = render_device
                                         .new_uniform(opt_layout_handle.value(),
                                                      sizeof(glm::mat4),
                                                      glm::value_ptr(view))
                                         .value();

    auto pipeline_handle = render_device.get_pipeline_handle("pong_shader").value();

    auto object_vertex_buffer = make_vertex_buffer(render_device, vertices);
    auto object_index_buffer  = make_index_buffer(render_device, indices);

    auto number_vertex_buffer = make_vertex_buffer(render_device, number_vertices);
    auto zero_index_buffer    = make_index_buffer(render_device, zero_indices);
    auto one_index_buffer     = make_index_buffer(render_device, one_indices);
    auto two_index_buffer     = make_index_buffer(render_device, two_indices);
    auto three_index_buffer   = make_index_buffer(render_device, three_indices);
    auto four_index_buffer    = make_index_buffer(render_device, four_indices);
    auto five_index_buffer    = make_index_buffer(render_device, five_indices);
    auto six_index_buffer     = make_index_buffer(render_device, six_indices);
    auto seven_index_buffer   = make_index_buffer(render_device, seven_indices);

    Object paddle_1{
        object_vertex_buffer, object_index_buffer, indices.size(), {1.f, 17.5f}, {1.f, 5.f}};

    Object paddle_2{
        object_vertex_buffer, object_index_buffer, indices.size(), {58.f, 17.5f}, {1.f, 5.f}};

    Ball ball{object_vertex_buffer,
              object_index_buffer,
              static_cast<uint32_t>(indices.size()),
              {2.f, 19.5f},
              {1.f, 1.f},
              {.2f, .2f}};

    std::vector<Object> score_objects;

    glm::vec2 empty_vec{0.f, 0.f};

    score_objects.emplace_back(
        number_vertex_buffer, zero_index_buffer, zero_indices.size(), empty_vec, empty_vec);
    score_objects.emplace_back(
        number_vertex_buffer, one_index_buffer, one_indices.size(), empty_vec, empty_vec);
    score_objects.emplace_back(
        number_vertex_buffer, two_index_buffer, two_indices.size(), empty_vec, empty_vec);
    score_objects.emplace_back(
        number_vertex_buffer, three_index_buffer, three_indices.size(), empty_vec, empty_vec);
    score_objects.emplace_back(
        number_vertex_buffer, four_index_buffer, four_indices.size(), empty_vec, empty_vec);
    score_objects.emplace_back(
        number_vertex_buffer, five_index_buffer, five_indices.size(), empty_vec, empty_vec);
    score_objects.emplace_back(
        number_vertex_buffer, six_index_buffer, six_indices.size(), empty_vec, empty_vec);
    score_objects.emplace_back(
        number_vertex_buffer, seven_index_buffer, seven_indices.size(), empty_vec, empty_vec);

    glm::mat4 left_player_score_transform  = glm::translate(glm::mat4{1.f},
                                                           glm::vec3{24.f, 2.f, 0.f});
    glm::mat4 right_player_score_transform = glm::translate(glm::mat4{1.f},
                                                            glm::vec3{33.f, 2.f, 0.f});

    States   current_state = States::WAIT;
    uint32_t player_score{0};
    uint32_t ai_score{0};

    bool draw_success{true};
    while (!glfwWindowShouldClose(window) && draw_success)
    {
        glfwPollEvents();

        if (current_state == States::WAIT)
        {
            current_state = wait_to_start_step(paddle_1, paddle_2, ball, player_score, ai_score);
        }
        else if (current_state == States::PLAY)
        {
            current_state = game_step(paddle_1, paddle_2, ball, player_score, ai_score);
        }
        else if (current_state == States::LOSE_POINT)
        {
            current_state = lose_point_step(paddle_1, paddle_2, ball, player_score, ai_score);
        }
        else if (current_state == States::LOSE)
        {
            current_state = lose_step(paddle_1, paddle_2, ball, player_score, ai_score);
        }

        paddle_1.draw(render_device, pipeline_handle, view_handle);
        paddle_2.draw(render_device, pipeline_handle, view_handle);
        ball.draw(render_device, pipeline_handle, view_handle);

        score_objects[player_score].draw(
            render_device, pipeline_handle, view_handle, left_player_score_transform);
        score_objects[ai_score].draw(
            render_device, pipeline_handle, view_handle, right_player_score_transform);

        draw_success = render_device.submit_frame();
    }

    render_device.wait_for_idle();

    render_device.quit();

    glfwDestroyWindow(window);

    glfwTerminate();

    LOG_INFO("Stopping Pong\n");
}