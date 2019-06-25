
#include "catch2/catch.hpp"
#include "gfx/render_device.hpp"

TEST_CASE("SubpassInfos can be compared equal")
{
    auto first_subpass = gfx::SubpassInfo{
        .color_attachments = {VkAttachmentReference{
            .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}},
        .color_resolve_attachment
        = VkAttachmentReference{.attachment = 1,
                                .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        .depth_stencil_attachment = VkAttachmentReference{
            .attachment = 2, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}};

    auto second_subpass = gfx::SubpassInfo{
        .color_attachments = {VkAttachmentReference{
            .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}},
        .color_resolve_attachment
        = VkAttachmentReference{.attachment = 1,
                                .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        .depth_stencil_attachment = VkAttachmentReference{
            .attachment = 2, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}};

    auto third_subpass = gfx::SubpassInfo{
        .color_attachments        = {VkAttachmentReference{
            .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}},
        .color_resolve_attachment = VkAttachmentReference{
            .attachment = 1, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};

    REQUIRE(first_subpass == second_subpass);
    REQUIRE(third_subpass != first_subpass);
}

TEST_CASE("SubpassInfos can be initialized from json")
{
    auto static_subpass = gfx::SubpassInfo{
        .color_attachments = {VkAttachmentReference{
            .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}},
        .color_resolve_attachment
        = VkAttachmentReference{.attachment = 1,
                                .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        .depth_stencil_attachment = VkAttachmentReference{
            .attachment = 2, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}};

    auto first_json_subpass = gfx::SubpassInfo{};

    std::string first_json = R"(
        {
            "color_attachments":
            [
                {
                    "attachment_index": 0,
                    "layout": "COLOR_ATTACHMENT_OPTIMAL"
                }
            ],
            "resolve_attachment": 
            {
                "attachment_index": 1,
                "layout": "COLOR_ATTACHMENT_OPTIMAL"
            },
            "depth_stencil_attachment":
            {
                "attachment_index": 2,
                "layout": "DEPTH_STENCIL_ATTACHMENT_OPTIMAL"
            }
        }
    )";

    rapidjson::Document document;

    REQUIRE(!document.Parse(first_json.c_str()).HasParseError());

    first_json_subpass.init(document);

    REQUIRE(first_json_subpass == static_subpass);

    auto second_json_subpass = gfx::SubpassInfo{};

    std::string second_json = R"(
        {}
    )";

    REQUIRE(!document.Parse(second_json.c_str()).HasParseError());

    second_json_subpass.init(document);

    REQUIRE(second_json_subpass != static_subpass);
}

TEST_CASE("Renderpasses can be compared")
{
    auto first_renderpass = gfx::Renderpass{
        .attachments          = {0, 1, 2},
        .subpasses            = {gfx::SubpassInfo{
            .color_attachments = {VkAttachmentReference{0,
                                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}},
            .color_resolve_attachment
            = VkAttachmentReference{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            .depth_stencil_attachment
            = VkAttachmentReference{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}}},
        .subpass_dependencies = {
            VkSubpassDependency{.srcSubpass    = VK_SUBPASS_EXTERNAL,
                                .dstSubpass    = 0,
                                .srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .srcAccessMask = 0,
                                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                                 | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT}}};

    auto second_renderpass = gfx::Renderpass{
        .attachments          = {0, 1, 2},
        .subpasses            = {gfx::SubpassInfo{
            .color_attachments = {VkAttachmentReference{0,
                                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}},
            .color_resolve_attachment
            = VkAttachmentReference{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            .depth_stencil_attachment
            = VkAttachmentReference{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}}},
        .subpass_dependencies = {
            VkSubpassDependency{.srcSubpass    = VK_SUBPASS_EXTERNAL,
                                .dstSubpass    = 0,
                                .srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .srcAccessMask = 0,
                                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                                 | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT}}};

    auto third_renderpass = gfx::Renderpass{
        .attachments = {0, 1, 2},
        .subpasses   = {
            gfx::SubpassInfo{.color_attachments = {VkAttachmentReference{
                                 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}},
                             .color_resolve_attachment
                             = VkAttachmentReference{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
                             .depth_stencil_attachment = VkAttachmentReference{
                                 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}}}};

    REQUIRE(first_renderpass == second_renderpass);
    REQUIRE(first_renderpass != third_renderpass);
}

TEST_CASE("Renderpasses can be initialized from json")
{
    std::string json = R"(
        {
            "attachment_infos": [0, 1, 2],
            "subpasses":
            [
                {
                    "color_attachments":
                    [
                        {
                            "attachment_index": 0,
                            "layout": "COLOR_ATTACHMENT_OPTIMAL"
                        }
                    ],
                    "resolve_attachment": 
                    {
                        "attachment_index": 2,
                        "layout": "COLOR_ATTACHMENT_OPTIMAL"
                    },
                    "depth_stencil_attachment":
                    {
                        "attachment_index": 1,
                        "layout": "DEPTH_STENCIL_ATTACHMENT_OPTIMAL"
                    }
                }
            ],
            "subpass_dependencies":
            [
                {
                    "src_subpass": "EXTERNAL_SUBPASS",
                    "dst_subpass": 0,
                    "src_stage_mask": ["COLOR_ATTACHMENT_OUTPUT"],
                    "dst_stage_mask": ["COLOR_ATTACHMENT_OUTPUT"],
                    "dst_access_mask":
                    [
                        "COLOR_ATTACHMENT_READ",
                        "COLOR_ATTACHMENT_WRITE"
                    ]
                }
            ]
        }
    )";

    rapidjson::Document document;

    REQUIRE(!document.Parse(json.c_str()).HasParseError());

    gfx::Renderpass json_renderpass;
    json_renderpass.init(document);

    auto first_renderpass = gfx::Renderpass{
        .attachments          = {0, 1, 2},
        .subpasses            = {gfx::SubpassInfo{
            .color_attachments = {VkAttachmentReference{0,
                                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}},
            .color_resolve_attachment
            = VkAttachmentReference{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            .depth_stencil_attachment
            = VkAttachmentReference{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}}},
        .subpass_dependencies = {
            VkSubpassDependency{.srcSubpass    = VK_SUBPASS_EXTERNAL,
                                .dstSubpass    = 0,
                                .srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                .srcAccessMask = 0,
                                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                                 | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT}}};

    REQUIRE(json_renderpass == first_renderpass);
}

TEST_CASE("AttachmentInfos can be compared")
{
    auto first_attachment_info = gfx::AttachmentInfo{
        .format = gfx::Format::USE_COLOR,
        .use_samples = true,
        .description = {
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_UNDEFINED
        }
    };

    auto second_attachment_info = gfx::AttachmentInfo{
        .format = gfx::Format::USE_COLOR,
        .use_samples = true,
        .description = {
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_UNDEFINED
        }
    };

    auto third_attachment_info = gfx::AttachmentInfo{
        .format = gfx::Format::USE_COLOR,
        .use_samples = true,
        .description = {
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_UNDEFINED
        }
    };

    REQUIRE(first_attachment_info == second_attachment_info);
    REQUIRE(first_attachment_info != third_attachment_info);
}

TEST_CASE("AttachmentInfos can be initialized from json")
{
    std::string json = R"(
        {
            "format": "color",
            "multisampled": true,
            "description":
            {
                "load_op": "LOAD",
                "store_op": "STORE",
                "initial_layout": "UNDEFINED",
                "final_layout": "UNDEFINED"
            }
        }
    )";

    rapidjson::Document document;

    REQUIRE(!document.Parse(json.c_str()).HasParseError());

    gfx::AttachmentInfo json_attachment_info;
    json_attachment_info.init(document);

    auto first_attachment_info = gfx::AttachmentInfo{
        .format = gfx::Format::USE_COLOR,
        .use_samples = true,
        .description = {
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_UNDEFINED
        }
    };

    REQUIRE(json_attachment_info == first_attachment_info);
}

TEST_CASE("Attachments can be compared")
{
    auto first_attachment = gfx::Attachment{
        .format = gfx::Format::USE_COLOR,
        .use_samples = true
    };

    auto second_attachment = gfx::Attachment{
        .format = gfx::Format::USE_COLOR,
        .use_samples = true
    };

    auto third_attachment = gfx::Attachment{
        .format = gfx::Format::USE_COLOR,
        .use_samples = false
    };

    REQUIRE(first_attachment == second_attachment);
    REQUIRE(first_attachment != third_attachment);
}

TEST_CASE("Attachments can be initialized from json")
{
    std::string json = R"(
        {
            "format": "color",
            "multisampled": true
        }
    )";

    rapidjson::Document document;

    REQUIRE(!document.Parse(json.c_str()).HasParseError());

    gfx::Attachment json_attachment;
    json_attachment.init(document);

    auto first_attachment = gfx::Attachment{
        .format = gfx::Format::USE_COLOR,
        .use_samples = true
    };

    REQUIRE(json_attachment == first_attachment);
}