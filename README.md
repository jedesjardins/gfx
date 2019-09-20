# gfx

A single file Open Source C++ wrapper for [Vulkan](https://www.khronos.org/vulkan/), for abstracting away boilerplate, and easing use.

At this moment it's not fully featured, and still requires a lot of knowledge of Vulkan, but there are some [examples](examples) that can be referenced that should make it a bit easier.

## Usage

This is an .hpp only library, so if you have the single file header in your include path, along with it's dependencies, it should be plug and play. More information can be found [here](docs/usage.md)

## API

There are three main parts of this library: [the json configuration file for your project](docs/json_configuration.md), [the RenderConfig class](docs/render_config.md), and [the Renderer class](docs/render_device.md)