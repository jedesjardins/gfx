# The RenderConfig class

The RenderConfig class, in the gfx namespace, is simply used to initialize the graphics resources from [the json configuration file for your project](docs/json_configuration.md). It's usage is simple, construct it with the path to the json configuration file, initialize it, and pass it to the [the Renderer class](docs/render_device.md) init function.

## API

##### `void gfx::RenderConfig::RenderConfig(char const* json_config_file_name);`

Creates the RenderConfig class.

##### `void gfx::RenderConfig::init();`

Initializes the RenderConfig class with the elements in the json configuration file.

