# ngff-zarr MCP Server

`ngff-zarr-mcp` is a Model Context Protocol (MCP) server that provides AI agents
with the ability to convert images to OME-Zarr format using the ngff-zarr
library.

<!-- mcp-name: io.github.thewtex/ngff-zarr -->

[![asciicast](https://asciinema.org/a/726628.png)](https://asciinema.org/a/726628)

## Features

### Tools

- **convert_images_to_ome_zarr**: Convert various image formats to OME-Zarr with
  full control over metadata, compression, and multiscale generation
- **read_ome_zarr_store**: Read OME-Zarr data with support for remote storage
  options
- **get_ome_zarr_info**: Inspect existing OME-Zarr stores and get detailed
  information
- **validate_ome_zarr_store**: Validate OME-Zarr structure and metadata
- **optimize_ome_zarr_store**: Optimize existing stores with new compression and
  chunking

### Resources

- **supported-formats**: List of supported input/output formats and backends
- **downsampling-methods**: Available downsampling methods for multiscale
  generation
- **compression-codecs**: Available compression codecs and their characteristics

### Input Support

- Local files (all formats supported by ngff-zarr)
- Local directories (Zarr stores)
- Network URLs (HTTP/HTTPS)
- S3 URLs (with optional s3fs dependency)
- Remote storage with authentication (AWS S3, Google Cloud Storage, Azure)

### Advanced Features

- **RFC 4 - Anatomical Orientation**: Support for medical imaging orientation
  systems (LPS, RAS)
- **Method Metadata**: Enhanced multiscale metadata with downsampling method
  information
- **Storage Options**: Cloud storage authentication and configuration support
- **Multiscale Type Tracking**: Automatic detection and preservation of
  downsampling methods

### Output Optimization

- Multiple compression codecs (gzip, lz4, zstd, blosc variants)
- Configurable compression levels
- Flexible chunk sizing
- Sharding support (Zarr v3/OME-Zarr v0.5)
- OME-Zarr version selection (0.4 or 0.5)

## Installation

### Requirements

- Python >= 3.9
- Cursor, Windsurf, Claude Desktop, VS Code, or another MCP Client

### Quick Install

The easiest way to use ngff-zarr MCP server is with `uvx`:

```bash
# Install uvx if not already installed
pip install uvx

# Run the MCP server directly from PyPI
uvx ngff-zarr-mcp
```

<details>
<summary><b>Install in Cursor</b></summary>

Go to: `Settings` -> `Cursor Settings` -> `MCP` -> `Add new global MCP server`

Pasting the following configuration into your Cursor `~/.cursor/mcp.json` file
is the recommended approach. You may also install in a specific project by
creating `.cursor/mcp.json` in your project folder. See
[Cursor MCP docs](https://docs.cursor.com/context/model-context-protocol) for
more info.

#### Using uvx (recommended)

```json
{
  "mcpServers": {
    "ngff-zarr": {
      "command": "uvx",
      "args": ["ngff-zarr-mcp"]
    }
  }
}
```

#### Using direct Python

```json
{
  "mcpServers": {
    "ngff-zarr": {
      "command": "python",
      "args": ["-m", "pip", "install", "ngff-zarr-mcp", "&&", "ngff-zarr-mcp"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in Windsurf</b></summary>

Add this to your Windsurf MCP config file. See
[Windsurf MCP docs](https://docs.windsurf.com/windsurf/mcp) for more info.

#### Using uvx (recommended)

```json
{
  "mcpServers": {
    "ngff-zarr": {
      "command": "uvx",
      "args": ["ngff-zarr-mcp"]
    }
  }
}
```

#### SSE Transport

```json
{
  "mcpServers": {
    "ngff-zarr": {
      "url": "http://localhost:8000/sse",
      "description": "ngff-zarr server running with SSE transport"
    }
  }
}
```

</details>

<details>
<summary><b>Install in VS Code</b></summary>

Add this to your VS Code MCP config file. See
[VS Code MCP docs](https://code.visualstudio.com/docs/copilot/chat/mcp-servers)
for more info.

#### Using uvx (recommended)

```json
"mcp": {
  "servers": {
    "ngff-zarr": {
      "command": "uvx",
      "args": ["ngff-zarr-mcp"]
    }
  }
}
```

#### Using pip install

```json
"mcp": {
  "servers": {
    "ngff-zarr": {
      "command": "python",
      "args": ["-c", "import subprocess; subprocess.run(['pip', 'install', 'ngff-zarr-mcp']); import ngff_zarr_mcp.server; ngff_zarr_mcp.server.main()"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in OpenCode</b></summary>

OpenCode is a Go-based CLI application that provides an AI-powered coding
assistant in the terminal. It supports MCP servers through JSON configuration
files. See [OpenCode MCP docs](https://opencode.ai/docs/mcp-servers/) for more
details.

Add this to your OpenCode configuration file (`~/.config/opencode/config.json`
for global or `opencode.json` for project-specific):

#### Using uvx (recommended)

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "ngff-zarr": {
      "type": "local",
      "command": ["uvx", "ngff-zarr-mcp"],
      "enabled": true
    }
  }
}
```

#### Using pip install

```json
{
  "mcp": {
    "ngff-zarr": {
      "type": "local",
      "command": [
        "python",
        "-c",
        "import subprocess; subprocess.run(['pip', 'install', 'ngff-zarr-mcp']); import ngff_zarr_mcp.server; ngff_zarr_mcp.server.main()"
      ],
      "enabled": true
    }
  }
}
```

After adding the configuration, restart OpenCode. The ngff-zarr tools will be
available in the terminal interface with automatic permission prompts for tool
execution.

</details>

<details>
<summary><b>Install in Claude Desktop</b></summary>

Add this to your Claude Desktop `claude_desktop_config.json` file. See
[Claude Desktop MCP docs](https://modelcontextprotocol.io/quickstart/user) for
more info.

#### Using uvx (recommended)

```json
{
  "mcpServers": {
    "ngff-zarr": {
      "command": "uvx",
      "args": ["ngff-zarr-mcp"]
    }
  }
}
```

#### Using direct installation

```json
{
  "mcpServers": {
    "ngff-zarr": {
      "command": "ngff-zarr-mcp"
    }
  }
}
```

</details>

<details>
<summary><b>Install in Claude Code</b></summary>

Run this command. See
[Claude Code MCP docs](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#set-up-model-context-protocol-mcp)
for more info.

#### Using uvx

```sh
claude mcp add ngff-zarr -- uvx ngff-zarr-mcp
```

#### Using pip

```sh
claude mcp add ngff-zarr -- python -m pip install ngff-zarr-mcp && ngff-zarr-mcp
```

</details>

<details>
<summary><b>Install in Gemini CLI</b></summary>

Add this to your _.gemini/settings.json_ Gemini CLI MCP configuration. See the
[Gemini CLI configuration docs](https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/configuration.md)
for more info.

#### Using uvx (recommended)

```json
{
  "mcpServers": {
    "ngff-zarr": {
      "command": "uvx",
      "args": ["ngff-zarr-mcp"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in Cline</b></summary>

1. Open **Cline**.
2. Click the hamburger menu icon (☰) to enter the **MCP Servers** section.
3. Add a new server with the following configuration:

#### Using uvx (recommended)

```json
{
  "mcpServers": {
    "ngff-zarr": {
      "command": "uvx",
      "args": ["ngff-zarr-mcp"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in BoltAI</b></summary>

Open the "Settings" page of the app, navigate to "Plugins," and enter the
following JSON:

```json
{
  "mcpServers": {
    "ngff-zarr": {
      "command": "uvx",
      "args": ["ngff-zarr-mcp"]
    }
  }
}
```

Once saved, you can start using ngff-zarr tools in your conversations. More
information is available on
[BoltAI's Documentation site](https://docs.boltai.com/docs/plugins/mcp-servers).

</details>

<details>
<summary><b>Install in Zed</b></summary>

Add this to your Zed `settings.json`. See
[Zed Context Server docs](https://zed.dev/docs/assistant/context-servers) for
more info.

```json
{
  "context_servers": {
    "ngff-zarr": {
      "command": "uvx",
      "args": ["ngff-zarr-mcp"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in Augment Code</b></summary>

### **A. Using the Augment Code UI**

1. Click the hamburger menu.
2. Select **Settings**.
3. Navigate to the **Tools** section.
4. Click the **+ Add MCP** button.
5. Enter the following command: `uvx ngff-zarr-mcp`
6. Name the MCP: **ngff-zarr**.
7. Click the **Add** button.

### **B. Manual Configuration**

1. Press Cmd/Ctrl Shift P or go to the hamburger menu in the Augment panel
2. Select Edit Settings
3. Under Advanced, click Edit in settings.json
4. Add the server configuration:

```json
"augment.advanced": {
  "mcpServers": [
    {
      "name": "ngff-zarr",
      "command": "uvx",
      "args": ["ngff-zarr-mcp"]
    }
  ]
}
```

</details>

<details>
<summary><b>Install in JetBrains AI Assistant</b></summary>

See
[JetBrains AI Assistant Documentation](https://www.jetbrains.com/help/ai-assistant/configure-an-mcp-server.html)
for more details.

1. In JetBrains IDEs go to `Settings` -> `Tools` -> `AI Assistant` ->
   `Model Context Protocol (MCP)`
2. Click `+ Add`.
3. Click on `Command` in the top-left corner of the dialog and select the As
   JSON option from the list
4. Add this configuration and click `OK`

```json
{
  "command": "uvx",
  "args": ["ngff-zarr-mcp"]
}
```

5. Click `Apply` to save changes.

</details>

<details>
<summary><b>Install in Qodo Gen</b></summary>

See
[Qodo Gen docs](https://docs.qodo.ai/qodo-documentation/qodo-gen/qodo-gen-chat/agentic-mode/agentic-tools-mcps)
for more details.

1. Open Qodo Gen chat panel in VSCode or IntelliJ.
2. Click Connect more tools.
3. Click + Add new MCP.
4. Add the following configuration:

```json
{
  "command": "uvx",
  "args": ["ngff-zarr-mcp"]
}
```

</details>

<details>
<summary><b>Install in Roo Code</b></summary>

Add this to your Roo Code MCP configuration file. See
[Roo Code MCP docs](https://docs.roocode.com/features/mcp/using-mcp-in-roo) for
more info.

```json
{
  "mcpServers": {
    "ngff-zarr": {
      "command": "uvx",
      "args": ["ngff-zarr-mcp"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in Amazon Q Developer CLI</b></summary>

Add this to your Amazon Q Developer CLI configuration file. See
[Amazon Q Developer CLI docs](https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-mcp-configuration.html)
for more details.

```json
{
  "mcpServers": {
    "ngff-zarr": {
      "command": "uvx",
      "args": ["ngff-zarr-mcp"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in Zencoder</b></summary>

To configure ngff-zarr MCP in Zencoder, follow these steps:

1. Go to the Zencoder menu (...)
2. From the dropdown menu, select Agent tools
3. Click on the Add custom MCP
4. Add the name and server configuration from below, and make sure to hit the
   Install button

```json
{
  "command": "uvx",
  "args": ["ngff-zarr-mcp"]
}
```

Once the MCP server is added, you can easily continue using it.

</details>

<details>
<summary><b>Install in Warp</b></summary>

See
[Warp Model Context Protocol Documentation](https://docs.warp.dev/knowledge-and-collaboration/mcp#adding-an-mcp-server)
for details.

1. Navigate `Settings` > `AI` > `Manage MCP servers`.
2. Add a new MCP server by clicking the `+ Add` button.
3. Paste the configuration given below:

```json
{
  "ngff-zarr": {
    "command": "uvx",
    "args": ["ngff-zarr-mcp"]
  }
}
```

4. Click `Save` to apply the changes.

</details>

<details>
<summary><b>Development Installation</b></summary>

For development work, use pixi (recommended) or pip:

#### Using pixi (Recommended)

```bash
# Install pixi if not already installed
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and setup environment
git clone <repository>
cd mcp/
pixi install

# Development environment (includes all dev tools)
pixi shell -e dev

# Run development server
pixi run dev-server

# Run tests and checks
pixi run test
pixi run lint
pixi run typecheck
```

#### Using pip

```bash
# Clone and install in development mode
git clone <repository>
cd mcp/
pip install -e ".[all]"

# Run the server
ngff-zarr-mcp
```

</details>

## Usage

### As MCP Server

The server can be run in different transport modes:

```bash
# STDIO transport (default)
ngff-zarr-mcp

# Server-Sent Events transport
ngff-zarr-mcp --transport sse --host localhost --port 8000
```

### Transport Options

- **STDIO**: Default transport for most MCP clients
- **SSE**: Server-Sent Events for web-based clients or when HTTP transport is
  preferred

See the installation section above for client-specific configuration examples.

## Examples

### Convert a Single Image

```python
# Through MCP client, the agent can:
result = await convert_images_to_ome_zarr(
    input_paths=["image.tif"],
    output_path="output.ome.zarr",
    ome_zarr_version="0.4",
    scale_factors=[2, 4, 8],
    method="itkwasm_gaussian",
    compression_codec="zstd"
)
```

### Convert with Metadata

```python
result = await convert_images_to_ome_zarr(
    input_paths=["image.nii.gz"],
    output_path="brain.ome.zarr",
    dims=["z", "y", "x"],
    scale={"z": 2.0, "y": 0.5, "x": 0.5},
    units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
    name="Brain MRI",
    scale_factors=[2, 4]
)
```

### Optimize Existing Store

```python
result = await optimize_ome_zarr_store(
    input_path="large.ome.zarr",
    output_path="optimized.ome.zarr",
    compression_codec="blosc:zstd",
    chunks=[64, 64, 64]
)
```

### Get Store Information

```python
info = await get_ome_zarr_info("data.ome.zarr")
print(f"Size: {info.size_bytes} bytes")
print(f"Scales: {info.num_scales}")
print(f"Dimensions: {info.dimensions}")
```

## Supported Formats

### Input Formats

- **ITK/ITK-Wasm**: .nii, .nii.gz, .mha, .mhd, .nrrd, .dcm, .jpg, .png, .bmp,
  etc.
- **TIFF**: .tif, .tiff, .svs, .ndpi, .scn, etc. via tifffile
- **Video**: .webm, .mp4, .avi, .mov, .gif, etc. via imageio
- **Zarr**: .zarr, .ome.zarr

### Output Formats

- OME-Zarr (.ome.zarr, .zarr)

## Performance Options

### Memory Management

- Set memory targets to control RAM usage
- Use caching for large datasets
- Configure Dask LocalCluster for distributed processing

### Compression

- Choose from multiple codecs: gzip, lz4, zstd, blosc variants
- Adjust compression levels for speed vs. size tradeoffs
- Use sharding to reduce file count (Zarr v3)

### Chunking

- Optimize chunk sizes for your access patterns
- Configure sharding for better performance with cloud storage

## Development

### Using pixi (Recommended)

Pixi provides reproducible, cross-platform environment management. All Python
dependencies are defined in `pyproject.toml` and automatically managed by pixi.

```bash
# Clone and setup environment
git clone <repository>
cd mcp/
pixi install

# Development environment (includes all dev tools)
pixi shell -e dev

# Run tests
pixi run test
pixi run test-cov

# Lint and format code
pixi run lint
pixi run format
pixi run typecheck

# Run all checks
pixi run all-checks
```

#### Pixi Environments

- **default**: Runtime dependencies only (from `[project.dependencies]`)
- **dev**: Development tools (pytest, black, mypy, ruff)
- **cloud**: Cloud storage support (s3fs, gcsfs)
- **all**: Complete feature set (all ngff-zarr dependencies + cloud)

```bash
pixi shell -e dev      # Development work
pixi shell -e cloud    # Cloud storage testing
pixi shell -e all      # Full feature testing
```

### Using traditional tools

```bash
# Clone and install in development mode
git clone <repository>
cd mcp/
pip install -e ".[all]"

# Run tests
pytest

# Lint code
black .
ruff check .
```

## Dependencies

### Core

- mcp: Model Context Protocol implementation
- ngff-zarr: Core image conversion functionality
- pydantic: Data validation
- httpx: HTTP client for remote files
- aiofiles: Async file operations

### Optional

- s3fs: S3 storage support
- gcsfs: Google Cloud Storage support
- dask[distributed]: Distributed processing

## 🚨 Troubleshooting

<details>
<summary><b>Python Version Issues</b></summary>

The ngff-zarr-mcp server requires Python 3.9 or higher. If you encounter version
errors:

```bash
# Check your Python version
python --version

# Use uvx to automatically handle Python environments
uvx ngff-zarr-mcp
```

</details>

<details>
<summary><b>Package Not Found Errors</b></summary>

If you encounter package not found errors with uvx:

```bash
# Update uvx
pip install --upgrade uvx

# Try installing the package explicitly first
uvx install ngff-zarr-mcp
uvx ngff-zarr-mcp
```

</details>

<details>
<summary><b>Permission Issues</b></summary>

If you encounter permission errors during installation:

```bash
# Use user installation
pip install --user uvx

# Or create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install ngff-zarr-mcp
```

</details>

<details>
<summary><b>Memory Issues with Large Images</b></summary>

For large images, you may need to adjust memory settings:

```bash
# Start server with memory limit
ngff-zarr-mcp --memory-target 8GB

# Or use chunked processing in your conversion calls
# convert_images_to_ome_zarr(chunks=[512, 512, 64])
```

</details>

<details>
<summary><b>Network Issues with Remote Files</b></summary>

If you have issues accessing remote files:

```bash
# Test basic connectivity
curl -I <your-url>

# For S3 URLs, ensure s3fs is installed
pip install s3fs

# Configure AWS credentials if needed
aws configure
```

</details>

<details>
<summary><b>General MCP Client Errors</b></summary>

1. Ensure your MCP client supports the latest MCP protocol version
2. Check that the server starts correctly: `uvx ngff-zarr-mcp --help`
3. Verify JSON configuration syntax in your client config
4. Try restarting your MCP client after configuration changes
5. Check client logs for specific error messages

</details>

## License

MIT License - see LICENSE file for details.
