<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# 🤖 MCP Server

The [`ngff-zarr-mcp`] Python package provides a Model Context Protocol (MCP)
server that enables seamless integration with AI agents and development tools.
This server provides a standardized interface for AI agents to interact with
ngff-zarr's capabilities for converting datasets to the OME-Zarr scientific
imaging data format.

## What is MCP?

The Model Context Protocol (MCP) is an open standard that allows AI agents to
securely access external data sources and tools. It creates a bridge between AI
models and various applications, enabling them to work together more
effectively.

## Integration with AI Agents

The [`ngff-zarr-mcp`] server can be integrated with various AI agents and
development environments, including:

- **GitHub Copilot** in VS Code
- **OpenCode**
- **Cursor**
- **Claude Code**
- Any other agent that supports the MCP standard

This integration allows you to use natural language to:

- Convert datasets to OME-Zarr format
- Optimize compression codecs
- Configure sharding to limit file count
- Generate Python scripts for batch processing
- Validate OME-Zarr stores
- ... and more!

## Example Usage

Here's an example of how you might interact with the MCP server through an AI
agent:

### OpenCode Demo

The `ngff-zarr-mcp` Model Context Protocol (MCP) server converts datasets to the
OME-Zarr scientific imaging data format, optimizes the compression codec,
ensures a limited number of files are generated, and creates a Python script for
re-use.

<script src="https://asciinema.org/a/726628.js" id="asciicast-726628" async="true"></script>

#### Sample Prompts

> Convert LIDCFull.tif to OME-Zarr

> Find the optimal codec to use for this data.

> Use sharding to keep the number of files under 20.

> Create a Python script to convert all files in a provided input directory to a
> provided output directory. Use the codec we found to be optimal.

## Configuration

The MCP server can be configured for use with different AI agents. Here's an
example configuration for OpenCode:

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

A similar configuration can be used for other agents like GitHub Copilot or
Cursor, adjusting the command and environment as needed. The [`ngff-zarr-mcp`]
provides guidance on how to set up the MCP server for different environments.

## Available Functions

The MCP server provides the following functions:

- **convert_images_to_ome_zarr**: Convert images to OME-Zarr format with
  customizable parameters
- **get_ome_zarr_info**: Get detailed information about an OME-Zarr store
- **validate_ome_zarr_store**: Validate OME-Zarr store structure and metadata
- **optimize_ome_zarr_store**: Optimize existing stores with new
  compression/chunking

## Benefits

Using the MCP server provides several advantages:

1. **Natural Language Interface**: Interact with ngff-zarr using conversational
   prompts
2. **Automated Workflows**: Let AI agents handle complex conversion tasks
3. **Intelligent Optimization**: Get AI-driven recommendations for compression
   and chunking
4. **Code Generation**: Automatically generate Python scripts for batch
   processing
5. **Seamless Integration**: Works within your existing development environment

## Getting Started

To use the MCP server:

1. Install ngff-zarr with MCP support
2. Configure your AI agent to use the [`ngff-zarr-mcp`] server
3. Start interacting with natural language prompts

For detailed installation and setup instructions, the
[MCP server README](https://github.com/thewtex/ngff-zarr/tree/main/mcp/README.md)
for comprehensive setup instructions and examples.

[`ngff-zarr-mcp`]: https://pypi.org/project/ngff-zarr-mcp/
