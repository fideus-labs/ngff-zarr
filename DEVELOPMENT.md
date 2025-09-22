<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# Development Guide

This repository contains multiple packages implementing NGFF-Zarr support across
different languages and use cases.

## Repository Structure

```
ngff-zarr/
├── py/                    # Python package (ngff-zarr)
│   ├── ngff_zarr/        # Main Python library
│   ├── test/             # Python tests
│   ├── pyproject.toml    # Python package configuration
│   ├── pixi.lock         # Python environment lock
│   └── README.md         # Python package documentation
├── mcp/                   # Model Context Protocol server (ngff-zarr-mcp)
│   ├── ngff_zarr_mcp/    # MCP server implementation
│   ├── tests/            # MCP tests
│   ├── pyproject.toml    # MCP package configuration
│   └── README.md         # MCP documentation
├── docs/                  # Documentation (for Python package)
├── .github/workflows/     # CI/CD pipelines
└── README.md             # Repository overview
```

## Development Workflows

### Python Package (`py/`)

The main Python package providing NGFF-Zarr functionality.

```bash
# Navigate to Python package
cd py/

# Install dependencies
pixi install

# Run tests
pixi run test

# Run linting
pixi run lint

# Build documentation
pixi run build-docs
```

### MCP Server (`mcp/`)

The Model Context Protocol server for AI integration.

```bash
# Navigate to MCP directory
cd mcp/

# Install dependencies
pixi install

# Run tests
pixi run test

# Run linting
pixi run lint
```

## CI/CD

- **Python package**: Uses `pixi-test.yml` and `test.yml` workflows
- **MCP server**: Uses `mcp-ci.yml` workflow
- Both are automatically triggered on push/PR

## Future Packages

When adding new language implementations (e.g., TypeScript):

1. Create a new directory (e.g., `ts/`)
2. Add appropriate build and test configurations
3. Update CI workflows to include the new package
4. Update this guide and the main README

## Key Configuration Updates

After the migration to `py/`, the following configurations were updated:

- CI workflows to use `cd py && ...` for Python package operations
- ReadTheDocs configuration to install from `py/` directory
- Documentation build to reference `../py/ngff_zarr`
- Pre-commit hooks to exclude `py/test/conftest.py`

## Notes

- The main README.md provides an overview of all packages
- Each package directory contains its own README.md with specific instructions
- Dependencies and environments are isolated per package using pixi
- The documentation is built from the Python package but hosted at the
  repository level
