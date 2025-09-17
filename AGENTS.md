<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# AGENTS.md - Development Guide for AI Coding Agents

## Build/Test Commands

- **All commands use pixi**: `pixi run <command>`
- **Main package (py/)**: `pixi run test` (uses pytest with test environment)
- **MCP package (mcp/)**: `cd mcp && pixi run test` (dev environment)
- **Single test**: `pixi run pytest path/to/test_file.py::test_function_name`
- **Lint**: `pixi run lint` (runs pre-commit hooks)
- **Format**: `pixi run format` (MCP only), or `ruff --fix` + `black`
- **Type check**: `pixi run typecheck` (MCP only, uses mypy)
- **TypeScript**: `cd ts && pixi run test`, `pixi run lint`, `pixi run fmt`

## Python Code Style Guidelines

- **Line length**: 88 characters (Black/Ruff standard)
- **Imports**: Use absolute imports, group by standard/third-party/local
- **Types**: Use type hints, especially for public APIs
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error handling**: Use specific exceptions, avoid bare except clauses
- **Docstrings**: Use for public functions/classes
- **Comments**: Minimal, focus on why not what

## Key Python Conventions

- Use `pytest` for testing with fixtures in conftest.py
- Follow Ruff linting rules (see pyproject.toml [tool.lint])
- Use `dask.array` for large array processing
- Import from `.__about__` for version info
- Use `pathlib.Path` over os.path
- Pre-commit hooks enforce style automatically

## TypeScript Code Style Guidelines

- Use Deno's standard style (80 char line width, 2 space indent, semicolons)
- Strict TypeScript compiler options enabled
- Use JSR imports (@std/assert) and npm: prefix for npm packages

## Important Notes from Copilot Instructions

- For debugging Python issues, use `pixi run -e test python debug_script.py` (not `pixi run python ...`)
- Core workflow: Input → NgffImage → Multiscales → OME-Zarr via `to_multiscales()` and `to_ngff_zarr()`
- Multi-language project: py/ (core), mcp/ (MCP server), ts/ (TypeScript/Deno)
- Memory management via `config.memory_target` (default: 50% available memory)
- Import patterns: `from .__about__ import __version__` and function-based exports
- Test with pooch fixtures and parametrized tests for shape/chunk combinations
