<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# Contributing to ngff-zarr

Welcome! We're glad you're interested in contributing to ngff-zarr. Whether
you're fixing bugs, adding features, improving documentation, or helping with
testing, your contributions are greatly appreciated.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md). We are
committed to providing a welcoming and inclusive environment for everyone.

## Getting Started

### Prerequisites

Install [pixi](https://pixi.sh) for environment management.

### Setup

1. Fork and clone the repository
2. Install pre-commit hooks:
   ```bash
   cd py && pixi run pre-commit-install
   ```

## Contributing Workflow

We use the standard GitHub pull request workflow:

1. **Open an Issue First** - For significant changes, open a GitHub Issue to
   discuss your proposal before starting work
2. **Fork the Repository** - Create your own fork
3. **Create a Branch** - Create a feature branch from `main`
4. **Make Changes** - Implement your changes with tests
5. **Commit** - Use Conventional Commit messages
6. **Push** - Push to your fork
7. **Open a Pull Request** - Submit a PR against `main`

### Pull Request Guidelines

- **CI must pass** - All checks must be green before merge
- **Be responsive** - Please respond to review comments in a timely manner
- **Be patient** - Reviews may take time; we appreciate your patience
- **Copilot reviews** - GitHub Copilot may flag false positives; if you
  believe a suggestion is incorrect, leave a comment explaining why and
  resolve as appropriate

## Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/)
standard. All commit messages are validated by pre-commit hooks using Commitizen.

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code style changes (formatting, etc.)
- `refactor` - Code refactoring
- `perf` - Performance improvements
- `test` - Adding or updating tests
- `build` - Build system changes
- `ci` - CI/CD changes
- `chore` - Maintenance tasks

### Scopes (optional but recommended)

- `py` - Python package (ngff-zarr)
- `mcp` - MCP server package (ngff-zarr-mcp)
- `ts` - TypeScript package (@fideus-labs/ngff-zarr)

### Examples

```bash
feat(py): add support for zarr v3 sharding
fix(mcp): handle missing metadata gracefully
docs: update installation instructions
chore(ts): update dependencies
```

### Interactive Commit Helper

If you need help writing compliant commit messages, use the interactive CLI:

```bash
cd py && pixi run commit
# or from other directories:
cd ts && pixi run commit
cd mcp && pixi run commit
```

This will guide you through creating a properly formatted commit message.

### Version Management

Check the current version of a package:

```bash
cd py && pixi run version-check   # Shows Python package version
cd ts && pixi run version-check   # Shows TypeScript package version
cd mcp && pixi run version-check  # Shows MCP package version
```

### Pre-commit Validation

The pre-commit hooks will automatically validate your commit messages. If a
commit message doesn't follow the Conventional Commits format, the commit will
be rejected with helpful error messages.

## Project Overview

ngff-zarr is a multi-language implementation of the OME-NGFF Zarr
specification:

```
ngff-zarr/
├── py/          # Python package (ngff-zarr)
├── mcp/         # Model Context Protocol server (ngff-zarr-mcp)
├── ts/          # TypeScript/Deno package (@fideus-labs/ngff-zarr)
└── docs/        # Documentation
```

### Core Architecture

The central workflow follows this pattern across all implementations:

1. **Input → NgffImage** - Convert various formats to `NgffImage`
2. **NgffImage → Multiscales** - Generate resolution levels via `to_multiscales()`
3. **Multiscales → OME-Zarr** - Write to zarr stores via `to_ngff_zarr()`
4. **OME-Zarr → Multiscales** - Read back via `from_ngff_zarr()`

## Development Commands

All development uses pixi for consistent environments.

### Python (`py/`)

```bash
cd py
pixi install                    # Install dependencies
pixi run test                   # Run test suite
pixi run lint                   # Run linting
pixi run build-docs             # Build documentation
```

### MCP Server (`mcp/`)

```bash
cd mcp
pixi install                    # Install dependencies
pixi run test                   # Run tests
pixi run typecheck              # Type checking (mypy)
pixi run format                 # Format code
pixi run lint                   # Run linting
```

### TypeScript (`ts/`)

```bash
cd ts
pixi run test                   # Deno test suite
pixi run lint                   # Deno lint
pixi run fmt                    # Deno format
pixi run check                  # Type checking
pixi run build                  # Full build
pixi run test-browser           # Browser tests
```

## Code Style

### Python

- **Line length**: 88 characters (Black/Ruff standard)
- **Imports**: Absolute imports, grouped by standard/third-party/local
- **Types**: Use type hints for public APIs
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Docstrings**: Required for public functions and classes

Pre-commit hooks enforce style automatically via Ruff and Black.

### TypeScript

- **Style**: Deno standard (80 char width, 2 space indent, semicolons)
- **Types**: Strict TypeScript compiler options
- **Imports**: JSR imports (`@std/assert`) and `npm:` prefix for npm packages

## Testing

- **Python**: Uses pytest with fixtures in `conftest.py`
- **TypeScript**: Uses Deno's built-in test runner
- **MCP**: Uses pytest with async patterns

Run tests before submitting PRs to ensure nothing is broken.

## Questions?

If you have questions, please open a
[GitHub Issue](https://github.com/thewtex/ngff-zarr/issues).

Thank you for contributing!
