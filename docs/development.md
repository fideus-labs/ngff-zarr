<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# 🔨 Development

Welcome! 👋 We're glad you're interested in contributing to ngff-zarr. Whether
you're fixing bugs, adding features, improving documentation, or helping with
testing, your contributions are greatly appreciated. 🎉

## 📜 Code of Conduct

Please read and follow our [Code of Conduct]. We are committed to providing a
welcoming and inclusive environment for everyone.

## 🗂️ Project overview

ngff-zarr is a multi-language implementation of the OME-NGFF Zarr
specification:

```text
ngff-zarr/
├── py/          # Python package (ngff-zarr)
├── mcp/         # Model Context Protocol server (ngff-zarr-mcp)
├── ts/          # TypeScript/Deno package (@fideus-labs/ngff-zarr)
└── docs/        # Documentation
```

### 🏛️ Core architecture

The central workflow follows this pattern across all implementations:

1. **Input → NgffImage** - Convert various formats to `NgffImage`
2. **NgffImage → NgffMultiscales** - Generate resolution levels via
   `to_multiscales()`
3. **NgffMultiscales → OME-Zarr** - Write to Zarr stores via `to_ome_zarr()`
4. **OME-Zarr → NgffMultiscales** - Read back via `from_ome_zarr()`

## 🚀 Getting started

### Prerequisites

Install [pixi] for environment management.

### Get the source code

```shell
git clone https://github.com/fideus-labs/ngff-zarr
cd ngff-zarr
```

### ⚙️ Install dependencies

Each package directory (`py/`, `ts/`, `mcp/`) carries its own pixi manifest.
Install the Python environments and the Git hooks, which are managed by
[prek]:

```shell
cd py
pixi install -a
pixi run prek-install
```

`prek install` reads `default_install_hook_types` from
`.pre-commit-config.yaml`, so it installs shims for both `pre-commit`
(linting and formatting) and `commit-msg` (commit message validation).

## 🔄 Contributing workflow

We use the standard [GitHub flow]:

1. 💬 **Open an issue first** - For significant changes, open a GitHub issue to
   discuss your proposal before starting work
2. 🍴 **Fork the repository** - Create your own fork
3. 🌿 **Create a branch** - Create a feature branch from `main`
4. ✏️ **Make changes** - Implement your changes with tests
5. 💾 **Commit** - Use Conventional Commit messages
6. 📤 **Push** - Push to your fork
7. 📬 **Open a pull request** - Submit a PR against `main`

### 📋 Pull request guidelines

- ✅ **CI must pass** - All checks must be green before merge
- 💬 **Be responsive** - Please respond to review comments in a timely manner
- ⏳ **Be patient** - Reviews may take time; we appreciate your patience
- 🤖 **Copilot reviews** - GitHub Copilot may flag false positives; if you
  believe a suggestion is incorrect, leave a comment explaining why and
  resolve as appropriate

## 📝 Commit messages

We follow the [Conventional Commits] standard. All commit messages are
validated by Commitizen hooks, which run via prek.

### 📐 Format

```text
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### 🏷️ Types

- ✨ `feat` - New feature
- 🐛 `fix` - Bug fix
- 📖 `docs` - Documentation changes
- 🎨 `style` - Code style changes (formatting, etc.)
- ♻️ `refactor` - Code refactoring
- ⚡ `perf` - Performance improvements
- 🧪 `test` - Adding or updating tests
- 🏗️ `build` - Build system changes
- 🔧 `ci` - CI/CD changes
- 🧹 `chore` - Maintenance tasks

### 🎯 Scopes (optional but recommended)

- 🐍 `py` - Python package (ngff-zarr)
- 🔌 `mcp` - MCP server package (ngff-zarr-mcp)
- 🟦 `ts` - TypeScript package (@fideus-labs/ngff-zarr)

### 💡 Examples

```bash
feat(py): add support for zarr v3 sharding
fix(mcp): handle missing metadata gracefully
docs: update installation instructions
chore(ts): update dependencies
```

### 🧙 Interactive commit helper

If you need help writing compliant commit messages, use the interactive CLI:

```bash
# From the repository root, for whichever package you are committing:
(cd py && pixi run commit)
(cd ts && pixi run commit)
(cd mcp && pixi run commit)
```

This will guide you through creating a properly formatted commit message.

### 🛡️ Commit message validation

The Git hooks automatically validate your commit messages. If a commit message
doesn't follow the Conventional Commits format, the commit is rejected with a
helpful error message.

### 🏷️ Version management

Each package is versioned independently. Check the current version of a
package with:

```bash
(cd py && pixi run version-check)   # Python package version
(cd ts && pixi run version-check)   # TypeScript package version
(cd mcp && pixi run version-check)  # MCP package version
```

## 🛠️ Development commands

All development uses pixi for consistent environments. Run the commands from
the corresponding package directory.

### 🐍 Python (`py/`)

```bash
cd py
pixi install -a                 # Install all environments
pixi run test                   # Run the test suite
pixi run lint                   # Run the prek hooks (ruff, codespell, ...)
pixi run build-docs             # Build the documentation
```

Run a single test with pytest directly:

```bash
pixi run -e test pytest test/test_to_multiscales.py::test_downsamples_when_size_is_exactly_double_chunk
```

### 🔌 MCP server (`mcp/`)

```bash
cd mcp
pixi install -e dev             # Install the development environment
pixi run test                   # Run tests
pixi run typecheck              # Type checking (mypy)
pixi run format                 # Format code
pixi run lint                   # Run linting
```

### 🟦 TypeScript (`ts/`)

```bash
cd ts
pixi run test                   # Deno test suite
pixi run lint                   # Deno lint
pixi run fmt                    # Deno format
pixi run check                  # Type checking
pixi run build                  # Full build
pixi run test-browser           # Browser tests
```

## 🧪 Testing

- 🐍 **Python**: pytest, with fixtures in `py/test/conftest.py`
- 🟦 **TypeScript**: Deno's built-in test runner
- 🔌 **MCP**: pytest with async patterns

Run the tests before submitting a pull request to ensure nothing is broken. ✅

## 🎨 Code style

### 🐍 Python

- 📏 **Line length**: 88 characters (Ruff standard)
- 📦 **Imports**: Absolute imports, grouped by standard/third-party/local
- 🔤 **Types**: Use type hints for public APIs
- 🏷️ **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- 📝 **Docstrings**: Required for public functions and classes

Ruff, via the prek hooks, enforces the style automatically. Run
`cd py && pixi run lint` before committing.

### 🟦 TypeScript

- 🎨 **Style**: Deno standard (80 char width, 2 space indent, semicolons)
- 🔤 **Types**: Strict TypeScript compiler options
- 📦 **Imports**: JSR imports (`@std/assert`) and the `npm:` prefix for npm
  packages

## 📖 Build the documentation

If needed, build and update the documentation. This serves the docs locally
and rebuilds them as you edit:

```shell
cd py
pixi run dev-docs
```

To build the HTML once, without the auto-reloading server:

```shell
cd py
pixi run build-docs
```

## 🗃️ Update the test data

If needed, update the testing data.

1. Add the new data to _py/test/data_, then generate a new tarball and compute
   its sha256 hash:

   ```shell
   cd py
   pixi run hash-data
   ```

2. Upload the resulting `data.tar.gz` as an asset on the
   [`testing-data` release], renamed with the next
   `ngff-zarr-testing-data-v<version>.tar.gz` version.

3. Update the `url` and `test_data_sha256` variables in _py/test/\_data.py_ to
   point at the new asset and its hash.

## ❓ Questions?

If you have questions, please open a [GitHub issue].

Thank you for contributing! 💖

[Code of Conduct]: https://github.com/fideus-labs/ngff-zarr/blob/main/CODE_OF_CONDUCT.md
[Conventional Commits]: https://www.conventionalcommits.org/
[GitHub flow]: https://docs.github.com/en/get-started/using-github/github-flow
[GitHub issue]: https://github.com/fideus-labs/ngff-zarr/issues
[pixi]: https://pixi.sh
[prek]: https://prek.j178.dev/
[`testing-data` release]: https://github.com/fideus-labs/ngff-zarr/releases/tag/testing-data
