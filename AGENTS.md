<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# AGENTS.md - Development Guide for AI Coding Agents

## Repository Overview

This is a multi-language implementation of the OME Next Generation File Format
(NGFF) Zarr specification with three main packages:

- **`py/`** - Core Python implementation with CLI and library (ngff-zarr)
- **`mcp/`** - Model Context Protocol server for AI integration (ngff-zarr-mcp)
- **`ts/`** - TypeScript/Deno implementation for web/Node environments
  (@fideus-labs/ngff-zarr)

## Core Architecture Patterns

### Data Flow: The Multiscale Pipeline

The central workflow follows this pattern across all implementations:

1. **Input → NgffImage**: Convert various formats to `NgffImage` (single scale +
   metadata)
2. **NgffImage → NgffMultiscales**: Generate multiple resolution levels via
   `to_multiscales()`
3. **NgffMultiscales → OME-Zarr**: Write to zarr stores via `to_ngff_zarr()`
4. **OME-Zarr → NgffMultiscales**: Read back via `from_ngff_zarr()`

### Key Data Classes

- **`NgffImage`**: Single-scale image with dims, scale, translation, and
  dask/lazy array data
- **`NgffMultiscales`**: Container for multiple `NgffImage` scales + OME-Zarr
  metadata
- **`Metadata`**: OME-Zarr spec metadata (axes, datasets, coordinate
  transformations)

## Build & Test Commands

**All development uses pixi for consistent environments:**

### Python (py/)

```bash
pixi run --as-is test                    # Run pytest test suite
pixi run --as-is pytest path/to/test.py::test_name  # Single test
pixi run --as-is lint                    # Pre-commit hooks (ruff)
pixi run --as-is format                  # Format code
```

> **Note:** The `lint` task runs all pre-commit hooks including `ruff format`.
> Always run `lint` before committing to catch both linting and formatting issues.

### MCP Server (mcp/)

```bash
cd mcp && pixi run --as-is test         # Run MCP tests
cd mcp && pixi run --as-is typecheck    # mypy type checking
cd mcp && pixi run --as-is format       # Format code
cd mcp && pixi run --as-is dev          # Run MCP server in dev mode
```

### TypeScript (ts/)

```bash
cd ts && pixi run --as-is test          # Deno test suite
cd ts && pixi run --as-is lint          # Deno lint
cd ts && pixi run --as-is fmt           # Deno format
cd ts && pixi run --as-is build         # Full build pipeline
cd ts && pixi run --as-is test:browser  # Browser compatibility tests
cd ts && pixi run --as-is check         # Type checking
```

## Commit Message Format & Version Management

This repository follows [Conventional Commits](https://www.conventionalcommits.org/)
specification. All commit messages are validated by Commitizen pre-commit hooks.

### Commit Message Format

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test changes
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Other changes (dependencies, etc.)

**Scopes (optional but recommended):**
- `py`: Python package (ngff-zarr)
- `mcp`: MCP server package (ngff-zarr-mcp)
- `ts`: TypeScript package (@fideus-labs/ngff-zarr)

**Examples:**
```bash
feat(py): add support for RFC-9 OME-Zarr format
fix(ts): resolve memory leak in multiscale generation
docs: update installation instructions
chore(mcp): update dependencies
```

### Interactive Commit CLI

For help writing compliant commit messages:

```bash
cd py && pixi run commit
cd ts && pixi run commit
cd mcp && pixi run commit
```

### Version Management

Each package is versioned independently using Commitizen:

```bash
# Check current version
cd py && pixi run version-check
cd ts && pixi run version-check
cd mcp && pixi run version-check

# Bump version (analyzes commits, updates changelog, creates tag)
cd py && pixi run bump   # Python package
cd ts && pixi run bump   # TypeScript package
cd mcp && pixi run bump  # MCP package
```

The `bump` task will:
- Analyze commits since last tag
- Determine appropriate version bump (major/minor/patch)
- Update version files automatically
- Generate/update CHANGELOG.md (filtered by package scope)
- Create a git tag (py-v*, mcp-v*, or ts-v*)

### Changelog Filtering

Each package has its own changelog that only includes relevant commits:

**Filtering Rules:**
- **py**: Includes commits with scope `py` or files in `py/`
- **mcp**: Includes commits with scope `mcp` or files in `mcp/`
- **ts**: Includes commits with scope `ts` or files in `ts/`
- **All packages**: Include commits affecting multiple packages (CI, docs, root-level files)

**GitHub Links:**
All changelog entries include clickable GitHub commit links with short hashes:
```markdown
- **py**: add feature ([abc1234](https://github.com/fideus-labs/ngff-zarr/commit/abc1234...))
```

**Custom Plugin:**
The filtering is implemented via a custom Commitizen plugin in `.commitizen/cz_ngff_zarr.py`.
To modify filtering logic, edit the `_should_include_for_*` methods in that file.

### Pre-commit Hook Installation

The pre-commit hooks are now configured to validate commit messages and branch
names. Install them with:

```bash
cd py && pixi run pre-commit-install
```

This will install hooks for:
- **commit-msg**: Validates commit message format
- **pre-push**: Validates branch naming (if configured)
- **pre-commit**: Standard linting and formatting checks

## Python Code Style Guidelines

- **Line length**: 88 characters (Ruff standard)
- **Imports**: Use absolute imports, group by standard/third-party/local
- **Types**: Use type hints, especially for public APIs
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error handling**: Use specific exceptions, avoid bare except clauses
- **Docstrings**: Use for public functions/classes
- **Comments**: Minimal, focus on why not what

## Python Formatting Requirements (CRITICAL for AI Agents)

All Python code MUST pass `ruff format` and `ruff check` before committing.
The pre-commit hooks enforce this automatically for local commits, but AI
coding agents that commit via GitHub (Copilot, etc.) bypass these hooks. This
causes a ping-pong cycle where human maintainers must repeatedly reformat
AI-written code.

**Before committing any Python changes, always run:**

```bash
pixi run --as-is lint      # from py/ directory — runs all pre-commit hooks
```

Run it until it passes cleanly with no file modifications. If it modifies
files, stage the changes and run it again.

### Common AI Formatting Mistakes

AI agents frequently produce code that conflicts with `ruff format`. Follow
these conventions to prevent reformatting churn:

1. **No trailing whitespace on blank lines** — blank lines inside indented
   blocks must be completely empty (zero characters before the newline):
   ```python
   # WRONG — blank line has spaces:
   def foo():
       x = 1
       ····
       y = 2

   # RIGHT — blank line is truly empty:
   def foo():
       x = 1

       y = 2
   ```

2. **No backslash line continuations** — use parentheses instead:
   ```python
   # WRONG:
   assert img.shape == expected, \
       f"Shape mismatch: {img.shape}"

   # RIGHT:
   assert (
       img.shape == expected
   ), f"Shape mismatch: {img.shape}"
   ```

3. **No spaces around the `**` (power) operator**:
   ```python
   # WRONG:
   x = 2 ** i

   # RIGHT:
   x = 2**i
   ```

4. **Long lines (>88 chars) in assert statements** must use parenthesized
   form, not backslash continuations.

5. **Blank line after mid-function imports**:
   ```python
   # WRONG:
   from foo import Bar
   assert isinstance(x, Bar)

   # RIGHT:
   from foo import Bar

   assert isinstance(x, Bar)
   ```

## Key Python Conventions

- Use `pytest` for testing with fixtures in conftest.py
- Follow Ruff linting and formatting rules (see pyproject.toml [tool.ruff.lint])
- Use `dask.array` for large array processing
- Import from `.__about__` for version info
- Use `pathlib.Path` over os.path
- Pre-commit hooks enforce style automatically

## TypeScript Code Style Guidelines

- Use Deno's standard style (80 char line width, 2 space indent, semicolons)
- Strict TypeScript compiler options enabled
- Use JSR imports (@std/assert) and npm: prefix for npm packages

## Project-Specific Conventions

### Import Patterns

```python
# Python: Always import from .__about__ for version
from .__about__ import __version__

# Import core functions, not classes
from ngff_zarr import from_ngff_zarr, to_ngff_zarr, to_multiscales

# TypeScript: Function-based exports (not classes)
import { fromNgffZarr, toNgffZarr } from "./io/from_ngff_zarr.ts";
```

### Configuration & Memory Management

- Global config via `ngff_zarr.config` (memory_target, task_target, cache_store)
- Large images automatically trigger disk caching via memory usage estimation
- Controlled by `config.memory_target` (default: 50% available memory)
- Chunking optimized for visualization: 128px (3D) or 256px (2D)

### Zarr Store Handling

```python
# Python: Auto-detects zarr v2/v3, uses appropriate store type
from zarr.storage import DirectoryStore, LocalStore  # v2 vs v3
store = zarr.open.v2() if zarr_v2 else zarr.open()

# TypeScript: Auto-detects HTTP vs local paths
import { FetchStore, FileSystemStore } from "@zarrita/storage";
```

## Testing Infrastructure

- **Fixtures**: `input_images` fixture provides test datasets via pooch
  downloads
- **Test Data**: Located in `py/test/_data.py` with `extract_dir` and
  `test_data_dir`
- **Baseline Testing**: Compare outputs to known-good results
- **Version Skips**: Use `@pytest.mark.skipif(zarr_version < ...)` for version
  compatibility
- **Parametrized Tests**: Heavy use of `@pytest.mark.parametrize` for
  shape/chunk combinations

## Multi-Backend Support & Methods

```python
# Auto-detection in cli_input_to_ngff_image()
ConversionBackend.NGFF_ZARR     # Existing OME-Zarr
ConversionBackend.ITK           # Medical images via ITK
ConversionBackend.ITKWASM       # WebAssembly processing
ConversionBackend.TIFFFILE      # TIFF via tifffile

# Downsampling methods (in Methods enum)
Methods.ITKWASM_GAUSSIAN        # Default, web-compatible
Methods.ITK_GAUSSIAN           # Native ITK
Methods.DASK_IMAGE_GAUSSIAN    # scipy-based fallback
```

## Critical Integration Points

### MCP Server Tools (mcp/ngff_zarr_mcp/tools.py)

- `convert_to_ome_zarr()`: Main conversion function for AI agents
- `ConversionOptions`: Pydantic model for structured parameters
- `setup_dask_config()`: Configures dask for optimal performance
- Async/await patterns throughout for non-blocking operations

### RFC-4 Anatomical Orientation

```python
from ngff_zarr.rfc4 import LPS, RAS, AnatomicalOrientation
# Use add_anatomical_orientation_to_axis() to add spatial context
# Enable via is_rfc4_enabled() configuration
```

### Error Handling Patterns

- **TypeScript**: `zarrita` operations wrapped with version-specific fallbacks
- **Python**: Specific exceptions for zarr version compatibility issues
- **Rich Progress**: Use `NgffProgress` and `NgffProgressCallback` for CLI
  feedback
- **Store validation**: Always check for consolidated metadata first

## Cross-Component Communication

### TypeScript ↔ Python Equivalence

```typescript
// TypeScript (mirrors Python dataclasses)
export class NgffImage {
  data: LazyArray; // Equivalent to dask.array
  dims: string[]; // ["t", "c", "z", "y", "x"]
  scale: Record<string, number>;
  translation: Record<string, number>;
}
```

### MCP Server Integration

- Wraps core `ngff-zarr` functions for AI assistant access
- Provides format detection, validation, and optimization tools
- Uses Pydantic models for type-safe parameter validation

## Development Debugging

### Common Issues

1. **"Node not found: v3 array"** → Use `zarr.open.v2()` for zarr format 2
2. **Memory errors** → Check `config.memory_target`, enable caching
3. **TypeScript import errors** → Use relative imports with `.ts` extension
4. **Test fixture failures** → Ensure test data downloaded via pooch

### Performance Optimization

- Enable `use_tensorstore=True` for very large datasets
- Use `chunks_per_shard` for zarr v3 sharding
- Set appropriate `chunks` parameter for your use case

## Essential Files for Understanding

- **`py/ngff_zarr/to_ngff_zarr.py`** - Core write implementation
- **`py/ngff_zarr/to_multiscales.py`** - Downsampling pipeline
- **`ts/src/io/from_ngff_zarr.ts`** - TypeScript read implementation
- **`py/test/_data.py`** - Test infrastructure and baselines

## Agent Instructions

- When problem-solving Python issues, write temporary test scripts and run them
  with `pixi run --as-is -e test python debug_script.py`, for example. Or, run individual tests
  with `pixi run --as-is -e test pytest tests/test_*.py`. Do not try to use
  `pixi run --as-is pytest ...` or `pixi run --as-is python -c '<command>'` as these will not
  work correctly.
- **Before every `git commit`, ALWAYS run the pre-commit lint tasks.** Run
  `pixi run -e lint lint` from `py/` and `pixi run lint` from `ts/` until they
  pass cleanly with no file modifications. When pre-commit hooks modify files,
  re-stage them with `git add` and re-run lint until all hooks pass. Failure to
  do this causes the `ci` to fail.
