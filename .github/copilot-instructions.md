# AI Coding Agent Instructions for ngff-zarr

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
2. **NgffImage → Multiscales**: Generate multiple resolution levels via
   `to_multiscales()`
3. **Multiscales → OME-Zarr**: Write to zarr stores via `to_ngff_zarr()`
4. **OME-Zarr → Multiscales**: Read back via `from_ngff_zarr()`

### Key Data Classes

- **`NgffImage`**: Single-scale image with dims, scale, translation, and
  dask/lazy array data
- **`Multiscales`**: Container for multiple `NgffImage` scales + OME-Zarr
  metadata
- **`Metadata`**: OME-Zarr spec metadata (axes, datasets, coordinate
  transformations)

## Build & Test Commands

**All development uses pixi for consistent environments:**

### Python (py/)

```bash
pixi run test                    # Run pytest test suite
pixi run pytest path/to/test.py::test_name  # Single test
pixi run lint                    # Pre-commit hooks (ruff, black)
pixi run format                  # Format code
```

### MCP Server (mcp/)

```bash
cd mcp && pixi run test         # Run MCP tests
cd mcp && pixi run typecheck    # mypy type checking
cd mcp && pixi run format       # Format code
cd mcp && pixi run dev          # Run MCP server in dev mode
```

### TypeScript (ts/)

```bash
cd ts && pixi run test          # Deno test suite
cd ts && pixi run lint          # Deno lint
cd ts && pixi run fmt           # Deno format
cd ts && pixi run build         # Full build pipeline
cd ts && pixi run test:browser  # Browser compatibility tests
cd ts && pixi run check         # Type checking
```

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

### Testing Infrastructure

- **Fixtures**: `input_images` fixture provides test datasets via pooch
  downloads
- **Test Data**: Located in `py/test/_data.py` with `extract_dir` and
  `test_data_dir`
- **Baseline Testing**: Compare outputs to known-good results
- **Version Skips**: Use `@pytest.mark.skipif(zarr_version < ...)` for version
  compatibility
- **Parametrized Tests**: Heavy use of `@pytest.mark.parametrize` for
  shape/chunk combinations

### Multi-Backend Support & Methods

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
  with `pixi run -e test python debug_script.py`, for example. Or, run individual tests
  with `pixi run -e test pytest tests/test_*.py`. Do not try to use
  `pixi run pytest ...` or `pixi run python -c '<command>'` as these will not
  work correctly.
