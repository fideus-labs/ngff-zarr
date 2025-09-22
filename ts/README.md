<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->

# ngff-zarr TypeScript

A TypeScript implementation of ngff-zarr for reading and writing OME-Zarr files,
compatible with Deno, Node.js, and the browser.

## Features

- 🦕 **Deno-first**: Built for Deno with first-class TypeScript support
- 📦 **npm compatible**: Automatically builds npm packages using @deno/dnt
- 🔍 **Type-safe**: Full TypeScript support with Zod schema validation
- 🗂️ **OME-Zarr support**: Read and write OME-Zarr files using zarrita
- 🧪 **Well-tested**: Comprehensive test suite using @std/assert
- 🏗️ **Mirrors Python API**: TypeScript classes and types mirror the Python
  dataclasses

## Installation

### Deno

```typescript
import * as ngffZarr from "@fideus-labs/ngff-zarr";
```

### Node.js/npm

```bash
npm install @fideus-labs/ngff-zarr
```

```typescript
import * as ngffZarr from "@fideus-labs/ngff-zarr";
```

## Quick Start

### Reading OME-Zarr files

```typescript
import { OMEZarrReader } from "@fideus-labs/ngff-zarr";

const reader = new OMEZarrReader({ validate: true });
const multiscales = await reader.fromNgffZarr("path/to/image.ome.zarr");

console.log(`Loaded ${multiscales.images.length} scale levels`);
console.log(`Image shape: ${multiscales.images[0].data.shape}`);
```

### Writing OME-Zarr files

```typescript
import {
  createMetadata,
  createMultiscales,
  createNgffImage,
  OMEZarrWriter,
} from "@fideus-labs/ngff-zarr";

// Create a simple 2D image
const image = createNgffImage(
  new ArrayBuffer(256 * 256),
  [256, 256],
  "uint8",
  ["y", "x"],
  { y: 1.0, x: 1.0 },
  { y: 0.0, x: 0.0 },
);

// Create metadata
const axes = [
  { name: "y", type: "space" },
  { name: "x", type: "space" },
];
const datasets = [{
  path: "0",
  coordinateTransformations: [
    { type: "scale", scale: [1.0, 1.0] },
    { type: "translation", translation: [0.0, 0.0] },
  ],
}];
const metadata = createMetadata(axes, datasets);

// Create multiscales
const multiscales = createMultiscales([image], metadata);

// Write to OME-Zarr
const writer = new OMEZarrWriter();
await writer.toNgffZarr("output.ome.zarr", multiscales);
```

### Schema Validation

```typescript
import { MetadataSchema, validateMetadata } from "@fideus-labs/ngff-zarr";

const metadata = {
  axes: [
    { name: "y", type: "space" },
    { name: "x", type: "space" },
  ],
  datasets: [{
    path: "0",
    coordinateTransformations: [
      { type: "scale", scale: [1.0, 1.0] },
      { type: "translation", translation: [0.0, 0.0] },
    ],
  }],
  name: "test_image",
  version: "0.4",
};

// Validate using Zod schema
const validatedMetadata = MetadataSchema.parse(metadata);

// Or use utility function
const validatedMetadata2 = validateMetadata(metadata);
```

## API Reference

### Core Types

- `NgffImage`: Represents an image with associated metadata
- `Multiscales`: Container for multiple scale levels of an image
- `DaskArray`: Metadata representation of dask arrays
- `Metadata`: OME-Zarr metadata structure

### I/O Classes

- `OMEZarrReader`: Read OME-Zarr files using zarrita
- `OMEZarrWriter`: Write OME-Zarr files using zarrita

### Validation

- Zod schemas for all data structures
- Utility functions for validation
- Type-safe parsing and validation

### Factory Functions

- `createNgffImage()`: Create NgffImage instances
- `createMetadata()`: Create metadata structures
- `createMultiscales()`: Create multiscale containers

## Development

### Prerequisites

- [Deno](https://deno.land/) >= 1.40.0
- [Pixi](https://pixi.sh/) (optional, for environment management)

### Setup

```bash
cd ts/
pixi install  # or ensure Deno is installed
```

### Commands

```bash
# Run tests
pixi run test
# or
deno test --allow-all

# Format code
pixi run fmt
# or
deno fmt

# Lint code
pixi run lint
# or
deno lint

# Type check
pixi run check
# or
deno check src/mod.ts

# Build npm package
pixi run build-npm
# or
deno task build:npm

# Browser testing with Playwright
pixi run test-browser
# or
deno task test:browser

# Browser testing with UI
deno task test:browser:ui

# Run all tests (including browser tests)
deno task test:all
```

### Browser Testing

This project includes comprehensive browser testing using Playwright. The
browser tests verify that the npm-transpiled package works correctly in real
browser environments.

**Features:**

- Multi-browser testing (Chrome, Firefox, Safari, Mobile)
- NPM package validation in browser context
- Import resolution testing for dependencies
- Schema validation in browser environment

**Quick Start:**

```bash
# Build npm package and run browser tests
deno task test:browser:npm

# Run browser tests with interactive UI
deno task test:browser:ui

# Debug browser tests
deno task test:browser:debug
```

**Test Files:**

- `/test/browser-npm/` - Browser test files and configuration
- `/test/browser/server.ts` - Test server for serving assets
- `/scripts/run_playwright.mjs` - Node.js wrapper for Playwright execution

See [test/browser/README.md](test/browser/README.md) for detailed browser
testing documentation.

### Project Structure

```
ts/
├── src/
│   ├── types/          # TypeScript type definitions
│   ├── schemas/        # Zod validation schemas
│   ├── io/            # Zarr I/O operations
│   ├── utils/         # Utility functions
│   └── mod.ts         # Main module exports
├── test/              # Test files
├── scripts/           # Build scripts
├── deno.json          # Deno configuration
├── pixi.toml          # Pixi environment
└── README.md
```

## Relationship to Python Package

This TypeScript package mirrors the Python ngff-zarr package:

- **Types**: TypeScript interfaces/classes correspond to Python dataclasses
- **API**: Similar function names and patterns
- **Validation**: Zod schemas provide runtime validation like Python
- **I/O**: zarrita provides Zarr operations like zarr-python

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite: `deno test --allow-all`
6. Submit a pull request

## License

MIT License - see LICENSE file for details.
