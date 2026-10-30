<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->

# ngff-zarr TypeScript

[![npm - Version](https://img.shields.io/npm/v/@fideus-labs/ngff-zarr.svg)](https://www.npmjs.com/package/@fideus-labs/ngff-zarr)

A TypeScript implementation of ngff-zarr for reading and writing OME-Zarr files,
compatible with Node.js and the browser.

## Features

- 📘 **TypeScript-first**: Written in TypeScript, shipped with declarations
- 📦 **npm package**: ESM build for Node.js, bundlers, and the browser
- 🔍 **Type-safe**: Full TypeScript support with Zod schema validation
- 🗂️ **OME-Zarr support**: Read OME-Zarr v0.4, v0.5, and v0.6, and write v0.5 and v0.6 (Zarr v3), using fizarrita
- 🧪 **Well-tested**: Comprehensive test suite run with Vitest, Node, and Playwright
- 🏗️ **Mirrors Python API**: TypeScript classes and types mirror the Python
  dataclasses

## Installation

### Node.js/npm

```bash
npm install @fideus-labs/ngff-zarr
```

```typescript
import * as ngffZarr from "@fideus-labs/ngff-zarr";
```

The codec worker ships as a separate file in `dist/workers/` and is spawned
with `new Worker(new URL("../workers/omero_codec_worker.js", import.meta.url), { type: "module" })`,
the form bundlers such as Vite and webpack 5 recognise as a worker entry
point and emit alongside your bundle without configuration.

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
const datasets = [
  {
    path: "0",
    coordinateTransformations: [
      { type: "scale", scale: [1.0, 1.0] },
      { type: "translation", translation: [0.0, 0.0] },
    ],
  },
];
const metadata = createMetadata(axes, datasets);

// Create multiscales
const multiscales = createMultiscales([image], metadata);

// Write to OME-Zarr
const writer = new OMEZarrWriter();
await writer.toOmeZarr("output.ome.zarr", multiscales);
```

### Schema Validation

```typescript
import { MetadataSchema, validateMetadata } from "@fideus-labs/ngff-zarr";

const metadata = {
  axes: [
    { name: "y", type: "space" },
    { name: "x", type: "space" },
  ],
  datasets: [
    {
      path: "0",
      coordinateTransformations: [
        { type: "scale", scale: [1.0, 1.0] },
        { type: "translation", translation: [0.0, 0.0] },
      ],
    },
  ],
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

- `OMEZarrReader`: Read OME-Zarr files using fizarrita
- `OMEZarrWriter`: Write OME-Zarr files using fizarrita

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

- [Node.js](https://nodejs.org/) 22.18, 24.11, or newer (`npm` included)
- [Pixi](https://pixi.sh/) (optional; provides Node.js and the task aliases
  below)

The toolchain is [Vite+](https://viteplus.dev) (`vp`), installed as a dev
dependency: Vitest for tests, Oxlint and Oxfmt for linting and formatting,
tsdown for the build.

### Setup

```bash
cd ts/
npm ci          # or: pixi run install
```

### Commands

```bash
# Run tests
npm test                 # or: pixi run test

# Format code
npm run fmt              # or: pixi run fmt

# Lint code
npm run lint             # or: pixi run lint

# Format + lint + type check in one pass
npm run check            # or: pixi run check

# Build the npm package into dist/
npm run build            # or: pixi run build

# Node tests against the built package
npm run test:node        # or: pixi run test-node

# Browser testing with Playwright
npm run test:browser     # or: pixi run test-browser

# Browser testing with UI
npm run test:browser:ui

# Run all tests (including browser tests)
npm run test:all         # or: pixi run test-all
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
# Build the package, inline the worker, bundle, and run the browser tests
npm run test:browser:bundle

# Run browser tests with interactive UI
npm run test:browser:ui

# Debug browser tests
npm run test:browser:debug
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
├── package.json       # npm package manifest and scripts
├── vite.config.ts     # Vite+ configuration (test, lint, fmt, pack)
├── tsconfig.json      # TypeScript configuration
├── pixi.toml          # Pixi environment
└── README.md
```

## Relationship to Python Package

This TypeScript package mirrors the Python ngff-zarr package:

- **Types**: TypeScript interfaces/classes correspond to Python dataclasses
- **API**: Similar function names and patterns
- **Validation**: Zod schemas provide runtime validation like Python
- **I/O**: fizarrita provides Zarr operations like zarr-python

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite: `npm test`
6. Submit a pull request

## License

MIT License - see LICENSE file for details.
