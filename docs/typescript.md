<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# 🦕 TypeScript / JavaScript Interface

NGFF-Zarr provides a TypeScript implementation for working with OME-Zarr data structures in Deno, Node.js, and browser environments. The TypeScript API mirrors the Python interface, providing a familiar and consistent experience across languages.

## ✨ Features

- 🦕 **Deno-first**: Built for Deno with first-class TypeScript support
- 📦 **Universal compatibility**: Works in Deno, Node.js, and browsers
- 🔍 **Type-safe**: Full TypeScript support with Zod schema validation
- 🗂️ **OME-Zarr support**: Read and write OME-Zarr v0.4 and v0.5
- 🧪 **Well-tested**: Comprehensive test suite with browser validation
- 🏗️ **Mirrors Python API**: Familiar interfaces for Python users
- 📖 **Lazy loading**: Efficient handling of large datasets
- 🌐 **Web ready**: No filesystem dependencies, works with remote stores

## 📦 Installation

### JSR (Deno and Node.js)

The package is published to [JSR (JavaScript Registry)](https://jsr.io/) for use with Deno and modern Node.js:

**Deno:**
```typescript
import * as ngffZarr from "jsr:@fideus-labs/ngff-zarr";
```

**Node.js with JSR support:**
```bash
npx jsr add @fideus-labs/ngff-zarr
```

```typescript
import * as ngffZarr from "@fideus-labs/ngff-zarr";
```

### npm (Node.js and Bundlers)

For traditional Node.js projects and bundlers like webpack, Vite, or esbuild:

```bash
npm install @fideus-labs/ngff-zarr
```

```typescript
import * as ngffZarr from "@fideus-labs/ngff-zarr";
```

### Browser (CDN)

For direct browser usage via CDN:

```html
<script type="module">
  import * as ngffZarr from "https://esm.sh/@fideus-labs/ngff-zarr";

  // Your code here
</script>
```

## 🚀 Quick Start

### Reading OME-Zarr Files

Read an OME-Zarr file and access the multiscale data:

```typescript
import { fromNgffZarr } from "@fideus-labs/ngff-zarr";

// Read from a local file (Deno/Node.js)
const multiscales = await fromNgffZarr("path/to/image.ome.zarr");

// Read from a remote URL (works in all environments)
const remoteMultiscales = await fromNgffZarr(
  "https://example.com/data.ome.zarr"
);

// Access the image data
console.log(`Loaded ${multiscales.images.length} scale levels`);
console.log(`Image shape: ${multiscales.images[0].data.shape}`);
console.log(`Data type: ${multiscales.images[0].data.dtype}`);

// Access metadata
console.log(`Axes: ${JSON.stringify(multiscales.metadata.axes)}`);
console.log(`Version: ${multiscales.metadata.version}`);
```

### Validating OME-Zarr Metadata

Enable validation to ensure OME-Zarr files conform to the specification:

```typescript
import { fromNgffZarr } from "@fideus-labs/ngff-zarr";

// Validate during reading
const multiscales = await fromNgffZarr("image.ome.zarr", {
  validate: true,
});

// Specify expected version
const multiscalesV5 = await fromNgffZarr("image.ome.zarr", {
  validate: true,
  version: "0.5",
});
```

### Creating and Writing OME-Zarr Files

Create a simple OME-Zarr file from scratch:

```typescript
import {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
  createNgffImage,
  toNgffZarr,
} from "@fideus-labs/ngff-zarr";

// Create image data (256x256 grayscale)
const data = new Uint8Array(256 * 256);
for (let i = 0; i < data.length; i++) {
  data[i] = Math.floor(Math.random() * 256);
}

// Create an NgffImage with metadata
const image = createNgffImage(
  data.buffer,
  [256, 256],           // shape
  "uint8",              // dtype
  ["y", "x"],          // dimension names
  { y: 1.0, x: 1.0 },  // scale (pixel spacing)
  { y: 0.0, x: 0.0 }   // translation (origin)
);

// Create OME-Zarr metadata
const axes = [
  createAxis("y", "space", "micrometer"),
  createAxis("x", "space", "micrometer"),
];

const datasets = [
  createDataset(
    "0",                           // path
    [1.0, 1.0],                   // scale
    [0.0, 0.0]                    // translation
  ),
];

const metadata = createMetadata(axes, datasets, "my_image");

// Create multiscales container
const multiscales = createMultiscales([image], metadata);

// Write to OME-Zarr
await toNgffZarr("output.ome.zarr", multiscales);
```

### Generating Multiscale Pyramids

Create multiscale image pyramids with downsampling:

```typescript
import {
  createNgffImage,
  toMultiscales,
  toNgffZarr,
  Methods,
} from "@fideus-labs/ngff-zarr";

// Create base image
const data = new Uint8Array(512 * 512);
const image = createNgffImage(
  data.buffer,
  [512, 512],
  "uint8",
  ["y", "x"],
  { y: 1.0, x: 1.0 },
  { y: 0.0, x: 0.0 }
);

// Generate multiscale pyramid with 2x and 4x downsampling
const multiscales = await toMultiscales(image, {
  scaleFactors: [2, 4],
  method: Methods.ITKWASM_GAUSSIAN,
  chunks: 128,
});

// Write the pyramid
await toNgffZarr("pyramid.ome.zarr", multiscales);
```

## 🌍 Platform Support

### Deno

Full native support with TypeScript:

```typescript
import * as ngffZarr from "@fideus-labs/ngff-zarr";

const multiscales = await ngffZarr.fromNgffZarr("./data.ome.zarr");
```

### Node.js

Compatible with Node.js 18+ (ESM and CommonJS):

```typescript
// ESM
import { fromNgffZarr } from "@fideus-labs/ngff-zarr";

// CommonJS
const { fromNgffZarr } = require("@fideus-labs/ngff-zarr");
```

### Browser

Works in modern browsers with ES modules:

```html
<!DOCTYPE html>
<html>
<head>
  <title>OME-Zarr in Browser</title>
</head>
<body>
  <script type="module">
    import { fromNgffZarr } from "https://esm.sh/@fideus-labs/ngff-zarr";

    // Load from remote URL
    const multiscales = await fromNgffZarr(
      "https://example.com/data.ome.zarr"
    );

    console.log("Loaded:", multiscales.images.length, "scales");
  </script>
</body>
</html>
```

**Browser Limitations:**
- Cannot read from local filesystem (use HTTP URLs instead)
- FileSystemStore is not available in browsers
- Use FetchStore for remote data access

## 📚 API Reference

### Core Types

#### `NgffImage`

Represents a single-scale image with associated metadata:

```typescript
interface NgffImage {
  data: DaskArray;              // Lazy array data
  dims: string[];               // Dimension names ["t", "c", "z", "y", "x"]
  scale: Record<string, number>; // Pixel spacing per dimension
  translation: Record<string, number>; // Origin offset
  name?: string;                 // Optional image name
  axesUnits?: Record<string, Units>; // Optional units per axis
}
```

**Example:**
```typescript
import { createNgffImage } from "@fideus-labs/ngff-zarr";

const image = createNgffImage(
  arrayBuffer,
  [100, 100],
  "uint16",
  ["y", "x"],
  { y: 0.5, x: 0.5 },
  { y: 0.0, x: 0.0 }
);
```

#### `NgffMultiscales`

Container for multiple resolution levels:

```typescript
interface NgffMultiscales {
  images: NgffImage[];          // Array of scale levels
  metadata: Metadata;           // OME-Zarr metadata
  scaleFactors?: (number | Record<string, number>)[]; // Scale factors
  method?: Methods;             // Downsampling method
  chunks?: Record<string, number>; // Chunk sizes
}
```

#### `Metadata`

OME-Zarr metadata structure:

```typescript
interface Metadata {
  axes: Axis[];                 // Dimension definitions
  datasets: Dataset[];          // Scale level paths and transforms
  coordinateTransformations?: CoordinateTransformation[];
  name?: string;
  version?: "0.4" | "0.5";
  type?: string;                // Downsampling method type
  metadata?: Record<string, unknown>;
}
```

#### `Axis`

Dimension axis definition:

```typescript
interface Axis {
  name: string;                 // "t", "c", "z", "y", "x"
  type?: "time" | "channel" | "space";
  unit?: Units;                 // UDUNITS-2 identifier
}
```

### I/O Functions

#### `fromNgffZarr()`

Read an OME-Zarr file:

```typescript
async function fromNgffZarr(
  store: string | MemoryStore | FetchStore,
  options?: {
    validate?: boolean;
    version?: "0.4" | "0.5";
  }
): Promise<NgffMultiscales>
```

**Parameters:**
- `store`: Path to OME-Zarr (string), URL (string), or Zarr store object
- `options.validate`: Enable metadata validation (default: false)
- `options.version`: Expected OME-Zarr version

**Returns:** `NgffMultiscales` object with all scale levels

**Example:**
```typescript
// Basic reading
const ms = await fromNgffZarr("data.ome.zarr");

// With validation
const validatedMs = await fromNgffZarr("data.ome.zarr", {
  validate: true,
  version: "0.5",
});

// From URL
const remoteMs = await fromNgffZarr("https://example.com/data.ome.zarr");
```

#### `toNgffZarr()`

Write an NgffMultiscales object to OME-Zarr:

```typescript
async function toNgffZarr(
  store: string,
  multiscales: NgffMultiscales,
  options?: {
    version?: "0.4" | "0.5";
    chunksPerShard?: number | number[] | Record<string, number>;
  }
): Promise<void>
```

**Parameters:**
- `store`: Output path for OME-Zarr
- `multiscales`: NgffMultiscales object to write
- `options.version`: OME-Zarr version (default: "0.5")
- `options.chunksPerShard`: Sharding configuration (v0.5 only)

**Example:**
```typescript
// Basic writing
await toNgffZarr("output.ome.zarr", multiscales);

// With version specification
await toNgffZarr("output.ome.zarr", multiscales, { version: "0.5" });

// With sharding (v0.5)
await toNgffZarr("output.ome.zarr", multiscales, {
  version: "0.5",
  chunksPerShard: { z: 2, y: 2, x: 2 },
});
```

#### `toMultiscales()`

Generate multiscale pyramid from a single image:

```typescript
async function toMultiscales(
  image: NgffImage,
  options?: {
    scaleFactors?: (number | Record<string, number>)[];
    method?: Methods;
    chunks?: number | number[] | Record<string, number>;
  }
): Promise<NgffMultiscales>
```

**Parameters:**
- `image`: Base NgffImage to downsample
- `options.scaleFactors`: Downsampling factors (default: [2, 4])
- `options.method`: Downsampling method (default: ITKWASM_GAUSSIAN)
- `options.chunks`: Chunk sizes for output

**Returns:** NgffMultiscales with generated pyramid levels

**Example:**
```typescript
// Basic pyramid generation
const ms = await toMultiscales(image);

// Custom scale factors
const ms2 = await toMultiscales(image, {
  scaleFactors: [2, 4, 8],
  method: Methods.ITKWASM_GAUSSIAN,
});

// With custom chunks
const ms3 = await toMultiscales(image, {
  scaleFactors: [{ x: 2, y: 2 }, { x: 4, y: 4 }],
  chunks: 256,
});
```

### Factory Functions

#### `createNgffImage()`

Create an NgffImage from raw data:

```typescript
function createNgffImage(
  buffer: ArrayBuffer,
  shape: number[],
  dtype: string,
  dims: string[],
  scale: Record<string, number>,
  translation: Record<string, number>,
  name?: string
): NgffImage
```

#### `createAxis()`

Create an axis definition:

```typescript
function createAxis(
  name: string,
  type: "time" | "channel" | "space",
  unit?: Units
): Axis
```

#### `createDataset()`

Create a dataset entry:

```typescript
function createDataset(
  path: string,
  scale: number[],
  translation: number[]
): Dataset
```

#### `createMetadata()`

Create OME-Zarr metadata:

```typescript
function createMetadata(
  axes: Axis[],
  datasets: Dataset[],
  name?: string
): Metadata
```

#### `createMultiscales()`

Create an NgffMultiscales container:

```typescript
function createMultiscales(
  images: NgffImage[],
  metadata: Metadata,
  scaleFactors?: (number | Record<string, number>)[],
  method?: Methods
): NgffMultiscales
```

### Validation

#### Schema Validation with Zod

The package includes Zod schemas for runtime validation:

```typescript
import {
  MetadataSchema,
  NgffImageSchema,
  MultiscalesSchema,
  validateMetadata,
} from "@fideus-labs/ngff-zarr";

// Validate metadata
const metadata = { axes: [...], datasets: [...] };
const validated = MetadataSchema.parse(metadata);

// Safe parsing (returns result object)
const result = MetadataSchema.safeParse(metadata);
if (result.success) {
  console.log("Valid metadata:", result.data);
} else {
  console.error("Validation errors:", result.error.issues);
}

// Utility function
const validMetadata = validateMetadata(metadata);
```

### Downsampling Methods

Available downsampling methods via the `Methods` enum:

```typescript
enum Methods {
  ITKWASM_GAUSSIAN = "itkwasm_gaussian",        // Gaussian smoothing (default)
  ITKWASM_BIN_SHRINK = "itkwasm_bin_shrink",    // Bin shrinking
  ITKWASM_LABEL_IMAGE = "itkwasm_label_image",  // Label-aware downsampling
}
```

**Example:**
```typescript
import { toMultiscales, Methods } from "@fideus-labs/ngff-zarr";

const pyramid = await toMultiscales(image, {
  method: Methods.ITKWASM_GAUSSIAN,
  scaleFactors: [2, 4],
});
```

## 💡 Examples

### Example 1: Convert Image Format

Read an image and convert to OME-Zarr:

```typescript
import {
  itkImageToNgffImage,
  toMultiscales,
  toNgffZarr,
} from "@fideus-labs/ngff-zarr";

// Load image using itk-wasm
const itkImage = await loadImageFromFile("input.png");

// Convert to NgffImage
const ngffImage = await itkImageToNgffImage(itkImage);

// Generate pyramid
const multiscales = await toMultiscales(ngffImage);

// Write to OME-Zarr
await toNgffZarr("output.ome.zarr", multiscales);
```

### Example 2: Process Remote Data

Work with remote OME-Zarr files:

```typescript
import { fromNgffZarr } from "@fideus-labs/ngff-zarr";

// Read from S3 or other HTTP-accessible storage
const multiscales = await fromNgffZarr(
  "https://s3.amazonaws.com/bucket/data.ome.zarr"
);

// Access the highest resolution
const fullRes = multiscales.images[0];
console.log(`Shape: ${fullRes.data.shape}`);
console.log(`Spacing: ${JSON.stringify(fullRes.scale)}`);

// Access a lower resolution
const lowRes = multiscales.images[multiscales.images.length - 1];
console.log(`Low-res shape: ${lowRes.data.shape}`);
```

### Example 3: Metadata Inspection

Examine OME-Zarr metadata:

```typescript
import { fromNgffZarr } from "@fideus-labs/ngff-zarr";

const multiscales = await fromNgffZarr("data.ome.zarr");

// Inspect axes
multiscales.metadata.axes.forEach(axis => {
  console.log(`Axis ${axis.name}: type=${axis.type}, unit=${axis.unit}`);
});

// Inspect datasets
multiscales.metadata.datasets.forEach((dataset, i) => {
  console.log(`Scale ${i}: path=${dataset.path}`);
  dataset.coordinateTransformations.forEach(transform => {
    if (transform.type === "scale") {
      console.log(`  Scale: ${transform.scale}`);
    }
  });
});

// Check version
console.log(`OME-Zarr version: ${multiscales.metadata.version}`);
```

### Example 4: Custom Chunk Sizes

Control chunking for optimal performance:

```typescript
import {
  createNgffImage,
  toMultiscales,
  toNgffZarr,
} from "@fideus-labs/ngff-zarr";

const image = createNgffImage(
  buffer,
  [1024, 1024, 1024], // Large 3D volume
  "uint16",
  ["z", "y", "x"],
  { z: 1.0, y: 0.5, x: 0.5 },
  { z: 0.0, y: 0.0, x: 0.0 }
);

// Use smaller chunks for better streaming
const multiscales = await toMultiscales(image, {
  chunks: { z: 64, y: 128, x: 128 },
  scaleFactors: [2, 4, 8],
});

await toNgffZarr("large_volume.ome.zarr", multiscales);
```

### Example 5: TypeScript Type Safety

Leverage TypeScript's type system:

```typescript
import type {
  NgffImage,
  NgffMultiscales,
  Metadata,
  Axis,
} from "@fideus-labs/ngff-zarr";
import { fromNgffZarr, validateMetadata } from "@fideus-labs/ngff-zarr";

// Type-safe function
async function processImage(path: string): Promise<NgffMultiscales> {
  const multiscales: NgffMultiscales = await fromNgffZarr(path);

  // TypeScript knows the structure
  const axes: Axis[] = multiscales.metadata.axes;
  const firstImage: NgffImage = multiscales.images[0];

  return multiscales;
}

// Type-safe metadata validation
function ensureValidMetadata(metadata: unknown): Metadata {
  return validateMetadata(metadata);
}
```

### Example 6: JavaScript Usage

Pure JavaScript usage without TypeScript:

```javascript
// JavaScript (Node.js or browser)
import { fromNgffZarr, toNgffZarr, createNgffImage } from "@fideus-labs/ngff-zarr";

// Read OME-Zarr
const multiscales = await fromNgffZarr("data.ome.zarr");

// Create new image
const data = new Uint8Array(100 * 100);
const image = createNgffImage(
  data.buffer,
  [100, 100],
  "uint8",
  ["y", "x"],
  { y: 1.0, x: 1.0 },
  { y: 0.0, x: 0.0 }
);

// Write to OME-Zarr
await toNgffZarr("output.ome.zarr", image);
```

## 🔧 Advanced Usage

### Working with Different Store Types

```typescript
import * as zarr from "zarrita";
import { fromNgffZarr } from "@fideus-labs/ngff-zarr";

// Memory store
const memoryStore = new Map<string, Uint8Array>();
const ms1 = await fromNgffZarr(memoryStore);

// Fetch store (HTTP/HTTPS)
const fetchStore = new zarr.FetchStore("https://example.com/data.ome.zarr");
const ms2 = await fromNgffZarr(fetchStore);

// File system store (Node.js/Deno only)
import { FileSystemStore } from "@zarrita/storage";
const fsStore = new FileSystemStore("/path/to/data.ome.zarr");
const ms3 = await fromNgffZarr(fsStore);
```

### Converting Between OME-Zarr Versions

```typescript
import { fromNgffZarr, toNgffZarr } from "@fideus-labs/ngff-zarr";

// Convert from v0.4 to v0.5
const multiscales = await fromNgffZarr("data_v04.ome.zarr");
await toNgffZarr("data_v05.ome.zarr", multiscales, { version: "0.5" });

// Convert from v0.5 to v0.4
const multiscalesV5 = await fromNgffZarr("data_v05.ome.zarr");
await toNgffZarr("data_v04.ome.zarr", multiscalesV5, { version: "0.4" });
```

### High Content Screening (HCS)

Work with plate and well data:

```typescript
import { fromHcsZarr } from "@fideus-labs/ngff-zarr";

// Read HCS plate
const plate = await fromHcsZarr("plate.ome.zarr");

console.log(`Plate: ${plate.metadata.name}`);
console.log(`Wells: ${plate.metadata.wells.length}`);

// Access specific well
const well = plate.getWell("A", "1");
if (well) {
  console.log(`Well A/1 has ${well.images.length} field(s)`);
}
```

## 🆚 Comparison with Python

The TypeScript API closely mirrors the Python interface:

| Python | TypeScript |
|--------|-----------|
| `from_ngff_zarr()` | `fromNgffZarr()` |
| `to_ngff_zarr()` | `toNgffZarr()` |
| `to_multiscales()` | `toMultiscales()` |
| `to_ngff_image()` | `createNgffImage()` |
| `NgffImage` dataclass | `NgffImage` interface |
| `NgffMultiscales` dataclass | `NgffMultiscales` interface |
| `dask.array.Array` | `DaskArray` (metadata) |
| JSON Schema validation | JSON Schema validation |

### Key Differences

1. **Naming conventions**: Python uses snake_case, TypeScript uses camelCase
2. **Data handling**: Python uses Dask arrays, TypeScript uses array metadata
3. **Async**: TypeScript I/O operations are async by default

## 🐛 Troubleshooting

### Browser CORS Issues

When accessing remote OME-Zarr files in the browser, ensure CORS headers are set:

```typescript
// This may fail due to CORS
const ms = await fromNgffZarr("https://example.com/data.ome.zarr");

// Solution: Ensure server sends proper CORS headers
// Access-Control-Allow-Origin: *
```

### File System Access in Browser

Browsers cannot access local filesystem:

```typescript
// ❌ Won't work in browser
const ms = await fromNgffZarr("/path/to/local/file.ome.zarr");

// ✅ Use HTTP URLs instead
const ms = await fromNgffZarr("http://localhost:8000/file.ome.zarr");
```

### Memory Management

For large datasets, zarrita handles lazy loading automatically:

```typescript
// Data is loaded lazily - metadata read immediately
const multiscales = await fromNgffZarr("large_file.ome.zarr");

// Actual array data loaded on access
const shape = multiscales.images[0].data.shape; // Fast
// Actual data retrieval would happen when reading chunks
```

## 📖 Related Documentation

- [Python Interface](./python.md) - Python API documentation
- [Installation](./installation.md) - Installation and setup guide
- [Methods](./methods.md) - Downsampling methods
- [HCS Support](./hcs.md) - High Content Screening
- [FAQ](./faq.md) - Frequently asked questions

## 📚 External Resources

- [OME-Zarr Specification](https://ngff.openmicroscopy.org/)
- [Zarrita Documentation](https://github.com/manzt/zarrita.js)
- [Deno Manual](https://deno.land/manual)
- [JSR Registry](https://jsr.io/)
- [ITK-Wasm](https://wasm.itk.org/)

## 🤝 Contributing

Contributions are welcome! See the [development documentation](./development.md) for details on:

- Setting up the development environment
- Running tests
- Building the npm package
- Browser testing with Playwright
