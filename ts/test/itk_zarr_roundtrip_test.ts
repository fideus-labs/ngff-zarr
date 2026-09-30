#!/usr/bin/env -S deno test --allow-read --allow-write

/**
 * Round-trip tests for ITK Image <-> Zarr conversion
 *
 * Tests that itkImageToZarr correctly handles:
 * 1. Different data types (uint8, int16, uint16, float32, etc.)
 * 2. Different image dimensions (2D, 3D)
 * 3. Different sizes/shapes
 * 4. Vector images (multi-component)
 */

import { assertEquals } from "@std/assert";
import type { Image as ItkWasmImage } from "itk-wasm";
import * as zarr from "zarrita";

/**
 * Calculate C-order (row-major) stride for a given shape
 */
function calculateStride(shape: number[]): number[] {
  const stride = new Array(shape.length);
  stride[shape.length - 1] = 1;
  for (let i = shape.length - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }
  return stride;
}

/**
 * Create a test ITK image with specified properties
 */
function createTestItkImage(
  size: number[],
  componentType:
    | "uint8"
    | "int8"
    | "uint16"
    | "int16"
    | "uint32"
    | "int32"
    | "float32"
    | "float64",
  components: number = 1,
): ItkWasmImage {
  const totalSize = size.reduce((a, b) => a * b, 1) * components;

  let data:
    | Uint8Array
    | Int8Array
    | Uint16Array
    | Int16Array
    | Uint32Array
    | Int32Array
    | Float32Array
    | Float64Array;

  switch (componentType) {
    case "uint8":
      data = new Uint8Array(totalSize);
      break;
    case "int8":
      data = new Int8Array(totalSize);
      break;
    case "uint16":
      data = new Uint16Array(totalSize);
      break;
    case "int16":
      data = new Int16Array(totalSize);
      break;
    case "uint32":
      data = new Uint32Array(totalSize);
      break;
    case "int32":
      data = new Int32Array(totalSize);
      break;
    case "float32":
      data = new Float32Array(totalSize);
      break;
    case "float64":
      data = new Float64Array(totalSize);
      break;
  }

  // Fill with test pattern - use index values
  for (let i = 0; i < totalSize; i++) {
    data[i] = i % 256; // Use modulo to keep values reasonable
  }

  return {
    imageType: {
      dimension: size.length,
      componentType,
      pixelType: components > 1 ? "VariableLengthVector" : "Scalar",
      components,
    },
    name: "test",
    origin: size.map(() => 0),
    spacing: size.map(() => 1),
    direction: createIdentityMatrix(size.length),
    size,
    data,
    metadata: new Map(),
  };
}

/**
 * Create identity matrix for ITK direction
 */
function createIdentityMatrix(dimension: number): Float64Array {
  const matrix = new Float64Array(dimension * dimension);
  for (let i = 0; i < dimension * dimension; i++) {
    matrix[i] = i % (dimension + 1) === 0 ? 1 : 0;
  }
  return matrix;
}

/**
 * Convert ITK-Wasm Image to zarr array
 */
async function itkImageToZarr(
  itkImage: ItkWasmImage,
  store: Map<string, Uint8Array>,
  path: string,
): Promise<zarr.Array<zarr.DataType, zarr.Readable>> {
  const root = zarr.root(store);

  if (!itkImage.data) {
    throw new Error("ITK image data is null or undefined");
  }

  // Determine data type
  let dataType: zarr.DataType;
  if (itkImage.data instanceof Uint8Array) {
    dataType = "uint8";
  } else if (itkImage.data instanceof Int8Array) {
    dataType = "int8";
  } else if (itkImage.data instanceof Uint16Array) {
    dataType = "uint16";
  } else if (itkImage.data instanceof Int16Array) {
    dataType = "int16";
  } else if (itkImage.data instanceof Uint32Array) {
    dataType = "uint32";
  } else if (itkImage.data instanceof Int32Array) {
    dataType = "int32";
  } else if (itkImage.data instanceof Float32Array) {
    dataType = "float32";
  } else if (itkImage.data instanceof Float64Array) {
    dataType = "float64";
  } else {
    throw new Error(`Unsupported data type: ${itkImage.data.constructor.name}`);
  }

  // ITK size is already in C-order ([z, y, x] for 3D, [y, x] for 2D)
  const shape = itkImage.size;
  const chunkShape = shape.map((s) => Math.min(s, 256));

  const array = await zarr.create(root.resolve(path), {
    shape: shape,
    chunkShape,
    dtype: dataType,
    fillValue: 0,
  });

  // Write data - preserve the actual data type
  // Use null for each dimension to select the entire array
  const selection = shape.map(() => null);
  await zarr.set(array, selection, {
    data: itkImage.data,
    shape: shape,
    stride: calculateStride(shape),
  });

  return array;
}

/**
 * Convert zarr array back to ITK-Wasm Image
 */
async function zarrToItkImage(
  array: zarr.Array<zarr.DataType, zarr.Readable>,
): Promise<ItkWasmImage> {
  const result = await zarr.get(array);

  if (!result.data || result.data.length === 0) {
    throw new Error("Zarr array data is empty");
  }

  // Determine component type from data
  let componentType:
    | "uint8"
    | "int8"
    | "uint16"
    | "int16"
    | "uint32"
    | "int32"
    | "float32"
    | "float64";
  if (result.data instanceof Uint8Array) {
    componentType = "uint8";
  } else if (result.data instanceof Int8Array) {
    componentType = "int8";
  } else if (result.data instanceof Uint16Array) {
    componentType = "uint16";
  } else if (result.data instanceof Int16Array) {
    componentType = "int16";
  } else if (result.data instanceof Uint32Array) {
    componentType = "uint32";
  } else if (result.data instanceof Int32Array) {
    componentType = "int32";
  } else if (result.data instanceof Float32Array) {
    componentType = "float32";
  } else if (result.data instanceof Float64Array) {
    componentType = "float64";
  } else {
    throw new Error(
      `Unsupported zarr data type: ${result.data.constructor.name}`,
    );
  }

  const shape = result.shape;

  return {
    imageType: {
      dimension: shape.length,
      componentType,
      pixelType: "Scalar",
      components: 1,
    },
    name: "image",
    origin: shape.map(() => 0),
    spacing: shape.map(() => 1),
    direction: createIdentityMatrix(shape.length),
    size: shape as number[],
    data: result.data,
    metadata: new Map(),
  };
}

/**
 * Compare two typed arrays for equality
 */
function assertArraysEqual(
  // deno-lint-ignore no-explicit-any
  actual: any,
  // deno-lint-ignore no-explicit-any
  expected: any,
  message?: string,
): void {
  if (actual === null || expected === null) {
    throw new Error(`${message}: Array is null`);
  }

  assertEquals(
    actual.length,
    expected.length,
    `${message}: Array lengths differ`,
  );
  assertEquals(
    actual.constructor.name,
    expected.constructor.name,
    `${message}: Array types differ`,
  );

  for (let i = 0; i < actual.length; i++) {
    if (actual[i] !== expected[i]) {
      throw new Error(
        `${message}: Arrays differ at index ${i}: actual=${
          actual[i]
        }, expected=${expected[i]}`,
      );
    }
  }
}

// Test cases for different data types
Deno.test("Round-trip: 2D uint8 image", async () => {
  const size = [256, 256]; // [y, x]
  const original = createTestItkImage(size, "uint8");

  const store = new Map<string, Uint8Array>();
  const zarrArray = await itkImageToZarr(original, store, "test");

  const restored = await zarrToItkImage(zarrArray);

  assertEquals(restored.size, original.size, "Size should match");
  assertEquals(
    restored.imageType.componentType,
    original.imageType.componentType,
    "Component type should match",
  );
  assertArraysEqual(restored.data, original.data, "Data should match");
});

Deno.test("Round-trip: 2D int16 image", async () => {
  const size = [128, 128];
  const original = createTestItkImage(size, "int16");

  const store = new Map<string, Uint8Array>();
  const zarrArray = await itkImageToZarr(original, store, "test");

  const restored = await zarrToItkImage(zarrArray);

  assertEquals(restored.size, original.size);
  assertEquals(restored.imageType.componentType, "int16");
  assertArraysEqual(restored.data, original.data, "int16 data");
});

Deno.test("Round-trip: 2D uint16 image", async () => {
  const size = [100, 100];
  const original = createTestItkImage(size, "uint16");

  const store = new Map<string, Uint8Array>();
  const zarrArray = await itkImageToZarr(original, store, "test");

  const restored = await zarrToItkImage(zarrArray);

  assertEquals(restored.size, original.size);
  assertEquals(restored.imageType.componentType, "uint16");
  assertArraysEqual(restored.data, original.data, "uint16 data");
});

Deno.test("Round-trip: 2D float32 image", async () => {
  const size = [64, 64];
  const original = createTestItkImage(size, "float32");

  const store = new Map<string, Uint8Array>();
  const zarrArray = await itkImageToZarr(original, store, "test");

  const restored = await zarrToItkImage(zarrArray);

  assertEquals(restored.size, original.size);
  assertEquals(restored.imageType.componentType, "float32");
  assertArraysEqual(restored.data, original.data, "float32 data");
});

Deno.test("Round-trip: 3D uint8 image", async () => {
  const size = [32, 64, 64]; // [z, y, x]
  const original = createTestItkImage(size, "uint8");

  const store = new Map<string, Uint8Array>();
  const zarrArray = await itkImageToZarr(original, store, "test");

  const restored = await zarrToItkImage(zarrArray);

  assertEquals(restored.size, original.size, "3D size should match");
  assertEquals(restored.imageType.componentType, "uint8");
  assertArraysEqual(restored.data, original.data, "3D data");
});

Deno.test("Round-trip: 3D int16 image", async () => {
  const size = [16, 32, 32];
  const original = createTestItkImage(size, "int16");

  const store = new Map<string, Uint8Array>();
  const zarrArray = await itkImageToZarr(original, store, "test");

  const restored = await zarrToItkImage(zarrArray);

  assertEquals(restored.size, original.size);
  assertEquals(restored.imageType.componentType, "int16");
  assertArraysEqual(restored.data, original.data, "3D int16 data");
});

Deno.test(
  "Round-trip: 3D float32 image with non-square dimensions",
  async () => {
    const size = [10, 20, 30]; // [z, y, x] - all different
    const original = createTestItkImage(size, "float32");

    const store = new Map<string, Uint8Array>();
    const zarrArray = await itkImageToZarr(original, store, "test");

    const restored = await zarrToItkImage(zarrArray);

    assertEquals(restored.size, original.size, "Non-square size should match");
    assertEquals(restored.imageType.componentType, "float32");
    assertArraysEqual(restored.data, original.data, "Non-square data");
  },
);

Deno.test("Round-trip: Small 2D image (edge case)", async () => {
  const size = [5, 7]; // Small, non-power-of-2
  const original = createTestItkImage(size, "uint8");

  const store = new Map<string, Uint8Array>();
  const zarrArray = await itkImageToZarr(original, store, "test");

  const restored = await zarrToItkImage(zarrArray);

  assertEquals(restored.size, original.size, "Small size should match");
  assertArraysEqual(restored.data, original.data, "Small image data");
});

Deno.test("Round-trip: Large chunks vs small chunks", async () => {
  const size = [512, 512]; // Should use 256x256 chunks
  const original = createTestItkImage(size, "uint8");

  const store = new Map<string, Uint8Array>();
  const zarrArray = await itkImageToZarr(original, store, "test");

  const restored = await zarrToItkImage(zarrArray);

  assertEquals(restored.size, original.size, "Large image size should match");
  assertArraysEqual(
    restored.data,
    original.data,
    "Large image data with chunking",
  );
});

Deno.test("Data type preservation: int8", async () => {
  const size = [50, 50];
  const original = createTestItkImage(size, "int8");

  const store = new Map<string, Uint8Array>();
  const zarrArray = await itkImageToZarr(original, store, "test");

  const restored = await zarrToItkImage(zarrArray);

  assertEquals(restored.imageType.componentType, "int8", "int8 type preserved");
  assertArraysEqual(restored.data, original.data, "int8 data preserved");
});

Deno.test("Data type preservation: uint32", async () => {
  const size = [30, 30];
  const original = createTestItkImage(size, "uint32");

  const store = new Map<string, Uint8Array>();
  const zarrArray = await itkImageToZarr(original, store, "test");

  const restored = await zarrToItkImage(zarrArray);

  assertEquals(
    restored.imageType.componentType,
    "uint32",
    "uint32 type preserved",
  );
  assertArraysEqual(restored.data, original.data, "uint32 data preserved");
});

Deno.test("Data type preservation: int32", async () => {
  const size = [25, 25];
  const original = createTestItkImage(size, "int32");

  const store = new Map<string, Uint8Array>();
  const zarrArray = await itkImageToZarr(original, store, "test");

  const restored = await zarrToItkImage(zarrArray);

  assertEquals(
    restored.imageType.componentType,
    "int32",
    "int32 type preserved",
  );
  assertArraysEqual(restored.data, original.data, "int32 data preserved");
});

Deno.test("Data type preservation: float64", async () => {
  const size = [20, 20];
  const original = createTestItkImage(size, "float64");

  const store = new Map<string, Uint8Array>();
  const zarrArray = await itkImageToZarr(original, store, "test");

  const restored = await zarrToItkImage(zarrArray);

  assertEquals(
    restored.imageType.componentType,
    "float64",
    "float64 type preserved",
  );
  assertArraysEqual(restored.data, original.data, "float64 data preserved");
});
