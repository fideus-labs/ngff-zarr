// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import { assert, assertEquals } from "@std/assert";
import {
  FloatTypes,
  Image as ITKImage,
  ImageType as ITKImageType,
  IntTypes,
  PixelTypes,
} from "itk-wasm";
import { ngffImageToItkImage } from "../src/io/ngff_image_to_itk_image.ts";
import { itkImageToNgffImage } from "../src/io/itk_image_to_ngff_image.ts";
import {
  AnatomicalOrientationValues,
  createAnatomicalOrientation,
  LPS,
  RAS,
} from "../src/types/rfc4.ts";
import { NgffImage } from "../src/types/ngff_image.ts";
import * as zarr from "zarrita";
import { defaultCodecs } from "../src/utils/codecs.ts";
import { zarrSet } from "../src/utils/worker_pool.ts";
import { _zarrita_internal_get_strides as getStrides } from "zarrita";

// Basic test to verify function is exportable
Deno.test("ngffImageToItkImage function exports", async () => {
  // Test that the function can be imported
  const { ngffImageToItkImage } = await import(
    "../src/io/ngff_image_to_itk_image.ts"
  );

  assertEquals(typeof ngffImageToItkImage, "function");
});

// Test the module can be imported from mod.ts
Deno.test("ngffImageToItkImage exported from mod.ts", async () => {
  const mod = await import("../src/mod.ts");

  assertEquals(typeof mod.ngffImageToItkImage, "function");
});

// Basic conversion tests using round-trip approach to avoid zarrita complexity
Deno.test("ngffImageToItkImage 2D basic conversion", async () => {
  // Create an ITK image and convert it to NGFF, then back to ITK
  const originalSize = [64, 32]; // x, y
  const originalSpacing = [1.5, 2.0];
  const originalOrigin = [10.0, 20.0];
  const dataLength = originalSize.reduce((a, b) => a * b, 1);
  const originalData = new Float32Array(dataLength);

  for (let i = 0; i < dataLength; i++) {
    originalData[i] = i + 1;
  }

  const originalImageType: ITKImageType = {
    dimension: 2,
    componentType: FloatTypes.Float32,
    pixelType: PixelTypes.Scalar,
    components: 1,
  };

  const originalItkImage: ITKImage = {
    imageType: originalImageType,
    name: "test_2d",
    origin: originalOrigin,
    spacing: originalSpacing,
    direction: new Float64Array([1, 0, 0, 1]),
    size: originalSize,
    metadata: new Map(),
    data: originalData,
  };

  // Convert ITK -> NGFF -> ITK
  const ngffImage = await itkImageToNgffImage(originalItkImage);
  const reconvertedItkImage = await ngffImageToItkImage(ngffImage);

  // Verify basic properties
  assertEquals(reconvertedItkImage.imageType.dimension, 2);
  assertEquals(reconvertedItkImage.imageType.componentType, FloatTypes.Float32);
  assertEquals(reconvertedItkImage.imageType.pixelType, PixelTypes.Scalar);

  // Check data integrity
  assert(reconvertedItkImage.data, "Reconverted image data should not be null");
  assertEquals(reconvertedItkImage.data.length, dataLength);
});

Deno.test("ngffImageToItkImage 3D basic conversion", async () => {
  const originalSize = [32, 48, 16]; // x, y, z
  const originalSpacing = [1.0, 1.5, 2.0];
  const originalOrigin = [0.0, 10.0, 20.0];
  const dataLength = originalSize.reduce((a, b) => a * b, 1);
  const originalData = new Uint16Array(dataLength);

  for (let i = 0; i < dataLength; i++) {
    originalData[i] = i % 65536;
  }

  const originalImageType: ITKImageType = {
    dimension: 3,
    componentType: IntTypes.UInt16,
    pixelType: PixelTypes.Scalar,
    components: 1,
  };

  const originalItkImage: ITKImage = {
    imageType: originalImageType,
    name: "test_3d",
    origin: originalOrigin,
    spacing: originalSpacing,
    direction: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
    size: originalSize,
    metadata: new Map(),
    data: originalData,
  };

  // Convert ITK -> NGFF -> ITK
  const ngffImage = await itkImageToNgffImage(originalItkImage);
  const reconvertedItkImage = await ngffImageToItkImage(ngffImage);

  // Verify basic properties
  assertEquals(reconvertedItkImage.imageType.dimension, 3);
  assertEquals(reconvertedItkImage.imageType.componentType, IntTypes.UInt16);
  assertEquals(reconvertedItkImage.imageType.pixelType, PixelTypes.Scalar);

  // Check data integrity
  assert(reconvertedItkImage.data, "Reconverted image data should not be null");
  assertEquals(reconvertedItkImage.data.length, dataLength);
});

Deno.test(
  "ngffImageToItkImage different data types via round-trip",
  async () => {
    // Test different data types using round-trip conversion
    type ComponentType =
      | (typeof IntTypes)[keyof typeof IntTypes]
      | (typeof FloatTypes)[keyof typeof FloatTypes];
    type ArrayConstructor =
      | Uint8ArrayConstructor
      | Int16ArrayConstructor
      | Float32ArrayConstructor
      | Float64ArrayConstructor;

    const testCases: Array<[ComponentType, ArrayConstructor]> = [
      [IntTypes.UInt8, Uint8Array],
      [IntTypes.Int16, Int16Array],
      [FloatTypes.Float32, Float32Array],
      [FloatTypes.Float64, Float64Array],
    ];

    for (const [componentType, arrayType] of testCases) {
      const originalSize = [16, 16];
      const dataLength = originalSize.reduce((a, b) => a * b, 1);
      const originalData = new arrayType(dataLength);

      for (let i = 0; i < dataLength; i++) {
        originalData[i] = i + 1;
      }

      const originalImageType: ITKImageType = {
        dimension: 2,
        componentType,
        pixelType: PixelTypes.Scalar,
        components: 1,
      };

      const originalItkImage: ITKImage = {
        imageType: originalImageType,
        name: "data_type_test",
        origin: [0, 0],
        spacing: [1, 1],
        direction: new Float64Array([1, 0, 0, 1]),
        size: originalSize,
        metadata: new Map(),
        data: originalData,
      };

      // Round-trip conversion
      const ngffImage = await itkImageToNgffImage(originalItkImage);
      const reconvertedItkImage = await ngffImageToItkImage(ngffImage);

      assertEquals(
        reconvertedItkImage.imageType.componentType,
        componentType,
        `Failed for component type: ${componentType}`,
      );
      assert(
        reconvertedItkImage.data,
        `Data should not be null for ${componentType}`,
      );
      assert(
        reconvertedItkImage.data instanceof arrayType,
        `Expected ${arrayType.name} for ${componentType}, got ${reconvertedItkImage.data.constructor.name}`,
      );
    }
  },
);

Deno.test("ngffImageToItkImage round-trip conversion", async () => {
  // Create an ITK image
  const originalSize = [64, 32, 16]; // x, y, z
  const originalSpacing = [2.5, 1.5, 0.8];
  const originalOrigin = [10.0, 20.0, 30.0];
  const dataLength = originalSize.reduce((a, b) => a * b, 1);
  const originalData = new Float32Array(dataLength);

  for (let i = 0; i < dataLength; i++) {
    originalData[i] = Math.sin(i * 0.1) * 1000; // Some pattern for testing
  }

  const originalImageType: ITKImageType = {
    dimension: 3,
    componentType: FloatTypes.Float32,
    pixelType: PixelTypes.Scalar,
    components: 1,
  };

  const originalItkImage: ITKImage = {
    imageType: originalImageType,
    name: "round_trip_test",
    origin: originalOrigin,
    spacing: originalSpacing,
    direction: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), // Identity matrix
    size: originalSize,
    metadata: new Map(),
    data: originalData,
  };

  // Round trip: ITK -> NGFF -> ITK
  const ngffImage = await itkImageToNgffImage(originalItkImage);
  const reconvertedItkImage = await ngffImageToItkImage(ngffImage);

  // Verify properties are preserved
  assertEquals(
    reconvertedItkImage.imageType.dimension,
    originalImageType.dimension,
  );
  assertEquals(
    reconvertedItkImage.imageType.componentType,
    originalImageType.componentType,
  );
  assertEquals(
    reconvertedItkImage.imageType.pixelType,
    originalImageType.pixelType,
  );
  assertEquals(
    reconvertedItkImage.imageType.components,
    originalImageType.components,
  );

  // Note: ITK->NGFF->ITK may reorder dimensions - check that spatial info is preserved
  // The actual order depends on how itkImageToNgffImage maps dimensions
  assertEquals(
    reconvertedItkImage.size.reduce((a, b) => a * b, 1),
    originalSize.reduce((a, b) => a * b, 1),
  );

  // Check spacing and origin arrays have same values (may be reordered)
  const originalSpacingSet = new Set(originalSpacing);
  const reconvertedSpacingSet = new Set(reconvertedItkImage.spacing);
  assertEquals(originalSpacingSet, reconvertedSpacingSet);

  const originalOriginSet = new Set(originalOrigin);
  const reconvertedOriginSet = new Set(reconvertedItkImage.origin);
  assertEquals(originalOriginSet, reconvertedOriginSet);

  assertEquals(reconvertedItkImage.name, "image"); // itkImageToNgffImage hardcodes this name

  // Verify data integrity
  assert(reconvertedItkImage.data, "Reconverted image data should not be null");
  assertEquals(reconvertedItkImage.data.length, originalData.length);

  // Check data values (allow small floating point differences)
  for (let i = 0; i < Math.min(100, originalData.length); i++) {
    // Check first 100 values
    const reconvertedValue = reconvertedItkImage.data[i] as number;
    const originalValue = originalData[i];
    const diff = Math.abs(reconvertedValue - originalValue);
    assert(
      diff < 1e-6,
      `Data mismatch at index ${i}: ${reconvertedValue} != ${originalValue}`,
    );
  }
});

Deno.test("ngffImageToItkImage 2D round-trip conversion", async () => {
  // Test 2D round-trip with different data type
  const originalSize = [128, 64]; // x, y
  const originalSpacing = [0.5, 1.25];
  const originalOrigin = [-5.0, 15.0];
  const dataLength = originalSize.reduce((a, b) => a * b, 1);
  const originalData = new Uint16Array(dataLength);

  for (let i = 0; i < dataLength; i++) {
    originalData[i] = i % 65536; // Cycle through uint16 range
  }

  const originalImageType: ITKImageType = {
    dimension: 2,
    componentType: IntTypes.UInt16,
    pixelType: PixelTypes.Scalar,
    components: 1,
  };

  const originalItkImage: ITKImage = {
    imageType: originalImageType,
    name: "2d_round_trip",
    origin: originalOrigin,
    spacing: originalSpacing,
    direction: new Float64Array([1, 0, 0, 1]), // 2D identity matrix
    size: originalSize,
    metadata: new Map(),
    data: originalData,
  };

  // Round trip: ITK -> NGFF -> ITK
  const ngffImage = await itkImageToNgffImage(originalItkImage);
  const reconvertedItkImage = await ngffImageToItkImage(ngffImage);

  // Verify all properties
  assertEquals(reconvertedItkImage.imageType.dimension, 2);
  assertEquals(reconvertedItkImage.imageType.componentType, IntTypes.UInt16);

  // Check that total data size is preserved
  assertEquals(
    reconvertedItkImage.size.reduce((a, b) => a * b, 1),
    originalSize.reduce((a, b) => a * b, 1),
  );

  // Check spacing and origin values are preserved (may be reordered)
  const originalSpacingSet = new Set(originalSpacing);
  const reconvertedSpacingSet = new Set(reconvertedItkImage.spacing);
  assertEquals(originalSpacingSet, reconvertedSpacingSet);

  const originalOriginSet = new Set(originalOrigin);
  const reconvertedOriginSet = new Set(reconvertedItkImage.origin);
  assertEquals(originalOriginSet, reconvertedOriginSet);

  // Verify exact data match for integer types
  assert(reconvertedItkImage.data, "Reconverted image data should not be null");
  assertEquals(reconvertedItkImage.data.length, originalData.length);
  for (let i = 0; i < Math.min(200, originalData.length); i++) {
    assertEquals(
      reconvertedItkImage.data[i],
      originalData[i],
      `Data mismatch at index ${i}`,
    );
  }
});

Deno.test("ngffImageToItkImage metadata preservation", async () => {
  const originalSize = [32, 48];
  const dataLength = originalSize.reduce((a, b) => a * b, 1);
  const originalData = new Int32Array(dataLength);

  for (let i = 0; i < dataLength; i++) {
    originalData[i] = i + 42;
  }

  const originalImageType: ITKImageType = {
    dimension: 2,
    componentType: IntTypes.Int32,
    pixelType: PixelTypes.Scalar,
    components: 1,
  };

  const originalItkImage: ITKImage = {
    imageType: originalImageType,
    name: "metadata_test",
    origin: [5.0, 15.0],
    spacing: [2.5, 1.8],
    direction: new Float64Array([1, 0, 0, 1]),
    size: originalSize,
    metadata: new Map(),
    data: originalData,
  };

  // Convert ITK -> NGFF -> ITK
  const ngffImage = await itkImageToNgffImage(originalItkImage);
  const reconvertedItkImage = await ngffImageToItkImage(ngffImage);

  assertEquals(reconvertedItkImage.name, "image"); // itkImageToNgffImage sets this
  assertEquals(reconvertedItkImage.imageType.componentType, IntTypes.Int32);

  // Direction matrix should be identity
  const expectedDirection = new Float64Array([1, 0, 0, 1]);
  for (let i = 0; i < expectedDirection.length; i++) {
    assertEquals(reconvertedItkImage.direction[i], expectedDirection[i]);
  }
});

// --- Tests for direction matrix from anatomical orientation ---

/**
 * Helper to create a simple 3D NgffImage backed by a zarr array.
 */
async function createTestNgffImage(
  shape: number[],
  dims: string[],
  axesOrientations?: Record<
    string,
    { readonly type: "anatomical"; readonly value: AnatomicalOrientationValues }
  >,
): Promise<NgffImage> {
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);
  const chunkShape = shape.map((s) => Math.min(s, 256));

  const zarrArray = await zarr.create(root.resolve("test"), {
    shape,
    chunk_shape: chunkShape,
    data_type: "uint8",
    fill_value: 0,
    codecs: defaultCodecs("uint8"),
  });

  const totalSize = shape.reduce((a, b) => a * b, 1);
  const data = new Uint8Array(totalSize);
  const selection = new Array(shape.length).fill(null);
  const chunk = {
    data,
    shape,
    stride: getStrides(shape, "C"),
  };
  await zarrSet(zarrArray, selection, chunk);

  const scale: Record<string, number> = {};
  const translation: Record<string, number> = {};
  for (const d of dims) {
    scale[d] = 1.0;
    translation[d] = 0.0;
  }

  return new NgffImage({
    data: zarrArray,
    dims,
    scale,
    translation,
    name: "test",
    axesUnits: undefined,
    axesOrientations,
    computedCallbacks: undefined,
  });
}

Deno.test("direction matrix - without orientation → identity", async () => {
  const ngff = await createTestNgffImage([4, 8, 8], ["z", "y", "x"]);
  const itk = await ngffImageToItkImage(ngff);

  const expected = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  for (let i = 0; i < expected.length; i++) {
    assertEquals(itk.direction[i], expected[i]);
  }
});

Deno.test("direction matrix - LPS orientation → identity", async () => {
  const ngff = await createTestNgffImage(
    [4, 8, 8],
    ["z", "y", "x"],
    LPS,
  );
  const itk = await ngffImageToItkImage(ngff);

  const expected = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  for (let i = 0; i < expected.length; i++) {
    assertEquals(itk.direction[i], expected[i]);
  }
});

Deno.test("direction matrix - RAS orientation → flipped X/Y", async () => {
  const ngff = await createTestNgffImage(
    [4, 8, 8],
    ["z", "y", "x"],
    RAS,
  );
  const itk = await ngffImageToItkImage(ngff);

  // RAS: X=-1, Y=-1, Z=+1
  const expected = new Float64Array([-1, 0, 0, 0, -1, 0, 0, 0, 1]);
  for (let i = 0; i < expected.length; i++) {
    assertEquals(itk.direction[i], expected[i]);
  }
});

Deno.test("direction matrix - permuted orientations", async () => {
  const axesOrientations = {
    x: createAnatomicalOrientation(
      AnatomicalOrientationValues.AnteriorToPosterior,
    ),
    y: createAnatomicalOrientation(
      AnatomicalOrientationValues.InferiorToSuperior,
    ),
    z: createAnatomicalOrientation(
      AnatomicalOrientationValues.RightToLeft,
    ),
  };

  const ngff = await createTestNgffImage(
    [4, 8, 8],
    ["z", "y", "x"],
    axesOrientations,
  );
  const itk = await ngffImageToItkImage(ngff);

  // itk_dims sorted = [x, y, z]
  // col 0 (x) -> A/P -> [0, 1, 0]
  // col 1 (y) -> I/S -> [0, 0, 1]
  // col 2 (z) -> R/L -> [1, 0, 0]
  const expected = new Float64Array([0, 0, 1, 1, 0, 0, 0, 1, 0]);
  for (let i = 0; i < expected.length; i++) {
    assertEquals(itk.direction[i], expected[i]);
  }
});

Deno.test("direction matrix - non-LPS fallback → identity", async () => {
  const axesOrientations = {
    x: createAnatomicalOrientation(
      AnatomicalOrientationValues.RightToLeft,
    ),
    y: createAnatomicalOrientation(
      AnatomicalOrientationValues.DorsalToVentral,
    ),
    z: createAnatomicalOrientation(
      AnatomicalOrientationValues.InferiorToSuperior,
    ),
  };

  const ngff = await createTestNgffImage(
    [4, 8, 8],
    ["z", "y", "x"],
    axesOrientations,
  );
  const itk = await ngffImageToItkImage(ngff);

  const expected = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  for (let i = 0; i < expected.length; i++) {
    assertEquals(itk.direction[i], expected[i]);
  }
});

Deno.test("direction matrix - 2D RAS orientation", async () => {
  const axesOrientations = {
    x: RAS.x,
    y: RAS.y,
  };

  const ngff = await createTestNgffImage(
    [8, 8],
    ["y", "x"],
    axesOrientations,
  );
  const itk = await ngffImageToItkImage(ngff);

  const expected = new Float64Array([-1, 0, 0, -1]);
  for (let i = 0; i < expected.length; i++) {
    assertEquals(itk.direction[i], expected[i]);
  }
});

Deno.test("direction round-trip - identity preserved", async () => {
  const originalDirection = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const originalImage: ITKImage = {
    imageType: {
      dimension: 3,
      componentType: "uint8" as typeof IntTypes.UInt8,
      pixelType: "Scalar" as typeof PixelTypes.Scalar,
      components: 1,
    },
    name: "test",
    origin: [0, 0, 0],
    spacing: [1, 1, 1],
    direction: originalDirection,
    size: [4, 8, 8],
    metadata: new Map(),
    data: new Uint8Array(4 * 8 * 8),
  };

  const ngff = await itkImageToNgffImage(originalImage);
  const itkBack = await ngffImageToItkImage(ngff);

  for (let i = 0; i < originalDirection.length; i++) {
    assertEquals(itkBack.direction[i], originalDirection[i]);
  }
});

Deno.test("direction round-trip - flipped Y preserved", async () => {
  const originalDirection = new Float64Array([1, 0, 0, 0, -1, 0, 0, 0, 1]);
  const originalImage: ITKImage = {
    imageType: {
      dimension: 3,
      componentType: "uint8" as typeof IntTypes.UInt8,
      pixelType: "Scalar" as typeof PixelTypes.Scalar,
      components: 1,
    },
    name: "test",
    origin: [0, 0, 0],
    spacing: [1, 1, 1],
    direction: originalDirection,
    size: [4, 8, 8],
    metadata: new Map(),
    data: new Uint8Array(4 * 8 * 8),
  };

  const ngff = await itkImageToNgffImage(originalImage);
  const itkBack = await ngffImageToItkImage(ngff);

  for (let i = 0; i < originalDirection.length; i++) {
    assertEquals(itkBack.direction[i], originalDirection[i]);
  }
});
