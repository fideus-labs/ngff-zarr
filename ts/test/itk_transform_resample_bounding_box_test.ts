// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * RFC-5 to ITK transform bridge and resample bounding box tests.
 *
 * Mirrors `py/test/test_itk_transform_resample_bounding_box.py`. The expected
 * regions are not taken from the pipeline itself: they come either from the
 * worked examples in the ITK-Wasm `resample-bounding-box` documentation, or
 * from `oracleRegion` below, which recomputes the region from first principles
 * in NGFF axis order. That keeps the tests honest about the two conventions
 * that differ between RFC-5 and ITK: axis order and sequence composition
 * order.
 */

import { assertAlmostEquals, assertEquals, assertRejects } from "@std/assert";
import * as zarr from "zarrita";
import { itkTransformResampleBoundingBox, NgffImage } from "../src/mod.ts";
import { resampleBoundingBoxShared } from "../src/io/itk_transform_resample_bounding_box-shared.ts";
import { RAS } from "../src/types/rfc4.ts";
import type { AnatomicalOrientation } from "../src/types/rfc4.ts";

/** An ITK-Wasm translation, in ITK (fastest-axis-first) order. */
// deno-lint-ignore no-explicit-any
function itkTranslation(offset: number[]): any {
  return [{
    transformType: {
      transformParameterization: "Translation",
      parametersValueType: "float64",
      inputDimension: offset.length,
      outputDimension: offset.length,
    },
    name: "TranslationTransform",
    inputSpaceName: "",
    outputSpaceName: "",
    numberOfFixedParameters: 0,
    numberOfParameters: offset.length,
    fixedParameters: new Float64Array(0),
    parameters: new Float64Array(offset),
    metadata: new Map(),
  }];
}

// deno-lint-ignore no-explicit-any
const identity = (dimension: number): any =>
  itkTranslation(new Array(dimension).fill(0));

/** An ITK-Wasm affine, row-major matrix then translation, centre at the origin. */
// deno-lint-ignore no-explicit-any
function itkAffine(matrix: number[][], offset: number[]): any {
  const dimension = offset.length;
  return [{
    transformType: {
      transformParameterization: "Affine",
      parametersValueType: "float64",
      inputDimension: dimension,
      outputDimension: dimension,
    },
    name: "AffineTransform",
    inputSpaceName: "",
    outputSpaceName: "",
    numberOfFixedParameters: dimension,
    numberOfParameters: dimension * dimension + dimension,
    fixedParameters: new Float64Array(dimension),
    parameters: new Float64Array([...matrix.flat(), ...offset]),
    metadata: new Map(),
  }];
}

/** Reverse a square matrix's rows and columns: NGFF order <-> ITK order. */
function reversed(matrix: number[][]): number[][] {
  const order = matrix.map((_, i) => matrix.length - 1 - i);
  return order.map((r) => order.map((c) => matrix[r][c]));
}

/** A geometry-only NgffImage; the data is never meant to be read. */
async function geometryImage(
  dims: string[],
  shape: Record<string, number>,
  scale: Record<string, number>,
  translation: Record<string, number>,
  axesOrientations?: Record<string, AnatomicalOrientation>,
): Promise<NgffImage> {
  const store = new Map();
  const root = zarr.root(store);
  const arrayShape = dims.map((dim) => shape[dim]);
  const data = await zarr.create(root.resolve("data"), {
    shape: arrayShape,
    chunk_shape: arrayShape.map((n) => Math.min(n, 32)),
    data_type: "uint8",
    fill_value: 0,
  });
  return new NgffImage({
    data,
    dims,
    scale,
    translation,
    name: "image",
    axesUnits: undefined,
    axesOrientations,
    computedCallbacks: undefined,
  });
}

/** Recompute the region in NGFF order, independently of the pipeline. */
function oracleRegion(
  matrix: number[][],
  offset: number[],
  fixedShape: number[],
  fixedScale: number[],
  fixedTranslation: number[],
  movingScale: number[],
  movingTranslation: number[],
  padding: number,
): { start: number[]; size: number[] } {
  const ndim = fixedShape.length;
  const indexMin = new Array(ndim).fill(Infinity);
  const indexMax = new Array(ndim).fill(-Infinity);

  // A linear map sends the fixed rectangle to a convex region, so sampling
  // the corners is exact.
  for (let mask = 0; mask < 1 << ndim; mask++) {
    const corner = Array.from(
      { length: ndim },
      (_, i) => (mask >> i) & 1 ? fixedShape[i] - 1 : 0,
    );
    const point = corner.map((c, i) => fixedTranslation[i] + fixedScale[i] * c);
    for (let row = 0; row < ndim; row++) {
      let moved = offset[row];
      for (let col = 0; col < ndim; col++) {
        moved += matrix[row][col] * point[col];
      }
      const continuousIndex = (moved - movingTranslation[row]) /
        movingScale[row];
      indexMin[row] = Math.min(indexMin[row], continuousIndex);
      indexMax[row] = Math.max(indexMax[row], continuousIndex);
    }
  }

  const start = indexMin.map((v) => Math.floor(v) - padding);
  const end = indexMax.map((v) => Math.ceil(v) + padding);
  return { start, size: end.map((e, i) => Math.max(e - start[i] + 1, 0)) };
}

Deno.test("mismatched spatial dims are rejected", async () => {
  const fixed = await geometryImage(["z", "y", "x"], { z: 4, y: 4, x: 4 }, {
    z: 1,
    y: 1,
    x: 1,
  }, { z: 0, y: 0, x: 0 });
  const moving = await geometryImage(["y", "x"], { y: 4, x: 4 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });

  await assertRejects(
    () => itkTransformResampleBoundingBox(identity(2), fixed, moving),
    Error,
    "they must match",
  );
});

Deno.test("pixel data is never read", async () => {
  // A store that throws on any chunk read: if the pipeline touched pixels,
  // this would reject rather than return a region.
  const backing = new Map<string, Uint8Array>();
  const poison = {
    get(key: string): Promise<Uint8Array | undefined> {
      if (key.endsWith("zarr.json")) {
        return Promise.resolve(backing.get(key));
      }
      throw new Error("pixel data was read");
    },
    set(key: string, value: Uint8Array): Promise<void> {
      backing.set(key, value);
      return Promise.resolve();
    },
    delete(key: string): Promise<boolean> {
      return Promise.resolve(backing.delete(key));
    },
  };
  const root = zarr.root(poison as never);
  const data = await zarr.create(root.resolve("data"), {
    shape: [32, 32],
    chunk_shape: [8, 8],
    data_type: "uint8",
    fill_value: 0,
  });
  const image = new NgffImage({
    data,
    dims: ["y", "x"],
    scale: { y: 1, x: 1 },
    translation: { y: 0, x: 0 },
    name: "image",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });

  const boundingBox = await itkTransformResampleBoundingBox(
    identity(2),
    image,
    image,
    { padding: 1 },
  );

  assertEquals(boundingBox.startIndex, { y: -1, x: -1 });
  assertEquals(boundingBox.size, { y: 34, x: 34 });
});

Deno.test("non-spatial axes are passed through", async () => {
  const dims = ["t", "c", "z", "y", "x"];
  const fixed = await geometryImage(dims, { t: 3, c: 2, z: 4, y: 8, x: 8 }, {
    t: 1,
    c: 1,
    z: 1,
    y: 1,
    x: 1,
  }, { t: 0, c: 0, z: 0, y: 0, x: 0 });
  const moving = await geometryImage(
    dims,
    { t: 3, c: 2, z: 16, y: 32, x: 32 },
    {
      t: 1,
      c: 1,
      z: 1,
      y: 1,
      x: 1,
    },
    { t: 0, c: 0, z: 0, y: 0, x: 0 },
  );

  const boundingBox = await itkTransformResampleBoundingBox(
    itkTranslation([3, 2, 1]),
    fixed,
    moving,
    { padding: 0 },
  );

  assertEquals(boundingBox.dims, ["z", "y", "x"]);
  assertEquals(boundingBox.startIndex, { z: 1, y: 2, x: 3 });

  const selection = boundingBox.selection(moving.dims);
  assertEquals(selection[0], null); // t
  assertEquals(selection[1], null); // c
  assertEquals(selection[2], zarr.slice(1, 5)); // z
  assertEquals(selection[3], zarr.slice(2, 10)); // y
  assertEquals(selection[4], zarr.slice(3, 11)); // x
});

Deno.test("a region outside the moving image is empty", async () => {
  const fixed = await geometryImage(["y", "x"], { y: 4, x: 4 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });
  const moving = await geometryImage(["y", "x"], { y: 32, x: 32 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });

  const boundingBox = await itkTransformResampleBoundingBox(
    itkTranslation([1000, 1000]),
    fixed,
    moving,
    { padding: 1 },
  );

  assertEquals(boundingBox.isEmpty, true);
});

Deno.test("a negative start index is clamped, not wrapped", async () => {
  const fixed = await geometryImage(["y", "x"], { y: 4, x: 4 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });
  const moving = await geometryImage(["y", "x"], { y: 32, x: 32 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });

  const boundingBox = await itkTransformResampleBoundingBox(
    identity(2),
    fixed,
    moving,
    { padding: 2 },
  );

  assertEquals(boundingBox.startIndex, { y: -2, x: -2 });
  assertEquals(boundingBox.clamped(), { y: [0, 6], x: [0, 6] });
  assertEquals(boundingBox.isEmpty, false);
});

Deno.test("the cropped translation shifts by start * scale", async () => {
  const fixed = await geometryImage(["y", "x"], { y: 64, x: 64 }, {
    y: 1,
    x: 1,
  }, { y: 512, x: 1024 });
  const moving = await geometryImage(["y", "x"], { y: 1024, x: 1024 }, {
    y: 2,
    x: 2,
  }, { y: 0, x: 0 });

  const boundingBox = await itkTransformResampleBoundingBox(
    itkTranslation([0, 0]),
    fixed,
    moving,
    { padding: 1 },
  );
  const translation = boundingBox.croppedTranslation(moving);
  const bounds = boundingBox.clamped();

  for (const dim of ["y", "x"]) {
    assertAlmostEquals(
      translation[dim],
      moving.translation[dim] + bounds[dim][0] * moving.scale[dim],
    );
  }
});

Deno.test("RAS orientation yields a non-identity direction", async () => {
  const { itkDirection } = await import(
    "../src/io/itk_transform_resample_bounding_box-shared.ts"
  );
  const image = await geometryImage(
    ["z", "y", "x"],
    { z: 4, y: 8, x: 16 },
    { z: 1, y: 1, x: 1 },
    { z: 0, y: 0, x: 0 },
    RAS,
  );
  const direction = itkDirection(image, ["x", "y", "z"]);
  assertEquals(Array.from(direction), [-1, 0, 0, 0, -1, 0, 0, 0, 1]);
});

Deno.test("a 3D-only orientation on a 2D image falls back to identity", async () => {
  const { itkDirection } = await import(
    "../src/io/itk_transform_resample_bounding_box-shared.ts"
  );
  const { AnatomicalOrientationValues, createAnatomicalOrientation } =
    await import("../src/types/rfc4.ts");
  // An inferior/superior axis points along LPS z; truncating its column to
  // 2D would produce a singular matrix.
  const image = await geometryImage(
    ["y", "x"],
    { y: 8, x: 16 },
    { y: 1, x: 1 },
    { y: 0, x: 0 },
    {
      x: createAnatomicalOrientation(AnatomicalOrientationValues.LeftToRight),
      y: createAnatomicalOrientation(
        AnatomicalOrientationValues.InferiorToSuperior,
      ),
    },
  );
  const direction = itkDirection(image, ["x", "y"]);
  assertEquals(Array.from(direction), [1, 0, 0, 1]);
});

Deno.test("an index range overflow is reported, not silently empty", async () => {
  // The region is computed in 32-bit index space while the corners come back
  // as doubles, so a fixed/moving scale mismatch this large wraps the integer
  // region. Reporting "no overlap" for a grid covering the whole moving image
  // would be a silent, wrong answer.
  const fixed = await geometryImage(["y", "x"], { y: 16, x: 16 }, {
    y: 1e9,
    x: 1e9,
  }, { y: 0, x: 0 });
  const moving = await geometryImage(["y", "x"], { y: 64, x: 64 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });

  await assertRejects(
    () =>
      itkTransformResampleBoundingBox(identity(2), fixed, moving, {
        padding: 1,
      }),
    Error,
    "does not contain the transformed grid",
  );
});

Deno.test("padding that is not a non-negative integer is rejected", async () => {
  const fixed = await geometryImage(["y", "x"], { y: 4, x: 4 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });
  for (const padding of [-1, 1.5, NaN, Infinity]) {
    await assertRejects(
      () =>
        itkTransformResampleBoundingBox(identity(2), fixed, fixed, {
          padding,
        }),
      Error,
      "padding must be a non-negative integer",
    );
  }
});

Deno.test("unsupported spatial dimensionality is rejected", async () => {
  const fixed = await geometryImage(["x"], { x: 8 }, { x: 1 }, { x: 0 });
  await assertRejects(
    () => itkTransformResampleBoundingBox(identity(2), fixed, fixed),
    Error,
    "only 2 and 3 are supported",
  );
});

Deno.test("a missing scale entry is rejected rather than defaulted", async () => {
  // Python raises here; TypeScript used to substitute spacing 1 and return a
  // plausible but wrong region.
  const fixed = await geometryImage(["y", "x"], { y: 4, x: 4 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });
  const partial = new NgffImage({
    data: fixed.data,
    dims: ["y", "x"],
    scale: { y: 2 } as Record<string, number>,
    translation: { y: 0, x: 0 },
    name: "image",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });

  await assertRejects(
    () => itkTransformResampleBoundingBox(identity(2), partial, fixed),
    Error,
    "no entry for dimension 'x'",
  );
});

Deno.test("a non-finite scale is rejected", async () => {
  const fixed = await geometryImage(["y", "x"], { y: 4, x: 4 }, {
    y: 1,
    x: NaN,
  }, { y: 0, x: 0 });
  await assertRejects(
    () => itkTransformResampleBoundingBox(identity(2), fixed, fixed),
    Error,
    "must be finite",
  );
});

Deno.test("a zero scale is rejected", async () => {
  const fixed = await geometryImage(["y", "x"], { y: 4, x: 4 }, {
    y: 1,
    x: 0,
  }, { y: 0, x: 0 });
  await assertRejects(
    () => itkTransformResampleBoundingBox(identity(2), fixed, fixed),
    Error,
    "scale for dimension 'x' is zero",
  );
});

Deno.test("a degenerate fixed grid yields an empty region", async () => {
  const fixed = await geometryImage(["y", "x"], { y: 0, x: 4 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });
  const moving = await geometryImage(["y", "x"], { y: 32, x: 32 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });

  const boundingBox = await itkTransformResampleBoundingBox(
    identity(2),
    fixed,
    moving,
  );

  assertEquals(boundingBox.isEmpty, true);
});

Deno.test("an ITK-Wasm transform list is accepted", async () => {
  const fixed = await geometryImage(["y", "x"], { y: 16, x: 16 }, {
    y: 2,
    x: 2,
  }, { y: 20, x: 10 });
  const moving = await geometryImage(["y", "x"], { y: 64, x: 64 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });

  // Stated directly in ITK order (x, y).
  const transformList = [{
    transformType: {
      transformParameterization: "Translation",
      parametersValueType: "float64",
      inputDimension: 2,
      outputDimension: 2,
    },
    name: "TranslationTransform",
    inputSpaceName: "",
    outputSpaceName: "",
    numberOfFixedParameters: 0,
    numberOfParameters: 2,
    fixedParameters: new Float64Array(0),
    parameters: new Float64Array([10, 5]),
    metadata: new Map(),
  }];

  const boundingBox = await itkTransformResampleBoundingBox(
    transformList as never,
    fixed,
    moving,
    { padding: 1 },
  );

  assertEquals(boundingBox.startIndex, { y: 24, x: 19 });
  assertEquals(boundingBox.size, { y: 33, x: 33 });
});

Deno.test("an asymmetric 3D affine matches the oracle", () => {
  // Stated in NGFF (z, y, x) order for the oracle; the transform itself is
  // built in ITK order, so both row and column ordering reverse.
  const matrix = [[1, 0.2, 0], [0, 2, 0.3], [0.5, 0, 1]];
  const offset = [4, -6, 11];
  return (async () => {
    const dims = ["z", "y", "x"];
    const fixed = await geometryImage(dims, { z: 4, y: 8, x: 16 }, {
      z: 3,
      y: 2,
      x: 1,
    }, { z: 30, y: 20, x: 10 });
    const moving = await geometryImage(dims, { z: 64, y: 128, x: 256 }, {
      z: 1.5,
      y: 0.5,
      x: 0.25,
    }, { z: -5, y: 7, x: 3 });

    const boundingBox = await itkTransformResampleBoundingBox(
      itkAffine(reversed(matrix), [...offset].reverse()),
      fixed,
      moving,
      { padding: 2 },
    );

    const expected = oracleRegion(
      matrix,
      offset,
      [4, 8, 16],
      [3, 2, 1],
      [30, 20, 10],
      [1.5, 0.5, 0.25],
      [-5, 7, 3],
      2,
    );
    assertEquals(dims.map((d) => boundingBox.startIndex[d]), expected.start);
    assertEquals(dims.map((d) => boundingBox.size[d]), expected.size);
  })();
});

Deno.test("unusable pipeline index arrays are rejected", async () => {
  const fixed = await geometryImage(["y", "x"], { y: 4, x: 4 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });
  const corners = { min: [0, 0], max: [1, 1] };
  const cases: [number[], number[], string][] = [
    [[0], [4, 4], "1 paddedStartIndex values for 2 dimensions"],
    [[0, 0], [4], "1 paddedSize values for 2 dimensions"],
    [[0, NaN], [4, 4], "paddedStartIndex for dimension 'y'"],
    [[0, 0], [4, Infinity], "paddedSize for dimension 'y'"],
  ];

  for (const [paddedStartIndex, paddedSize, message] of cases) {
    const pipeline = () =>
      Promise.resolve({
        boundingBox: {
          paddedStartIndex,
          paddedSize,
          corners,
          paddedCorners: corners,
        },
      });
    await assertRejects(
      () => resampleBoundingBoxShared(pipeline, identity(2), fixed, fixed),
      Error,
      message,
    );
  }
});
