// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * RFC-5 to ITK transform bridge and resample bounding box tests.
 *
 * Mirrors `py/test/test_resample_bounding_box.py`. The expected
 * regions are not taken from the pipeline itself: they come either from the
 * worked examples in the ITK-Wasm `resample-bounding-box` documentation, or
 * from `oracleRegion` below, which recomputes the region from first principles
 * in NGFF axis order. That keeps the tests honest about the two conventions
 * that differ between RFC-5 and ITK: axis order and sequence composition
 * order.
 */

import {
  assertAlmostEquals,
  assertEquals,
  assertRejects,
  assertThrows,
} from "@std/assert";
import * as zarr from "zarrita";
import {
  createAffine,
  createBijection,
  createByDimension,
  createIdentity,
  createMapAxis,
  createRotation,
  createScale,
  createTransformSequence,
  createTranslation,
  itkDisplacementFieldToNgffTransform,
  itkTransformToNgffTransform,
  NgffImage,
  ngffTransformToItkTransform,
  resampleBoundingBox,
} from "../src/mod.ts";
import { ngffTransformToItkMatrix } from "../src/utils/ngff_transform_to_itk_transform.ts";
import { resampleBoundingBoxShared } from "../src/io/resample_bounding_box-shared.ts";
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

Deno.test("an NGFF translation reproduces the documented 2D example", async () => {
  // The documentation states this in ITK order: fixed 16x16 spacing (2,2)
  // origin (10,20), moving 64x64 spacing (1,1) origin 0, translation (10,5),
  // padding 1. Written as RFC-5 the axes reverse, so the translation is
  // [5, 10] over ["y", "x"].
  const fixed = await geometryImage(["y", "x"], { y: 16, x: 16 }, {
    y: 2,
    x: 2,
  }, { y: 20, x: 10 });
  const moving = await geometryImage(["y", "x"], { y: 64, x: 64 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });

  const boundingBox = await resampleBoundingBox(
    createTranslation([5, 10]),
    fixed,
    moving,
    { padding: 1 },
  );

  assertEquals(boundingBox.startIndex, { y: 24, x: 19 });
  assertEquals(boundingBox.size, { y: 33, x: 33 });
  assertEquals(boundingBox.cornersMin, { y: 25, x: 20 });
  assertEquals(boundingBox.cornersMax, { y: 55, x: 50 });
  assertEquals(boundingBox.paddedCornersMin, { y: 24, x: 19 });
  assertEquals(boundingBox.paddedCornersMax, { y: 56, x: 51 });
});

Deno.test("padding is applied symmetrically", async () => {
  const fixed = await geometryImage(["y", "x"], { y: 16, x: 16 }, {
    y: 2,
    x: 2,
  }, { y: 20, x: 10 });
  const moving = await geometryImage(["y", "x"], { y: 64, x: 64 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });
  const transform = createTranslation([5, 10]);

  const padded = await resampleBoundingBox(
    transform,
    fixed,
    moving,
    {
      padding: 1,
    },
  );
  const tight = await resampleBoundingBox(
    transform,
    fixed,
    moving,
    {
      padding: 0,
    },
  );

  for (const dim of ["y", "x"]) {
    assertEquals(tight.startIndex[dim], padded.startIndex[dim] + 1);
    assertEquals(tight.size[dim], padded.size[dim] - 2);
  }
  // The tight corners are padding independent.
  assertEquals(tight.cornersMin, padded.cornersMin);
  assertEquals(tight.cornersMax, padded.cornersMax);
});

Deno.test("an asymmetric 3D NGFF affine matches the oracle", async () => {
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

  // Deliberately asymmetric, with shear, so any axis-order slip shows.
  const matrix = [[1, 0.2, 0], [0, 2, 0.3], [0.5, 0, 1]];
  const offset = [4, -6, 11];
  const affine = matrix.map((row, i) => [...row, offset[i]]);

  const boundingBox = await resampleBoundingBox(
    createAffine(affine),
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
});

Deno.test("affine translation is the last column", () => {
  const { matrix, offset } = ngffTransformToItkMatrix(
    createAffine([[1, 2, 3], [4, 5, 6]]),
    ["y", "x"],
  );
  // In NGFF terms: y = 1*y + 2*x + 3 and x = 4*y + 5*x + 6. ITK reverses the
  // axes, so both the matrix and the offset come back reversed.
  assertEquals(offset, [6, 3]);
  assertEquals(matrix, [[5, 4], [2, 1]]);
});

Deno.test("an affine of the wrong shape is rejected", () => {
  let message = "";
  try {
    ngffTransformToItkMatrix(createAffine([[1, 0], [0, 1]]), ["y", "x"]);
  } catch (error) {
    message = (error as Error).message;
  }
  assertEquals(message.includes("translation is the last column"), true);
});

Deno.test("a sequence applies its first entry first", () => {
  // RFC-5 composes [f0, f1] as f1(f0(x)). ITK transform lists compose the
  // other way round, so a sequence passed through unreversed would silently
  // give 2x + 10 here instead of 2(x + 10).
  const sequence = createTransformSequence([
    createTranslation([0, 10]),
    createScale([1, 2]),
  ]);
  const { matrix, offset } = ngffTransformToItkMatrix(sequence, ["y", "x"]);

  // ITK order is (x, y): translate by 10 then scale by 2 gives an offset of 20.
  assertAlmostEquals(offset[0], 20);
  assertAlmostEquals(matrix[0][0], 2);

  const reversed = createTransformSequence([
    createScale([1, 2]),
    createTranslation([0, 10]),
  ]);
  assertAlmostEquals(
    ngffTransformToItkMatrix(reversed, ["y", "x"]).offset[0],
    10,
  );
});

Deno.test("sequence order survives the whole pipeline", async () => {
  // Every other sequence test here reads the matrix out of
  // ngffTransformToItkMatrix. If the order inverted, the region the caller
  // actually gets would move, and nothing below that helper would notice.
  // Applying [translate 10, scale 2] the wrong way round gives 2x + 10
  // instead of 2(x + 10), so the x start moves by 10.
  const fixed = await geometryImage(["y", "x"], { y: 16, x: 16 }, {
    y: 2,
    x: 2,
  }, { y: 20, x: 10 });
  const moving = await geometryImage(["y", "x"], { y: 512, x: 512 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });

  const region = await resampleBoundingBox(
    createTransformSequence([
      createTranslation([0, 10]),
      createScale([1, 2]),
    ]),
    fixed,
    moving,
    { padding: 0 },
  );

  // First entry first: y = x, x = 2(x + 10) = 2x + 20.
  const { start, size } = oracleRegion(
    [[1, 0], [0, 2]],
    [0, 20],
    [16, 16],
    [2, 2],
    [20, 10],
    [1, 1],
    [0, 0],
    0,
  );
  assertEquals([region.startIndex.y, region.startIndex.x], start);
  assertEquals([region.size.y, region.size.x], size);
  assertEquals(region.startIndex.x, 40); // 30 if the order inverted
});

Deno.test("nested sequences compose", () => {
  const inner = createTransformSequence([
    createScale([1, 2]),
    createTranslation([0, 1]),
  ]);
  const outer = createTransformSequence([inner, createTranslation([0, 100])]);
  assertAlmostEquals(
    ngffTransformToItkMatrix(outer, ["y", "x"]).offset[0],
    101,
  );
});

Deno.test("identity, scale and rotation convert", () => {
  const identity = ngffTransformToItkMatrix(createIdentity(), ["y", "x"]);
  assertEquals(identity.matrix, [[1, 0], [0, 1]]);
  assertEquals(identity.offset, [0, 0]);

  // Reversed to ITK order (x, y).
  const scale = ngffTransformToItkMatrix(createScale([2, 3]), ["y", "x"]);
  assertEquals(scale.matrix, [[3, 0], [0, 2]]);

  const rotation = ngffTransformToItkMatrix(
    createRotation([[0, -1], [1, 0]]),
    ["y", "x"],
  );
  // Reversing both rows and columns of [[0,-1],[1,0]] gives [[0,1],[-1,0]].
  assertEquals(rotation.matrix, [[0, 1], [-1, 0]]);
});

Deno.test("a non-linear transformation is rejected", () => {
  let message = "";
  try {
    ngffTransformToItkMatrix(
      { type: "displacements", path: "field" },
      ["y", "x"],
    );
  } catch (error) {
    message = (error as Error).message;
  }
  assertEquals(message.includes("cannot be converted"), true);
});

Deno.test("a transform coupling spatial and non-spatial axes is rejected", () => {
  // y would depend on c.
  const affine = [[1, 0, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0]];
  let message = "";
  try {
    ngffTransformToItkMatrix(createAffine(affine), ["c", "y", "x"]);
  } catch (error) {
    message = (error as Error).message;
  }
  assertEquals(message.includes("couples spatial and non-spatial"), true);
});

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
    () => resampleBoundingBox(identity(2), fixed, moving),
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

  const boundingBox = await resampleBoundingBox(
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

  const boundingBox = await resampleBoundingBox(
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

  const boundingBox = await resampleBoundingBox(
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

  const boundingBox = await resampleBoundingBox(
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

  const boundingBox = await resampleBoundingBox(
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
    "../src/io/resample_bounding_box-shared.ts"
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
    "../src/io/resample_bounding_box-shared.ts"
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
      resampleBoundingBox(identity(2), fixed, moving, {
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
        resampleBoundingBox(identity(2), fixed, fixed, {
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
    () => resampleBoundingBox(identity(2), fixed, fixed),
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
    () => resampleBoundingBox(identity(2), partial, fixed),
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
    () => resampleBoundingBox(identity(2), fixed, fixed),
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
    () => resampleBoundingBox(identity(2), fixed, fixed),
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

  const boundingBox = await resampleBoundingBox(
    identity(2),
    fixed,
    moving,
  );

  assertEquals(boundingBox.isEmpty, true);
});

Deno.test("the RFC-5 branch ignores anatomical orientation", async () => {
  // An RFC-5 transformation acts on the intrinsic coordinate system, where
  // a point is translation + scale * index and no direction matrix applies,
  // so the region it selects cannot depend on RFC-4 anatomical orientation:
  // the oriented region must equal the unoriented one.
  const ras: Record<string, AnatomicalOrientation> = RAS;
  const geometry = async (oriented: boolean) => ({
    fixed: await geometryImage(
      ["z", "y", "x"],
      { z: 8, y: 16, x: 24 },
      { z: 1, y: 1, x: 1 },
      { z: 0, y: 0, x: 0 },
      oriented ? ras : undefined,
    ),
    moving: await geometryImage(
      ["z", "y", "x"],
      { z: 32, y: 64, x: 96 },
      { z: 1, y: 1, x: 1 },
      { z: 0, y: 0, x: 0 },
      oriented ? ras : undefined,
    ),
  });
  const transform = createTranslation([2, 3, 4]);

  const plainImages = await geometry(false);
  const orientedImages = await geometry(true);
  const plain = await resampleBoundingBox(
    transform,
    plainImages.fixed,
    plainImages.moving,
    { padding: 0 },
  );
  const oriented = await resampleBoundingBox(
    transform,
    orientedImages.fixed,
    orientedImages.moving,
    { padding: 0 },
  );

  assertEquals(plain.startIndex, { z: 2, y: 3, x: 4 });
  assertEquals(oriented.startIndex, plain.startIndex);
  assertEquals(oriented.size, plain.size);
});

Deno.test("an RFC-5 displacements transform reaches the same region", async () => {
  // The RFC-5 branch has to reach the same region for a field as for an
  // affine, which means the field it points at has to reach the pipeline.
  const size = [8, 8];
  const spacing = [8.0, 8.0];
  const shift = [5.0, -3.0];
  const parameters = new Float64Array(size[0] * size[1] * 2);
  for (let i = 0; i < parameters.length; i += 2) {
    parameters[i] = shift[0];
    parameters[i + 1] = shift[1];
  }
  const warp = {
    transformType: {
      transformParameterization: "DisplacementField",
      parametersValueType: "float64",
      inputDimension: 2,
      outputDimension: 2,
    },
    name: "DisplacementFieldTransform",
    inputSpaceName: "",
    outputSpaceName: "",
    numberOfFixedParameters: 10,
    numberOfParameters: parameters.length,
    fixedParameters: new Float64Array([
      ...size,
      0,
      0,
      ...spacing,
      1,
      0,
      0,
      1,
    ]),
    parameters,
    metadata: new Map(),
    // deno-lint-ignore no-explicit-any
  } as any;

  const fixed = await geometryImage(
    ["y", "x"],
    { y: 8, x: 8 },
    { y: 8, x: 8 },
    {
      y: 0,
      x: 0,
    },
  );
  const moving = await geometryImage(["y", "x"], { y: 64, x: 64 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });

  const viaItk = await resampleBoundingBox([warp], fixed, moving);
  const { transform, field } = await itkDisplacementFieldToNgffTransform(
    warp,
    ["y", "x"],
    { path: "warp" },
  );
  const viaRfc5 = await resampleBoundingBox(
    transform,
    fixed,
    moving,
    { fields: { warp: field } },
  );

  assertEquals(viaRfc5.startIndex, viaItk.startIndex);
  assertEquals(viaRfc5.size, viaItk.size);

  // Without the field there is nothing to convert, and the message says so.
  await assertRejects(
    () => resampleBoundingBox(transform, fixed, moving),
    Error,
    "no field was passed",
  );
});

Deno.test("converting with frames matches the ITK path on oriented images", async () => {
  // The acceptance test for the change of frame: an ITK transform acts on
  // physical space (direction matrix included), the RFC-5 branch on the
  // intrinsic systems. Converted with { fixed, moving }, the two paths must
  // select the same region of the same RAS-oriented images. Fractional
  // translations keep every corner off an integer, away from floor/ceil
  // knife edges.
  const ras: Record<string, AnatomicalOrientation> = RAS;
  const fixed = await geometryImage(
    ["z", "y", "x"],
    { z: 8, y: 8, x: 8 },
    { z: 1, y: 2, x: 3 },
    { z: 1.3, y: -2.7, x: 5.1 },
    ras,
  );
  const moving = await geometryImage(
    ["z", "y", "x"],
    { z: 256, y: 256, x: 256 },
    { z: 1, y: 1, x: 1 },
    { z: -4.2, y: 7.6, x: 3.4 },
    ras,
  );
  // A sheared affine with translation, in ITK order: every frame error shows.
  const transform = [{
    transformType: {
      transformParameterization: "Affine",
      parametersValueType: "float64",
      inputDimension: 3,
      outputDimension: 3,
    },
    name: "AffineTransform",
    inputSpaceName: "",
    outputSpaceName: "",
    numberOfFixedParameters: 3,
    numberOfParameters: 12,
    fixedParameters: new Float64Array(3),
    parameters: new Float64Array([
      0.9,
      0.1,
      0,
      -0.1,
      1.1,
      0,
      0,
      0,
      1,
      2,
      -3,
      4,
    ]),
    metadata: new Map(),
    // deno-lint-ignore no-explicit-any
  } as any];

  const viaItk = await resampleBoundingBox(
    transform,
    fixed,
    moving,
    {
      padding: 0,
    },
  );
  const converted = itkTransformToNgffTransform(
    transform,
    ["z", "y", "x"],
    true,
    {
      fixed,
      moving,
    },
  );
  const viaRfc5 = await resampleBoundingBox(
    converted,
    fixed,
    moving,
    { padding: 0 },
  );

  assertEquals(viaRfc5.startIndex, viaItk.startIndex);
  assertEquals(viaRfc5.size, viaItk.size);

  // Passing only one image is refused.
  assertThrows(
    () =>
      itkTransformToNgffTransform(transform, ["z", "y", "x"], true, { fixed }),
    Error,
    "both fixed and moving",
  );

  // Passing an unoriented pair is a no-op: every direction is the identity
  // and the change of frame comes back with the mapping it was given.
  const plain = await geometryImage(
    ["z", "y", "x"],
    { z: 8, y: 8, x: 8 },
    { z: 1, y: 2, x: 3 },
    { z: 1.3, y: -2.7, x: 5.1 },
  );
  assertEquals(
    itkTransformToNgffTransform(transform, ["z", "y", "x"], false, {
      fixed: plain,
      moving: plain,
    }),
    itkTransformToNgffTransform(transform, ["z", "y", "x"], false),
  );
});

Deno.test("frames round trip through both converters", async () => {
  // RFC-5 -> ITK -> RFC-5 with frames must return the identical mapping.
  // On a 3-cycle orientation (z, y, x onto LPS x, z, y): a diagonal
  // direction such as RAS is its own inverse, so it cannot tell the reverse
  // conversion's D^-1 from D.
  const { AnatomicalOrientationValues, createAnatomicalOrientation } =
    await import("../src/types/rfc4.ts");
  const cycle = {
    z: createAnatomicalOrientation(AnatomicalOrientationValues.LeftToRight),
    y: createAnatomicalOrientation(
      AnatomicalOrientationValues.InferiorToSuperior,
    ),
    x: createAnatomicalOrientation(
      AnatomicalOrientationValues.PosteriorToAnterior,
    ),
  };
  const fixed = await geometryImage(
    ["z", "y", "x"],
    { z: 8, y: 8, x: 8 },
    { z: 1, y: 2, x: 3 },
    { z: 1.3, y: -2.7, x: 5.1 },
    cycle,
  );
  const moving = await geometryImage(
    ["z", "y", "x"],
    { z: 16, y: 16, x: 16 },
    { z: 1, y: 1, x: 1 },
    { z: -4.2, y: 7.6, x: 3.4 },
    cycle,
  );
  const original = createAffine([
    [1, 0.2, 0, 4.1],
    [0, 2, 0.3, -6.2],
    [0.5, 0, 1, 11.3],
  ]);

  const itkList = ngffTransformToItkTransform(original, ["z", "y", "x"], {
    fixed,
    moving,
  });
  const back = itkTransformToNgffTransform(itkList, ["z", "y", "x"], false, {
    fixed,
    moving,
  });

  assertEquals(back.type, "affine");
  const rows = (back as { affine: number[][] }).affine;
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 4; col++) {
      assertAlmostEquals(rows[row][col], original.affine![row][col], 1e-9);
    }
  }
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

  const boundingBox = await resampleBoundingBox(
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

    const boundingBox = await resampleBoundingBox(
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

Deno.test("a mapAxis region matches the oracle", async () => {
  const dims = ["z", "y", "x"];
  const fixed = await geometryImage(dims, { z: 12, y: 20, x: 16 }, {
    z: 2,
    y: 1,
    x: 0.5,
  }, { z: 3, y: -4, x: 6 });
  const moving = await geometryImage(dims, { z: 64, y: 64, x: 64 }, {
    z: 1,
    y: 1,
    x: 1,
  }, { z: 0, y: 0, x: 0 });

  const region = await resampleBoundingBox(
    createMapAxis([1, 2, 0]),
    fixed,
    moving,
    { padding: 1 },
  );

  // Output axis i takes input axis mapAxis[i], so row i selects that column.
  const { start, size } = oracleRegion(
    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    [0, 0, 0],
    [12, 20, 16],
    [2, 1, 0.5],
    [3, -4, 6],
    [1, 1, 1],
    [0, 0, 0],
    1,
  );
  assertEquals(dims.map((dim) => region.startIndex[dim]), start);
  assertEquals(dims.map((dim) => region.size[dim]), size);
});

Deno.test("a byDimension region matches the oracle", async () => {
  const dims = ["z", "y", "x"];
  const fixed = await geometryImage(dims, { z: 10, y: 24, x: 24 }, {
    z: 1,
    y: 0.5,
    x: 0.5,
  }, { z: -2, y: 1, x: 4 });
  const moving = await geometryImage(dims, { z: 64, y: 128, x: 128 }, {
    z: 1,
    y: 0.25,
    x: 0.25,
  }, { z: 0, y: 0, x: 0 });

  const region = await resampleBoundingBox(
    createByDimension([
      {
        transformation: createTranslation([5]),
        inputAxes: [0],
        outputAxes: [0],
      },
      {
        transformation: createAffine([[0.8, -0.6, 2], [0.6, 0.8, -3]]),
        inputAxes: [1, 2],
        outputAxes: [1, 2],
      },
    ]),
    fixed,
    moving,
    { padding: 2 },
  );

  const { start, size } = oracleRegion(
    [[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]],
    [5, 2, -3],
    [10, 24, 24],
    [1, 0.5, 0.5],
    [-2, 1, 4],
    [1, 0.25, 0.25],
    [0, 0, 0],
    2,
  );
  assertEquals(dims.map((dim) => region.startIndex[dim]), start);
  assertEquals(dims.map((dim) => region.size[dim]), size);
});

Deno.test("a bijection region follows its forward direction", async () => {
  const fixed = await geometryImage(["y", "x"], { y: 16, x: 16 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });
  const moving = await geometryImage(["y", "x"], { y: 128, x: 128 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });
  const forward = createScale([2, 4]);

  const region = await resampleBoundingBox(
    createBijection(forward, createScale([0.5, 0.25])),
    fixed,
    moving,
    { padding: 1 },
  );
  const directly = await resampleBoundingBox(forward, fixed, moving, {
    padding: 1,
  });

  assertEquals(region.startIndex, directly.startIndex);
  assertEquals(region.size, directly.size);
});

/** A field image over `dims`, its component axis first, from raw values. */
async function bumpField(
  values: Float32Array,
  dims: string[],
  extent: number,
  chunk: number,
): Promise<NgffImage> {
  const shape = [dims.length, ...dims.map(() => extent)];
  const voxels = shape.slice(1).reduce((a, b) => a * b, 1);
  const strides = new Array<number>(dims.length);
  let step = 1;
  for (let axis = dims.length - 1; axis >= 0; axis--) {
    strides[axis] = step;
    step *= extent;
  }
  const data = await zarr.create(zarr.root(new Map()).resolve("warp"), {
    shape,
    chunk_shape: [dims.length, ...dims.map(() => chunk)],
    data_type: "float32",
    fill_value: 0,
  });
  await zarr.set(data, null, {
    data: values,
    shape,
    stride: [voxels, ...strides],
  });
  const geometry: Record<string, number> = { c: 1 };
  const origin: Record<string, number> = { c: 0 };
  for (const dim of dims) {
    geometry[dim] = 1;
    origin[dim] = 0;
  }
  return new NgffImage({
    data,
    dims: ["c", ...dims],
    scale: geometry,
    translation: origin,
    name: "warp",
    axesUnits: undefined,
    axesTypes: { c: "displacement" },
    axesOrientations: undefined,
    computedCallbacks: undefined,
  });
}

/** A displacement that is zero on the grid boundary and peaks in the middle. */
function interiorBump(extent: number, radius: number, peaks: number[]) {
  const values = new Float32Array(peaks.length * extent * extent);
  const voxels = extent * extent;
  for (let y = 0; y < extent; y++) {
    for (let x = 0; x < extent; x++) {
      const distance = Math.hypot(y - extent / 2, x - extent / 2);
      const profile = distance < radius
        ? 0.5 * (1 + Math.cos(Math.PI * distance / radius))
        : 0;
      peaks.forEach((peak, component) => {
        values[component * voxels + y * extent + x] = peak * profile;
      });
    }
  }
  return values;
}

Deno.test("a bump inside the grid widens the region", async () => {
  // The pipeline sizes a region by walking the boundary of the transformed
  // grid, so a displacement that is zero there and large inside left the
  // region unchanged and the pixels it reaches unread.
  const extent = 64;
  // The peak carries the grid past its own extent, so a region that only
  // covers the grid is not enough.
  const values = interiorBump(extent, 20, [60, -60]);
  const field = await bumpField(values, ["y", "x"], extent, 16);
  const fixed = await geometryImage(
    ["y", "x"],
    { y: extent, x: extent },
    { y: 1, x: 1 },
    { y: 0, x: 0 },
  );
  const moving = await geometryImage(["y", "x"], { y: 160, x: 160 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });

  const region = await resampleBoundingBox(
    { type: "displacements", path: "warp" },
    fixed,
    moving,
    { fields: { warp: field } },
  );

  const voxels = extent * extent;
  ["y", "x"].forEach((dim, component) => {
    let lowest = Number.POSITIVE_INFINITY;
    let highest = Number.NEGATIVE_INFINITY;
    for (let y = 0; y < extent; y++) {
      for (let x = 0; x < extent; x++) {
        const index = component === 0 ? y : x;
        const reached = index + values[component * voxels + y * extent + x];
        lowest = Math.min(lowest, reached);
        highest = Math.max(highest, reached);
      }
    }
    assertEquals(region.startIndex[dim] <= lowest, true);
    assertEquals(region.startIndex[dim] + region.size[dim] >= highest, true);
  });
});

Deno.test("a field's chunking does not change the region it reports", async () => {
  // The range is read a chunk at a time; over the whole grid that has to be
  // the same range whatever the chunks are.
  const extent = 64;
  const values = interiorBump(extent, 20, [60, -60]);
  const fixed = await geometryImage(
    ["y", "x"],
    { y: extent, x: extent },
    { y: 1, x: 1 },
    { y: 0, x: 0 },
  );
  const moving = await geometryImage(["y", "x"], { y: 160, x: 160 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });
  const transform = { type: "displacements" as const, path: "warp" };

  const chunked = await resampleBoundingBox(transform, fixed, moving, {
    fields: { warp: await bumpField(values, ["y", "x"], extent, 16) },
  });
  const whole = await resampleBoundingBox(transform, fixed, moving, {
    fields: { warp: await bumpField(values, ["y", "x"], extent, extent) },
  });

  assertEquals(chunked.startIndex, whole.startIndex);
  assertEquals(chunked.size, whole.size);
});

Deno.test("a grid reaching past the field keeps its undisplaced part", async () => {
  // ITK displaces a point beyond the field by nothing, so zero belongs in the
  // range as much as the values do. A field that displaces every point it
  // covers the same way otherwise carries the region off the points it does
  // not cover.
  const extent = 32;
  const grid = 64;
  const values = new Float32Array(2 * extent * extent).fill(-5);
  const field = await bumpField(values, ["y", "x"], extent, 16);
  const fixed = await geometryImage(
    ["y", "x"],
    { y: grid, x: grid },
    { y: 1, x: 1 },
    { y: 0, x: 0 },
  );
  const moving = await geometryImage(["y", "x"], { y: 160, x: 160 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });

  const region = await resampleBoundingBox(
    { type: "displacements", path: "warp" },
    fixed,
    moving,
    { fields: { warp: field } },
  );

  // The half beyond the field maps identically, so the region has to reach
  // the grid's own last index, not only that index shifted by -5.
  for (const dim of ["y", "x"]) {
    assertEquals(region.startIndex[dim] <= -5, true);
    assertEquals(region.startIndex[dim] + region.size[dim] >= grid - 1, true);
  }
});

Deno.test("an oriented field is refused by the bounding box", async () => {
  // This branch works on the intrinsic systems, where no orientation applies,
  // so a field carrying one cannot be placed here.
  const { AnatomicalOrientationValues, createAnatomicalOrientation } =
    await import("../src/types/rfc4.ts");
  const values = new Float32Array(2 * 32 * 32);
  const field = await bumpField(values, ["y", "x"], 32, 16);
  const oriented = new NgffImage({
    data: field.data,
    dims: field.dims,
    scale: field.scale,
    translation: field.translation,
    name: field.name,
    axesUnits: undefined,
    axesTypes: field.axesTypes,
    axesOrientations: {
      y: createAnatomicalOrientation(
        AnatomicalOrientationValues.AnteriorToPosterior,
      ),
      x: createAnatomicalOrientation(AnatomicalOrientationValues.LeftToRight),
    },
    computedCallbacks: undefined,
  });
  const fixed = await geometryImage(["y", "x"], { y: 32, x: 32 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });
  const moving = await geometryImage(["y", "x"], { y: 64, x: 64 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });

  await assertRejects(
    () =>
      resampleBoundingBox(
        { type: "displacements", path: "warp" },
        fixed,
        moving,
        { fields: { warp: oriented } },
      ),
    Error,
    "anatomical orientation",
  );
});

function displacementFieldEntry(
  vectors: Float64Array,
  size: number[],
  spacing: number[],
  // deno-lint-ignore no-explicit-any
): any {
  const dimension = size.length;
  return {
    transformType: {
      transformParameterization: "DisplacementField",
      parametersValueType: "float64",
      inputDimension: dimension,
      outputDimension: dimension,
    },
    name: "DisplacementFieldTransform",
    inputSpaceName: "",
    outputSpaceName: "",
    numberOfFixedParameters: 3 * dimension + dimension * dimension,
    numberOfParameters: vectors.length,
    fixedParameters: new Float64Array([
      ...size,
      ...size.map(() => 0),
      ...spacing,
      ...Array.from(
        { length: dimension * dimension },
        (_, index) => index % (dimension + 1) === 0 ? 1 : 0,
      ),
    ]),
    parameters: vectors,
    metadata: new Map(),
  };
}

Deno.test("the stage interval reads the layouts itk writes", async () => {
  // A field interleaves components per point; a B-spline blocks them per
  // component. Misreading the layout would produce plausible wrong bounds.
  const { resampleBoundingBox } = await import(
    "../src/io/resample_bounding_box.ts"
  );
  const vectors = new Float64Array(4 * 3 * 2);
  vectors[(1 * 4 + 2) * 2] = 7.0;
  vectors[(1 * 4 + 2) * 2 + 1] = -9.0;
  const warp = displacementFieldEntry(vectors, [4, 3], [8, 8]);
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 4, x: 4 },
    { y: 1, x: 1 },
    {
      y: 0,
      x: 0,
    },
  );
  const moving = await geometryImage(["y", "x"], { y: 64, x: 64 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });

  const region = await resampleBoundingBox([warp], fixed, moving, {
    padding: 0,
  });

  // The x range is [0, 7] and the y range [-9, 0], zero joined since the
  // 4x4 grid leaves nothing to check against the field's own 32x24 reach:
  // the grid is inside, so the values' range stands alone.
  assertEquals(region.startIndex, { y: -9, x: 0 });
  assertEquals(region.size, { y: 13, x: 11 });
});

Deno.test("an interior bump in an itk field widens the region", async () => {
  const { resampleBoundingBox } = await import(
    "../src/io/resample_bounding_box.ts"
  );
  const extent = 64;
  const vectors = new Float64Array(extent * extent * 2);
  const reach = { y: [0, 0], x: [0, 0] };
  for (let y = 0; y < extent; y++) {
    for (let x = 0; x < extent; x++) {
      const distance = Math.hypot(y - extent / 2, x - extent / 2);
      const profile = distance < 20
        ? 0.5 * (1 + Math.cos(Math.PI * distance / 20))
        : 0;
      const index = (y * extent + x) * 2;
      vectors[index] = -60 * profile;
      vectors[index + 1] = 60 * profile;
      reach.y = [
        Math.min(reach.y[0], y + 60 * profile),
        Math.max(reach.y[1], y + 60 * profile),
      ];
      reach.x = [
        Math.min(reach.x[0], x - 60 * profile),
        Math.max(reach.x[1], x - 60 * profile),
      ];
    }
  }
  const warp = displacementFieldEntry(vectors, [extent, extent], [1, 1]);
  const fixed = await geometryImage(
    ["y", "x"],
    { y: extent, x: extent },
    { y: 1, x: 1 },
    { y: 0, x: 0 },
  );
  const moving = await geometryImage(["y", "x"], { y: 256, x: 256 }, {
    y: 1,
    x: 1,
  }, { y: 0, x: 0 });

  const region = await resampleBoundingBox([warp], fixed, moving, {
    padding: 1,
  });

  for (const dim of ["y", "x"] as const) {
    assertEquals(region.startIndex[dim] <= reach[dim][0], true);
    assertEquals(
      region.startIndex[dim] + region.size[dim] >= reach[dim][1],
      true,
    );
  }
});

/** A field that shifts every point of its lattice by the same vector. */
function constantFieldEntry(
  shift: number[],
  size: number[],
  spacing: number[],
  // deno-lint-ignore no-explicit-any
): any {
  const points = size.reduce((total, extent) => total * extent, 1);
  const vectors = new Float64Array(points * shift.length);
  for (let point = 0; point < points; point++) {
    for (let component = 0; component < shift.length; component++) {
      vectors[point * shift.length + component] = shift[component];
    }
  }
  return displacementFieldEntry(vectors, size, spacing);
}

/** A field that is zero on its boundary and peaks at its centre.
 *
 * The boundary walk sees only the zero, so a stage outside this one has to
 * carry the peak for the region to hold it. `peaks` is in ITK component
 * order (x, y); the returned entry's lattice is unit-spaced at the origin. */
function bumpFieldEntry(
  extent: number,
  peaks: number[],
  radius: number,
  // deno-lint-ignore no-explicit-any
): any {
  const vectors = new Float64Array(extent * extent * 2);
  for (let y = 0; y < extent; y++) {
    for (let x = 0; x < extent; x++) {
      const distance = Math.hypot(y - extent / 2, x - extent / 2);
      const profile = distance < radius
        ? 0.5 * (1 + Math.cos(Math.PI * distance / radius))
        : 0;
      const index = (y * extent + x) * 2;
      vectors[index] = peaks[0] * profile;
      vectors[index + 1] = peaks[1] * profile;
    }
  }
  return displacementFieldEntry(vectors, [extent, extent], [1, 1]);
}

Deno.test("a field outside an interior bump carries its range", async () => {
  // ITK applies the last entry first, so the bump's range has to travel
  // through the constant field outside it, which is itself non-linear:
  // folding stage by stage stops there and the walk, which sees only the
  // bump's zero boundary, reports a region the composition leaves.
  const { resampleBoundingBox } = await import(
    "../src/io/resample_bounding_box.ts"
  );
  const extent = 64;
  const peaks = [-60, 60];
  const bump = bumpFieldEntry(extent, peaks, 20);
  const shift = [-3, 11];
  const outer = constantFieldEntry(shift, [256, 256], [1, 1]);
  const fixed = await geometryImage(
    ["y", "x"],
    { y: extent, x: extent },
    { y: 1, x: 1 },
    { y: 0, x: 0 },
  );
  const moving = await geometryImage(["y", "x"], { y: 512, x: 512 }, {
    y: 1,
    x: 1,
  }, { y: -128, x: -128 });

  const region = await resampleBoundingBox([outer, bump], fixed, moving, {
    padding: 0,
  });

  // The grid centre takes the whole peak, then the constant shift.
  const centre = extent / 2;
  assertEquals(region.cornersMax.y >= centre + peaks[1] + shift[1], true);
  assertEquals(region.cornersMin.x <= centre + peaks[0] + shift[0], true);
});

Deno.test("an affine outside two fields carries both their ranges", async () => {
  const { resampleBoundingBox } = await import(
    "../src/io/resample_bounding_box.ts"
  );
  const extent = 64;
  const peaks = [-60, 60];
  const bump = bumpFieldEntry(extent, peaks, 20);
  const shift = [-3, 11];
  const outer = constantFieldEntry(shift, [256, 256], [1, 1]);
  const [affine] = itkAffine([[2, 0], [0, 3]], [4, -6]);
  const fixed = await geometryImage(
    ["y", "x"],
    { y: extent, x: extent },
    { y: 1, x: 1 },
    { y: 0, x: 0 },
  );
  const moving = await geometryImage(["y", "x"], { y: 1024, x: 1024 }, {
    y: 1,
    x: 1,
  }, { y: -512, x: -512 });

  const region = await resampleBoundingBox(
    [affine, outer, bump],
    fixed,
    moving,
    { padding: 0 },
  );

  // The centre takes the peak and the shift, then the affine scales it:
  // x' = 2 (x + peak_x + shift_x) + 4, y' = 3 (y + peak_y + shift_y) - 6.
  const centre = extent / 2;
  assertEquals(
    region.cornersMax.y >= 3 * (centre + peaks[1] + shift[1]) - 6,
    true,
  );
  assertEquals(
    region.cornersMin.x <= 2 * (centre + peaks[0] + shift[0]) + 4,
    true,
  );
});
