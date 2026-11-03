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

import { test } from "vite-plus/test";
import {
  assert,
  assertAlmostEquals,
  assertEquals,
  assertRejects,
  assertThrows,
} from "./_assert.ts";
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
import { resampleBoundingBoxNode } from "@itk-wasm/downsample";
import {
  metadataOnlyItkImage,
  resampleBoundingBoxShared,
  windowedFields,
} from "../src/io/resample_bounding_box-shared.ts";
import { RAS } from "../src/types/rfc4.ts";
import type { AnatomicalOrientation } from "../src/types/rfc4.ts";

/** An ITK-Wasm translation, in ITK (fastest-axis-first) order. */
function itkTranslation(offset: number[]): any {
  return [
    {
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
    },
  ];
}

/**
 * An ITK Euler2D entry at angle zero: the identity, in a parameterization the
 * staged bound does not decode, so the pipeline sizes the region.
 */
const pipelineOnly = (): any => [
  {
    transformType: {
      transformParameterization: "Euler2D",
      parametersValueType: "float64",
      inputDimension: 2,
      outputDimension: 2,
    },
    name: "Euler2DTransform",
    inputSpaceName: "",
    outputSpaceName: "",
    numberOfFixedParameters: 2,
    numberOfParameters: 3,
    fixedParameters: new Float64Array([0, 0]),
    parameters: new Float64Array([0, 0, 0]),
    metadata: new Map(),
  },
];

const identity = (dimension: number): any =>
  itkTranslation(new Array(dimension).fill(0));

/** An ITK-Wasm affine, row-major matrix then translation, centre at the origin. */
function itkAffine(matrix: number[][], offset: number[]): any {
  const dimension = offset.length;
  return [
    {
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
    },
  ];
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
    chunkShape: arrayShape.map((n) => Math.min(n, 32)),
    dtype: "uint8",
    fillValue: 0,
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
    const corner = Array.from({ length: ndim }, (_, i) =>
      (mask >> i) & 1 ? fixedShape[i] - 1 : 0,
    );
    const point = corner.map((c, i) => fixedTranslation[i] + fixedScale[i] * c);
    for (let row = 0; row < ndim; row++) {
      let moved = offset[row];
      for (let col = 0; col < ndim; col++) {
        moved += matrix[row][col] * point[col];
      }
      const continuousIndex =
        (moved - movingTranslation[row]) / movingScale[row];
      indexMin[row] = Math.min(indexMin[row], continuousIndex);
      indexMax[row] = Math.max(indexMax[row], continuousIndex);
    }
  }

  const start = indexMin.map((v) => Math.floor(v) - padding);
  const end = indexMax.map((v) => Math.ceil(v) + padding);
  return { start, size: end.map((e, i) => Math.max(e - start[i] + 1, 0)) };
}

test("an NGFF translation reproduces the documented 2D example", async () => {
  // The documentation states this in ITK order: fixed 16x16 spacing (2,2)
  // origin (10,20), moving 64x64 spacing (1,1) origin 0, translation (10,5),
  // padding 1. Written as RFC-5 the axes reverse, so the translation is
  // [5, 10] over ["y", "x"].
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 16, x: 16 },
    {
      y: 2,
      x: 2,
    },
    { y: 20, x: 10 },
  );
  const moving = await geometryImage(
    ["y", "x"],
    { y: 64, x: 64 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );

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

test("padding is applied symmetrically", async () => {
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 16, x: 16 },
    {
      y: 2,
      x: 2,
    },
    { y: 20, x: 10 },
  );
  const moving = await geometryImage(
    ["y", "x"],
    { y: 64, x: 64 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );
  const transform = createTranslation([5, 10]);

  const padded = await resampleBoundingBox(transform, fixed, moving, {
    padding: 1,
  });
  const tight = await resampleBoundingBox(transform, fixed, moving, {
    padding: 0,
  });

  for (const dim of ["y", "x"]) {
    assertEquals(tight.startIndex[dim], padded.startIndex[dim] + 1);
    assertEquals(tight.size[dim], padded.size[dim] - 2);
  }
  // The tight corners are padding independent.
  assertEquals(tight.cornersMin, padded.cornersMin);
  assertEquals(tight.cornersMax, padded.cornersMax);
});

test("an asymmetric 3D NGFF affine matches the oracle", async () => {
  const dims = ["z", "y", "x"];
  const fixed = await geometryImage(
    dims,
    { z: 4, y: 8, x: 16 },
    {
      z: 3,
      y: 2,
      x: 1,
    },
    { z: 30, y: 20, x: 10 },
  );
  const moving = await geometryImage(
    dims,
    { z: 64, y: 128, x: 256 },
    {
      z: 1.5,
      y: 0.5,
      x: 0.25,
    },
    { z: -5, y: 7, x: 3 },
  );

  // Deliberately asymmetric, with shear, so any axis-order slip shows.
  const matrix = [
    [1, 0.2, 0],
    [0, 2, 0.3],
    [0.5, 0, 1],
  ];
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
  assertEquals(
    dims.map((d) => boundingBox.startIndex[d]),
    expected.start,
  );
  assertEquals(
    dims.map((d) => boundingBox.size[d]),
    expected.size,
  );
});

test("affine translation is the last column", () => {
  const { matrix, offset } = ngffTransformToItkMatrix(
    createAffine([
      [1, 2, 3],
      [4, 5, 6],
    ]),
    ["y", "x"],
  );
  // In NGFF terms: y = 1*y + 2*x + 3 and x = 4*y + 5*x + 6. ITK reverses the
  // axes, so both the matrix and the offset come back reversed.
  assertEquals(offset, [6, 3]);
  assertEquals(matrix, [
    [5, 4],
    [2, 1],
  ]);
});

test("an affine of the wrong shape is rejected", () => {
  let message = "";
  try {
    ngffTransformToItkMatrix(
      createAffine([
        [1, 0],
        [0, 1],
      ]),
      ["y", "x"],
    );
  } catch (error) {
    message = (error as Error).message;
  }
  assertEquals(message.includes("translation is the last column"), true);
});

test("a sequence applies its first entry first", () => {
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

test("sequence order survives the whole pipeline", async () => {
  // Every other sequence test here reads the matrix out of
  // ngffTransformToItkMatrix. If the order inverted, the region the caller
  // actually gets would move, and nothing below that helper would notice.
  // Applying [translate 10, scale 2] the wrong way round gives 2x + 10
  // instead of 2(x + 10), so the x start moves by 10.
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 16, x: 16 },
    {
      y: 2,
      x: 2,
    },
    { y: 20, x: 10 },
  );
  const moving = await geometryImage(
    ["y", "x"],
    { y: 512, x: 512 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );

  const region = await resampleBoundingBox(
    createTransformSequence([createTranslation([0, 10]), createScale([1, 2])]),
    fixed,
    moving,
    { padding: 0 },
  );

  // First entry first: y = x, x = 2(x + 10) = 2x + 20.
  const { start, size } = oracleRegion(
    [
      [1, 0],
      [0, 2],
    ],
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

test("nested sequences compose", () => {
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

test("identity, scale and rotation convert", () => {
  const identity = ngffTransformToItkMatrix(createIdentity(), ["y", "x"]);
  assertEquals(identity.matrix, [
    [1, 0],
    [0, 1],
  ]);
  assertEquals(identity.offset, [0, 0]);

  // Reversed to ITK order (x, y).
  const scale = ngffTransformToItkMatrix(createScale([2, 3]), ["y", "x"]);
  assertEquals(scale.matrix, [
    [3, 0],
    [0, 2],
  ]);

  const rotation = ngffTransformToItkMatrix(
    createRotation([
      [0, -1],
      [1, 0],
    ]),
    ["y", "x"],
  );
  // Reversing both rows and columns of [[0,-1],[1,0]] gives [[0,1],[-1,0]].
  assertEquals(rotation.matrix, [
    [0, 1],
    [-1, 0],
  ]);
});

test("a non-linear transformation is rejected", () => {
  let message = "";
  try {
    ngffTransformToItkMatrix({ type: "displacements", path: "field" }, [
      "y",
      "x",
    ]);
  } catch (error) {
    message = (error as Error).message;
  }
  assertEquals(message.includes("cannot be converted"), true);
});

test("a transform coupling spatial and non-spatial axes is rejected", () => {
  // y would depend on c.
  const affine = [
    [1, 0, 0, 0],
    [0.5, 1, 0, 0],
    [0, 0, 1, 0],
  ];
  let message = "";
  try {
    ngffTransformToItkMatrix(createAffine(affine), ["c", "y", "x"]);
  } catch (error) {
    message = (error as Error).message;
  }
  assertEquals(message.includes("couples spatial and non-spatial"), true);
});

test("mismatched spatial dims are rejected", async () => {
  const fixed = await geometryImage(
    ["z", "y", "x"],
    { z: 4, y: 4, x: 4 },
    {
      z: 1,
      y: 1,
      x: 1,
    },
    { z: 0, y: 0, x: 0 },
  );
  const moving = await geometryImage(
    ["y", "x"],
    { y: 4, x: 4 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );

  await assertRejects(
    () => resampleBoundingBox(identity(2), fixed, moving),
    Error,
    "they must match",
  );
});

test("pixel data is never read", async () => {
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
    chunkShape: [8, 8],
    dtype: "uint8",
    fillValue: 0,
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

  const boundingBox = await resampleBoundingBox(identity(2), image, image, {
    padding: 1,
  });

  assertEquals(boundingBox.startIndex, { y: -1, x: -1 });
  assertEquals(boundingBox.size, { y: 34, x: 34 });
});

test("non-spatial axes are passed through", async () => {
  const dims = ["t", "c", "z", "y", "x"];
  const fixed = await geometryImage(
    dims,
    { t: 3, c: 2, z: 4, y: 8, x: 8 },
    {
      t: 1,
      c: 1,
      z: 1,
      y: 1,
      x: 1,
    },
    { t: 0, c: 0, z: 0, y: 0, x: 0 },
  );
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

test("a region outside the moving image is empty", async () => {
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 4, x: 4 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );
  const moving = await geometryImage(
    ["y", "x"],
    { y: 32, x: 32 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );

  const boundingBox = await resampleBoundingBox(
    itkTranslation([1000, 1000]),
    fixed,
    moving,
    { padding: 1 },
  );

  assertEquals(boundingBox.isEmpty, true);
});

test("a negative start index is clamped, not wrapped", async () => {
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 4, x: 4 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );
  const moving = await geometryImage(
    ["y", "x"],
    { y: 32, x: 32 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );

  const boundingBox = await resampleBoundingBox(identity(2), fixed, moving, {
    padding: 2,
  });

  assertEquals(boundingBox.startIndex, { y: -2, x: -2 });
  assertEquals(boundingBox.clamped(), { y: [0, 6], x: [0, 6] });
  assertEquals(boundingBox.isEmpty, false);
});

test("the cropped translation shifts by start * scale", async () => {
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 64, x: 64 },
    {
      y: 1,
      x: 1,
    },
    { y: 512, x: 1024 },
  );
  const moving = await geometryImage(
    ["y", "x"],
    { y: 1024, x: 1024 },
    {
      y: 2,
      x: 2,
    },
    { y: 0, x: 0 },
  );

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

test("RAS orientation yields a non-identity direction", async () => {
  const { itkDirection } =
    await import("../src/io/resample_bounding_box-shared.ts");
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

test("a 3D-only orientation on a 2D image falls back to identity", async () => {
  const { itkDirection } =
    await import("../src/io/resample_bounding_box-shared.ts");
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

test("a grid far larger than the moving image reads all of it", async () => {
  // The region is computed in floating point from the grid's corners, so a
  // grid that covers the whole moving image many times over clamps to all of
  // it rather than wrapping an integer index.
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 16, x: 16 },
    {
      y: 1e9,
      x: 1e9,
    },
    { y: 0, x: 0 },
  );
  const moving = await geometryImage(
    ["y", "x"],
    { y: 64, x: 64 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );

  const region = await resampleBoundingBox(identity(2), fixed, moving, {
    padding: 1,
  });

  assert(!region.isEmpty);
  assertEquals(region.clamped(), { y: [0, 64], x: [0, 64] });
});

test("an index range overflow is reported, not silently empty", async () => {
  // On the pipeline path the region is computed in 32-bit index space while
  // the corners come back as doubles, so a fixed/moving scale mismatch this
  // large wraps the integer region. Reporting "no overlap" for a grid
  // covering the whole moving image would be a silent, wrong answer.
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 16, x: 16 },
    {
      y: 1e9,
      x: 1e9,
    },
    { y: 0, x: 0 },
  );
  const moving = await geometryImage(
    ["y", "x"],
    { y: 64, x: 64 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );

  await assertRejects(
    () =>
      resampleBoundingBox(pipelineOnly(), fixed, moving, {
        padding: 1,
      }),
    Error,
    "does not contain the transformed grid",
  );
});

test("padding that is not a non-negative integer is rejected", async () => {
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 4, x: 4 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );
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

test("unsupported spatial dimensionality is rejected", async () => {
  const fixed = await geometryImage(["x"], { x: 8 }, { x: 1 }, { x: 0 });
  await assertRejects(
    () => resampleBoundingBox(identity(2), fixed, fixed),
    Error,
    "only 2 and 3 are supported",
  );
});

test("a missing scale entry is rejected rather than defaulted", async () => {
  // Python raises here; TypeScript used to substitute spacing 1 and return a
  // plausible but wrong region.
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 4, x: 4 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );
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

test("a non-finite scale is rejected", async () => {
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 4, x: 4 },
    {
      y: 1,
      x: NaN,
    },
    { y: 0, x: 0 },
  );
  await assertRejects(
    () => resampleBoundingBox(identity(2), fixed, fixed),
    Error,
    "must be finite",
  );
});

test("a zero scale is rejected", async () => {
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 4, x: 4 },
    {
      y: 1,
      x: 0,
    },
    { y: 0, x: 0 },
  );
  await assertRejects(
    () => resampleBoundingBox(identity(2), fixed, fixed),
    Error,
    "scale for dimension 'x' is zero",
  );
});

test("a degenerate fixed grid yields an empty region", async () => {
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 0, x: 4 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );
  const moving = await geometryImage(
    ["y", "x"],
    { y: 32, x: 32 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );

  const boundingBox = await resampleBoundingBox(identity(2), fixed, moving);

  assertEquals(boundingBox.isEmpty, true);
});

test("the RFC-5 branch ignores anatomical orientation", async () => {
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

test("an RFC-5 displacements transform reaches the same region", async () => {
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
    fixedParameters: new Float64Array([...size, 0, 0, ...spacing, 1, 0, 0, 1]),
    parameters,
    metadata: new Map(),
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
  const moving = await geometryImage(
    ["y", "x"],
    { y: 64, x: 64 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );

  const viaItk = await resampleBoundingBox([warp], fixed, moving);
  const { transform, field } = await itkDisplacementFieldToNgffTransform(
    warp,
    ["y", "x"],
    { path: "warp" },
  );
  const viaRfc5 = await resampleBoundingBox(transform, fixed, moving, {
    fields: { warp: field },
  });

  assertEquals(viaRfc5.startIndex, viaItk.startIndex);
  assertEquals(viaRfc5.size, viaItk.size);

  // Without the field there is nothing to convert, and the message says so.
  await assertRejects(
    () => resampleBoundingBox(transform, fixed, moving),
    Error,
    "no field was passed",
  );
});

test("converting with frames matches the ITK path on oriented images", async () => {
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
  const transform = [
    {
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
        0.9, 0.1, 0, -0.1, 1.1, 0, 0, 0, 1, 2, -3, 4,
      ]),
      metadata: new Map(),
    } as any,
  ];

  const viaItk = await resampleBoundingBox(transform, fixed, moving, {
    padding: 0,
  });
  const converted = itkTransformToNgffTransform(
    transform,
    ["z", "y", "x"],
    true,
    {
      fixed,
      moving,
    },
  );
  const viaRfc5 = await resampleBoundingBox(converted, fixed, moving, {
    padding: 0,
  });

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

test("frames round trip through both converters", async () => {
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

test("an ITK-Wasm transform list is accepted", async () => {
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 16, x: 16 },
    {
      y: 2,
      x: 2,
    },
    { y: 20, x: 10 },
  );
  const moving = await geometryImage(
    ["y", "x"],
    { y: 64, x: 64 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );

  // Stated directly in ITK order (x, y).
  const transformList = [
    {
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
    },
  ];

  const boundingBox = await resampleBoundingBox(
    transformList as never,
    fixed,
    moving,
    { padding: 1 },
  );

  assertEquals(boundingBox.startIndex, { y: 24, x: 19 });
  assertEquals(boundingBox.size, { y: 33, x: 33 });
});

test("an asymmetric 3D affine matches the oracle", () => {
  // Stated in NGFF (z, y, x) order for the oracle; the transform itself is
  // built in ITK order, so both row and column ordering reverse.
  const matrix = [
    [1, 0.2, 0],
    [0, 2, 0.3],
    [0.5, 0, 1],
  ];
  const offset = [4, -6, 11];
  return (async () => {
    const dims = ["z", "y", "x"];
    const fixed = await geometryImage(
      dims,
      { z: 4, y: 8, x: 16 },
      {
        z: 3,
        y: 2,
        x: 1,
      },
      { z: 30, y: 20, x: 10 },
    );
    const moving = await geometryImage(
      dims,
      { z: 64, y: 128, x: 256 },
      {
        z: 1.5,
        y: 0.5,
        x: 0.25,
      },
      { z: -5, y: 7, x: 3 },
    );

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
    assertEquals(
      dims.map((d) => boundingBox.startIndex[d]),
      expected.start,
    );
    assertEquals(
      dims.map((d) => boundingBox.size[d]),
      expected.size,
    );
  })();
});

test("unusable pipeline index arrays are rejected", async () => {
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 4, x: 4 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );
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
      () => resampleBoundingBoxShared(pipeline, pipelineOnly(), fixed, fixed),
      Error,
      message,
    );
  }
});

test("a mapAxis region matches the oracle", async () => {
  const dims = ["z", "y", "x"];
  const fixed = await geometryImage(
    dims,
    { z: 12, y: 20, x: 16 },
    {
      z: 2,
      y: 1,
      x: 0.5,
    },
    { z: 3, y: -4, x: 6 },
  );
  const moving = await geometryImage(
    dims,
    { z: 64, y: 64, x: 64 },
    {
      z: 1,
      y: 1,
      x: 1,
    },
    { z: 0, y: 0, x: 0 },
  );

  const region = await resampleBoundingBox(
    createMapAxis([1, 2, 0]),
    fixed,
    moving,
    { padding: 1 },
  );

  // Output axis i takes input axis mapAxis[i], so row i selects that column.
  const { start, size } = oracleRegion(
    [
      [0, 1, 0],
      [0, 0, 1],
      [1, 0, 0],
    ],
    [0, 0, 0],
    [12, 20, 16],
    [2, 1, 0.5],
    [3, -4, 6],
    [1, 1, 1],
    [0, 0, 0],
    1,
  );
  assertEquals(
    dims.map((dim) => region.startIndex[dim]),
    start,
  );
  assertEquals(
    dims.map((dim) => region.size[dim]),
    size,
  );
});

test("a byDimension region matches the oracle", async () => {
  const dims = ["z", "y", "x"];
  const fixed = await geometryImage(
    dims,
    { z: 10, y: 24, x: 24 },
    {
      z: 1,
      y: 0.5,
      x: 0.5,
    },
    { z: -2, y: 1, x: 4 },
  );
  const moving = await geometryImage(
    dims,
    { z: 64, y: 128, x: 128 },
    {
      z: 1,
      y: 0.25,
      x: 0.25,
    },
    { z: 0, y: 0, x: 0 },
  );

  const region = await resampleBoundingBox(
    createByDimension([
      {
        transformation: createTranslation([5]),
        inputAxes: [0],
        outputAxes: [0],
      },
      {
        transformation: createAffine([
          [0.8, -0.6, 2],
          [0.6, 0.8, -3],
        ]),
        inputAxes: [1, 2],
        outputAxes: [1, 2],
      },
    ]),
    fixed,
    moving,
    { padding: 2 },
  );

  const { start, size } = oracleRegion(
    [
      [1, 0, 0],
      [0, 0.8, -0.6],
      [0, 0.6, 0.8],
    ],
    [5, 2, -3],
    [10, 24, 24],
    [1, 0.5, 0.5],
    [-2, 1, 4],
    [1, 0.25, 0.25],
    [0, 0, 0],
    2,
  );
  assertEquals(
    dims.map((dim) => region.startIndex[dim]),
    start,
  );
  assertEquals(
    dims.map((dim) => region.size[dim]),
    size,
  );
});

test("a bijection region follows its forward direction", async () => {
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 16, x: 16 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );
  const moving = await geometryImage(
    ["y", "x"],
    { y: 128, x: 128 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );
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
    chunkShape: [dims.length, ...dims.map(() => chunk)],
    dtype: "float32",
    fillValue: 0,
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
      const profile =
        distance < radius
          ? 0.5 * (1 + Math.cos((Math.PI * distance) / radius))
          : 0;
      peaks.forEach((peak, component) => {
        values[component * voxels + y * extent + x] = peak * profile;
      });
    }
  }
  return values;
}

test("a bump strictly inside the grid widens the region", async () => {
  // A region is the box of the landing points of the samples under the grid,
  // so a displacement that is zero on the boundary and large inside reaches
  // the region all the same.
  const extent = 64;
  const values = interiorBump(extent, 20, [60, -60]);
  const field = await bumpField(values, ["y", "x"], extent, 16);
  const fixed = await geometryImage(
    ["y", "x"],
    { y: extent, x: extent },
    { y: 1, x: 1 },
    { y: 0, x: 0 },
  );
  const moving = await geometryImage(
    ["y", "x"],
    { y: 160, x: 160 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );

  const region = await resampleBoundingBox(
    { type: "displacements", path: "warp" },
    fixed,
    moving,
    { fields: { warp: field } },
  );
  // Every point the field displaces onto lies in the region.
  const voxels = extent * extent;
  ["y", "x"].forEach((dim, component) => {
    let reachedMin = Infinity;
    let reachedMax = -Infinity;
    for (let y = 0; y < extent; y++) {
      for (let x = 0; x < extent; x++) {
        const reached =
          (component === 0 ? y : x) +
          values[component * voxels + y * extent + x];
        reachedMin = Math.min(reachedMin, reached);
        reachedMax = Math.max(reachedMax, reached);
      }
    }
    assert(region.startIndex[dim] <= reachedMin, `${dim} start`);
    assert(
      region.startIndex[dim] + region.size[dim] >= reachedMax,
      `${dim} end`,
    );
  });
});

test("a field's chunking does not change the region it reports", async () => {
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
  const moving = await geometryImage(
    ["y", "x"],
    { y: 160, x: 160 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );
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

test("a grid reaching past the field keeps its undisplaced part", async () => {
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
  const moving = await geometryImage(
    ["y", "x"],
    { y: 160, x: 160 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );

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

test("an oriented field is refused by the bounding box", async () => {
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
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 32, x: 32 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );
  const moving = await geometryImage(
    ["y", "x"],
    { y: 64, x: 64 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );

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

test("an ITK displacement field reaches the pipeline as the grid's window", async () => {
  const fixed = await geometryImage(
    ["y", "x"],
    { y: 16, x: 16 },
    {
      y: 1,
      x: 1,
    },
    { y: 20, x: 30 },
  );
  const moving = await geometryImage(
    ["y", "x"],
    { y: 64, x: 64 },
    {
      y: 1,
      x: 1,
    },
    { y: 0, x: 0 },
  );

  // A 64 x 64 field in ITK layout: one (x, y) vector per point, x fastest. It
  // varies by less than a pixel per pixel, so it does not fold.
  const parameters = new Float64Array(2 * 64 * 64);
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      parameters[2 * (y * 64 + x)] = 3 * Math.sin(y / 5);
      parameters[2 * (y * 64 + x) + 1] = 2 * Math.cos(x / 7);
    }
  }
  const whole = [
    {
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
      fixedParameters: new Float64Array([64, 64, 0, 0, 1, 1, 1, 0, 0, 1]),
      parameters,
      metadata: new Map(),
    },
  ];

  const received: number[] = [];
  const recording: Parameters<typeof resampleBoundingBoxShared>[0] = (
    transform,
    ...rest
  ) => {
    received.push(transform[0].numberOfParameters);
    return resampleBoundingBoxNode(transform, ...rest);
  };
  // A stage the staged bound cannot read keeps the list on the pipeline path,
  // where the field is windowed before the call.
  const cut = await resampleBoundingBoxShared(
    recording,
    [...whole, ...pipelineOnly()] as never,
    fixed,
    moving,
  );

  // The identity that finds the window, then the window itself: sixteen
  // samples, the three pixels the field moves them by at most, and a sample
  // of margin on either side.
  assertEquals(received.length, 2);
  assert(received[1] <= 2 * 24 * 24, `the pipeline received ${received[1]}`);

  const direction = new Float64Array([1, 0, 0, 1]);
  const { boundingBox } = await resampleBoundingBoxNode(
    whole as never,
    metadataOnlyItkImage(fixed, ["x", "y"], direction),
    metadataOnlyItkImage(moving, ["x", "y"], direction),
    { padding: 1 },
  );
  const uncut = boundingBox as {
    paddedStartIndex: number[];
    paddedSize: number[];
  };
  assertEquals(cut.startIndex, {
    y: uncut.paddedStartIndex[1],
    x: uncut.paddedStartIndex[0],
  });
  assertEquals(cut.size, { y: uncut.paddedSize[1], x: uncut.paddedSize[0] });
});

test("a B-spline stage reports the region Python reports", async () => {
  // The same control points as py/test/test_lattice_bound.py's cross-port
  // case; the regions are Python's, to the pixel.
  const size = [7, 6];
  const parameters = new Float64Array([
    2.0, 7.347244, 6.850978, 1.053526, -3.709612, -2.233242, 3.869248, 5.620471,
    7.598251, 3.190661, -2.78565, -3.799723, 1.256629, 6.857785, 6.993303,
    4.671437, -1.574518, -4.918935, -1.707008, 4.551243, 7.016754, 5.320731,
    -0.20877, -5.355521, -4.495113, 1.432194, 5.948992, 4.11929, 1.177635,
    -4.958891, -6.585839, -1.925269, 3.929714, 4.580737, -0.683641, -3.706987,
    -7.564433, -4.9064, 1.362393, 4.391361, 0.870424, -5.352711, -5.0,
    -1.939795, 3.288079, 7.316478, 7.482513, 3.781871, -1.147075, -4.126678,
    -3.608427, 0.34028, 5.320348, 7.948683, 6.357554, 1.588895, -1.811789,
    -3.608427, -1.97461, 2.442431, 6.685689, 7.665291, 4.477694, 1.13601,
    -1.939795, -2.847932, -0.311933, 4.134731, 7.248253, 6.510182, 3.686969,
    0.814569, -1.97461, -1.980565, 1.186932, 5.252123, 6.976352, 4.949962,
    3.692486, 0.34028, -1.980565, -1.127957, 2.374206, 5.713358,
  ]);
  const spline = [
    {
      transformType: {
        transformParameterization: "BSpline",
        parametersValueType: "float64",
        inputDimension: 2,
        outputDimension: 2,
      },
      name: "BSplineTransform",
      inputSpaceName: "",
      outputSpaceName: "",
      numberOfFixedParameters: 10,
      numberOfParameters: parameters.length,
      fixedParameters: new Float64Array([
        ...size,
        -16,
        -16,
        16,
        16,
        1,
        0,
        0,
        1,
      ]),
      parameters,
      metadata: new Map(),
    },
  ];
  const moving = await geometryImage(
    ["y", "x"],
    { y: 128, x: 128 },
    {
      y: 1,
      x: 1,
    },
    { y: -32, x: -32 },
  );
  const cases: [
    number,
    number,
    number,
    number,
    Record<string, number>,
    Record<string, number>,
  ][] = [
    [0, 0, 32, 32, { y: 28, x: 30 }, { y: 38, x: 34 }],
    [20, 24, 24, 40, { y: 50, x: 50 }, { y: 33, x: 52 }],
    [40, 8, 16, 48, { y: 67, x: 31 }, { y: 30, x: 66 }],
  ];
  for (const [y, x, height, width, startIndex, size] of cases) {
    const fixed = await geometryImage(
      ["y", "x"],
      { y: height, x: width },
      {
        y: 1,
        x: 1,
      },
      { y, x },
    );
    const region = await resampleBoundingBox(spline as never, fixed, moving);
    assertEquals(region.startIndex, startIndex);
    assertEquals(region.size, size);
  }
});

test("a field's window covers what a B-spline bump before it sends inside", async () => {
  // One control point of 60 at (24, 24), x component, on 11 x 11 control
  // points 8 apart from (-8, -8): the block [16, 32)^2 folds onto itself, and
  // its center lands at x = 24 + 60 * (2/3)^2, past the 41 its boundary
  // reaches, which a boundary walk does not see.
  const coefficients = new Float64Array(2 * 11 * 11);
  coefficients[4 * 11 + 4] = 60;
  const spline = {
    transformType: {
      transformParameterization: "BSpline",
      parametersValueType: "float64",
      inputDimension: 2,
      outputDimension: 2,
    },
    name: "BSplineTransform",
    inputSpaceName: "",
    outputSpaceName: "",
    numberOfFixedParameters: 10,
    numberOfParameters: coefficients.length,
    fixedParameters: new Float64Array([11, 11, -8, -8, 8, 8, 1, 0, 0, 1]),
    parameters: coefficients,
    metadata: new Map(),
  };
  const field = {
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
    numberOfParameters: 2 * 64 * 64,
    fixedParameters: new Float64Array([64, 64, 0, 0, 1, 1, 1, 0, 0, 1]),
    parameters: new Float64Array(2 * 64 * 64),
    metadata: new Map(),
  };
  const block = await geometryImage(
    ["y", "x"],
    { y: 16, x: 16 },
    { y: 1, x: 1 },
    { y: 16, x: 16 },
  );
  const direction = new Float64Array([1, 0, 0, 1]);

  // ITK applies the spline first, then the field.
  const [cut] = await windowedFields(
    resampleBoundingBoxNode,
    [field, spline] as never,
    metadataOnlyItkImage(block, ["x", "y"], direction),
  );

  const [width, , first] = Array.from(cut.fixedParameters ?? [], Number);
  const last = first + width - 1;
  assert(last >= 51, `the window ends at x = ${last}`);
  assert(width < 64, "the whole field was handed over");
});
