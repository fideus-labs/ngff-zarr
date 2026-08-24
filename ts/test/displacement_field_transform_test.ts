// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
// ITK displacement fields to and from RFC-5 `displacements` transformations.
//
// TypeScript has no `itk` to evaluate a field against, so the anchors are
// arithmetic: a field defines T(p) = p + v at its grid points, and every
// converted field must satisfy phi_out(q + d(q)) = phi_in(q) + v(q) point by
// point. Layout and component order are pinned with one known vector, and
// the reverse conversion must give back the entry it was built from.

import { assertAlmostEquals, assertEquals, assertRejects } from "@std/assert";
import type { Image, Transform } from "itk-wasm";
import * as zarr from "zarrita";
import {
  itkDisplacementFieldToNgffTransform,
  itkTransformToNgffTransform,
  ngffDisplacementFieldToItkTransform,
  NgffImage,
  ngffTransformToItkTransform,
} from "../src/mod.ts";
import { type AnatomicalOrientation, RAS } from "../src/types/rfc4.ts";
import type { Displacements } from "../src/types/zarr_metadata.ts";
import { directionRows, itkDirection } from "../src/utils/itk_direction.ts";

const CANONICAL: Record<number, string[]> = {
  2: ["y", "x"],
  3: ["z", "y", "x"],
};

/** A deterministic pseudo-random sequence, so a field is reproducible. */
function noise(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000 * 6 - 3;
  };
}

/** A field on a grid of `size` (ITK order), as a `DisplacementField` entry. */
function fieldTransform(
  size: number[],
  spacing: number[],
  origin: number[],
  direction?: number[][],
  seed = 0,
  Values: typeof Float64Array | typeof Float32Array = Float64Array,
): Transform {
  const dimension = size.length;
  const voxels = size.reduce((a, b) => a * b, 1);
  const next = noise(seed + 1);
  const parameters = new Values(voxels * dimension);
  for (let i = 0; i < parameters.length; i++) parameters[i] = next();
  const rows = direction ??
    Array.from(
      { length: dimension },
      (_, r) => Array.from({ length: dimension }, (_, c) => +(r === c)),
    );
  const fixed = new Float64Array([
    ...size,
    ...origin,
    ...spacing,
    ...rows.flat(),
  ]);
  return {
    transformType: {
      transformParameterization: "DisplacementField",
      parametersValueType: Values === Float32Array ? "float32" : "float64",
      inputDimension: dimension,
      outputDimension: dimension,
    },
    name: "DisplacementFieldTransform",
    inputSpaceName: "",
    outputSpaceName: "",
    numberOfFixedParameters: fixed.length,
    numberOfParameters: parameters.length,
    fixedParameters: fixed,
    parameters,
    metadata: new Map(),
  } as unknown as Transform;
}

/** The vector the entry holds at ITK index `index`, in ITK component order. */
function vectorAt(entry: Transform, size: number[], index: number[]): number[] {
  const dimension = size.length;
  let flat = 0;
  let stride = 1;
  for (let axis = 0; axis < dimension; axis++) {
    flat += index[axis] * stride;
    stride *= size[axis];
  }
  const parameters = entry.parameters as unknown as ArrayLike<number>;
  return Array.from(
    { length: dimension },
    (_, c) => parameters[flat * dimension + c],
  );
}

async function fieldData(field: NgffImage): Promise<ArrayLike<number>> {
  return (await zarr.get(field.data, null)).data as ArrayLike<number>;
}

/** A geometry-only image on the grid a field is sampled on. */
async function frameImage(
  size: number[],
  spacing: number[],
  origin: number[],
  orientations?: Record<string, AnatomicalOrientation>,
): Promise<NgffImage> {
  const dims = CANONICAL[size.length];
  const itkOrder = ["x", "y", "z"].filter((dim) => dims.includes(dim));
  const shape = dims.map((dim) => size[itkOrder.indexOf(dim)]);
  const data = await zarr.create(zarr.root(new Map()).resolve("data"), {
    shape,
    chunk_shape: shape,
    data_type: "uint8",
    fill_value: 0,
  });
  return new NgffImage({
    data,
    dims,
    scale: Object.fromEntries(
      dims.map((dim) => [dim, spacing[itkOrder.indexOf(dim)]]),
    ),
    translation: Object.fromEntries(
      dims.map((dim) => [dim, origin[itkOrder.indexOf(dim)]]),
    ),
    name: "image",
    axesUnits: undefined,
    axesOrientations: orientations,
    computedCallbacks: undefined,
  });
}

/** An image's intrinsic point to ITK physical space, `D (q - o) + o`. */
function phi(
  image: NgffImage,
  pointItk: number[],
  itkDims: string[],
): number[] {
  const direction = directionRows(itkDirection(image, itkDims), itkDims.length);
  const origin = itkDims.map((dim) => image.translation[dim]);
  return direction.map((row, i) =>
    row.reduce((sum, value, j) => sum + value * (pointItk[j] - origin[j]), 0) +
    origin[i]
  );
}

function assertAllClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
) {
  assertEquals(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    assertAlmostEquals(actual[i], expected[i], 1e-9);
  }
}

Deno.test("components follow dims order", async () => {
  // One vector (dx=7, dy=9) at ITK index x=2, y=1. RFC-5 components follow
  // the axes of dims, so a yx field reads (9, 7) at [y=1, x=2] and an xy
  // field reads (7, 9) at [x=2, y=1].
  const entry = fieldTransform([3, 2], [0.5, 2.0], [10.0, 20.0]);
  (entry.parameters as unknown as Float64Array).fill(0);
  (entry.parameters as unknown as Float64Array).set([7, 9], (1 * 3 + 2) * 2);

  const yx = await itkDisplacementFieldToNgffTransform(entry, ["y", "x"], {
    path: "w",
  });
  const xy = await itkDisplacementFieldToNgffTransform(entry, ["x", "y"], {
    path: "w",
  });

  assertEquals(yx.transform, {
    type: "displacements",
    path: "w",
    interpolation: "linear",
  });
  assertEquals(yx.field.dims, ["c", "y", "x"]);
  assertEquals(yx.field.axesTypes, { c: "displacement" });
  assertEquals(yx.field.data.shape, [2, 2, 3]);
  const yxData = await fieldData(yx.field);
  // (c, y, x): component c at [1, 2] sits at c*6 + 1*3 + 2.
  assertEquals([yxData[0 * 6 + 5], yxData[1 * 6 + 5]], [9, 7]);
  assertEquals(
    Array.from(yxData as ArrayLike<number>).filter((v) => v !== 0).length,
    2,
  );
  assertEquals(yx.field.scale, { c: 1, y: 2.0, x: 0.5 });
  assertEquals(yx.field.translation, { c: 0, y: 20.0, x: 10.0 });

  assertEquals(xy.field.data.shape, [2, 3, 2]);
  const xyData = await fieldData(xy.field);
  // (c, x, y): component c at [2, 1] sits at c*6 + 2*2 + 1.
  assertEquals([xyData[0 * 6 + 5], xyData[1 * 6 + 5]], [7, 9]);
  assertEquals(xy.field.scale, { c: 1, x: 0.5, y: 2.0 });
});

for (const ndim of [2, 3]) {
  Deno.test(`round trip gives back the entry (${ndim}D)`, async () => {
    const size = [5, 4, 3].slice(0, ndim);
    const spacing = [0.5, 2.0, 1.5].slice(0, ndim);
    const origin = [10.0, 20.0, -3.0].slice(0, ndim);
    const dims = CANONICAL[ndim];
    const entry = fieldTransform(size, spacing, origin);

    const { transform, field } = await itkDisplacementFieldToNgffTransform(
      entry,
      dims,
      { path: "warp" },
    );
    assertEquals(field.data.shape, [ndim, ...size.slice().reverse()]);
    const itkOrder = ["x", "y", "z"].slice(0, ndim);
    for (const dim of dims) {
      assertEquals(field.scale[dim], spacing[itkOrder.indexOf(dim)]);
      assertEquals(field.translation[dim], origin[itkOrder.indexOf(dim)]);
    }

    const [back] = await ngffDisplacementFieldToItkTransform(
      transform,
      field,
      dims,
    );
    assertEquals(
      String(back.transformType.transformParameterization),
      "DisplacementField",
    );
    assertAllClose(
      back.fixedParameters as unknown as ArrayLike<number>,
      entry.fixedParameters as unknown as ArrayLike<number>,
    );
    assertAllClose(
      back.parameters as unknown as ArrayLike<number>,
      entry.parameters as unknown as ArrayLike<number>,
    );
  });
}

Deno.test("a vector image converts like the transform holding it", async () => {
  const size = [4, 3];
  const entry = fieldTransform(size, [1.0, 1.5], [2.0, -1.0], undefined, 3);
  const image = {
    imageType: {
      dimension: 2,
      componentType: "float64",
      pixelType: "Vector",
      components: 2,
    },
    name: "warp",
    origin: [2.0, -1.0],
    spacing: [1.0, 1.5],
    direction: new Float64Array([1, 0, 0, 1]),
    size,
    metadata: new Map(),
    data: entry.parameters,
  } as unknown as Image;

  const fromEntry = await itkDisplacementFieldToNgffTransform(
    entry,
    ["y", "x"],
    {
      path: "w",
    },
  );
  const fromImage = await itkDisplacementFieldToNgffTransform(
    image,
    ["y", "x"],
    {
      path: "w",
    },
  );
  const fromList = await itkDisplacementFieldToNgffTransform([entry], [
    "y",
    "x",
  ], {
    path: "w",
  });
  for (const candidate of [fromImage, fromList]) {
    assertEquals(candidate.transform, fromEntry.transform);
    assertEquals(candidate.field.dims, fromEntry.field.dims);
    assertEquals(candidate.field.scale, fromEntry.field.scale);
    assertEquals(candidate.field.translation, fromEntry.field.translation);
    assertAllClose(
      await fieldData(candidate.field),
      await fieldData(fromEntry.field),
    );
  }
});

for (const moving of ["same", "none"] as const) {
  Deno.test(`frames are applied point by point (moving: ${moving})`, async () => {
    // phi_out(q + d(q)) must equal phi_in(q) + v(q) at every grid point q,
    // whether the two images share a frame (d = D^-1 v) or not.
    const size = [4, 3, 5];
    const spacing = [1.0, 2.0, 0.5];
    const origin = [5.0, -2.0, 8.0];
    const dims = CANONICAL[3];
    const itkDims = ["x", "y", "z"];
    const fixed = await frameImage(size, spacing, origin, RAS);
    const movingImage = await frameImage(
      size,
      spacing,
      [1.0, 1.0, 1.0],
      moving === "same" ? RAS : undefined,
    );
    const directionIn = directionRows(itkDirection(fixed, itkDims), 3);
    assertEquals(directionIn, [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]);
    const entry = fieldTransform(size, spacing, origin, directionIn, 7);

    const { transform, field } = await itkDisplacementFieldToNgffTransform(
      entry,
      dims,
      { path: "warp", fixed, moving: movingImage },
    );
    assertEquals(field.axesOrientations, RAS);

    const data = await fieldData(field);
    const shape = field.data.shape;
    const voxels = shape[1] * shape[2] * shape[3];
    for (const index of [[0, 0, 0], [1, 2, 3], [4, 1, 0], [2, 2, 2]]) {
      // index is in dims order (z, y, x).
      const q = dims.map((dim, i) =>
        field.translation[dim] + field.scale[dim] * index[i]
      );
      const offset = (index[0] * shape[2] + index[1]) * shape[3] + index[2];
      const d = [0, 1, 2].map((c) => data[c * voxels + offset]);
      const qItk = [...q].reverse();
      const dItk = [...d].reverse();
      const v = vectorAt(entry, size, [...index].reverse());
      const expected = phi(fixed, qItk, itkDims).map((value, i) =>
        value + v[i]
      );
      assertAllClose(
        phi(movingImage, qItk.map((value, i) => value + dItk[i]), itkDims),
        expected,
      );
    }

    const [back] = await ngffDisplacementFieldToItkTransform(
      transform,
      field,
      dims,
      {
        fixed,
        moving: movingImage,
      },
    );
    assertAllClose(
      back.fixedParameters as unknown as ArrayLike<number>,
      entry.fixedParameters as unknown as ArrayLike<number>,
    );
    assertAllClose(
      back.parameters as unknown as ArrayLike<number>,
      entry.parameters as unknown as ArrayLike<number>,
    );
  });
}

Deno.test("the grid direction must match the fixed image", async () => {
  const size = [4, 3, 5];
  const flipped = fieldTransform(size, [1, 1, 1], [0, 0, 0], [
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1],
  ]);
  await assertRejects(
    () =>
      itkDisplacementFieldToNgffTransform(flipped, CANONICAL[3], { path: "w" }),
    Error,
    "non-identity direction",
  );
  const fixed = await frameImage(size, [1, 1, 1], [0, 0, 0]);
  await assertRejects(
    () =>
      itkDisplacementFieldToNgffTransform(flipped, CANONICAL[3], {
        path: "w",
        fixed,
        moving: fixed,
      }),
    Error,
    "not oriented like the fixed image",
  );
});

Deno.test("refusals", async () => {
  const entry = fieldTransform([3, 2], [1, 1], [0, 0]);
  const { transform, field } = await itkDisplacementFieldToNgffTransform(
    entry,
    ["y", "x"],
    { path: "w" },
  );

  await assertRejects(
    () =>
      itkDisplacementFieldToNgffTransform(entry, ["c", "y", "x"], {
        path: "w",
      }),
    Error,
    "spatial axes only",
  );
  await assertRejects(
    () =>
      itkDisplacementFieldToNgffTransform(entry, ["z", "y", "x"], {
        path: "w",
      }),
    Error,
    "name 3 axes",
  );
  await assertRejects(
    () =>
      itkDisplacementFieldToNgffTransform(entry, ["y", "x"], {
        path: "w",
        fixed: field,
      }),
    Error,
    "pass both fixed and moving",
  );
  await assertRejects(
    () =>
      itkDisplacementFieldToNgffTransform([entry, entry], ["y", "x"], {
        path: "w",
      }),
    Error,
    "transform list of 2 entries",
  );
  let message = "";
  try {
    itkTransformToNgffTransform(entry, ["y", "x"]);
  } catch (error) {
    message = (error as Error).message;
  }
  assertEquals(message.includes("itkDisplacementFieldToNgffTransform"), true);
  message = "";
  try {
    ngffTransformToItkTransform(transform, ["y", "x"]);
  } catch (error) {
    message = (error as Error).message;
  }
  assertEquals(message.includes("ngffDisplacementFieldToItkTransform"), true);
  await assertRejects(
    () => ngffDisplacementFieldToItkTransform(transform, field, ["x", "y"]),
    Error,
    "component axis first",
  );
  const untyped = new NgffImage({
    data: field.data,
    dims: field.dims,
    scale: field.scale,
    translation: field.translation,
    name: "untyped",
    axesUnits: undefined,
    computedCallbacks: undefined,
  });
  await assertRejects(
    () => ngffDisplacementFieldToItkTransform(transform, untyped, ["y", "x"]),
    Error,
    "exactly one axis of type 'displacement'",
  );
});

Deno.test("interpolation other than linear still converts", async () => {
  const entry = fieldTransform([3, 2], [1, 1], [0, 0]);
  const { field } = await itkDisplacementFieldToNgffTransform(
    entry,
    ["y", "x"],
    {
      path: "w",
    },
  );
  const nearest: Displacements = {
    type: "displacements",
    path: "w",
    interpolation: "nearest",
  };
  const warn = console.warn;
  const warnings: string[] = [];
  console.warn = (message: string) => warnings.push(message);
  try {
    const [back] = await ngffDisplacementFieldToItkTransform(nearest, field, [
      "y",
      "x",
    ]);
    assertAllClose(
      back.parameters as unknown as ArrayLike<number>,
      entry.parameters as unknown as ArrayLike<number>,
    );
  } finally {
    console.warn = warn;
  }
  assertEquals(warnings.length, 1);
  assertEquals(warnings[0].includes("'nearest' interpolation"), true);
});

Deno.test("single precision is preserved", async () => {
  const entry = fieldTransform(
    [3, 2],
    [1, 1],
    [0, 0],
    undefined,
    0,
    Float32Array,
  );
  const { transform, field } = await itkDisplacementFieldToNgffTransform(
    entry,
    ["y", "x"],
    { path: "w" },
  );
  assertEquals(field.data.dtype, "float32");
  const [back] = await ngffDisplacementFieldToItkTransform(transform, field, [
    "y",
    "x",
  ]);
  assertEquals(String(back.transformType.parametersValueType), "float32");
  assertEquals(back.parameters instanceof Float32Array, true);
});
