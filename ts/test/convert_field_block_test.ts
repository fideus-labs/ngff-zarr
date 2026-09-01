// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Converting one block of a field between ITK's convention and RFC-5's.
 *
 * Mirrors `py/test/test_convert_field_block.py`. The reference is the
 * whole-field converter itself, so the two cannot drift apart: a block
 * converted where it sits must equal that block of the converted whole. The
 * framed cases are the ones that matter -- without frames the positional term
 * is zero and any axis order passes.
 */

import { assertAlmostEquals, assertThrows } from "@std/assert";
import type { Transform } from "itk-wasm";
import * as zarr from "zarrita";
import {
  convertFieldBlock,
  itkDisplacementFieldToNgffTransform,
  NgffImage,
} from "../src/mod.ts";
import { type AnatomicalOrientation, LPS, RAS } from "../src/types/rfc4.ts";

const CANONICAL: Record<number, string[]> = {
  2: ["y", "x"],
  3: ["z", "y", "x"],
};

function noise(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return (state / 0xffffffff) * 6 - 3;
  };
}

/** A random deformation on a grid of `size` (ITK order), as a transform. */
function fieldTransform(
  size: number[],
  spacing: number[],
  origin: number[],
  seed = 0,
): Transform {
  const dimension = size.length;
  const voxels = size.reduce((a, b) => a * b, 1);
  const next = noise(seed + 1);
  const parameters = new Float64Array(voxels * dimension);
  for (let i = 0; i < parameters.length; i++) parameters[i] = next();
  const rows = Array.from(
    { length: dimension },
    (_, r) => Array.from({ length: dimension }, (_, c) => +(r === c)),
  );
  return {
    transformType: {
      transformParameterization: "DisplacementField",
      parametersValueType: "float64",
      inputDimension: dimension,
      outputDimension: dimension,
    },
    numberOfFixedParameters: 3 * dimension + dimension * dimension,
    numberOfParameters: parameters.length,
    fixedParameters: new Float64Array([
      ...size,
      ...origin,
      ...spacing,
      ...rows.flat(),
    ]),
    parameters,
    name: "DisplacementFieldTransform",
  } as unknown as Transform;
}

/** A geometry-only image on the grid the field is sampled on. */
async function frameImage(
  dims: string[],
  orientations?: Record<string, AnatomicalOrientation>,
): Promise<NgffImage> {
  const shape = dims.map(() => 4);
  const data = await zarr.create(zarr.root(new Map()).resolve("frame"), {
    shape,
    chunk_shape: shape,
    data_type: "uint8",
    fill_value: 0,
  });
  return new NgffImage({
    data,
    dims,
    scale: Object.fromEntries(dims.map((dim) => [dim, 1])),
    translation: Object.fromEntries(dims.map((dim) => [dim, 0])),
    name: "frame",
    axesUnits: undefined,
    axesOrientations: orientations,
    computedCallbacks: undefined,
  });
}

/**
 * The block a transform holds, channel-first with ITK components and the
 * spatial axes in `dims` order.
 *
 * ITK lays a field out canonically (`[z][y][x]`, its size given fastest-axis
 * first) and interleaves the components; the block API takes the spatial axes
 * in the caller's own `dims` order, so this is where the two meet. For
 * canonical `dims` it is a de-interleave and nothing more.
 */
function itkBlock(
  transform: Transform,
  dims: string[],
  size: number[],
): Float64Array {
  const dimension = dims.length;
  const values =
    (transform as unknown as { parameters: Float64Array }).parameters;
  const voxels = values.length / dimension;
  const itkOrder = ["x", "y", "z"].slice(0, dimension);
  const canonical = [...itkOrder].reverse();
  const canonicalShape = canonical.map((dim) => size[itkOrder.indexOf(dim)]);
  const dimsShape = dims.map((dim) => size[itkOrder.indexOf(dim)]);
  const strideOf = (shape: number[]) => {
    const out = new Array<number>(shape.length);
    let step = 1;
    for (let axis = shape.length - 1; axis >= 0; axis--) {
      out[axis] = step;
      step *= shape[axis];
    }
    return out;
  };
  const canonicalStrides = strideOf(canonicalShape);
  const dimsStrides = strideOf(dimsShape);
  const out = new Float64Array(values.length);
  for (let voxel = 0; voxel < voxels; voxel++) {
    let offset = 0;
    for (let axis = 0; axis < dimension; axis++) {
      const index = Math.floor(voxel / canonicalStrides[axis]) %
        canonicalShape[axis];
      offset += index * dimsStrides[dims.indexOf(canonical[axis])];
    }
    for (let axis = 0; axis < dimension; axis++) {
      out[axis * voxels + offset] = values[voxel * dimension + axis];
    }
  }
  return out;
}

for (const ndim of [2, 3]) {
  Deno.test(`blockwise equals the whole-field converter (${ndim}D)`, async () => {
    const dims = CANONICAL[ndim];
    const size = [6, 5, 4].slice(0, ndim);
    const spacing = [1.0, 1.5, 2.0].slice(0, ndim);
    const origin = [7.0, -3.0, 5.0].slice(0, ndim);
    const transform = fieldTransform(size, spacing, origin, 3);
    const { field } = await itkDisplacementFieldToNgffTransform(
      transform,
      dims,
      {
        path: "DVF",
      },
    );
    const reference = (await zarr.get(field.data, null)).data as ArrayLike<
      number
    >;

    const converted = convertFieldBlock(
      itkBlock(transform, dims, size),
      [...field.data.shape],
      dims,
      field.translation,
      field.scale,
    );
    for (let i = 0; i < reference.length; i++) {
      assertAlmostEquals(converted[i], reference[i], 1e-12);
    }
  });
}

for (const dims of [["z", "y", "x"], ["x", "y", "z"]]) {
  Deno.test(
    `the positional term stays on its own axes for dims ${dims.join("")}`,
    async () => {
      // The framed case: without frames the term is zero and any order passes.
      const size = [6, 5, 4];
      const transform = fieldTransform(
        size,
        [1.0, 1.5, 2.0],
        [7.0, -3.0, 5.0],
        3,
      );
      const fixed = await frameImage(dims, LPS);
      const moving = await frameImage(dims, RAS);
      const { field } = await itkDisplacementFieldToNgffTransform(
        transform,
        dims,
        {
          path: "DVF",
          fixed,
          moving,
        },
      );
      const reference = (await zarr.get(field.data, null)).data as ArrayLike<
        number
      >;

      const converted = convertFieldBlock(
        itkBlock(transform, dims, size),
        [...field.data.shape],
        dims,
        field.translation,
        field.scale,
        { fixed, moving },
      );
      for (let i = 0; i < reference.length; i++) {
        assertAlmostEquals(converted[i], reference[i], 1e-12);
      }
    },
  );
}

Deno.test("forward then inverse is the identity", async () => {
  const dims = CANONICAL[3];
  const shape = [3, 4, 5, 6];
  const next = noise(7);
  const block = new Float64Array(shape.reduce((a, b) => a * b, 1));
  for (let i = 0; i < block.length; i++) block[i] = next();
  const geometry = {
    translation: { z: 5.0, y: -3.0, x: 7.0 },
    scale: { z: 2.0, y: 1.5, x: 1.0 },
  };
  const fixed = await frameImage(dims);
  const moving = await frameImage(dims);
  const stored = convertFieldBlock(
    block,
    shape,
    dims,
    geometry.translation,
    geometry.scale,
    { fixed, moving },
  );
  const back = convertFieldBlock(
    stored,
    shape,
    dims,
    geometry.translation,
    geometry.scale,
    { fixed, moving, inverse: true },
  );
  for (let i = 0; i < block.length; i++) {
    assertAlmostEquals(back[i], block[i], 1e-12);
  }
});

Deno.test("a block shape that is not the grid is refused", () => {
  assertThrows(
    () =>
      convertFieldBlock(
        new Float64Array(12),
        [3, 4],
        CANONICAL[3],
        { z: 0, y: 0, x: 0 },
        { z: 1, y: 1, x: 1 },
      ),
    Error,
    "one component axis",
  );
});

Deno.test("geometry that does not cover the axes is refused", () => {
  assertThrows(
    () =>
      convertFieldBlock(
        new Float64Array(3 * 4 * 5 * 6),
        [3, 4, 5, 6],
        CANONICAL[3],
        { z: 0, y: 0 },
        { z: 1, y: 1, x: 1 },
      ),
    Error,
    "translation and scale",
  );
});
