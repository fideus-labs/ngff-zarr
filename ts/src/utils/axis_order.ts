// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Normalization of an image's axes to the OME-Zarr specification order.
 *
 * Mirrors `py/ngff_zarr/methods/_support.py`.
 */

import * as zarr from "zarrita";

import { NgffImage } from "../types/ngff_image.ts";
import type { ZarrCodec } from "./codecs.ts";
import { defaultCodecs } from "./codecs.ts";
import {
  calculateStride,
  componentTypeOf,
  transposeArray,
} from "./transpose.ts";
import { zarrGet, zarrSet } from "./worker_pool.ts";

/** The OME-Zarr specification axis order: time, then channel, then space. */
export const CANONICAL_AXIS_ORDER = ["t", "c", "z", "y", "x"];

/**
 * Return `image` with its dims in the spec axis order `(t, c, z, y, x)`.
 *
 * OME-Zarr requires axes ordered by type: time, then channel, then space.
 * Conversion sources produce channel-last layouts, so the multiscale pipeline
 * normalizes them here and the generated metadata is spec-ordered.
 *
 * An image whose dims fall outside the canonical set is returned unchanged:
 * an axis model that was not expressible before RFC-3 carries no spec ordering
 * to normalize to.
 *
 * Where the Python port transposes lazily through dask, this reads the array
 * and writes the permuted buffer into a new in-memory zarr array. The
 * downsampling path materializes the image anyway.
 *
 * @param image - The image to normalize.
 * @param codecs - Codec pipeline for the reordered array; defaults to
 * {@link defaultCodecs}.
 * @returns The image itself when already ordered, otherwise a reordered copy.
 */
export async function canonicalAxisOrder(
  image: NgffImage,
  codecs?: ZarrCodec[],
): Promise<NgffImage> {
  const dims = image.dims;
  const newDims = CANONICAL_AXIS_ORDER.filter((dim) => dims.includes(dim));
  if (
    newDims.length !== dims.length ||
    newDims.every((dim, index) => dim === dims[index])
  ) {
    return image;
  }

  const permutation = newDims.map((dim) => dims.indexOf(dim));
  const result = await zarrGet(image.data);
  const componentType = componentTypeOf(result.data);
  const transposed = transposeArray(
    result.data,
    [...result.shape],
    permutation,
    componentType,
  );

  const shape = permutation.map((index) => result.shape[index]);
  const chunkShape = permutation.map((index) => image.data.chunks[index]);
  const store: Map<string, Uint8Array> = new Map();
  const array = await zarr.create(zarr.root(store).resolve("/0"), {
    shape,
    chunk_shape: chunkShape,
    data_type: image.data.dtype,
    fill_value: 0,
    codecs: codecs ?? defaultCodecs(image.data.dtype),
  });
  await zarrSet(array, shape.map(() => null), {
    data: transposed,
    shape,
    stride: calculateStride(shape),
  });

  return new NgffImage({
    data: array as zarr.Array<zarr.DataType, zarr.Readable>,
    dims: newDims,
    scale: image.scale,
    translation: image.translation,
    name: image.name,
    axesUnits: image.axesUnits,
    axesOrientations: image.axesOrientations,
    axesTypes: image.axesTypes,
    computedCallbacks: image.computedCallbacks,
  });
}
