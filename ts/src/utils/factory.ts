// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import * as zarr from "zarrita";
import { defaultCodecs } from "../utils/codecs.ts";
import { NgffImage } from "../types/ngff_image.ts";
import { NgffMultiscales } from "../types/multiscales.ts";
import type {
  Axis,
  AxisOrientation,
  Dataset,
  Metadata,
  Scale,
  Translation,
} from "../types/zarr_metadata.ts";
import type { AxesType, SupportedDims, Units } from "../types/units.ts";
import type { Methods } from "../types/methods.ts";
import type { AnatomicalOrientation } from "../types/rfc4.ts";

// Create a zarr.Array for testing using an in-memory store
async function createTestZarrArray(
  shape: number[],
  dtype: string,
  chunks: number[],
  name: string,
): Promise<zarr.Array<zarr.DataType, zarr.Readable>> {
  const store = new Map<string, Uint8Array>();
  const root = zarr.root(store);

  // Create the array using zarrita with correct options
  const array = await zarr.create(root.resolve(name), {
    shape,
    chunk_shape: chunks,
    data_type: dtype as zarr.DataType,
    codecs: defaultCodecs(dtype),
  });

  return array;
}

export async function createNgffImage(
  _data: ArrayBuffer | number[],
  shape: number[],
  dtype: string,
  dims: string[],
  scale: Record<string, number>,
  translation: Record<string, number>,
  name = "image",
  axesTypes?: Record<string, string>,
): Promise<NgffImage> {
  const zarrArray = await createTestZarrArray(shape, dtype, shape, name);

  return new NgffImage({
    data: zarrArray,
    dims,
    scale,
    translation,
    name,
    axesUnits: undefined,
    axesTypes,
    computedCallbacks: undefined,
  });
}

export function createAxis(
  name: SupportedDims,
  type: AxesType,
  unit?: Units,
  orientation?: AxisOrientation | AnatomicalOrientation,
): Axis {
  const axis: Axis = {
    name,
    type,
    unit: unit || undefined,
  };
  if (orientation) {
    axis.orientation = orientation;
  }
  return axis;
}

export function createScale(scale: number[]): Scale {
  return {
    scale: [...scale],
    type: "scale",
  };
}

export function createTranslation(translation: number[]): Translation {
  return {
    translation: [...translation],
    type: "translation",
  };
}

export function createDataset(
  path: string,
  scale: number[],
  translation: number[],
): Dataset {
  return {
    path,
    coordinateTransformations: [
      createScale(scale),
      createTranslation(translation),
    ],
  };
}

export function createMetadata(
  axes: Axis[],
  datasets: Dataset[],
  name = "image",
  version = "0.4",
): Metadata {
  return {
    axes: [...axes],
    datasets: [...datasets],
    coordinateTransformations: undefined,
    omero: undefined,
    name,
    version,
  };
}

export function createNgffMultiscales(
  images: NgffImage[],
  metadata: Metadata,
  scaleFactors?: (Record<string, number> | number)[],
  method?: Methods,
): NgffMultiscales {
  return new NgffMultiscales({
    images: [...images],
    metadata,
    scaleFactors: scaleFactors || undefined,
    method: method || undefined,
    chunks: undefined,
  });
}

// Backwards-compatible alias
export const createMultiscales = createNgffMultiscales;
