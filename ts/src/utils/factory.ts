// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import * as zarr from "zarrita";
import { NgffImage } from "../types/ngff_image.ts";
import { Multiscales } from "../types/multiscales.ts";
import type {
  Axis,
  Dataset,
  Metadata,
  Scale,
  Translation,
} from "../types/zarr_metadata.ts";
import type { AxesType, SupportedDims, Units } from "../types/units.ts";
import type { Methods } from "../types/methods.ts";

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
): Promise<NgffImage> {
  const zarrArray = await createTestZarrArray(shape, dtype, shape, name);

  return new NgffImage({
    data: zarrArray,
    dims,
    scale,
    translation,
    name,
    axesUnits: undefined,
    computedCallbacks: undefined,
  });
}

export function createAxis(
  name: SupportedDims,
  type: AxesType,
  unit?: Units,
): Axis {
  return {
    name,
    type,
    unit: unit || undefined,
  };
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

export function createMultiscales(
  images: NgffImage[],
  metadata: Metadata,
  scaleFactors?: (Record<string, number> | number)[],
  method?: Methods,
): Multiscales {
  return new Multiscales({
    images: [...images],
    metadata,
    scaleFactors: scaleFactors || undefined,
    method: method || undefined,
    chunks: undefined,
  });
}
