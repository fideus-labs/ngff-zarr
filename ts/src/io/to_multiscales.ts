import * as zarr from "zarrita";
import { NgffImage } from "../types/ngff_image.ts";
import { Multiscales } from "../types/multiscales.ts";
import { Methods } from "../types/methods.ts";
import type { MemoryStore } from "../io/from_ngff_zarr.ts";
import {
  createAxis,
  createDataset,
  createMetadata,
  createMultiscales,
} from "../utils/factory.ts";
import { getMethodMetadata } from "../utils/method_metadata.ts";

export interface ToNgffImageOptions {
  dims?: string[];
  scale?: Record<string, number>;
  translation?: Record<string, number>;
  name?: string;
}

/**
 * Convert array data to NgffImage
 *
 * @param data - Input data as typed array or regular array
 * @param options - Configuration options for NgffImage creation
 * @returns NgffImage instance
 */
export async function toNgffImage(
  data: ArrayLike<number> | number[][] | number[][][],
  options: ToNgffImageOptions = {},
): Promise<NgffImage> {
  const {
    dims = ["y", "x"],
    scale = {},
    translation = {},
    name = "image",
  } = options;

  // Determine data shape and create typed array
  let typedData: Float32Array;
  let shape: number[];

  if (Array.isArray(data)) {
    // Handle multi-dimensional arrays
    if (Array.isArray(data[0])) {
      if (Array.isArray((data[0] as unknown[])[0])) {
        // 3D array
        const d3 = data as number[][][];
        shape = [d3.length, d3[0].length, d3[0][0].length];
        typedData = new Float32Array(shape[0] * shape[1] * shape[2]);

        let idx = 0;
        for (let i = 0; i < shape[0]; i++) {
          for (let j = 0; j < shape[1]; j++) {
            for (let k = 0; k < shape[2]; k++) {
              typedData[idx++] = d3[i][j][k];
            }
          }
        }
      } else {
        // 2D array
        const d2 = data as number[][];
        shape = [d2.length, d2[0].length];
        typedData = new Float32Array(shape[0] * shape[1]);

        let idx = 0;
        for (let i = 0; i < shape[0]; i++) {
          for (let j = 0; j < shape[1]; j++) {
            typedData[idx++] = d2[i][j];
          }
        }
      }
    } else {
      // 1D array
      const d1 = data as unknown as number[];
      shape = [d1.length];
      typedData = new Float32Array(d1);
    }
  } else {
    // ArrayLike (already a typed array)
    typedData = new Float32Array(data as ArrayLike<number>);
    shape = [typedData.length];
  }

  // Adjust shape to match dims length
  while (shape.length < dims.length) {
    shape.unshift(1);
  }

  if (shape.length > dims.length) {
    throw new Error(
      `Data dimensionality (${shape.length}) exceeds dims length (${dims.length})`,
    );
  }

  // Create in-memory zarr store and array
  const store: MemoryStore = new Map();
  const root = zarr.root(store);

  // Calculate appropriate chunk size
  const chunkShape = shape.map((dim) => Math.min(dim, 256));

  const zarrArray = await zarr.create(root.resolve("data"), {
    shape,
    chunk_shape: chunkShape,
    data_type: "float32",
    fill_value: 0,
  });

  // Write data to zarr array
  await zarr.set(zarrArray, [], {
    data: typedData,
    shape,
    stride: calculateStride(shape),
  });

  // Create scale and translation records with defaults
  const fullScale: Record<string, number> = {};
  const fullTranslation: Record<string, number> = {};

  for (const dim of dims) {
    fullScale[dim] = scale[dim] ?? 1.0;
    fullTranslation[dim] = translation[dim] ?? 0.0;
  }

  return new NgffImage({
    data: zarrArray,
    dims,
    scale: fullScale,
    translation: fullTranslation,
    name,
    axesUnits: undefined,
    computedCallbacks: undefined,
  });
}

export interface ToMultiscalesOptions {
  scaleFactors?: (Record<string, number> | number)[];
  method?: Methods;
  chunks?: number | number[] | Record<string, number>;
}

/**
 * Generate multiple resolution scales for an NgffImage (simplified version for testing)
 *
 * @param image - Input NgffImage
 * @param options - Configuration options
 * @returns Multiscales object
 */
export function toMultiscales(
  image: NgffImage,
  options: ToMultiscalesOptions = {},
): Multiscales {
  const {
    scaleFactors = [2, 4],
    method = Methods.ITKWASM_GAUSSIAN,
    chunks: _chunks,
  } = options;

  // For now, create only the base image (no actual downsampling)
  // This is a simplified implementation for testing metadata functionality
  const images = [image];

  // Create axes from image dimensions
  const axes = image.dims.map((dim) => {
    if (dim === "x" || dim === "y" || dim === "z") {
      return createAxis(
        dim as "x" | "y" | "z",
        "space",
        image.axesUnits?.[dim],
      );
    } else if (dim === "c") {
      return createAxis(dim as "c", "channel");
    } else if (dim === "t") {
      return createAxis(dim as "t", "time");
    } else {
      throw new Error(`Unsupported dimension: ${dim}`);
    }
  });

  // Create datasets
  const datasets = [
    createDataset(
      "0",
      image.dims.map((dim) => image.scale[dim]),
      image.dims.map((dim) => image.translation[dim]),
    ),
  ];

  // Create metadata with method information
  const methodMetadata = getMethodMetadata(method);
  const metadata = createMetadata(axes, datasets, image.name);
  metadata.type = method;
  if (methodMetadata) {
    metadata.metadata = methodMetadata;
  }

  return createMultiscales(images, metadata, scaleFactors, method);
}

function calculateStride(shape: number[]): number[] {
  const stride = new Array(shape.length);
  stride[shape.length - 1] = 1;
  for (let i = shape.length - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }
  return stride;
}
