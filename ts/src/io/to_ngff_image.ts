import * as zarr from "zarrita";
import { DEFAULT_CODECS } from "../utils/codecs.ts";
import { NgffImage } from "../types/ngff_image.ts";
import type { MemoryStore } from "../io/from_ngff_zarr.ts";

export interface ToNgffImageOptions {
  dims?: string[];
  scale?: Record<string, number>;
  translation?: Record<string, number>;
  name?: string;
  shape?: number[]; // Explicit shape for typed arrays
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
    shape: explicitShape,
  } = options;

  // Determine data shape and create typed array
  let typedData: Float32Array | Uint8Array | Uint16Array;
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
    // Use explicit shape if provided, otherwise infer from data length and dims
    if (explicitShape) {
      shape = [...explicitShape];
    } else {
      // Try to infer shape - this is a best guess
      shape = [data.length];
    }

    // Preserve the original typed array type
    if (data instanceof Uint8Array) {
      typedData = data;
    } else if (data instanceof Uint16Array) {
      typedData = data;
    } else {
      typedData = new Float32Array(data as ArrayLike<number>);
    }
  }

  // Adjust shape to match dims length if not explicitly provided
  if (!explicitShape) {
    while (shape.length < dims.length) {
      shape.unshift(1);
    }
  }

  if (shape.length !== dims.length) {
    throw new Error(
      `Shape dimensionality (${shape.length}) must match dims length (${dims.length})`,
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
    codecs: [...DEFAULT_CODECS],
  });

  // Write data to zarr array
  await zarr.set(zarrArray, [], {
    data: typedData as Float32Array,
    shape,
    stride: calculateStride(shape),
  });

  // Create scale and translation records with defaults
  const fullScale: Record<string, number> = {};
  const fullTranslation: Record<string, number> = {};
  const spatialDims = new Set(["x", "y", "z"]);

  for (const dim of dims) {
    // Only set defaults for spatial dimensions
    if (spatialDims.has(dim)) {
      fullScale[dim] = scale[dim] ?? 1.0;
      fullTranslation[dim] = translation[dim] ?? 0.0;
    } else {
      // For non-spatial dimensions, only include if explicitly provided
      if (scale[dim] !== undefined) {
        fullScale[dim] = scale[dim];
      }
      if (translation[dim] !== undefined) {
        fullTranslation[dim] = translation[dim];
      }
    }
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

function calculateStride(shape: number[]): number[] {
  const stride = new Array(shape.length);
  stride[shape.length - 1] = 1;
  for (let i = shape.length - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }
  return stride;
}
