import * as zarr from "zarrita";
import { defaultCodecs } from "../utils/codecs.ts";
import { NgffImage } from "../types/ngff_image.ts";
import type { MemoryStore } from "../io/from_ngff_zarr.ts";
import { calculateStride } from "../utils/transpose.ts";
import type { NumericTypedArray } from "../utils/transpose.ts";
import { zarrSet } from "../utils/worker_pool.ts";

export interface ToNgffImageOptions {
  dims?: string[];
  scale?: Record<string, number>;
  translation?: Record<string, number>;
  name?: string;
  shape?: number[]; // Explicit shape for typed arrays
  axesTypes?: Record<string, string> | undefined;
}

/** A buffer paired with the zarr data type its elements are stored as. */
interface TypedInput {
  data: NumericTypedArray;
  dataType: zarr.DataType;
}

/**
 * Pair a typed array with the zarr data type of its elements.
 *
 * The buffer reaches the store as raw bytes sized by the array's declared
 * `data_type`, so the type named here has to be the type the buffer holds.
 *
 * Covers the typed arrays whose elements are numbers and that this package
 * can encode. A generic `ArrayLike` or a `Float16Array` returns `null`, and
 * the caller materializes a float32 copy. The 64-bit integer arrays hold
 * `bigint` rather than `number`, so they fall outside the declared input
 * type.
 */
function typedInputOf(data: ArrayLike<number>): TypedInput | null {
  if (data instanceof Uint8Array) return { data, dataType: "uint8" };
  if (data instanceof Uint8ClampedArray) {
    // Elements are 8-bit, but zarr writes "uint8" from a Uint8Array.
    return { data: new Uint8Array(data), dataType: "uint8" };
  }
  if (data instanceof Int8Array) return { data, dataType: "int8" };
  if (data instanceof Uint16Array) return { data, dataType: "uint16" };
  if (data instanceof Int16Array) return { data, dataType: "int16" };
  if (data instanceof Uint32Array) return { data, dataType: "uint32" };
  if (data instanceof Int32Array) return { data, dataType: "int32" };
  if (data instanceof Float32Array) return { data, dataType: "float32" };
  if (data instanceof Float64Array) return { data, dataType: "float64" };
  return null;
}

/**
 * Convert array data to NgffImage
 *
 * @param data - Input data as typed array or regular array. A typed array's
 * element type is preserved: the zarr array is created with the matching
 * `data_type` and holds the caller's values. Plain JavaScript arrays, and
 * any other `ArrayLike`, are converted to float32.
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
    axesTypes,
  } = options;

  // Determine data shape and create typed array
  let typedData: NumericTypedArray;
  let dataType: zarr.DataType = "float32";
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
    const typedInput = typedInputOf(data);
    if (typedInput) {
      typedData = typedInput.data;
      dataType = typedInput.dataType;
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
    data_type: dataType,
    fill_value: 0,
    codecs: defaultCodecs(dataType),
  });

  // Write data to zarr array; a null selection targets the full array (an
  // empty selection list writes nothing).
  await zarrSet(zarrArray, null, {
    data: typedData,
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
    axesTypes,
    computedCallbacks: undefined,
  });
}
