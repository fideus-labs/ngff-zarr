// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Shared helper functions for ITK-Wasm downsampling
 * Used by both browser and Node implementations
 */

import { castImage } from "itk-wasm";
import type { Image } from "itk-wasm";
import * as zarr from "zarrita";

import type { NgffImage } from "../types/ngff_image.ts";
import type { ZarrCodec } from "../utils/codecs.ts";
import { defaultCodecs } from "../utils/codecs.ts";
import { zarrGet, zarrSet } from "../utils/worker_pool.ts";
import {
  calculateStride,
  componentTypeOf,
  transposeArray,
} from "../utils/transpose.ts";

export { calculateStride, transposeArray };

export const SPATIAL_DIMS = ["x", "y", "z"];

/**
 * Maximum number of components to use vector mode for itkwasm downsampling.
 * Images with more components will iterate over channels individually to avoid
 * itkwasm limitations with VariableLengthVector images.
 */
export const MAX_VECTOR_COMPONENTS = 8;

export interface DimFactors {
  [key: string]: number;
}

/**
 * Calculate the incremental factor needed to reach the target size from the
 * previous size, so downsampling incrementally lands on exact target sizes.
 *
 * `nominal` is the requested factor relative to the previous level, floored
 * when the ladder step is not an integer: a `[2, 5]` ladder steps by 2.5 and
 * floors to 2. It is kept whenever `floor(previousSize / nominal)` is the
 * target, so a level asked for at 4x is shrunk 4x. Flooring the step can only
 * leave the level larger than the target, and the guard keeps it only when it
 * lands on the target exactly, so a floored step never shrinks past the
 * requested size. The search below runs only when the nominal factor misses
 * the target, which happens on ladders such as `[2, 3]`.
 *
 * Mirrors the Python port's `_incremental_factor`.
 */
export function calculateIncrementalFactor(
  previousSize: number,
  targetSize: number,
  nominal?: number,
): number {
  if (targetSize <= 0) {
    return 1;
  }

  if (
    nominal !== undefined && nominal >= 1 &&
    Math.floor(previousSize / nominal) === targetSize
  ) {
    return nominal;
  }

  // Start with the theoretical factor
  let factor = Math.floor(Math.ceil(previousSize / (targetSize + 0.5)));

  // Verify this gives us the right size
  let actualSize = Math.floor(previousSize / factor);
  if (actualSize !== targetSize) {
    // Adjust factor to get exact target
    factor = Math.max(1, Math.floor(previousSize / targetSize));
    actualSize = Math.floor(previousSize / factor);

    // If still not exact, try ceil
    if (actualSize !== targetSize) {
      factor = Math.max(1, Math.ceil(previousSize / targetSize));
    }
  }

  return Math.max(1, factor);
}

/**
 * Convert dimension scale factors to ITK-Wasm format
 * This computes the incremental scale factor relative to the previous scale,
 * not the absolute scale factor from the original image.
 *
 * When originalImage and previousImage are provided, calculates the exact
 * incremental factor needed to reach the target size from the previous size.
 * This ensures we get exact 1x, 2x, 3x, 4x sizes even with incremental downsampling.
 */
export function dimScaleFactors(
  dims: string[],
  scaleFactor: Record<string, number> | number,
  previousDimFactors: DimFactors,
  originalImage?: NgffImage,
  previousImage?: NgffImage,
): DimFactors {
  const dimFactors: DimFactors = {};

  if (typeof scaleFactor === "number") {
    if (originalImage !== undefined && previousImage !== undefined) {
      // Calculate target size: floor(original_size / scale_factor)
      // Then calculate incremental factor from previous size to target size
      for (const dim of dims) {
        if (SPATIAL_DIMS.includes(dim)) {
          const dimIndex = originalImage.dims.indexOf(dim);
          const originalSize = originalImage.data.shape[dimIndex];
          const targetSize = Math.floor(originalSize / scaleFactor);

          const prevDimIndex = previousImage.dims.indexOf(dim);
          const previousSize = previousImage.data.shape[prevDimIndex];

          const nominal = Math.floor(
            scaleFactor / (previousDimFactors[dim] || 1),
          );
          dimFactors[dim] = calculateIncrementalFactor(
            previousSize,
            targetSize,
            nominal,
          );
        } else {
          dimFactors[dim] = 1;
        }
      }
    } else {
      // Fallback to old behavior when images not provided
      for (const dim of dims) {
        if (SPATIAL_DIMS.includes(dim)) {
          // Divide by previous factor to get incremental scaling
          // Use Math.floor to truncate (matching Python's int() behavior)
          const incrementalFactor = scaleFactor /
            (previousDimFactors[dim] || 1);
          dimFactors[dim] = Math.max(1, Math.floor(incrementalFactor));
        } else {
          dimFactors[dim] = previousDimFactors[dim] || 1;
        }
      }
    }
  } else {
    if (originalImage !== undefined && previousImage !== undefined) {
      for (const dim in scaleFactor) {
        const dimIndex = originalImage.dims.indexOf(dim);
        const originalSize = originalImage.data.shape[dimIndex];
        const targetSize = Math.floor(originalSize / scaleFactor[dim]);

        const prevDimIndex = previousImage.dims.indexOf(dim);
        const previousSize = previousImage.data.shape[prevDimIndex];

        const nominal = Math.floor(
          scaleFactor[dim] / (previousDimFactors[dim] || 1),
        );
        dimFactors[dim] = calculateIncrementalFactor(
          previousSize,
          targetSize,
          nominal,
        );
      }
    } else {
      // Fallback to old behavior when images not provided
      for (const dim in scaleFactor) {
        // Divide by previous factor to get incremental scaling
        // Use Math.floor to truncate (matching Python's int() behavior)
        const incrementalFactor = scaleFactor[dim] /
          (previousDimFactors[dim] || 1);
        dimFactors[dim] = Math.max(1, Math.floor(incrementalFactor));
      }
    }
    // Add dims not in scale_factor with factor of 1
    for (const dim of dims) {
      if (!(dim in dimFactors)) {
        dimFactors[dim] = 1;
      }
    }
  }

  return dimFactors;
}

/**
 * Update previous dimension factors
 */
export function updatePreviousDimFactors(
  scaleFactor: Record<string, number> | number,
  spatialDims: string[],
  previousDimFactors: DimFactors,
): DimFactors {
  const updated = { ...previousDimFactors };

  if (typeof scaleFactor === "number") {
    for (const dim of spatialDims) {
      updated[dim] = scaleFactor;
    }
  } else {
    for (const dim in scaleFactor) {
      updated[dim] = scaleFactor[dim];
    }
  }

  return updated;
}

/**
 * Compute next scale metadata
 */
export function nextScaleMetadata(
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[],
): [Record<string, number>, Record<string, number>] {
  const translation: Record<string, number> = {};
  const scale: Record<string, number> = {};

  for (const dim of image.dims) {
    if (spatialDims.includes(dim)) {
      const factor = dimFactors[dim];
      scale[dim] = image.scale[dim] * factor;
      // Add offset to account for pixel center shift when downsampling
      translation[dim] = image.translation[dim] +
        0.5 * (factor - 1) * image.scale[dim];
    } else {
      scale[dim] = image.scale[dim];
      translation[dim] = image.translation[dim];
    }
  }

  return [translation, scale];
}

/**
 * Copy typed array to appropriate type
 */
function copyTypedArray(
  data: unknown,
):
  | Float32Array
  | Float64Array
  | Uint8Array
  | Int8Array
  | Uint16Array
  | Int16Array
  | Uint32Array
  | Int32Array {
  if (data instanceof Float32Array) {
    return new Float32Array(data);
  } else if (data instanceof Float64Array) {
    return new Float64Array(data);
  } else if (data instanceof Uint8Array) {
    return new Uint8Array(data);
  } else if (data instanceof Int8Array) {
    return new Int8Array(data);
  } else if (data instanceof Uint16Array) {
    return new Uint16Array(data);
  } else if (data instanceof Int16Array) {
    return new Int16Array(data);
  } else if (data instanceof Uint32Array) {
    return new Uint32Array(data);
  } else if (data instanceof Int32Array) {
    return new Int32Array(data);
  } else {
    // Convert to Float32Array as fallback
    return new Float32Array(data as ArrayLike<number>);
  }
}

/**
 * Integer component types eligible for the Gaussian float32 workaround
 */
export type IntegerComponentType =
  | "uint8"
  | "int8"
  | "uint16"
  | "int16"
  | "uint32"
  | "int32";

const INTEGER_COMPONENT_RANGES: Record<
  IntegerComponentType,
  [number, number]
> = {
  uint8: [0, 255],
  int8: [-128, 127],
  uint16: [0, 65535],
  int16: [-32768, 32767],
  uint32: [0, 4294967295],
  int32: [-2147483648, 2147483647],
};

export function isIntegerComponentType(
  componentType: unknown,
): componentType is IntegerComponentType {
  return typeof componentType === "string" &&
    Object.hasOwn(INTEGER_COMPONENT_RANGES, componentType);
}

/**
 * Round half to even, matching NumPy's rint used by the Python port.
 */
export function rintHalfToEven(value: number): number {
  const floor = Math.floor(value);
  const frac = value - floor;
  if (frac < 0.5) return floor;
  if (frac > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * Cast an ITK-Wasm image to float32.
 *
 * itkwasm-downsample's Gaussian filter performs internal arithmetic that
 * loses precision on integer inputs (e.g. a uint16 input of 1 produces 0).
 * The Python port casts integer inputs to float32 before the Gaussian
 * downsample, then rounds and casts back; this helper and
 * castImageToIntegerType mirror that so both ports produce identical
 * pixel values.
 */
export function castImageToFloat32(image: Image): Image {
  if (image.data === null) {
    throw new Error("Image data is null");
  }
  return castImage(image, { componentType: "float32" });
}

/**
 * Round a float image back to the given integer type, clamping to the
 * type's range. Matches np.clip(np.rint(data), min, max) in the Python
 * port's float workaround.
 *
 * castImage performs the type conversion; it truncates raw floats, so the
 * values are rounded and clamped first. The intermediate is float64 because
 * float32 cannot represent every uint32/int32 value exactly.
 */
export function castImageToIntegerType(
  image: Image,
  componentType: IntegerComponentType,
): Image {
  const src = image.data;
  if (src === null) {
    throw new Error("Image data is null");
  }
  const [min, max] = INTEGER_COMPONENT_RANGES[componentType];
  const rounded = new Float64Array(src.length);
  for (let i = 0; i < src.length; i++) {
    let value = rintHalfToEven(src[i] as number);
    if (value < min) {
      value = min;
    } else if (value > max) {
      value = max;
    }
    rounded[i] = value;
  }
  return castImage(
    {
      ...image,
      imageType: { ...image.imageType, componentType: "float64" },
      data: rounded,
    },
    { componentType },
  );
}

/**
 * Create identity matrix for ITK direction
 */
export function createIdentityMatrix(dimension: number): Float64Array {
  const matrix = new Float64Array(dimension * dimension);
  for (let i = 0; i < dimension * dimension; i++) {
    matrix[i] = i % (dimension + 1) === 0 ? 1 : 0;
  }
  return matrix;
}

/**
 * Convert zarr array to ITK-Wasm Image format
 * If isVector is true, ensures "c" dimension is last by transposing if needed
 */
export async function zarrToItkImage(
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  dims: string[],
  isVector = false,
): Promise<Image> {
  // Read the full array data
  const result = await zarrGet(array);

  // Ensure we have the data
  if (!result.data || result.data.length === 0) {
    throw new Error("Zarr array data is empty");
  }

  let data:
    | Float32Array
    | Float64Array
    | Uint8Array
    | Int8Array
    | Uint16Array
    | Int16Array
    | Uint32Array
    | Int32Array;
  let shape = result.shape;
  let _finalDims = dims;

  // If vector image, ensure "c" is last dimension
  if (isVector) {
    const cIndex = dims.indexOf("c");
    if (cIndex !== -1 && cIndex !== dims.length - 1) {
      // Need to transpose to move "c" to the end
      const permutation = dims.map((_, i) => i).filter((i) => i !== cIndex);
      permutation.push(cIndex);

      // Reorder dims
      _finalDims = permutation.map((i) => dims[i]);

      // Reorder shape
      shape = permutation.map((i) => result.shape[i]);

      // Transpose the data
      data = transposeArray(
        result.data,
        result.shape,
        permutation,
        componentTypeOf(result.data),
      );
    } else {
      // "c" already at end or not present, just copy data
      data = copyTypedArray(result.data);
    }
  } else {
    // Not a vector image, just copy data
    data = copyTypedArray(result.data);
  }

  // For vector images, the last dimension is the component count, not a spatial dimension
  const spatialShape = isVector ? shape.slice(0, -1) : shape;
  const components = isVector ? shape[shape.length - 1] : 1;

  // ITK expects size in physical space order [x, y, z], but spatialShape is in array order [z, y, x]
  // So we need to reverse it
  const itkSize = [...spatialShape].reverse();

  // Create ITK-Wasm image
  const itkImage: Image = {
    imageType: {
      dimension: spatialShape.length,
      componentType: componentTypeOf(data),
      pixelType: isVector ? "VariableLengthVector" : "Scalar",
      components,
    },
    name: "image",
    origin: spatialShape.map(() => 0),
    spacing: spatialShape.map(() => 1),
    direction: createIdentityMatrix(spatialShape.length),
    size: itkSize,
    data: data as
      | Float32Array
      | Float64Array
      | Uint8Array
      | Int8Array
      | Uint16Array
      | Int16Array
      | Uint32Array
      | Int32Array,
    metadata: new Map(),
  };

  return itkImage;
}

/**
 * Convert ITK-Wasm Image back to zarr array
 * Uses the provided store instead of creating a new one
 *
 * Important: ITK-Wasm stores size in physical space order [x, y, z], but data in
 * column-major order (x contiguous). This column-major layout with size [x, y, z]
 * is equivalent to C-order (row-major) with shape [z, y, x]. We reverse the size
 * to get the zarr shape and use C-order strides for that reversed shape.
 *
 * @param itkImage - The ITK-Wasm image to convert
 * @param store - The zarr store to write to
 * @param path - The path within the store
 * @param chunkShape - The chunk shape (in spatial dimension order, will be adjusted for components)
 * @param targetDims - The target dimension order (e.g., ["c", "z", "y", "x"])
 */
export async function itkImageToZarr(
  itkImage: Image,
  store: Map<string, Uint8Array>,
  path: string,
  chunkShape: number[],
  targetDims?: string[],
  codecs?: ZarrCodec[],
): Promise<zarr.Array<zarr.DataType, zarr.Readable>> {
  const root = zarr.root(store);

  if (!itkImage.data) {
    throw new Error("ITK image data is null or undefined");
  }

  // Determine data type - support all ITK TypedArray types
  let dataType: zarr.DataType;
  if (itkImage.data instanceof Uint8Array) {
    dataType = "uint8";
  } else if (itkImage.data instanceof Int8Array) {
    dataType = "int8";
  } else if (itkImage.data instanceof Uint16Array) {
    dataType = "uint16";
  } else if (itkImage.data instanceof Int16Array) {
    dataType = "int16";
  } else if (itkImage.data instanceof Uint32Array) {
    dataType = "uint32";
  } else if (itkImage.data instanceof Int32Array) {
    dataType = "int32";
  } else if (itkImage.data instanceof Float32Array) {
    dataType = "float32";
  } else if (itkImage.data instanceof Float64Array) {
    dataType = "float64";
  } else {
    throw new Error(`Unsupported data type: ${itkImage.data.constructor.name}`);
  }

  // ITK stores size/spacing/origin in physical space order [x, y, z],
  // but the data buffer is in C-order (row-major) which means [z, y, x] indexing.
  // We need to reverse the size to match the data layout, just like we do for spacing/origin.
  const shape = [...itkImage.size].reverse();

  // For vector images, the components are stored in the data but not in the size
  // The actual data length includes components
  const components = itkImage.imageType.components || 1;
  const isVector = components > 1;

  // Validate data length matches expected shape (including components for vector images)
  const spatialElements = shape.reduce((a, b) => a * b, 1);
  const expectedLength = spatialElements * components;
  if (itkImage.data.length !== expectedLength) {
    console.error(`[ERROR] Data length mismatch in itkImageToZarr:`);
    console.error(`  ITK image size (physical order):`, itkImage.size);
    console.error(`  Shape (reversed):`, shape);
    console.error(`  Components:`, components);
    console.error(`  Expected data length:`, expectedLength);
    console.error(`  Actual data length:`, itkImage.data.length);
    throw new Error(
      `Data length (${itkImage.data.length}) doesn't match expected shape ${shape} with ${components} components (${expectedLength} elements)`,
    );
  }

  // Determine the final shape and whether we need to transpose
  // ITK image data has shape [...spatialDimsReversed, components] (with c at end)
  // If targetDims is provided, we need to match that order
  let zarrShape: number[];
  let zarrChunkShape: number[];
  let finalData = itkImage.data;

  if (isVector && targetDims) {
    // Find where "c" should be in targetDims
    const cIndex = targetDims.indexOf("c");
    if (cIndex === -1) {
      throw new Error("Vector image but 'c' not found in targetDims");
    }

    // Current shape is [z, y, x, c] (spatial reversed + c at end)
    // Target shape should match targetDims order
    const currentShape = [...shape, components];

    // Build target shape based on targetDims
    zarrShape = new Array(targetDims.length);
    const spatialDims = shape.slice(); // [z, y, x]
    let spatialIdx = 0;

    for (let i = 0; i < targetDims.length; i++) {
      if (targetDims[i] === "c") {
        zarrShape[i] = components;
      } else {
        zarrShape[i] = spatialDims[spatialIdx++];
      }
    }

    // If c is not at the end, we need to transpose
    if (cIndex !== targetDims.length - 1) {
      // Build permutation: where does each target dim come from in current shape?
      const permutation: number[] = [];
      spatialIdx = 0;
      for (let i = 0; i < targetDims.length; i++) {
        if (targetDims[i] === "c") {
          permutation.push(currentShape.length - 1); // c is at end of current
        } else {
          permutation.push(spatialIdx++);
        }
      }

      // Transpose the data
      finalData = transposeArray(
        itkImage.data,
        currentShape,
        permutation,
        componentTypeOf(itkImage.data),
      );
    }

    // Chunk shape should match zarrShape
    zarrChunkShape = new Array(zarrShape.length);
    spatialIdx = 0;
    for (let i = 0; i < targetDims.length; i++) {
      if (targetDims[i] === "c") {
        zarrChunkShape[i] = components;
      } else {
        zarrChunkShape[i] = chunkShape[spatialIdx++];
      }
    }
  } else {
    // No targetDims or not a vector - use default behavior
    zarrShape = isVector ? [...shape, components] : shape;
    zarrChunkShape = isVector ? [...chunkShape, components] : chunkShape;
  }

  // Chunk shape should match the dimensionality of zarrShape
  if (zarrChunkShape.length !== zarrShape.length) {
    throw new Error(
      `chunkShape length (${zarrChunkShape.length}) must match shape length (${zarrShape.length})`,
    );
  }

  // Validate codecs if provided
  if (codecs !== undefined) {
    if (codecs.length === 0) {
      throw new Error(
        "codecs array must not be empty; at minimum, include the required 'bytes' array-to-bytes codec",
      );
    }
    const hasBytesCodec = codecs.some((codec) => codec.name === "bytes");
    if (!hasBytesCodec) {
      throw new Error(
        "codecs array must include the required 'bytes' array-to-bytes codec",
      );
    }
  }

  const array = await zarr.create(root.resolve(path), {
    shape: zarrShape,
    chunk_shape: zarrChunkShape,
    data_type: dataType,
    fill_value: 0,
    codecs: codecs ?? defaultCodecs(dataType),
  });

  // Write data - preserve the actual data type, don't cast to Float32Array
  // Shape and stride should match the ITK image size order
  // Use null for each dimension to select the entire array
  const selection = zarrShape.map(() => null);
  await zarrSet(array, selection, {
    data: finalData,
    shape: zarrShape,
    stride: calculateStride(zarrShape),
  });

  return array;
}
