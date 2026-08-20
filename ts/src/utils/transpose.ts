// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Typed-array transposition, independent of any image toolkit.
 *
 * Lives outside `methods/` so modules that must not pull in `itk-wasm` can
 * reorder array data.
 */

/** Every numeric typed array the zarr data types map onto. */
export type NumericTypedArray =
  | Float32Array
  | Float64Array
  | Uint8Array
  | Int8Array
  | Uint16Array
  | Int16Array
  | Uint32Array
  | Int32Array;

/** The element type of a {@link NumericTypedArray}. */
export type ComponentType =
  | "uint8"
  | "int8"
  | "uint16"
  | "int16"
  | "uint32"
  | "int32"
  | "float32"
  | "float64";

/** The {@link ComponentType} of a typed array; `float32` when unrecognized. */
export function componentTypeOf(data: unknown): ComponentType {
  if (data instanceof Uint8Array) return "uint8";
  if (data instanceof Int8Array) return "int8";
  if (data instanceof Uint16Array) return "uint16";
  if (data instanceof Int16Array) return "int16";
  if (data instanceof Uint32Array) return "uint32";
  if (data instanceof Int32Array) return "int32";
  if (data instanceof Float64Array) return "float64";
  return "float32";
}

/** Row-major (C-order) strides for `shape`. */
export function calculateStride(shape: number[]): number[] {
  const stride = new Array(shape.length);
  stride[shape.length - 1] = 1;
  for (let i = shape.length - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }
  return stride;
}

/** Allocate a typed array of `length` elements of `componentType`. */
function allocate(
  componentType: ComponentType,
  length: number,
): NumericTypedArray {
  switch (componentType) {
    case "uint8":
      return new Uint8Array(length);
    case "int8":
      return new Int8Array(length);
    case "uint16":
      return new Uint16Array(length);
    case "int16":
      return new Int16Array(length);
    case "uint32":
      return new Uint32Array(length);
    case "int32":
      return new Int32Array(length);
    case "float64":
      return new Float64Array(length);
    case "float32":
    default:
      return new Float32Array(length);
  }
}

/**
 * Reorder `data` so that axis `permutation[i]` of `shape` becomes axis `i`.
 *
 * @param data - Row-major buffer of `shape`.
 * @param shape - The buffer's current shape.
 * @param permutation - Source axis index per target axis.
 * @param componentType - The buffer's element type.
 * @returns A new buffer of the permuted shape.
 */
export function transposeArray(
  data: unknown,
  shape: number[],
  permutation: number[],
  componentType: ComponentType,
): NumericTypedArray {
  const typedData = data as NumericTypedArray;
  const totalSize = typedData.length;
  const output = allocate(componentType, totalSize);

  const sourceStride = calculateStride(shape);
  const newShape = permutation.map((i) => shape[i]);
  const targetStride = calculateStride(newShape);

  const indices = new Array(shape.length).fill(0);
  for (let i = 0; i < totalSize; i++) {
    let sourceIdx = 0;
    for (let j = 0; j < shape.length; j++) {
      sourceIdx += indices[j] * sourceStride[j];
    }

    let targetIdx = 0;
    for (let j = 0; j < permutation.length; j++) {
      targetIdx += indices[permutation[j]] * targetStride[j];
    }

    output[targetIdx] = typedData[sourceIdx];

    for (let j = shape.length - 1; j >= 0; j--) {
      indices[j]++;
      if (indices[j] < shape[j]) break;
      indices[j] = 0;
    }
  }

  return output;
}
