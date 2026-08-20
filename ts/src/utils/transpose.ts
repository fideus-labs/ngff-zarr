// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Typed-array transposition, independent of any image toolkit.
 *
 * Lives outside `methods/` so modules that must not pull in `itk-wasm` can
 * reorder array data.
 */

/** Every number-valued typed array the zarr data types map onto. */
export type NumericTypedArray =
  | Float32Array
  | Float64Array
  | Uint8Array
  | Int8Array
  | Uint16Array
  | Int16Array
  | Uint32Array
  | Int32Array;

/** The 64-bit integer arrays, whose elements are `bigint` rather than `number`. */
export type BigTypedArray = BigInt64Array | BigUint64Array;

/** Any typed array {@link transposeArray} accepts. */
export type AnyTypedArray = NumericTypedArray | BigTypedArray;

/** The element type of an {@link AnyTypedArray}. */
export type ComponentType =
  | "uint8"
  | "int8"
  | "uint16"
  | "int16"
  | "uint32"
  | "int32"
  | "int64"
  | "uint64"
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
  if (data instanceof BigInt64Array) return "int64";
  if (data instanceof BigUint64Array) return "uint64";
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
): AnyTypedArray {
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
    case "int64":
      return new BigInt64Array(length);
    case "uint64":
      return new BigUint64Array(length);
    case "float64":
      return new Float64Array(length);
    case "float32":
    default:
      return new Float32Array(length);
  }
}

/** An indexable view, so one loop can copy `number` and `bigint` elements. */
type ElementView = { [index: number]: number | bigint; length: number };

/**
 * Reorder `data` so that axis `permutation[i]` of `shape` becomes axis `i`.
 *
 * @param data - Row-major buffer of `shape`.
 * @param shape - The buffer's current shape.
 * @param permutation - Source axis index per target axis.
 * @param componentType - The buffer's element type.
 * @returns A new buffer of the permuted shape, of the same array type as
 * `data`. Callers that have already narrowed the element type keep it: `T`
 * appears only in the return position, so a caller holding, say, an ITK
 * component buffer does not widen to the `bigint` arrays.
 */
export function transposeArray<T extends AnyTypedArray = AnyTypedArray>(
  data: unknown,
  shape: number[],
  permutation: number[],
  componentType: ComponentType,
): T {
  const typedData = data as AnyTypedArray;
  const totalSize = typedData.length;
  const output = allocate(componentType, totalSize);
  // `int64`/`uint64` arrays hold `bigint`; the element type is uniform between
  // source and output, but TypeScript cannot narrow the union across the pair.
  const source = typedData as unknown as ElementView;
  const target = output as unknown as ElementView;

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

    target[targetIdx] = source[sourceIdx];

    for (let j = shape.length - 1; j >= 0; j--) {
      indices[j]++;
      if (indices[j] < shape[j]) break;
      indices[j] = 0;
    }
  }

  return output as T;
}
