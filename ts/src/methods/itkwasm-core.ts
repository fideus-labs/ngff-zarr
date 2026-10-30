// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * ITK-Wasm downsampling shared by the Node and browser entry points
 * (itkwasm-node.ts, itkwasm-browser.ts), which differ only in the
 * @itk-wasm/downsample functions they pass in.
 */

import type { Image } from "itk-wasm";
import * as zarr from "zarrita";

import { NgffImage } from "../types/ngff_image.ts";
import type { ZarrCodec } from "../utils/codecs.ts";
import { defaultCodecs } from "../utils/codecs.ts";
import { zarrGet, zarrSet } from "../utils/worker_pool.ts";
import {
  castImageToFloat32,
  castImageToIntegerType,
  type DimFactors,
  dimScaleFactors,
  isIntegerComponentType,
  itkImageToZarr,
  MAX_VECTOR_COMPONENTS,
  nextScaleMetadata,
  SPATIAL_DIMS,
  updatePreviousDimFactors,
  zarrToItkImage,
} from "./itkwasm-shared.ts";

type Downsample = (
  image: Image,
  options: { shrinkFactors: number[]; cropRadius?: number[] },
) => Promise<{ downsampled: Image }>;

/** The @itk-wasm/downsample functions for one environment. */
export interface ItkWasmDownsamplers {
  downsample: Downsample;
  downsampleBinShrink: Downsample;
  downsampleLabelImage: Downsample;
}

/**
 * Compute chunk shape from configurable chunks option
 * @param shape - The array shape
 * @param chunks - Optional chunk configuration (number, array, or record)
 * @param dims - Optional dimension names for record-based chunks
 * @param defaultChunk - Default chunk size if not specified (256)
 */
function computeChunkShape(
  shape: number[],
  chunks?: number | number[] | Record<string, number>,
  dims?: string[],
  defaultChunk = 256,
): number[] {
  if (chunks === undefined) {
    return shape.map((s) => Math.min(s, defaultChunk));
  }
  if (typeof chunks === "number") {
    return shape.map((s) => Math.min(s, chunks));
  }
  if (Array.isArray(chunks)) {
    return shape.map((s, i) => Math.min(s, chunks[i] ?? defaultChunk));
  }
  // Record<string, number> - use dims to look up
  if (dims) {
    return shape.map((s, i) => Math.min(s, chunks[dims[i]] ?? defaultChunk));
  }
  return shape.map((s) => Math.min(s, defaultChunk));
}

/**
 * The chunk shape of a downsampled image's spatial axes, in array order.
 * `chunks` is keyed or ordered by `dims`, which may include `c`; the
 * channel chunk is set where the components are stored.
 */
function spatialChunkShape(
  itkSize: number[],
  chunks: number | number[] | Record<string, number> | undefined,
  dims: string[],
): number[] {
  const spatialChunks = Array.isArray(chunks)
    ? chunks.filter((_, i) => dims[i] !== "c")
    : chunks;
  return computeChunkShape(
    // ITK sizes run x, y, z; arrays run z, y, x.
    [...itkSize].reverse(),
    spatialChunks,
    dims.filter((dim) => dim !== "c"),
  );
}

type DownsampleImage = (
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[],
  chunks?: number | number[] | Record<string, number>,
  codecs?: ZarrCodec[],
) => Promise<NgffImage>;

/**
 * Downsample each index along `dim` with `downsampleSlice` and stack the
 * results back along `dim`, which itself is not downsampled.
 */
async function downsamplePerSlice(
  dim: "t" | "c",
  downsampleSlice: DownsampleImage,
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[],
  chunks?: number | number[] | Record<string, number>,
  codecs?: ZarrCodec[],
): Promise<NgffImage> {
  const dimIndex = image.dims.indexOf(dim);
  const size = image.data.shape[dimIndex];
  const newDims = image.dims.filter((d) => d !== dim);

  const downsampledSlices: zarr.Array<zarr.DataType, zarr.Readable>[] = [];
  for (let index = 0; index < size; index++) {
    // Extract the slice
    const selection = new Array(image.data.shape.length).fill(null);
    selection[dimIndex] = index;
    const sliceData = await zarrGet(image.data, selection);

    // Create temporary zarr array for this slice
    const sliceStore = new Map<string, Uint8Array>();
    const sliceRoot = zarr.root(sliceStore);
    const sliceShape = image.data.shape.filter((_, i) => i !== dimIndex);
    const sliceChunks = Array.isArray(chunks)
      ? chunks.filter((_, i) => i !== dimIndex)
      : chunks;
    const sliceChunkShape = computeChunkShape(sliceShape, sliceChunks, newDims);

    const sliceArray = await zarr.create(sliceRoot.resolve("slice"), {
      shape: sliceShape,
      chunkShape: sliceChunkShape,
      dtype: image.data.dtype,
      fillValue: 0,
      codecs: codecs ?? defaultCodecs(image.data.dtype),
    });

    const fullSelection = new Array(sliceShape.length).fill(null);
    await zarrSet(sliceArray, fullSelection, sliceData);

    // Create NgffImage for this slice (without `dim`)
    const sliceImage = new NgffImage({
      data: sliceArray,
      dims: newDims,
      scale: Object.fromEntries(
        Object.entries(image.scale).filter(([d]) => d !== dim),
      ),
      translation: Object.fromEntries(
        Object.entries(image.translation).filter(([d]) => d !== dim),
      ),
      name: image.name,
      axesUnits: image.axesUnits
        ? Object.fromEntries(
            Object.entries(image.axesUnits).filter(([d]) => d !== dim),
          )
        : undefined,
      computedCallbacks: image.computedCallbacks,
    });

    // Recursively downsample this slice (without `dim`, so no infinite loop)
    const downsampledSlice = await downsampleSlice(
      sliceImage,
      dimFactors,
      spatialDims,
      sliceChunks,
      codecs,
    );
    downsampledSlices.push(downsampledSlice.data);
  }

  // Combine downsampled slices back into a single array with `dim`
  const firstSlice = downsampledSlices[0];
  const combinedShape = [...image.data.shape];
  combinedShape[dimIndex] = size;
  // Update spatial dimensions based on downsampled size
  for (let i = 0; i < image.dims.length; i++) {
    if (i !== dimIndex) {
      const sliceIndex = i < dimIndex ? i : i - 1;
      combinedShape[i] = firstSlice.shape[sliceIndex];
    }
  }

  // Create combined array
  const combinedStore = new Map<string, Uint8Array>();
  const combinedRoot = zarr.root(combinedStore);
  const combinedArray = await zarr.create(combinedRoot.resolve("combined"), {
    shape: combinedShape,
    chunkShape: computeChunkShape(combinedShape, chunks, image.dims),
    dtype: image.data.dtype,
    fillValue: 0,
    codecs: codecs ?? defaultCodecs(image.data.dtype),
  });

  // Copy each downsampled slice into the combined array
  for (let index = 0; index < size; index++) {
    const sliceData = await zarrGet(downsampledSlices[index]);
    const targetSelection = new Array(combinedShape.length).fill(null);
    targetSelection[dimIndex] = index;
    await zarrSet(combinedArray, targetSelection, sliceData);
  }

  // Compute new metadata (`dim` unchanged, spatial dimensions downsampled)
  const [translation, scale] = nextScaleMetadata(
    image,
    dimFactors,
    spatialDims,
  );

  return new NgffImage({
    data: combinedArray,
    dims: image.dims,
    scale: { ...image.scale, ...scale },
    translation: { ...image.translation, ...translation },
    name: image.name,
    axesUnits: image.axesUnits,
    computedCallbacks: image.computedCallbacks,
  });
}

/**
 * Perform Gaussian downsampling using ITK-Wasm
 */
async function downsampleGaussian(
  fns: ItkWasmDownsamplers,
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[],
  chunks?: number | number[] | Record<string, number>,
  codecs?: ZarrCodec[],
): Promise<NgffImage> {
  // Handle time dimension by processing each time slice independently
  if (image.dims.includes("t")) {
    return await downsamplePerSlice(
      "t",
      (...args) => downsampleGaussian(fns, ...args),
      image,
      dimFactors,
      spatialDims,
      chunks,
      codecs,
    );
  }

  // Determine if we should use vector mode for multi-channel images.
  // Only use vector mode for small channel counts (≤ MAX_VECTOR_COMPONENTS)
  // to avoid itkwasm limitations with VariableLengthVector images.
  const cIndex = image.dims.indexOf("c");
  const hasChannels = cIndex !== -1;
  const numChannels = hasChannels ? image.data.shape[cIndex] : 1;
  const isVector = hasChannels && numChannels <= MAX_VECTOR_COMPONENTS;

  // Handle channel dimension by processing each channel independently when not using vector mode
  if (hasChannels && !isVector) {
    return await downsamplePerSlice(
      "c",
      (...args) => downsampleGaussian(fns, ...args),
      image,
      dimFactors,
      spatialDims,
      chunks,
      codecs,
    );
  }

  // Convert to ITK-Wasm format
  const itkImage = await zarrToItkImage(image.data, image.dims, isVector);

  // Prepare shrink factors - need to be for ALL dimensions in ITK order (reversed)
  const shrinkFactors: number[] = [];
  for (let i = image.dims.length - 1; i >= 0; i--) {
    const dim = image.dims[i];
    if (SPATIAL_DIMS.includes(dim)) {
      shrinkFactors.push(dimFactors[dim] || 1);
    }
  }

  // Use all zeros for cropRadius
  const cropRadius = new Array(shrinkFactors.length).fill(0);

  // The Gaussian filter loses precision on integer inputs; cast to
  // float32 and round back afterwards, mirroring the Python port.
  const originalComponentType = itkImage.imageType.componentType;
  const needsFloatWorkaround = isIntegerComponentType(originalComponentType);
  const inputImage = needsFloatWorkaround
    ? castImageToFloat32(itkImage)
    : itkImage;

  const { downsampled: downsampledRaw } = await fns.downsample(inputImage, {
    shrinkFactors,
    cropRadius: cropRadius,
  });
  const downsampled = needsFloatWorkaround
    ? castImageToIntegerType(downsampledRaw, originalComponentType)
    : downsampledRaw;

  // Compute new metadata
  const [translation, scale] = nextScaleMetadata(
    image,
    dimFactors,
    spatialDims,
  );

  // Convert back to zarr array in a new in-memory store
  const store = new Map<string, Uint8Array>();
  const chunkShape = spatialChunkShape(downsampled.size, chunks, image.dims);
  const array = await itkImageToZarr(
    downsampled,
    store,
    "image",
    chunkShape,
    image.dims,
    codecs,
  );

  return new NgffImage({
    data: array,
    dims: image.dims,
    scale,
    translation,
    name: image.name,
    axesUnits: image.axesUnits,
    computedCallbacks: image.computedCallbacks,
  });
}

/**
 * Perform bin shrink downsampling using ITK-Wasm
 */
async function downsampleBinShrinkImpl(
  fns: ItkWasmDownsamplers,
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[],
  chunks?: number | number[] | Record<string, number>,
  codecs?: ZarrCodec[],
): Promise<NgffImage> {
  // Handle time dimension by processing each time slice independently
  if (image.dims.includes("t")) {
    return await downsamplePerSlice(
      "t",
      (...args) => downsampleBinShrinkImpl(fns, ...args),
      image,
      dimFactors,
      spatialDims,
      chunks,
      codecs,
    );
  }

  // Determine if we should use vector mode for multi-channel images.
  // Only use vector mode for small channel counts (≤ MAX_VECTOR_COMPONENTS)
  // to avoid itkwasm limitations with VariableLengthVector images.
  const cIndex = image.dims.indexOf("c");
  const hasChannels = cIndex !== -1;
  const numChannels = hasChannels ? image.data.shape[cIndex] : 1;
  const isVector = hasChannels && numChannels <= MAX_VECTOR_COMPONENTS;

  // Handle channel dimension by processing each channel independently when not using vector mode
  if (hasChannels && !isVector) {
    return await downsamplePerSlice(
      "c",
      (...args) => downsampleBinShrinkImpl(fns, ...args),
      image,
      dimFactors,
      spatialDims,
      chunks,
      codecs,
    );
  }

  // Convert to ITK-Wasm format
  const itkImage = await zarrToItkImage(image.data, image.dims, isVector);

  // Prepare shrink factors - only for spatial dimensions in ITK order (reversed)
  const shrinkFactors: number[] = [];
  for (let i = image.dims.length - 1; i >= 0; i--) {
    const dim = image.dims[i];
    if (SPATIAL_DIMS.includes(dim)) {
      shrinkFactors.push(dimFactors[dim] || 1);
    }
  }

  const { downsampled } = await fns.downsampleBinShrink(itkImage, {
    shrinkFactors,
  });

  // Compute new metadata
  const [translation, scale] = nextScaleMetadata(
    image,
    dimFactors,
    spatialDims,
  );

  // Convert back to zarr array in a new in-memory store
  const store = new Map<string, Uint8Array>();
  const chunkShape = spatialChunkShape(downsampled.size, chunks, image.dims);
  const array = await itkImageToZarr(
    downsampled,
    store,
    "image",
    chunkShape,
    image.dims,
    codecs,
  );

  return new NgffImage({
    data: array,
    dims: image.dims,
    scale,
    translation,
    name: image.name,
    axesUnits: image.axesUnits,
    computedCallbacks: image.computedCallbacks,
  });
}

/**
 * Perform label image downsampling using ITK-Wasm
 */
async function downsampleLabelImageImpl(
  fns: ItkWasmDownsamplers,
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[],
  chunks?: number | number[] | Record<string, number>,
  codecs?: ZarrCodec[],
): Promise<NgffImage> {
  // Handle time dimension by processing each time slice independently
  if (image.dims.includes("t")) {
    return await downsamplePerSlice(
      "t",
      (...args) => downsampleLabelImageImpl(fns, ...args),
      image,
      dimFactors,
      spatialDims,
      chunks,
      codecs,
    );
  }

  // downsampleLabelImage takes no vector images, so channels go one by one.
  if (image.dims.includes("c")) {
    return await downsamplePerSlice(
      "c",
      (...args) => downsampleLabelImageImpl(fns, ...args),
      image,
      dimFactors,
      spatialDims,
      chunks,
      codecs,
    );
  }

  // Convert to ITK-Wasm format
  const itkImage = await zarrToItkImage(image.data, image.dims, false);

  // Prepare shrink factors - need to be for ALL dimensions in ITK order (reversed)
  const shrinkFactors: number[] = [];
  for (let i = image.dims.length - 1; i >= 0; i--) {
    const dim = image.dims[i];
    if (SPATIAL_DIMS.includes(dim)) {
      shrinkFactors.push(dimFactors[dim] || 1);
    } else {
      shrinkFactors.push(1); // Non-spatial dimensions don't shrink
    }
  }

  // Use all zeros for cropRadius
  const cropRadius = new Array(shrinkFactors.length).fill(0);

  const { downsampled } = await fns.downsampleLabelImage(itkImage, {
    shrinkFactors,
    cropRadius: cropRadius,
  });

  // Compute new metadata
  const [translation, scale] = nextScaleMetadata(
    image,
    dimFactors,
    spatialDims,
  );

  // Convert back to zarr array in a new in-memory store
  const store = new Map<string, Uint8Array>();
  const chunkShape = spatialChunkShape(downsampled.size, chunks, image.dims);
  const array = await itkImageToZarr(
    downsampled,
    store,
    "image",
    chunkShape,
    image.dims,
    codecs,
  );

  return new NgffImage({
    data: array,
    dims: image.dims,
    scale,
    translation,
    name: image.name,
    axesUnits: image.axesUnits,
    computedCallbacks: image.computedCallbacks,
  });
}

/**
 * Main downsampling function for ITK-Wasm
 */
export async function downsampleItkWasmCore(
  fns: ItkWasmDownsamplers,
  ngffImage: NgffImage,
  scaleFactors: (Record<string, number> | number)[],
  smoothing: "gaussian" | "bin_shrink" | "label_image",
  chunks?: number | number[] | Record<string, number>,
  codecs?: ZarrCodec[],
): Promise<NgffImage[]> {
  const multiscales: NgffImage[] = [ngffImage];
  const dims = ngffImage.dims;
  const spatialDims = dims.filter((dim) => SPATIAL_DIMS.includes(dim));

  let previousImage = ngffImage;
  let previousDimFactors: DimFactors = {};
  for (const dim of dims) previousDimFactors[dim] = 1;

  for (let i = 0; i < scaleFactors.length; i++) {
    const scaleFactor = scaleFactors[i];
    let sourceImage: NgffImage;
    let sourceDimFactors: DimFactors;

    {
      // Scale factors are absolute, relative to the full-resolution image.
      const dimFactors = dimScaleFactors(
        dims,
        scaleFactor,
        previousDimFactors,
        ngffImage,
        previousImage,
      );

      let canDownsampleIncrementally = true;
      for (const dim of Object.keys(dimFactors)) {
        const dimIndex = ngffImage.dims.indexOf(dim);
        if (dimIndex >= 0 && SPATIAL_DIMS.includes(dim)) {
          const originalSize = ngffImage.data.shape[dimIndex];
          const targetSize = Math.floor(
            originalSize /
              (typeof scaleFactor === "number"
                ? scaleFactor
                : scaleFactor[dim]),
          );
          const prevDimIndex = previousImage.dims.indexOf(dim);
          const previousSize = previousImage.data.shape[prevDimIndex];
          if (Math.floor(previousSize / dimFactors[dim]) !== targetSize) {
            canDownsampleIncrementally = false;
            break;
          }
        }
      }
      if (canDownsampleIncrementally) {
        sourceImage = previousImage;
        sourceDimFactors = dimFactors;
      } else {
        sourceImage = ngffImage;
        const originalDimFactors: DimFactors = {};
        for (const dim of dims) originalDimFactors[dim] = 1;
        sourceDimFactors = dimScaleFactors(
          dims,
          scaleFactor,
          originalDimFactors,
        );
      }
    }

    let downsampled: NgffImage;
    if (smoothing === "gaussian") {
      downsampled = await downsampleGaussian(
        fns,
        sourceImage,
        sourceDimFactors,
        spatialDims,
        chunks,
        codecs,
      );
    } else if (smoothing === "bin_shrink") {
      downsampled = await downsampleBinShrinkImpl(
        fns,
        sourceImage,
        sourceDimFactors,
        spatialDims,
        chunks,
        codecs,
      );
    } else if (smoothing === "label_image") {
      downsampled = await downsampleLabelImageImpl(
        fns,
        sourceImage,
        sourceDimFactors,
        spatialDims,
        chunks,
        codecs,
      );
    } else {
      throw new Error(`Unknown smoothing method: ${smoothing}`);
    }

    multiscales.push(downsampled);

    previousImage = downsampled;
    previousDimFactors = updatePreviousDimFactors(
      scaleFactor,
      spatialDims,
      previousDimFactors,
    );
  }

  return multiscales;
}
