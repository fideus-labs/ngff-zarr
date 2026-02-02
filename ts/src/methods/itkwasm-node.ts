// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Node-compatible ITK-Wasm downsampling support
 * Uses native WASM implementations from @itk-wasm/downsample
 *
 * Note: Imports Node-specific functions that are available when the
 * "node" condition is active in package.json exports resolution.
 * The @itk-wasm/downsample package uses conditional exports to provide
 * different implementations for browser vs Node environments.
 */

import {
  downsampleBinShrinkNode as downsampleBinShrink,
  downsampleLabelImageNode as downsampleLabelImage,
  downsampleNode as downsample,
} from "@itk-wasm/downsample";
import * as zarr from "zarrita";
import { NgffImage } from "../types/ngff_image.ts";
import {
  type DimFactors,
  dimScaleFactors,
  itkImageToZarr,
  MAX_VECTOR_COMPONENTS,
  nextScaleMetadata,
  SPATIAL_DIMS,
  updatePreviousDimFactors,
  zarrToItkImage,
} from "./itkwasm-shared.ts";

/**
 * Perform Gaussian downsampling using ITK-Wasm (browser version)
 */
async function downsampleGaussian(
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[],
): Promise<NgffImage> {
  // Handle time dimension by processing each time slice independently
  if (image.dims.includes("t")) {
    const tDimIndex = image.dims.indexOf("t");
    const tSize = image.data.shape[tDimIndex];
    const newDims = image.dims.filter((dim) => dim !== "t");

    // Downsample each time slice
    const downsampledSlices: zarr.Array<zarr.DataType, zarr.Readable>[] = [];
    for (let t = 0; t < tSize; t++) {
      // Extract time slice
      const selection = new Array(image.data.shape.length).fill(null);
      selection[tDimIndex] = t;
      const sliceData = await zarr.get(image.data, selection);

      // Create temporary zarr array for this slice
      const sliceStore = new Map<string, Uint8Array>();
      const sliceRoot = zarr.root(sliceStore);
      const sliceShape = image.data.shape.filter((_, i) => i !== tDimIndex);
      const sliceChunkShape = sliceShape.map((s) => Math.min(s, 256));

      const sliceArray = await zarr.create(sliceRoot.resolve("slice"), {
        shape: sliceShape,
        chunk_shape: sliceChunkShape,
        data_type: image.data.dtype,
        fill_value: 0,
      });

      const fullSelection = new Array(sliceShape.length).fill(null);
      await zarr.set(sliceArray, fullSelection, sliceData);

      // Create NgffImage for this slice (without 't' dimension)
      const sliceImage = new NgffImage({
        data: sliceArray,
        dims: newDims,
        scale: Object.fromEntries(
          Object.entries(image.scale).filter(([dim]) => dim !== "t"),
        ),
        translation: Object.fromEntries(
          Object.entries(image.translation).filter(([dim]) => dim !== "t"),
        ),
        name: image.name,
        axesUnits: image.axesUnits
          ? Object.fromEntries(
            Object.entries(image.axesUnits).filter(([dim]) => dim !== "t"),
          )
          : undefined,
        computedCallbacks: image.computedCallbacks,
      });

      // Recursively downsample this slice (without 't', so no infinite loop)
      const downsampledSlice = await downsampleGaussian(
        sliceImage,
        dimFactors,
        spatialDims,
      );
      downsampledSlices.push(downsampledSlice.data);
    }

    // Combine downsampled slices back into a single array with 't' dimension
    const firstSlice = downsampledSlices[0];
    const combinedShape = [...image.data.shape];
    combinedShape[tDimIndex] = tSize;
    // Update spatial dimensions based on downsampled size
    for (let i = 0; i < image.dims.length; i++) {
      if (i !== tDimIndex) {
        const sliceIndex = i < tDimIndex ? i : i - 1;
        combinedShape[i] = firstSlice.shape[sliceIndex];
      }
    }

    // Create combined array
    const combinedStore = new Map<string, Uint8Array>();
    const combinedRoot = zarr.root(combinedStore);
    const combinedArray = await zarr.create(combinedRoot.resolve("combined"), {
      shape: combinedShape,
      chunk_shape: combinedShape.map((s) => Math.min(s, 256)),
      data_type: image.data.dtype,
      fill_value: 0,
    });

    // Copy each downsampled slice into the combined array
    for (let t = 0; t < tSize; t++) {
      const sliceData = await zarr.get(downsampledSlices[t]);
      const targetSelection = new Array(combinedShape.length).fill(null);
      targetSelection[tDimIndex] = t;
      await zarr.set(combinedArray, targetSelection, sliceData);
    }

    // Compute new metadata (time dimension unchanged, spatial dimensions downsampled)
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

  // Determine if we should use vector mode for multi-channel images.
  // Only use vector mode for small channel counts (≤ MAX_VECTOR_COMPONENTS)
  // to avoid itkwasm limitations with VariableLengthVector images.
  const cIndex = image.dims.indexOf("c");
  const hasChannels = cIndex !== -1;
  const numChannels = hasChannels ? image.data.shape[cIndex] : 1;
  const isVector = hasChannels && numChannels <= MAX_VECTOR_COMPONENTS;

  // Handle channel dimension by processing each channel independently when not using vector mode
  if (hasChannels && !isVector) {
    const cDimIndex = cIndex;
    const cSize = numChannels;
    const newDims = image.dims.filter((dim) => dim !== "c");

    // Downsample each channel
    const downsampledSlices: zarr.Array<zarr.DataType, zarr.Readable>[] = [];
    for (let c = 0; c < cSize; c++) {
      // Extract channel slice
      const selection = new Array(image.data.shape.length).fill(null);
      selection[cDimIndex] = c;
      const sliceData = await zarr.get(image.data, selection);

      // Create temporary zarr array for this slice
      const sliceStore = new Map<string, Uint8Array>();
      const sliceRoot = zarr.root(sliceStore);
      const sliceShape = image.data.shape.filter((_, i) => i !== cDimIndex);
      const sliceChunkShape = sliceShape.map((s) => Math.min(s, 256));

      const sliceArray = await zarr.create(sliceRoot.resolve("slice"), {
        shape: sliceShape,
        chunk_shape: sliceChunkShape,
        data_type: image.data.dtype,
        fill_value: 0,
      });

      const fullSelection = new Array(sliceShape.length).fill(null);
      await zarr.set(sliceArray, fullSelection, sliceData);

      // Create NgffImage for this slice (without 'c' dimension)
      const sliceImage = new NgffImage({
        data: sliceArray,
        dims: newDims,
        scale: Object.fromEntries(
          Object.entries(image.scale).filter(([dim]) => dim !== "c"),
        ),
        translation: Object.fromEntries(
          Object.entries(image.translation).filter(([dim]) => dim !== "c"),
        ),
        name: image.name,
        axesUnits: image.axesUnits
          ? Object.fromEntries(
            Object.entries(image.axesUnits).filter(([dim]) => dim !== "c"),
          )
          : undefined,
        computedCallbacks: image.computedCallbacks,
      });

      // Recursively downsample this slice (without 'c', so no infinite loop)
      const downsampledSlice = await downsampleGaussian(
        sliceImage,
        dimFactors,
        spatialDims,
      );
      downsampledSlices.push(downsampledSlice.data);
    }

    // Combine downsampled slices back into a single array with 'c' dimension
    const firstSlice = downsampledSlices[0];
    const combinedShape = [...image.data.shape];
    combinedShape[cDimIndex] = cSize;
    // Update spatial dimensions based on downsampled size
    for (let i = 0; i < image.dims.length; i++) {
      if (i !== cDimIndex) {
        const sliceIndex = i < cDimIndex ? i : i - 1;
        combinedShape[i] = firstSlice.shape[sliceIndex];
      }
    }

    // Create combined array
    const combinedStore = new Map<string, Uint8Array>();
    const combinedRoot = zarr.root(combinedStore);
    const combinedArray = await zarr.create(combinedRoot.resolve("combined"), {
      shape: combinedShape,
      chunk_shape: combinedShape.map((s) => Math.min(s, 256)),
      data_type: image.data.dtype,
      fill_value: 0,
    });

    // Copy each downsampled slice into the combined array
    for (let c = 0; c < cSize; c++) {
      const sliceData = await zarr.get(downsampledSlices[c]);
      const targetSelection = new Array(combinedShape.length).fill(null);
      targetSelection[cDimIndex] = c;
      await zarr.set(combinedArray, targetSelection, sliceData);
    }

    // Compute new metadata (channel dimension unchanged, spatial dimensions downsampled)
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

  // Perform downsampling using browser-compatible function
  const { downsampled } = await downsample(itkImage, {
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
  const chunkShape = downsampled.size.map((s) => Math.min(s, 256)).reverse();
  const array = await itkImageToZarr(
    downsampled,
    store,
    "image",
    chunkShape,
    image.dims,
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
 * Perform bin shrink downsampling using ITK-Wasm (browser version)
 */
async function downsampleBinShrinkImpl(
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[],
): Promise<NgffImage> {
  // Handle time dimension by processing each time slice independently
  if (image.dims.includes("t")) {
    const tDimIndex = image.dims.indexOf("t");
    const tSize = image.data.shape[tDimIndex];
    const newDims = image.dims.filter((dim) => dim !== "t");

    // Downsample each time slice
    const downsampledSlices: zarr.Array<zarr.DataType, zarr.Readable>[] = [];
    for (let t = 0; t < tSize; t++) {
      // Extract time slice
      const selection = new Array(image.data.shape.length).fill(null);
      selection[tDimIndex] = t;
      const sliceData = await zarr.get(image.data, selection);

      // Create temporary zarr array for this slice
      const sliceStore = new Map<string, Uint8Array>();
      const sliceRoot = zarr.root(sliceStore);
      const sliceShape = image.data.shape.filter((_, i) => i !== tDimIndex);
      const sliceChunkShape = sliceShape.map((s) => Math.min(s, 256));

      const sliceArray = await zarr.create(sliceRoot.resolve("slice"), {
        shape: sliceShape,
        chunk_shape: sliceChunkShape,
        data_type: image.data.dtype,
        fill_value: 0,
      });

      const fullSelection = new Array(sliceShape.length).fill(null);
      await zarr.set(sliceArray, fullSelection, sliceData);

      // Create NgffImage for this slice (without 't' dimension)
      const sliceImage = new NgffImage({
        data: sliceArray,
        dims: newDims,
        scale: Object.fromEntries(
          Object.entries(image.scale).filter(([dim]) => dim !== "t"),
        ),
        translation: Object.fromEntries(
          Object.entries(image.translation).filter(([dim]) => dim !== "t"),
        ),
        name: image.name,
        axesUnits: image.axesUnits
          ? Object.fromEntries(
            Object.entries(image.axesUnits).filter(([dim]) => dim !== "t"),
          )
          : undefined,
        computedCallbacks: image.computedCallbacks,
      });

      // Recursively downsample this slice
      const downsampledSlice = await downsampleBinShrinkImpl(
        sliceImage,
        dimFactors,
        spatialDims,
      );
      downsampledSlices.push(downsampledSlice.data);
    }

    // Combine downsampled slices back into a single array with 't' dimension
    const firstSlice = downsampledSlices[0];
    const combinedShape = [...image.data.shape];
    combinedShape[tDimIndex] = tSize;
    // Update spatial dimensions based on downsampled size
    for (let i = 0; i < image.dims.length; i++) {
      if (i !== tDimIndex) {
        const sliceIndex = i < tDimIndex ? i : i - 1;
        combinedShape[i] = firstSlice.shape[sliceIndex];
      }
    }

    // Create combined array
    const combinedStore = new Map<string, Uint8Array>();
    const combinedRoot = zarr.root(combinedStore);
    const combinedArray = await zarr.create(combinedRoot.resolve("combined"), {
      shape: combinedShape,
      chunk_shape: combinedShape.map((s) => Math.min(s, 256)),
      data_type: image.data.dtype,
      fill_value: 0,
    });

    // Copy each downsampled slice into the combined array
    for (let t = 0; t < tSize; t++) {
      const sliceData = await zarr.get(downsampledSlices[t]);
      const targetSelection = new Array(combinedShape.length).fill(null);
      targetSelection[tDimIndex] = t;
      await zarr.set(combinedArray, targetSelection, sliceData);
    }

    // Compute new metadata
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

  // Determine if we should use vector mode for multi-channel images.
  // Only use vector mode for small channel counts (≤ MAX_VECTOR_COMPONENTS)
  // to avoid itkwasm limitations with VariableLengthVector images.
  const cIndex = image.dims.indexOf("c");
  const hasChannels = cIndex !== -1;
  const numChannels = hasChannels ? image.data.shape[cIndex] : 1;
  const isVector = hasChannels && numChannels <= MAX_VECTOR_COMPONENTS;

  // Handle channel dimension by processing each channel independently when not using vector mode
  if (hasChannels && !isVector) {
    const cDimIndex = cIndex;
    const cSize = numChannels;
    const newDims = image.dims.filter((dim) => dim !== "c");

    // Downsample each channel
    const downsampledSlices: zarr.Array<zarr.DataType, zarr.Readable>[] = [];
    for (let c = 0; c < cSize; c++) {
      // Extract channel slice
      const selection = new Array(image.data.shape.length).fill(null);
      selection[cDimIndex] = c;
      const sliceData = await zarr.get(image.data, selection);

      // Create temporary zarr array for this slice
      const sliceStore = new Map<string, Uint8Array>();
      const sliceRoot = zarr.root(sliceStore);
      const sliceShape = image.data.shape.filter((_, i) => i !== cDimIndex);
      const sliceChunkShape = sliceShape.map((s) => Math.min(s, 256));

      const sliceArray = await zarr.create(sliceRoot.resolve("slice"), {
        shape: sliceShape,
        chunk_shape: sliceChunkShape,
        data_type: image.data.dtype,
        fill_value: 0,
      });

      const fullSelection = new Array(sliceShape.length).fill(null);
      await zarr.set(sliceArray, fullSelection, sliceData);

      // Create NgffImage for this slice (without 'c' dimension)
      const sliceImage = new NgffImage({
        data: sliceArray,
        dims: newDims,
        scale: Object.fromEntries(
          Object.entries(image.scale).filter(([dim]) => dim !== "c"),
        ),
        translation: Object.fromEntries(
          Object.entries(image.translation).filter(([dim]) => dim !== "c"),
        ),
        name: image.name,
        axesUnits: image.axesUnits
          ? Object.fromEntries(
            Object.entries(image.axesUnits).filter(([dim]) => dim !== "c"),
          )
          : undefined,
        computedCallbacks: image.computedCallbacks,
      });

      // Recursively downsample this slice (without 'c', so no infinite loop)
      const downsampledSlice = await downsampleBinShrinkImpl(
        sliceImage,
        dimFactors,
        spatialDims,
      );
      downsampledSlices.push(downsampledSlice.data);
    }

    // Combine downsampled slices back into a single array with 'c' dimension
    const firstSlice = downsampledSlices[0];
    const combinedShape = [...image.data.shape];
    combinedShape[cDimIndex] = cSize;
    // Update spatial dimensions based on downsampled size
    for (let i = 0; i < image.dims.length; i++) {
      if (i !== cDimIndex) {
        const sliceIndex = i < cDimIndex ? i : i - 1;
        combinedShape[i] = firstSlice.shape[sliceIndex];
      }
    }

    // Create combined array
    const combinedStore = new Map<string, Uint8Array>();
    const combinedRoot = zarr.root(combinedStore);
    const combinedArray = await zarr.create(combinedRoot.resolve("combined"), {
      shape: combinedShape,
      chunk_shape: combinedShape.map((s) => Math.min(s, 256)),
      data_type: image.data.dtype,
      fill_value: 0,
    });

    // Copy each downsampled slice into the combined array
    for (let c = 0; c < cSize; c++) {
      const sliceData = await zarr.get(downsampledSlices[c]);
      const targetSelection = new Array(combinedShape.length).fill(null);
      targetSelection[cDimIndex] = c;
      await zarr.set(combinedArray, targetSelection, sliceData);
    }

    // Compute new metadata (channel dimension unchanged, spatial dimensions downsampled)
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

  // Perform downsampling using browser-compatible function
  const { downsampled } = await downsampleBinShrink(itkImage, {
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
  const chunkShape = downsampled.size.map((s) => Math.min(s, 256)).reverse();
  const array = await itkImageToZarr(
    downsampled,
    store,
    "image",
    chunkShape,
    image.dims,
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
 * Perform label image downsampling using ITK-Wasm (browser version)
 */
async function downsampleLabelImageImpl(
  image: NgffImage,
  dimFactors: DimFactors,
  spatialDims: string[],
): Promise<NgffImage> {
  // Handle time dimension by processing each time slice independently
  if (image.dims.includes("t")) {
    const tDimIndex = image.dims.indexOf("t");
    const tSize = image.data.shape[tDimIndex];
    const newDims = image.dims.filter((dim) => dim !== "t");

    // Downsample each time slice
    const downsampledSlices: zarr.Array<zarr.DataType, zarr.Readable>[] = [];
    for (let t = 0; t < tSize; t++) {
      // Extract time slice
      const selection = new Array(image.data.shape.length).fill(null);
      selection[tDimIndex] = t;
      const sliceData = await zarr.get(image.data, selection);

      // Create temporary zarr array for this slice
      const sliceStore = new Map<string, Uint8Array>();
      const sliceRoot = zarr.root(sliceStore);
      const sliceShape = image.data.shape.filter((_, i) => i !== tDimIndex);
      const sliceChunkShape = sliceShape.map((s) => Math.min(s, 256));

      const sliceArray = await zarr.create(sliceRoot.resolve("slice"), {
        shape: sliceShape,
        chunk_shape: sliceChunkShape,
        data_type: image.data.dtype,
        fill_value: 0,
      });

      const fullSelection = new Array(sliceShape.length).fill(null);
      await zarr.set(sliceArray, fullSelection, sliceData);

      // Create NgffImage for this slice (without 't' dimension)
      const sliceImage = new NgffImage({
        data: sliceArray,
        dims: newDims,
        scale: Object.fromEntries(
          Object.entries(image.scale).filter(([dim]) => dim !== "t"),
        ),
        translation: Object.fromEntries(
          Object.entries(image.translation).filter(([dim]) => dim !== "t"),
        ),
        name: image.name,
        axesUnits: image.axesUnits
          ? Object.fromEntries(
            Object.entries(image.axesUnits).filter(([dim]) => dim !== "t"),
          )
          : undefined,
        computedCallbacks: image.computedCallbacks,
      });

      // Recursively downsample this slice
      const downsampledSlice = await downsampleLabelImageImpl(
        sliceImage,
        dimFactors,
        spatialDims,
      );
      downsampledSlices.push(downsampledSlice.data);
    }

    // Combine downsampled slices back into a single array with 't' dimension
    const firstSlice = downsampledSlices[0];
    const combinedShape = [...image.data.shape];
    combinedShape[tDimIndex] = tSize;
    // Update spatial dimensions based on downsampled size
    for (let i = 0; i < image.dims.length; i++) {
      if (i !== tDimIndex) {
        const sliceIndex = i < tDimIndex ? i : i - 1;
        combinedShape[i] = firstSlice.shape[sliceIndex];
      }
    }

    // Create combined array
    const combinedStore = new Map<string, Uint8Array>();
    const combinedRoot = zarr.root(combinedStore);
    const combinedArray = await zarr.create(combinedRoot.resolve("combined"), {
      shape: combinedShape,
      chunk_shape: combinedShape.map((s) => Math.min(s, 256)),
      data_type: image.data.dtype,
      fill_value: 0,
    });

    // Copy each downsampled slice into the combined array
    for (let t = 0; t < tSize; t++) {
      const sliceData = await zarr.get(downsampledSlices[t]);
      const targetSelection = new Array(combinedShape.length).fill(null);
      targetSelection[tDimIndex] = t;
      await zarr.set(combinedArray, targetSelection, sliceData);
    }

    // Compute new metadata
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

  // Determine if we should use vector mode for multi-channel images.
  // Only use vector mode for small channel counts (≤ MAX_VECTOR_COMPONENTS)
  // to avoid itkwasm limitations with VariableLengthVector images.
  const cIndex = image.dims.indexOf("c");
  const hasChannels = cIndex !== -1;
  const numChannels = hasChannels ? image.data.shape[cIndex] : 1;
  const isVector = hasChannels && numChannels <= MAX_VECTOR_COMPONENTS;

  // Handle channel dimension by processing each channel independently when not using vector mode
  if (hasChannels && !isVector) {
    const cDimIndex = cIndex;
    const cSize = numChannels;
    const newDims = image.dims.filter((dim) => dim !== "c");

    // Downsample each channel
    const downsampledSlices: zarr.Array<zarr.DataType, zarr.Readable>[] = [];
    for (let c = 0; c < cSize; c++) {
      // Extract channel slice
      const selection = new Array(image.data.shape.length).fill(null);
      selection[cDimIndex] = c;
      const sliceData = await zarr.get(image.data, selection);

      // Create temporary zarr array for this slice
      const sliceStore = new Map<string, Uint8Array>();
      const sliceRoot = zarr.root(sliceStore);
      const sliceShape = image.data.shape.filter((_, i) => i !== cDimIndex);
      const sliceChunkShape = sliceShape.map((s) => Math.min(s, 256));

      const sliceArray = await zarr.create(sliceRoot.resolve("slice"), {
        shape: sliceShape,
        chunk_shape: sliceChunkShape,
        data_type: image.data.dtype,
        fill_value: 0,
      });

      const fullSelection = new Array(sliceShape.length).fill(null);
      await zarr.set(sliceArray, fullSelection, sliceData);

      // Create NgffImage for this slice (without 'c' dimension)
      const sliceImage = new NgffImage({
        data: sliceArray,
        dims: newDims,
        scale: Object.fromEntries(
          Object.entries(image.scale).filter(([dim]) => dim !== "c"),
        ),
        translation: Object.fromEntries(
          Object.entries(image.translation).filter(([dim]) => dim !== "c"),
        ),
        name: image.name,
        axesUnits: image.axesUnits
          ? Object.fromEntries(
            Object.entries(image.axesUnits).filter(([dim]) => dim !== "c"),
          )
          : undefined,
        computedCallbacks: image.computedCallbacks,
      });

      // Recursively downsample this slice (without 'c', so no infinite loop)
      const downsampledSlice = await downsampleLabelImageImpl(
        sliceImage,
        dimFactors,
        spatialDims,
      );
      downsampledSlices.push(downsampledSlice.data);
    }

    // Combine downsampled slices back into a single array with 'c' dimension
    const firstSlice = downsampledSlices[0];
    const combinedShape = [...image.data.shape];
    combinedShape[cDimIndex] = cSize;
    // Update spatial dimensions based on downsampled size
    for (let i = 0; i < image.dims.length; i++) {
      if (i !== cDimIndex) {
        const sliceIndex = i < cDimIndex ? i : i - 1;
        combinedShape[i] = firstSlice.shape[sliceIndex];
      }
    }

    // Create combined array
    const combinedStore = new Map<string, Uint8Array>();
    const combinedRoot = zarr.root(combinedStore);
    const combinedArray = await zarr.create(combinedRoot.resolve("combined"), {
      shape: combinedShape,
      chunk_shape: combinedShape.map((s) => Math.min(s, 256)),
      data_type: image.data.dtype,
      fill_value: 0,
    });

    // Copy each downsampled slice into the combined array
    for (let c = 0; c < cSize; c++) {
      const sliceData = await zarr.get(downsampledSlices[c]);
      const targetSelection = new Array(combinedShape.length).fill(null);
      targetSelection[cDimIndex] = c;
      await zarr.set(combinedArray, targetSelection, sliceData);
    }

    // Compute new metadata (channel dimension unchanged, spatial dimensions downsampled)
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

  // Convert to ITK-Wasm format
  const itkImage = await zarrToItkImage(image.data, image.dims, isVector);

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

  // Perform downsampling using browser-compatible function
  const { downsampled } = await downsampleLabelImage(itkImage, {
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
  const chunkShape = downsampled.size.map((s) => Math.min(s, 256)).reverse();
  const array = await itkImageToZarr(
    downsampled,
    store,
    "image",
    chunkShape,
    image.dims,
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
 * Main downsampling function for ITK-Wasm (browser version)
 */
export async function downsampleItkWasm(
  ngffImage: NgffImage,
  scaleFactors: (Record<string, number> | number)[],
  smoothing: "gaussian" | "bin_shrink" | "label_image",
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

    if (smoothing === "bin_shrink") {
      // Purely incremental: scaleFactor is the shrink for this step
      sourceImage = previousImage;
      sourceDimFactors = {} as DimFactors;
      if (typeof scaleFactor === "number") {
        for (const dim of spatialDims) sourceDimFactors[dim] = scaleFactor;
      } else {
        for (const dim of spatialDims) {
          sourceDimFactors[dim] = scaleFactor[dim] || 1;
        }
      }
      for (const dim of dims) {
        if (!(dim in sourceDimFactors)) sourceDimFactors[dim] = 1;
      }
    } else {
      // Hybrid absolute strategy
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
        if (dimIndex >= 0) {
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
        sourceImage,
        sourceDimFactors,
        spatialDims,
      );
    } else if (smoothing === "bin_shrink") {
      downsampled = await downsampleBinShrinkImpl(
        sourceImage,
        sourceDimFactors,
        spatialDims,
      );
    } else if (smoothing === "label_image") {
      downsampled = await downsampleLabelImageImpl(
        sourceImage,
        sourceDimFactors,
        spatialDims,
      );
    } else {
      throw new Error(`Unknown smoothing method: ${smoothing}`);
    }

    multiscales.push(downsampled);

    previousImage = downsampled;
    if (smoothing === "bin_shrink") {
      if (typeof scaleFactor === "number") {
        for (const dim of spatialDims) {
          previousDimFactors[dim] *= scaleFactor;
        }
      } else {
        for (const dim of spatialDims) {
          previousDimFactors[dim] *= scaleFactor[dim] || 1;
        }
      }
    } else {
      previousDimFactors = updatePreviousDimFactors(
        scaleFactor,
        spatialDims,
        previousDimFactors,
      );
    }
  }

  return multiscales;
}
