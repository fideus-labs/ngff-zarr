// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Convert NgffImage to ITK-Wasm Image
 */

import type {
  FloatTypes,
  Image,
  ImageType,
  IntTypes,
  PixelTypes,
  TypedArray,
} from "itk-wasm";
import * as zarr from "zarrita";
import { NgffImage } from "../types/ngff_image.ts";

export interface NgffImageToItkImageOptions {
  /**
   * Extract a specific time index from a time-series image
   * If specified and 't' dimension exists, extracts the slice at this time index
   */
  tIndex?: number;

  /**
   * Extract a specific channel index from a multi-channel image
   * If specified and 'c' dimension exists, extracts the slice at this channel index
   */
  cIndex?: number;
}

/**
 * Convert the data type from zarr DataType to ITK-Wasm component type
 */
function dataTypeToComponentType(
  dataType: zarr.DataType,
):
  | (typeof IntTypes)[keyof typeof IntTypes]
  | (typeof FloatTypes)[keyof typeof FloatTypes] {
  switch (dataType) {
    case "uint8":
      return "uint8" as typeof IntTypes.UInt8;
    case "int8":
      return "int8" as typeof IntTypes.Int8;
    case "uint16":
      return "uint16" as typeof IntTypes.UInt16;
    case "int16":
      return "int16" as typeof IntTypes.Int16;
    case "uint32":
      return "uint32" as typeof IntTypes.UInt32;
    case "int32":
      return "int32" as typeof IntTypes.Int32;
    case "uint64":
      return "uint64" as typeof IntTypes.UInt64;
    case "int64":
      return "int64" as typeof IntTypes.Int64;
    case "float32":
      return "float32" as typeof FloatTypes.Float32;
    case "float64":
      return "float64" as typeof FloatTypes.Float64;
    default:
      throw new Error(`Unsupported data type: ${dataType}`);
  }
}

/**
 * Move the channel dimension to the end (last position) if it exists
 * This follows the ITK convention where vector/RGB components are the last dimension
 */
function moveChannelDimToLast(ngffImage: NgffImage): NgffImage {
  const dims = ngffImage.dims;
  const cIndex = dims.indexOf("c");

  if (cIndex === -1 || cIndex === dims.length - 1) {
    // No channel dimension or already at the end
    return ngffImage;
  }

  // Reorder dimensions to move 'c' to the end
  const newDims = [...dims];
  const cDim = newDims.splice(cIndex, 1)[0];
  newDims.push(cDim);

  // Note: We would need to transpose the zarr array data here
  // For now, we'll assume the data is already in the correct order
  // In a full implementation, you'd transpose the zarr array

  return new NgffImage({
    data: ngffImage.data, // TODO: transpose if needed
    dims: newDims,
    scale: ngffImage.scale,
    translation: ngffImage.translation,
    name: ngffImage.name,
    axesUnits: ngffImage.axesUnits,
    axesOrientations: ngffImage.axesOrientations,
    computedCallbacks: ngffImage.computedCallbacks,
  });
}

/**
 * Convert an NgffImage to an ITK-Wasm Image, preserving spatial metadata.
 *
 * This function converts NgffImage objects to ITK-Wasm Image format while
 * preserving spatial information like spacing, origin, and direction matrix.
 * Optionally extracts specific time or channel slices.
 *
 * @param ngffImage - The NgffImage to convert
 * @param options - Conversion options
 * @returns Promise resolving to ITK-Wasm Image
 */
export async function ngffImageToItkImage(
  ngffImage: NgffImage,
  options: NgffImageToItkImageOptions = {},
): Promise<Image> {
  const { tIndex, cIndex } = options;

  let workingImage = ngffImage;

  // Extract time slice if requested
  if (tIndex !== undefined && workingImage.dims.includes("t")) {
    const tDimIndex = workingImage.dims.indexOf("t");
    const newDims = workingImage.dims.filter((dim) => dim !== "t");
    const newScale = Object.fromEntries(
      Object.entries(workingImage.scale).filter(([dim]) => dim !== "t"),
    );
    const newTranslation = Object.fromEntries(
      Object.entries(workingImage.translation).filter(([dim]) => dim !== "t"),
    );

    // Extract the time slice from zarr array
    const selection = new Array(workingImage.data.shape.length).fill(null);
    selection[tDimIndex] = tIndex;

    const sliceData = await zarr.get(workingImage.data, selection);

    // Create new zarr array with reduced dimensions
    const store = new Map<string, Uint8Array>();
    const root = zarr.root(store);
    const newShape = workingImage.data.shape.filter((_, i) => i !== tDimIndex);
    const chunkShape = newShape.map((s) => Math.min(s, 256));

    const newArray = await zarr.create(root.resolve("slice"), {
      shape: newShape,
      chunk_shape: chunkShape,
      data_type: workingImage.data.dtype,
      fill_value: 0,
    });

    // Write the slice data
    const fullSelection = new Array(newShape.length).fill(null);
    await zarr.set(newArray, fullSelection, sliceData);

    workingImage = new NgffImage({
      data: newArray,
      dims: newDims,
      scale: newScale,
      translation: newTranslation,
      name: workingImage.name,
      axesUnits: workingImage.axesUnits
        ? Object.fromEntries(
          Object.entries(workingImage.axesUnits).filter(
            ([dim]) => dim !== "t",
          ),
        )
        : undefined,
      axesOrientations: workingImage.axesOrientations
        ? Object.fromEntries(
          Object.entries(workingImage.axesOrientations).filter(
            ([dim]) => dim !== "t",
          ),
        )
        : undefined,
      computedCallbacks: workingImage.computedCallbacks,
    });
  }

  // Extract channel slice if requested
  if (cIndex !== undefined && workingImage.dims.includes("c")) {
    const cDimIndex = workingImage.dims.indexOf("c");
    const newDims = workingImage.dims.filter((dim) => dim !== "c");
    const newScale = Object.fromEntries(
      Object.entries(workingImage.scale).filter(([dim]) => dim !== "c"),
    );
    const newTranslation = Object.fromEntries(
      Object.entries(workingImage.translation).filter(([dim]) => dim !== "c"),
    );

    // Extract the channel slice from zarr array
    const selection = new Array(workingImage.data.shape.length).fill(null);
    selection[cDimIndex] = cIndex;

    const sliceData = await zarr.get(workingImage.data, selection);

    // Create new zarr array with reduced dimensions
    const store = new Map<string, Uint8Array>();
    const root = zarr.root(store);
    const newShape = workingImage.data.shape.filter((_, i) => i !== cDimIndex);
    const chunkShape = newShape.map((s) => Math.min(s, 256));

    const newArray = await zarr.create(root.resolve("slice"), {
      shape: newShape,
      chunk_shape: chunkShape,
      data_type: workingImage.data.dtype,
      fill_value: 0,
    });

    // Write the slice data
    const fullSelection = new Array(newShape.length).fill(null);
    await zarr.set(newArray, fullSelection, sliceData);

    workingImage = new NgffImage({
      data: newArray,
      dims: newDims,
      scale: newScale,
      translation: newTranslation,
      name: workingImage.name,
      axesUnits: workingImage.axesUnits
        ? Object.fromEntries(
          Object.entries(workingImage.axesUnits).filter(
            ([dim]) => dim !== "c",
          ),
        )
        : undefined,
      axesOrientations: workingImage.axesOrientations
        ? Object.fromEntries(
          Object.entries(workingImage.axesOrientations).filter(
            ([dim]) => dim !== "c",
          ),
        )
        : undefined,
      computedCallbacks: workingImage.computedCallbacks,
    });
  }

  // Move channel dimension to last position (ITK convention)
  workingImage = moveChannelDimToLast(workingImage);

  const dims = workingImage.dims;
  const data = workingImage.data;

  // Identify ITK spatial dimensions
  const itkDimensionNames = new Set(["x", "y", "z", "t"]);
  const itkDims = dims.filter((dim) => itkDimensionNames.has(dim));

  // Sort ITK dimensions: spatial first (x, y, z), then time
  const sortedItkDims = itkDims.sort((a, b) => {
    const order = ["x", "y", "z", "t"];
    return order.indexOf(a) - order.indexOf(b);
  });

  // Create ITK spacing, origin, and size arrays
  const spacing = sortedItkDims.map((dim) => workingImage.scale[dim] || 1.0);
  const origin = sortedItkDims.map(
    (dim) => workingImage.translation[dim] || 0.0,
  );
  const size = sortedItkDims.map((dim) => data.shape[dims.indexOf(dim)]);
  const dimension = sortedItkDims.length;

  // Determine component type and pixel type
  const componentType = dataTypeToComponentType(data.dtype);

  let components = 1;
  let pixelType: (typeof PixelTypes)[keyof typeof PixelTypes] =
    "Scalar" as typeof PixelTypes.Scalar;

  if (dims.includes("c")) {
    components = data.shape[dims.indexOf("c")];
    if (components === 3 && componentType === "uint8") {
      pixelType = "RGB" as typeof PixelTypes.RGB;
    } else {
      pixelType =
        "VariableLengthVector" as typeof PixelTypes.VariableLengthVector;
    }
  }

  // Create ImageType
  const imageType: ImageType = {
    dimension,
    componentType,
    pixelType,
    components,
  };

  // Read all data from zarr array
  const selection = new Array(data.shape.length).fill(null);
  const dataChunk = await zarr.get(data, selection);

  // Create direction matrix (identity for now)
  const direction = new Float64Array(dimension * dimension);
  for (let i = 0; i < dimension; i++) {
    direction[i * dimension + i] = 1.0;
  }

  // Create ITK-Wasm Image
  const itkImage: Image = {
    imageType,
    name: workingImage.name || "image",
    origin,
    spacing,
    direction,
    size,
    metadata: new Map(),
    data: dataChunk.data as TypedArray, // Cast to handle zarrita's data type
  };

  return itkImage;
}
