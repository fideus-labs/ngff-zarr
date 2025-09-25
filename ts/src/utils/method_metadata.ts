import type { Methods } from "../types/methods.ts";
import type { MethodMetadata } from "../types/zarr_metadata.ts";

interface MethodInfo {
  description: string;
  package: string;
  method: string;
}

const METHOD_INFO: Record<string, MethodInfo> = {
  itkwasm_gaussian: {
    description:
      "Smoothed with a discrete gaussian filter to generate a scale space, ideal for intensity images. ITK-Wasm implementation is extremely portable and SIMD accelerated.",
    package: "itkwasm-downsample",
    method: "itkwasm_downsample.downsample",
  },
  itkwasm_bin_shrink: {
    description:
      "Uses the local mean for the output value. WebAssembly build. Fast but generates more artifacts than gaussian-based methods. Appropriate for intensity images.",
    package: "itkwasm-downsample",
    method: "itkwasm_downsample.downsample_bin_shrink",
  },
  itkwasm_label_image: {
    description:
      "A sample is the mode of the linearly weighted local labels in the image. Fast and minimal artifacts. For label images.",
    package: "itkwasm-downsample",
    method: "itkwasm_downsample.downsample_label_image",
  },
};

/**
 * Get metadata information for a given downsampling method.
 *
 * @param method - The downsampling method enum
 * @returns MethodMetadata with description, method (package.function), and version
 */
export function getMethodMetadata(method: Methods): MethodMetadata | undefined {
  const methodInfo = METHOD_INFO[method];
  if (!methodInfo) {
    return undefined;
  }

  // For TypeScript/browser environment, we can't easily get package versions
  // so we'll use a placeholder or try to infer from known versions
  const version = "unknown";

  return {
    description: methodInfo.description,
    method: methodInfo.method,
    version,
  };
}
