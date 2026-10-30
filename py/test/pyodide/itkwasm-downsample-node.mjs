// The itkwasm-downsample Pyodide wrapper imports the package's browser build,
// which runs each pipeline in a Web Worker; Node has no Worker global, so the
// harness hands it the Node build (local pipelines, no workers) under the
// browser names and signatures.
import * as pkg from "@itk-wasm/downsample";

const viaNode = (name) => (...args) => pkg[`${name}Node`](...args);
export const downsample = viaNode("downsample");
export const downsampleBinShrink = viaNode("downsampleBinShrink");
export const downsampleLabelImage = viaNode("downsampleLabelImage");
export const gaussianKernelRadius = viaNode("gaussianKernelRadius");
