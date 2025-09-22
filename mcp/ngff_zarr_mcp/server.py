# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""MCP server for ngff-zarr image conversion."""

from typing import List, Optional, Dict, Literal, Union

from mcp.server.fastmcp import FastMCP

from .models import (
    ConversionOptions,
    ConversionResult,
    StoreInfo,
    OptimizationOptions,
    ValidationResult,
)
from .tools import (
    convert_to_ome_zarr,
    read_ngff_zarr,
    inspect_ome_zarr,
    validate_ome_zarr,
    optimize_zarr_store,
)
from .utils import (
    get_supported_formats,
    get_available_methods,
    get_available_compression_codecs,
)

# Create the MCP server
mcp = FastMCP("ngff-zarr")


@mcp.tool()
async def convert_images_to_ome_zarr(
    input_paths: List[str],
    output_path: str,
    ome_zarr_version: str = "0.4",
    dims: Optional[List[str]] = None,
    scale: Optional[Dict[str, float]] = None,
    translation: Optional[Dict[str, float]] = None,
    units: Optional[Dict[str, str]] = None,
    name: Optional[str] = None,
    chunks: Optional[List[int]] = None,
    chunks_per_shard: Optional[List[int]] = None,
    method: str = "itkwasm_gaussian",
    scale_factors: Optional[List[int]] = None,
    compression_codec: Optional[str] = None,
    compression_level: Optional[int] = None,
    use_tensorstore: bool = False,
    use_local_cluster: bool = False,
    cache_dir: Optional[str] = None,
    # New RFC 4 and storage options
    anatomical_orientation: Optional[str] = None,
    enable_rfc4: bool = False,
    enabled_rfcs: Optional[List[int]] = None,
    storage_options: Optional[Dict[str, str]] = None,
) -> ConversionResult:
    """
    Convert images to OME-Zarr format.

    Args:
        input_paths: List of input image paths (local files, URLs, or S3 URLs)
        output_path: Output path for OME-Zarr store
        ome_zarr_version: OME-Zarr version (0.4 or 0.5)
        dims: Ordered NGFF dimensions from {t,z,y,x,c}
        scale: Scale/spacing for each dimension
        translation: Translation/origin for each dimension
        units: Units for each dimension
        name: Image name
        chunks: Dask array chunking specification
        chunks_per_shard: Chunks per shard for sharding (Zarr v3 only)
        method: Downsampling method
        scale_factors: Scale factors for multiscale generation
        compression_codec: Compression codec
        compression_level: Compression level
        use_tensorstore: Use TensorStore for I/O
        use_local_cluster: Use Dask LocalCluster for large datasets
        cache_dir: Directory for caching
        anatomical_orientation: Anatomical orientation preset (LPS, RAS)
        enable_rfc4: Enable RFC 4 anatomical orientation support
        enabled_rfcs: List of RFC numbers to enable
        storage_options: Storage options for remote stores (S3, GCS, etc.)

    Returns:
        ConversionResult with success status and store information
    """

    # Validate ome_zarr_version
    if ome_zarr_version not in ["0.4", "0.5"]:
        return ConversionResult(
            success=False,
            output_path="",
            store_info={},
            error=f"Invalid OME-Zarr version: {ome_zarr_version}. Must be '0.4' or '0.5'",
        )

    # Cast to proper type for mypy
    validated_version: Literal["0.4", "0.5"] = ome_zarr_version  # type: ignore[assignment]

    options = ConversionOptions(
        output_path=output_path,
        ome_zarr_version=validated_version,
        dims=dims,
        scale=scale,
        translation=translation,
        units=units,
        name=name,
        chunks=chunks,
        chunks_per_shard=chunks_per_shard,
        method=method,
        scale_factors=scale_factors,  # type: ignore[arg-type]
        compression_codec=compression_codec,
        compression_level=compression_level,
        use_tensorstore=use_tensorstore,
        use_local_cluster=use_local_cluster,
        cache_dir=cache_dir,
        anatomical_orientation=anatomical_orientation,
        enable_rfc4=enable_rfc4,
        enabled_rfcs=enabled_rfcs,
        storage_options=storage_options,  # type: ignore[arg-type]
    )

    return await convert_to_ome_zarr(input_paths, options)


@mcp.tool()
async def get_ome_zarr_info(store_path: str) -> StoreInfo:
    """
    Get detailed information about an OME-Zarr store.

    Args:
        store_path: Path to OME-Zarr store (local or URL)

    Returns:
        StoreInfo with detailed store information
    """
    return await inspect_ome_zarr(store_path)


@mcp.tool()
async def read_ome_zarr_store(
    store_path: str,
    storage_options: Optional[Dict[str, Union[str, int, bool]]] = None,
    validate: bool = False,
) -> ConversionResult:
    """
    Read OME-Zarr data from a store with optional storage options.

    Args:
        store_path: Path or URL to OME-Zarr store
        storage_options: Storage options for remote stores (S3, GCS, etc.)
        validate: Whether to validate the NGFF metadata

    Returns:
        ConversionResult with store information and any detected features
    """
    return await read_ngff_zarr(store_path, storage_options, validate)


@mcp.tool()
async def validate_ome_zarr_store(store_path: str) -> ValidationResult:
    """
    Validate an OME-Zarr store structure and metadata.

    Args:
        store_path: Path to OME-Zarr store (local or URL)

    Returns:
        ValidationResult with validation status and any errors/warnings
    """
    return await validate_ome_zarr(store_path)


@mcp.tool()
async def optimize_ome_zarr_store(
    input_path: str,
    output_path: str,
    compression_codec: Optional[str] = None,
    compression_level: Optional[int] = None,
    chunks: Optional[List[int]] = None,
    chunks_per_shard: Optional[List[int]] = None,
) -> ConversionResult:
    """
    Optimize an existing OME-Zarr store with new compression/chunking.

    Args:
        input_path: Path to input OME-Zarr store
        output_path: Path for optimized output store
        compression_codec: New compression codec
        compression_level: New compression level
        chunks: New chunk sizes
        chunks_per_shard: New sharding configuration

    Returns:
        ConversionResult with optimization results
    """

    options = OptimizationOptions(
        input_path=input_path,
        output_path=output_path,
        compression_codec=compression_codec,
        compression_level=compression_level,
        chunks=chunks,
        chunks_per_shard=chunks_per_shard,
        storage_options=None,  # Add the required parameter
    )

    return await optimize_zarr_store(options)


@mcp.resource("ngff-zarr://supported-formats")
def get_supported_input_output_formats() -> str:
    """Get information about supported input and output formats."""
    formats = get_supported_formats()

    output = ["# Supported Formats\n"]

    output.append("## Input Formats by Backend:\n")
    for backend, extensions in formats.input_formats.items():
        output.append(f"**{backend}**: {', '.join(extensions)}\n")

    output.append("\n## Output Formats:\n")
    output.append(f"{', '.join(formats.output_formats)}\n")

    output.append("\n## Available Backends:\n")
    output.append(f"{', '.join(formats.backends)}\n")

    return "".join(output)


@mcp.resource("ngff-zarr://downsampling-methods")
def get_downsampling_methods() -> str:
    """Get information about available downsampling methods."""
    methods = get_available_methods()

    output = ["# Available Downsampling Methods\n\n"]

    method_descriptions = {
        "itkwasm_gaussian": "Gaussian smoothing with ITK-Wasm (default, fast, portable)",
        "itkwasm_bin_shrink": "Bin shrinking with ITK-Wasm (fast, some artifacts)",
        "itkwasm_label_image": "Label-aware smoothing with ITK-Wasm (for label images)",
        "itk_gaussian": "Gaussian smoothing with ITK (native, GPU acceleration available)",
        "itk_bin_shrink": "Bin shrinking with ITK (native, fast)",
        "dask_image_gaussian": "Gaussian smoothing with dask-image (scipy-based)",
        "dask_image_mode": "Mode-based downsampling for labels (slower, fewer artifacts)",
        "dask_image_nearest": "Nearest neighbor for labels (artifacts, fast)",
    }

    for method in methods:
        description = method_descriptions.get(method, "No description available")
        output.append(f"- **{method}**: {description}\n")

    return "".join(output)


@mcp.resource("ngff-zarr://compression-codecs")
def get_compression_codecs() -> str:
    """Get information about available compression codecs."""
    codecs = get_available_compression_codecs()

    output = ["# Available Compression Codecs\n\n"]

    codec_descriptions = {
        "gzip": "General purpose, good compression ratio, slower",
        "lz4": "Fast compression/decompression, moderate compression",
        "zstd": "Good balance of speed and compression ratio",
        "blosc": "Meta-compressor with various algorithms",
    }

    for codec in codecs:
        base_codec = codec.split(":")[0]
        description = codec_descriptions.get(base_codec, "High-performance compression")
        output.append(f"- **{codec}**: {description}\n")

    return "".join(output)


def main():
    """Main entry point for the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="ngff-zarr MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mechanism to use",
    )
    parser.add_argument("--host", default="localhost", help="Host for SSE transport")
    parser.add_argument("--port", type=int, default=8000, help="Port for SSE transport")

    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run()
    elif args.transport == "sse":
        mcp.run_sse(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
