# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Data models for ngff-zarr MCP server."""

from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator


class ConversionOptions(BaseModel):
    """Options for image conversion to OME-Zarr."""

    # Output options
    output_path: str = Field(..., description="Output path for OME-Zarr store")
    ome_zarr_version: Literal["0.4", "0.5"] = Field(
        "0.4", description="OME-Zarr version"
    )

    # Metadata options
    dims: Optional[List[str]] = Field(
        None, description="Ordered NGFF dimensions from {t,z,y,x,c}"
    )
    scale: Optional[Dict[str, float]] = Field(
        None, description="Scale/spacing for each dimension"
    )
    translation: Optional[Dict[str, float]] = Field(
        None, description="Translation/origin for each dimension"
    )
    units: Optional[Dict[str, str]] = Field(
        None, description="Units for each dimension"
    )
    name: Optional[str] = Field(None, description="Image name")

    # RFC 4 - Anatomical Orientation support
    anatomical_orientation: Optional[str] = Field(
        None, description="Anatomical orientation preset (LPS, RAS) or custom mapping"
    )
    enable_rfc4: bool = Field(
        False, description="Enable RFC 4 anatomical orientation support"
    )

    # Storage options for cloud/remote storage
    storage_options: Optional[Dict[str, Union[str, int, bool]]] = Field(
        None, description="Storage options for remote stores (S3, GCS, etc.)"
    )

    # Processing options
    chunks: Optional[Union[int, List[int], tuple[int, ...]]] = Field(
        None, description="Dask array chunking specification"
    )
    chunks_per_shard: Optional[Union[int, List[int], tuple[int, ...]]] = Field(
        None, description="Chunks per shard for sharding"
    )
    method: str = Field("itkwasm_gaussian", description="Downsampling method")
    scale_factors: Optional[Union[int, List[Union[int, Dict[str, int]]]]] = Field(
        None, description="Scale factors for multiscale"
    )

    # Storage options
    compression_codec: Optional[str] = Field(
        None, description="Compression codec (gzip, lz4, zstd, blosc)"
    )
    compression_level: Optional[int] = Field(None, description="Compression level")
    use_tensorstore: bool = Field(False, description="Use TensorStore for I/O")

    # Performance options
    use_local_cluster: bool = Field(
        False, description="Use Dask LocalCluster for large datasets"
    )
    cache_dir: Optional[str] = Field(None, description="Directory for caching")

    # RFC and advanced features
    enabled_rfcs: Optional[List[int]] = Field(
        None,
        description="List of RFC numbers to enable (e.g., [4] for anatomical orientation)",
    )

    @field_validator("dims")
    @classmethod
    def validate_dims(cls, v):
        if v is not None:
            valid_dims = {"t", "z", "y", "x", "c"}
            if not all(dim in valid_dims for dim in v):
                raise ValueError(f"All dimensions must be from {valid_dims}")
        return v


class ConversionResult(BaseModel):
    """Result of image conversion."""

    success: bool = Field(..., description="Whether conversion succeeded")
    output_path: str = Field(..., description="Path to output OME-Zarr store")
    store_info: Dict = Field(..., description="Information about the created store")
    error: Optional[str] = Field(None, description="Error message if conversion failed")


class StoreInfo(BaseModel):
    """Information about an OME-Zarr store."""

    path: str = Field(..., description="Path to the store")
    version: str = Field(..., description="OME-Zarr version")
    size_bytes: int = Field(..., description="Total size in bytes")
    num_files: int = Field(..., description="Number of files")
    num_scales: int = Field(..., description="Number of scales in multiscale")
    dimensions: List[str] = Field(..., description="Image dimensions")
    shape: List[int] = Field(..., description="Image shape")
    dtype: str = Field(..., description="Data type")
    chunks: Union[List[int], tuple[int, ...]] = Field(..., description="Chunk sizes")
    compression: Optional[str] = Field(None, description="Compression codec")
    scale_info: Dict = Field(..., description="Scale/spacing information")
    translation_info: Dict = Field(..., description="Translation/origin information")
    method_type: Optional[str] = Field(None, description="Multiscale method type")
    method_metadata: Optional[Dict] = Field(
        None, description="Method metadata information"
    )
    anatomical_orientation: Optional[Dict] = Field(
        None, description="Anatomical orientation information"
    )
    rfc_support: Optional[List[int]] = Field(None, description="Enabled RFC features")


class SupportedFormats(BaseModel):
    """Supported input and output formats."""

    input_formats: Dict[str, List[str]] = Field(
        ..., description="Supported input formats by backend"
    )
    output_formats: List[str] = Field(..., description="Supported output formats")
    backends: List[str] = Field(..., description="Available conversion backends")


class OptimizationOptions(BaseModel):
    """Options for optimizing an existing Zarr store."""

    input_path: str = Field(..., description="Path to input Zarr store")
    output_path: str = Field(..., description="Path for optimized output store")
    compression_codec: Optional[str] = Field(None, description="New compression codec")
    compression_level: Optional[int] = Field(None, description="New compression level")
    chunks: Optional[Union[int, List[int], tuple[int, ...]]] = Field(
        None, description="New chunk sizes"
    )
    chunks_per_shard: Optional[Union[int, List[int], tuple[int, ...]]] = Field(
        None, description="New sharding configuration"
    )
    storage_options: Optional[Dict[str, Union[str, int, bool]]] = Field(
        None, description="Storage options for remote stores"
    )


class ValidationResult(BaseModel):
    """Result of OME-Zarr validation."""

    valid: bool = Field(..., description="Whether the store is valid")
    version: Optional[str] = Field(None, description="OME-Zarr version if valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
