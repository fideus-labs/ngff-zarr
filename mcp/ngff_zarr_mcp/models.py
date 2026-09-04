# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Data models for ngff-zarr MCP server."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _reject_level_without_codec(options):
    """A compression level is only ever read alongside a codec name.

    Both tools build the compressor from the two together and skip it
    entirely when the codec is absent, so a lone level asks for a setting
    the write would not apply -- and would report success anyway.
    """
    if options.compression_level is not None and not options.compression_codec:
        raise ValueError(
            "compression_level requires compression_codec: the level is an "
            "argument to the codec, and without one the store is written "
            "with the default compression."
        )
    return options


class ConversionOptions(BaseModel):
    """Options for image conversion to OME-Zarr."""

    # Output options
    output_path: str = Field(..., description="Output path for OME-Zarr store")
    ome_zarr_version: Literal["0.4", "0.5"] = Field(
        "0.5", description="OME-Zarr version"
    )

    # Metadata options
    dims: list[str] | None = Field(
        None, description="Ordered NGFF dimensions from {t,z,y,x,c}"
    )
    scale: dict[str, float] | None = Field(
        None, description="Scale/spacing for each dimension"
    )
    translation: dict[str, float] | None = Field(
        None, description="Translation/origin for each dimension"
    )
    units: dict[str, str] | None = Field(None, description="Units for each dimension")
    name: str | None = Field(None, description="Image name")

    # RFC 4 - Anatomical Orientation support. Anatomical orientation is written
    # automatically whenever it is present, so specifying a preset here is all
    # that is required to enable it.
    anatomical_orientation: str | None = Field(
        None,
        description=(
            "Anatomical orientation preset (LPS, RAS). Applies to file/array "
            "inputs only; existing zarr-store inputs keep their source "
            "orientation metadata."
        ),
    )

    # Storage options for cloud/remote storage
    storage_options: dict[str, str | int | bool] | None = Field(
        None, description="Storage options for remote stores (S3, GCS, etc.)"
    )

    # Processing options
    chunks: int | list[int] | tuple[int, ...] | None = Field(
        None, description="Dask array chunking specification"
    )
    chunks_per_shard: int | list[int] | tuple[int, ...] | None = Field(
        None, description="Chunks per shard for sharding"
    )
    method: str = Field("itkwasm_gaussian", description="Downsampling method")
    scale_factors: int | list[int | dict[str, int]] | None = Field(
        None, description="Scale factors for multiscale"
    )

    # Storage options
    compression_codec: str | None = Field(
        None, description="Compression codec (gzip, lz4, zstd, blosc)"
    )
    compression_level: int | None = Field(None, description="Compression level")
    use_tensorstore: bool = Field(
        False, description="Deprecated: uses the zarrista backend for I/O"
    )
    consolidate_metadata: bool = Field(
        True,
        description=(
            "Write consolidated metadata. Set False for stores destined for a "
            "backend that rejects it, such as Icechunk"
        ),
    )

    # Performance options
    use_local_cluster: bool = Field(
        False, description="Use Dask LocalCluster for large datasets"
    )
    cache_dir: str | None = Field(None, description="Directory for caching")

    @field_validator("dims")
    @classmethod
    def validate_dims(cls, v):
        if v is not None:
            valid_dims = {"t", "z", "y", "x", "c"}
            if not all(dim in valid_dims for dim in v):
                raise ValueError(f"All dimensions must be from {valid_dims}")
        return v

    @model_validator(mode="after")
    def validate_compression(self):
        return _reject_level_without_codec(self)


class ConversionResult(BaseModel):
    """Result of image conversion."""

    success: bool = Field(..., description="Whether conversion succeeded")
    output_path: str = Field(..., description="Path to output OME-Zarr store")
    store_info: dict = Field(..., description="Information about the created store")
    error: str | None = Field(None, description="Error message if conversion failed")


class StoreInfo(BaseModel):
    """Information about an OME-Zarr store."""

    path: str = Field(..., description="Path to the store")
    version: str = Field(..., description="OME-Zarr version")
    size_bytes: int = Field(..., description="Total size in bytes")
    num_files: int = Field(..., description="Number of files")
    num_scales: int = Field(..., description="Number of scales in multiscale")
    dimensions: list[str] = Field(..., description="Image dimensions")
    shape: list[int] = Field(..., description="Image shape")
    dtype: str = Field(..., description="Data type")
    chunks: list[int] | tuple[int, ...] = Field(..., description="Chunk sizes")
    compression: str | None = Field(None, description="Compression codec")
    scale_info: dict = Field(..., description="Scale/spacing information")
    translation_info: dict = Field(..., description="Translation/origin information")
    method_type: str | None = Field(None, description="Multiscale method type")
    method_metadata: dict | None = Field(
        None, description="Method metadata information"
    )
    anatomical_orientation: dict | None = Field(
        None, description="Anatomical orientation information"
    )
    rfc_support: list[int] | None = Field(None, description="Enabled RFC features")


class SupportedFormats(BaseModel):
    """Supported input and output formats."""

    input_formats: dict[str, list[str]] = Field(
        ..., description="Supported input formats by backend"
    )
    output_formats: list[str] = Field(..., description="Supported output formats")
    backends: list[str] = Field(..., description="Available conversion backends")


class OptimizationOptions(BaseModel):
    """Options for optimizing an existing Zarr store."""

    input_path: str = Field(..., description="Path to input Zarr store")
    output_path: str = Field(..., description="Path for optimized output store")
    compression_codec: str | None = Field(None, description="New compression codec")
    compression_level: int | None = Field(None, description="New compression level")
    chunks: int | list[int] | tuple[int, ...] | None = Field(
        None, description="New chunk sizes"
    )
    chunks_per_shard: int | list[int] | tuple[int, ...] | None = Field(
        None, description="New sharding configuration"
    )
    storage_options: dict[str, str | int | bool] | None = Field(
        None, description="Storage options for remote stores"
    )
    consolidate_metadata: bool = Field(
        True,
        description=(
            "Write consolidated metadata. Set False for stores destined for a "
            "backend that rejects it, such as Icechunk"
        ),
    )

    @model_validator(mode="after")
    def validate_compression(self):
        return _reject_level_without_codec(self)


class ValidationResult(BaseModel):
    """Result of OME-Zarr validation."""

    valid: bool = Field(..., description="Whether the store is valid")
    version: str | None = Field(None, description="OME-Zarr version if valid")
    errors: list[str] = Field(default_factory=list, description="Validation errors")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")
