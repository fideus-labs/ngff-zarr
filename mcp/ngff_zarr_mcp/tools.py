# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""MCP tools for ngff-zarr image conversion."""

import tempfile
import warnings
from pathlib import Path

from ngff_zarr import (  # type: ignore[import-untyped]
    Methods,
    cli_input_to_ngff_image,
    detect_cli_io_backend,
    from_ome_zarr,
    to_multiscales,
    to_ome_zarr,
)

# Import validation function if available
try:
    from ngff_zarr import validate as validate_ngff
except ImportError:
    # Fallback if ngff-zarr schema validation is unavailable (no-op stub whose
    # signature mirrors ngff_zarr.validate).
    def validate_ngff(
        ngff_dict: dict,
        version: str = "0.4",
        model: str = "image",
        strict: bool = False,
    ) -> None:
        pass


from .models import (
    ConversionOptions,
    ConversionResult,
    OptimizationOptions,
    StoreInfo,
    ValidationResult,
)
from .utils import (
    analyze_zarr_store,
    detect_ome_zarr_version,
    is_url,
    is_zarr_store,
    prepare_input_files,
    read_root_attrs,
    setup_dask_config,
    validate_conversion_options,
)


async def convert_to_ome_zarr(
    input_paths: list[str], options: ConversionOptions
) -> ConversionResult:
    """Convert input images to OME-Zarr format."""

    try:
        # Validate options
        validation_errors = validate_conversion_options(options.model_dump())
        if validation_errors:
            return ConversionResult(
                success=False,
                output_path="",
                store_info={},
                error=f"Validation errors: {'; '.join(validation_errors)}",
            )

        # Setup Dask if needed
        client = setup_dask_config(
            use_local_cluster=options.use_local_cluster,
            cache_dir=options.cache_dir,
        )

        try:
            # Check if input is already a zarr store
            multiscales_obj = None
            ngff_image = None
            is_zarr_input = False
            multiscales = None  # Initialize for scope

            if len(input_paths) == 1 and (
                is_zarr_store(input_paths[0]) or is_url(input_paths[0])
            ):
                input_path = input_paths[0]

                # Check if it's a zarr store URL by trying to load it
                try:
                    # Try to read as zarr store first
                    multiscales = from_ome_zarr(input_path)

                    # If successful, we have a zarr store input
                    # We'll work with the multiscales directly and apply transformations
                    ngff_image = multiscales.images[0]  # Start with the first scale
                    is_zarr_input = True

                    # For zarr input, we'll use the existing multiscales but may need to re-process
                    # depending on the requested output format and chunking

                except Exception:
                    # Not a zarr store, proceed with normal file handling
                    is_zarr_input = False

            if not is_zarr_input:
                # Prepare input files (download remote files if needed)
                with tempfile.TemporaryDirectory() as temp_dir:
                    prepared_inputs = await prepare_input_files(input_paths, temp_dir)

                    # Detect input backend
                    backend = detect_cli_io_backend(prepared_inputs)

                    # Load input image
                    ngff_image = cli_input_to_ngff_image(backend, prepared_inputs)

            # Apply metadata options (dims, scale, translation, name,
            # anatomical_orientation). These overrides apply only to file/array
            # inputs: for existing zarr-store inputs the source multiscales --
            # including any orientation already in its metadata -- is written
            # through unchanged.
            if not is_zarr_input and ngff_image is not None:
                if options.dims:
                    if len(options.dims) != len(ngff_image.dims):
                        raise ValueError(
                            f"Provided dims count ({len(options.dims)}) doesn't match image dims ({len(ngff_image.dims)})"
                        )
                    ngff_image.dims = options.dims

                if options.scale:
                    for dim, value in options.scale.items():
                        ngff_image.scale[dim] = value

                if options.translation:
                    for dim, value in options.translation.items():
                        ngff_image.translation[dim] = value

                if options.units:
                    # Note: axes_units assignment may need proper unit types
                    # For now, we'll skip this to avoid type errors
                    pass

                if options.name:
                    ngff_image.name = options.name

                # Apply anatomical orientation if requested (RFC 4). Setting the
                # orientations on the image is enough -- orientation is written
                # to the output automatically whenever it is present.
                if options.anatomical_orientation:
                    from ngff_zarr import orientation_from_name

                    ngff_image.axes_orientations = orientation_from_name(
                        options.anatomical_orientation
                    )

            # Setup chunking
            chunks = options.chunks
            if chunks is not None:
                if isinstance(chunks, list) and len(chunks) == 1:
                    chunks = chunks[0]
                elif isinstance(chunks, list):
                    chunks = tuple(chunks)

            # Determine if we need to generate multiscales
            scale_factors = options.scale_factors
            method = Methods(options.method) if options.method else None

            # For zarr inputs, we might want to preserve existing scales or regenerate
            if is_zarr_input and not scale_factors:
                # Use existing multiscales but may need to rechunk
                multiscales_obj = multiscales
            else:
                # Need to create multiscales
                if ngff_image is None:
                    raise ValueError("No valid input image found")

                # Setup caching for large datasets
                cache = False  # Let the system handle caching automatically

                # Always use to_multiscales to create proper NgffMultiscales object
                # If scale_factors is None, create single-scale multiscales
                if scale_factors is None:
                    scale_factors = []

                multiscales_obj = to_multiscales(
                    ngff_image,
                    scale_factors=scale_factors,
                    method=method,
                    chunks=chunks,
                    cache=cache,
                )

            # Ensure we have a valid multiscales object
            if multiscales_obj is None:
                raise ValueError("Failed to create or load multiscales object")

            # Setup output store
            output_path = Path(options.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # to_ome_zarr writes through the zarrista backend to a local path
            output_store = str(output_path)

            # Setup chunks per shard if specified
            chunks_per_shard = options.chunks_per_shard
            if chunks_per_shard is not None:
                if options.ome_zarr_version == "0.4":
                    raise ValueError("Sharding not supported in OME-Zarr v0.4")
                if isinstance(chunks_per_shard, list) and len(chunks_per_shard) == 1:
                    chunks_per_shard = chunks_per_shard[0]
                elif isinstance(chunks_per_shard, list):
                    chunks_per_shard = tuple(chunks_per_shard)

            # Convert to OME-Zarr
            kwargs = {}
            if options.compression_codec:
                from ngff_zarr.codecs import (
                    codec_from_name,  # type: ignore[import-untyped]
                )

                kwargs["compressor"] = codec_from_name(
                    options.compression_codec, options.compression_level
                )

            with warnings.catch_warnings():
                # use_tensorstore is a deprecated alias for the always-on
                # zarrista backend; keep MCP responses clean for agent
                # callers that still pass it.
                warnings.filterwarnings(
                    "ignore",
                    message="use_tensorstore is deprecated",
                    category=DeprecationWarning,
                )
                to_ome_zarr(
                    output_store,
                    multiscales_obj,
                    version=options.ome_zarr_version,
                    chunks_per_shard=chunks_per_shard,
                    use_tensorstore=options.use_tensorstore,
                    **kwargs,
                )

            # Analyze the created store
            store_info = analyze_zarr_store(str(output_path))

            return ConversionResult(
                success=True,
                output_path=str(output_path),
                store_info=store_info.model_dump(),
                error=None,
            )

        finally:
            # Cleanup Dask client
            if client:
                client.close()

    except Exception as e:
        return ConversionResult(
            success=False, output_path="", store_info={}, error=str(e)
        )


async def read_ngff_zarr(
    store_path: str,
    storage_options: dict | None = None,
    validate: bool = False,
) -> ConversionResult:
    """Read OME-Zarr NGFF data from a store.

    Parameters
    ----------
    store_path : str
        Path or URL to the OME-Zarr store
    storage_options : dict, optional
        Storage options for remote stores (S3, GCS, etc.)
    validate : bool
        Whether to validate the NGFF metadata

    Returns
    -------
    ConversionResult
        Result containing store information and any errors
    """
    try:
        # Read the multiscales
        multiscales = from_ome_zarr(store_path, validate=validate)

        # Analyze the store
        store_info = analyze_zarr_store(store_path)

        # Add additional information from multiscales
        store_info_dict = store_info.model_dump()
        if hasattr(multiscales, "method") and multiscales.method:
            store_info_dict["method_type"] = str(multiscales.method)

        # Note: Method metadata and RFC 4 features may not be available in current version
        # These will be added once the features are fully integrated

        return ConversionResult(
            success=True,
            output_path=store_path,
            store_info=store_info_dict,
            error=None,
        )

    except Exception as e:
        return ConversionResult(
            success=False,
            output_path=store_path,
            store_info={},
            error=f"Failed to read store: {str(e)}",
        )


async def inspect_ome_zarr(store_path: str) -> StoreInfo:
    """Inspect an OME-Zarr store and return detailed information."""

    try:
        return analyze_zarr_store(store_path)
    except Exception as e:
        raise ValueError(f"Failed to inspect store: {str(e)}")


async def validate_ome_zarr(store_path: str) -> ValidationResult:
    """Validate an OME-Zarr store."""

    try:
        errors = []
        warnings: list[str] = []
        version = None

        # Check if store exists and is accessible
        if is_url(store_path):
            # For remote stores, basic checks
            try:
                store_info = analyze_zarr_store(store_path)
                version = store_info.version
            except Exception as e:
                errors.append(f"Cannot access remote store: {str(e)}")
                return ValidationResult(
                    valid=False, version=None, errors=errors, warnings=warnings
                )
        else:
            store_path_obj = Path(store_path)
            if not store_path_obj.exists():
                errors.append(f"Store path does not exist: {store_path}")
                return ValidationResult(
                    valid=False, version=None, errors=errors, warnings=warnings
                )

            if not is_zarr_store(store_path):
                errors.append(f"Path is not a valid Zarr store: {store_path}")
                return ValidationResult(
                    valid=False, version=None, errors=errors, warnings=warnings
                )

        # Try to load as NGFF
        try:
            # Use store_path directly - from_ome_zarr can handle strings
            multiscales = from_ome_zarr(store_path)

            # Basic validation checks
            if len(multiscales.images) == 0:
                errors.append("No images found in multiscales")

            # Check each scale
            for i, image in enumerate(multiscales.images):
                if image.data is None:
                    errors.append(f"Scale {i} has no data")
                elif image.data.size == 0:
                    warnings.append(f"Scale {i} has empty data")

            # Determine version from the root metadata documents
            root_attrs: dict = {}
            try:
                root_attrs = read_root_attrs(store_path)
                detected = detect_ome_zarr_version(root_attrs)
                if detected is not None:
                    version = detected
            except Exception:
                version = "0.4"  # Default assumption

            # Try ngff-zarr schema validation if available. ngff_zarr.validate
            # expects the parsed NGFF metadata dict, not a store path.
            try:
                validate_ngff(root_attrs, version=version or "0.4")
            except Exception as validation_error:
                warnings.append(f"NGFF validation warning: {str(validation_error)}")

        except Exception as e:
            errors.append(f"Failed to load as NGFF: {str(e)}")

        return ValidationResult(
            valid=len(errors) == 0, version=version, errors=errors, warnings=warnings
        )

    except Exception as e:
        return ValidationResult(
            valid=False,
            version=None,
            errors=[f"Validation failed: {str(e)}"],
            warnings=[],
        )


async def optimize_zarr_store(options: OptimizationOptions) -> ConversionResult:
    """Optimize an existing Zarr store with new compression/chunking."""

    try:
        # Load existing store
        input_path = Path(options.input_path)
        if not input_path.exists():
            return ConversionResult(
                success=False,
                output_path="",
                store_info={},
                error=f"Input store does not exist: {options.input_path}",
            )

        # Load multiscales
        multiscales = from_ome_zarr(str(input_path))

        # Setup output store
        output_path = Path(options.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_store = str(output_path)

        # Apply optimizations by re-chunking/compressing the data
        optimized_images = []
        for image in multiscales.images:
            optimized_data = image.data

            # Apply new chunking if specified
            if options.chunks is not None:
                chunks = options.chunks
                if isinstance(chunks, list) and len(chunks) == 1:
                    chunks = chunks[0]
                elif isinstance(chunks, list):
                    chunks = tuple(chunks)
                optimized_data = optimized_data.rechunk(chunks)  # type: ignore[arg-type]

            # Create optimized image
            from ngff_zarr.ngff_image import NgffImage  # type: ignore[import-untyped]

            optimized_image = NgffImage(
                optimized_data,  # type: ignore[arg-type]
                image.dims,
                image.scale,
                image.translation,
                image.name,
                image.axes_units,
            )
            optimized_images.append(optimized_image)

        # Create new multiscales object with optimized images
        # For now, use the original multiscales and rely on to_ome_zarr to handle optimization
        # This preserves metadata and compatibility
        optimized_multiscales = multiscales  # type: ignore[assignment]

        # Setup chunks per shard if specified
        chunks_per_shard = options.chunks_per_shard
        if chunks_per_shard is not None:
            if isinstance(chunks_per_shard, list) and len(chunks_per_shard) == 1:
                chunks_per_shard = chunks_per_shard[0]
            elif isinstance(chunks_per_shard, list):
                chunks_per_shard = tuple(chunks_per_shard)

        # Save optimized store. Compression reaches to_ome_zarr as a codec
        # under `compressor`, the way convert_to_ome_zarr passes it;
        # `compression_codec`/`compression_level` are this module's own option
        # names and are not array-creation keywords the writer knows.
        compression_kwargs = {}
        if options.compression_codec:
            from ngff_zarr.codecs import (
                codec_from_name,  # type: ignore[import-untyped]
            )

            compression_kwargs["compressor"] = codec_from_name(
                options.compression_codec, options.compression_level
            )

        to_ome_zarr(
            output_store,
            optimized_multiscales,
            chunks_per_shard=chunks_per_shard,
            **compression_kwargs,
        )

        # Analyze the optimized store
        store_info = analyze_zarr_store(str(output_path))

        return ConversionResult(
            success=True,
            output_path=str(output_path),
            store_info=store_info.model_dump(),
            error=None,
        )

    except Exception as e:
        return ConversionResult(
            success=False, output_path="", store_info={}, error=str(e)
        )
