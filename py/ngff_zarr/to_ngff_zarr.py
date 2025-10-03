# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import sys
from collections.abc import MutableMapping
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Optional, Union, Tuple, Dict, List
from packaging import version
import warnings

from .methods._metadata import get_method_metadata

if sys.version_info < (3, 10):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

import dask.array
import numpy as np
from itkwasm import array_like_to_numpy_array

import zarr
import zarr.storage
from ._zarr_open_array import open_array
from .v04.zarr_metadata import Metadata as Metadata_v04
from .v05.zarr_metadata import Metadata as Metadata_v05
from .rfc4 import is_rfc4_enabled

# Zarr Python 3
if hasattr(zarr.storage, "StoreLike"):
    StoreLike = zarr.storage.StoreLike
else:
    StoreLike = Union[MutableMapping, str, Path, zarr.storage.BaseStore]
from ._zarr_kwargs import zarr_kwargs


from .config import config
from .memory_usage import memory_usage
from .methods._support import _dim_scale_factors
from .multiscales import Multiscales
from .rich_dask_progress import NgffProgress, NgffProgressCallback
from .to_multiscales import to_multiscales

zarr_version = version.parse(zarr.__version__)
zarr_version_major = zarr_version.major


def _pop_metadata_optionals(metadata_dict, enabled_rfcs: Optional[List[int]] = None):

    if "axes" in metadata_dict:
        for ax in metadata_dict["axes"]:
            if ax["unit"] is None:
                ax.pop("unit")

            # Handle RFC 4: Remove orientation if RFC 4 is not enabled
            if not is_rfc4_enabled(enabled_rfcs) and "orientation" in ax:
                ax.pop("orientation")

    if "coordinateTransformations" in metadata_dict:
        if metadata_dict["coordinateTransformations"] is None:
            metadata_dict.pop("coordinateTransformations")

    if "omero" in metadata_dict:
        if metadata_dict["omero"] is None:
            metadata_dict.pop("omero")

    return metadata_dict


def _prep_for_to_zarr(store: StoreLike, arr: dask.array.Array) -> dask.array.Array:
    try:
        importlib_metadata.distribution("kvikio")
        _KVIKIO_AVAILABLE = True
    except importlib_metadata.PackageNotFoundError:
        _KVIKIO_AVAILABLE = False

    if _KVIKIO_AVAILABLE:
        from kvikio.zarr import GDSStore

        if not isinstance(store, GDSStore):
            arr = dask.array.map_blocks(
                array_like_to_numpy_array,
                arr,
                dtype=arr.dtype,
                meta=np.empty(()),
            )
        return arr
    return dask.array.map_blocks(
        array_like_to_numpy_array, arr, dtype=arr.dtype, meta=np.empty(())
    )


def _numpy_to_zarr_dtype(dtype):
    dtype_map = {
        "bool": "bool",
        "int8": "int8",
        "int16": "int16",
        "int32": "int32",
        "int64": "int64",
        "uint8": "uint8",
        "uint16": "uint16",
        "uint32": "uint32",
        "uint64": "uint64",
        "float16": "float16",
        "float32": "float32",
        "float64": "float64",
        "complex64": "complex64",
        "complex128": "complex128",
    }

    dtype_str = str(dtype)

    # Handle endianness - strip byte order chars
    if dtype_str.startswith(("<", ">", "|")):
        dtype_str = dtype_str[1:]

    # Look up corresponding zarr dtype
    try:
        return dtype_map[dtype_str]
    except KeyError:
        raise ValueError(f"dtype {dtype} cannot be mapped to Zarr v3 core dtype")


def _write_with_tensorstore(
    store_path: str,
    array,
    region,
    chunks,
    zarr_format,
    dimension_names=None,
    internal_chunk_shape=None,
    full_array_shape=None,
    create_dataset=True,
    compressor=None,
    **kwargs,
) -> None:
    """Write array using tensorstore backend"""
    import tensorstore as ts

    # Filter out compression-related kwargs that don't apply to TensorStore
    # TensorStore handles compression through codecs in metadata
    filtered_kwargs = {
        k: v for k, v in kwargs.items()
        if k not in ['compressor', 'compression', 'filters']
    }

    # Use full array shape if provided, otherwise use the region array shape
    dataset_shape = full_array_shape if full_array_shape is not None else array.shape

    # Build the base spec
    spec = {
        "kvstore": {
            "driver": "file",
            "path": store_path,
        },
        "metadata": {
            "shape": dataset_shape,
        },
    }

    if zarr_format == 2:
        spec["driver"] = "zarr" if zarr_version_major < 3 else "zarr2"
        spec["metadata"]["dimension_separator"] = "/"
        spec["metadata"]["dtype"] = array.dtype.str
        # Only add chunk info when creating the dataset
        if create_dataset:
            spec["metadata"]["chunks"] = chunks
            # Add compression for zarr v2 with TensorStore
            if compressor is not None:
                # TensorStore zarr2 driver uses compressor in metadata
                if hasattr(compressor, 'codec_id'):
                    # numcodecs compressor object
                    spec["metadata"]["compressor"] = compressor.get_config()
                else:
                    # Simple compressor name or config dict
                    spec["metadata"]["compressor"] = compressor
    elif zarr_format == 3:
        spec["driver"] = "zarr3"
        spec["metadata"]["data_type"] = _numpy_to_zarr_dtype(array.dtype)
        spec["metadata"]["chunk_key_encoding"] = {
            "name": "default",
            "configuration": {"separator": "/"},
        }
        if dimension_names:
            spec["metadata"]["dimension_names"] = dimension_names
        # Only add chunk info when creating the dataset
        if create_dataset:
            spec["metadata"]["chunk_grid"] = {
                "name": "regular",
                "configuration": {"chunk_shape": chunks},
            }

            # Build codecs list for zarr v3
            codecs = []

            # Helper function to create compression codec
            def create_compression_codec(compressor):
                if compressor is None:
                    return None

                if hasattr(compressor, 'codec_id'):
                    # numcodecs compressor object
                    codec_id = compressor.codec_id
                    if codec_id == 'gzip':
                        return {
                            "name": "gzip",
                            "configuration": {"level": getattr(compressor, 'level', 6)}
                        }
                    elif codec_id == 'blosc':
                        return {
                            "name": "blosc",
                            "configuration": {
                                "cname": getattr(compressor, 'cname', 'lz4'),
                                "clevel": getattr(compressor, 'clevel', 5),
                                "shuffle": "shuffle" if getattr(compressor, 'shuffle', 1) == 1 else "noshuffle"
                            }
                        }
                    elif codec_id == 'zstd':
                        return {
                            "name": "zstd",
                            "configuration": {"level": getattr(compressor, 'level', 3)}
                        }
                    elif codec_id == 'lz4':
                        return {"name": "lz4"}
                    else:
                        # Fallback: try to use the codec_id as name
                        return {"name": codec_id}
                elif isinstance(compressor, str):
                    # Simple codec name
                    return {"name": compressor}
                elif isinstance(compressor, dict):
                    # Already in codec format
                    return compressor
                return None

            # Add sharding codec with inner codecs if needed
            if internal_chunk_shape:
                sharding_config = {"chunk_shape": internal_chunk_shape}

                # If compression is specified, add it as inner codec for sharding
                if compressor is not None:
                    compression_codec = create_compression_codec(compressor)
                    if compression_codec:
                        # For sharding, compression goes in the inner codecs
                        sharding_config["codecs"] = [compression_codec]

                codecs.append({
                    "name": "sharding_indexed",
                    "configuration": sharding_config,
                })
            else:
                # No sharding, add compression codec directly if specified
                if compressor is not None:
                    compression_codec = create_compression_codec(compressor)
                    if compression_codec:
                        codecs.append(compression_codec)

            # Set codecs if any were added
            if codecs:
                spec["metadata"]["codecs"] = codecs
    else:
        raise ValueError(f"Unsupported zarr format: {zarr_format}")

    # Try to open existing dataset first, create only if needed
    try:
        if create_dataset:
            dataset = ts.open(spec, create=True, dtype=array.dtype).result()
        else:
            # For existing datasets, use a minimal spec that just specifies the path
            existing_spec = {
                "kvstore": {
                    "driver": "file",
                    "path": store_path,
                },
                "driver": spec["driver"],
            }
            dataset = ts.open(existing_spec, create=False, dtype=array.dtype).result()
    except Exception as e:
        if "ALREADY_EXISTS" in str(e) and create_dataset:
            # Dataset already exists, open it without creating
            existing_spec = {
                "kvstore": {
                    "driver": "file",
                    "path": store_path,
                },
                "driver": spec["driver"],
            }
            dataset = ts.open(existing_spec, create=False, dtype=array.dtype).result()
        else:
            raise

    # Try to write the dask array directly first
    try:
        dataset[region] = array
    except Exception as e:
        # If we encounter dimension mismatch or shape-related errors,
        # compute the array and try again with corrective action
        error_msg = str(e).lower()
        if any(
            keyword in error_msg
            for keyword in [
                "dimension",
                "shape",
                "mismatch",
                "size",
                "extent",
                "rank",
                "invalid",
            ]
        ):
            # Compute the array to get the actual shape
            computed_array = array.compute()

            # Adjust region to match the actual computed array shape if needed
            if len(region) == len(computed_array.shape):
                adjusted_region = tuple(
                    slice(
                        region[i].start or 0,
                        (region[i].start or 0) + computed_array.shape[i],
                    )
                    if isinstance(region[i], slice)
                    else region[i]
                    for i in range(len(region))
                )
            else:
                adjusted_region = region

            # Try writing the computed array with adjusted region
            dataset[adjusted_region] = computed_array
        else:
            # Re-raise the exception if it's not related to dimension/shape issues
            raise


def _validate_ngff_parameters(
    version: str,
    chunks_per_shard: Optional[Union[int, Tuple[int, ...], Dict[str, int]]],
    use_tensorstore: bool,
    store: StoreLike,
) -> None:
    """Validate the parameters for the NGFF Zarr generation."""
    if version not in ["0.4", "0.5", "0.6.dev1"]:
        raise ValueError(f"Unsupported version: {version}")

    if chunks_per_shard is not None:
        if version == "0.4":
            raise ValueError(
                "Sharding is only supported for OME-Zarr version 0.5 and later"
            )
        if not use_tensorstore and zarr_version_major < 3:
            raise ValueError(
                "Sharding requires zarr-python version >= 3.0.0b1 for OME-Zarr version >= 0.5"
            )

    if use_tensorstore and not isinstance(store, (str, Path)):
        raise ValueError("use_tensorstore currently requires a path-like store")


def _prepare_metadata(
    multiscales: Multiscales, version: str, enabled_rfcs: Optional[List[int]] = None
) -> Tuple[Union[Metadata_v04, Metadata_v05], Tuple[str, ...], Dict]:
    """Prepare and convert metadata to the proper version format."""
    metadata = multiscales.metadata

    # Convert method enum to lowercase string for the type field
    method_type = None
    method_metadata = None
    if multiscales.method is not None:
        method_type = multiscales.method.value
        method_metadata = get_method_metadata(multiscales.method)

    if version == "0.4" and isinstance(metadata, Metadata_v05):
        metadata = Metadata_v04(
            axes=metadata.axes,
            datasets=metadata.datasets,
            coordinateTransformations=metadata.coordinateTransformations,
            name=metadata.name,
            type=method_type,
            metadata=method_metadata,
        )
    elif version == "0.5" and isinstance(metadata, Metadata_v04):
        metadata = Metadata_v05(
            axes=metadata.axes,
            datasets=metadata.datasets,
            coordinateTransformations=metadata.coordinateTransformations,
            name=metadata.name,
            type=method_type,
            metadata=method_metadata,
        )
    else:
        # Update the existing metadata object with the type
        if hasattr(metadata, 'type'):
            metadata.type = method_type

    dimension_names = tuple([ax.name for ax in metadata.axes])
    dimension_names_kwargs = (
        {"dimension_names": dimension_names} if version != "0.4" else {}
    )

    return metadata, dimension_names, dimension_names_kwargs


def _create_zarr_root(
    store: StoreLike,
    chunk_store: Optional[StoreLike],
    version: str,
    overwrite: bool,
) -> zarr.Group:
    """Create and configure the root Zarr group with proper attributes."""
    zarr_format = 2 if version == "0.4" else 3
    format_kwargs = {"zarr_format": zarr_format} if zarr_version_major >= 3 else {}
    _zarr_kwargs = zarr_kwargs.copy()

    if zarr_format == 2 and zarr_version_major >= 3:
        _zarr_kwargs["dimension_separator"] = "/"

    if version == "0.4":
        root = zarr.open_group(
            store,
            mode="w" if overwrite else "a",
            chunk_store=chunk_store,
            **format_kwargs,
        )
    else:
        if zarr_version_major < 3:
            raise ValueError(
                "zarr-python version >= 3.0.0b2 required for OME-Zarr version >= 0.5"
            )
        # For version >= 0.5, open root with Zarr v3
        root = zarr.open_group(
            store,
            mode="w" if overwrite else "a",
            chunk_store=chunk_store,
            **format_kwargs,
        )

    return root


def _configure_sharding(
    arr: dask.array.Array,
    chunks_per_shard: Optional[Union[int, Tuple[int, ...], Dict[str, int]]],
    dims: Tuple[str, ...],
    kwargs: Dict,
) -> Tuple[Dict, Optional[Tuple[int, ...]], dask.array.Array]:
    """Configure sharding parameters if sharding is enabled."""
    if chunks_per_shard is None:
        return {}, None, arr

    c0 = tuple([c[0] for c in arr.chunks])

    if isinstance(chunks_per_shard, int):
        shards = tuple([c * chunks_per_shard for c in c0])
    elif isinstance(chunks_per_shard, (tuple, list)):
        if len(chunks_per_shard) != arr.ndim:
            raise ValueError(f"chunks_per_shard must be a tuple of length {arr.ndim}")
        shards = tuple([c * c0[i] for i, c in enumerate(chunks_per_shard)])
    elif isinstance(chunks_per_shard, dict):
        shards = {d: c * chunks_per_shard.get(d, 1) for d, c in zip(dims, c0)}
        shards = tuple([shards[d] for d in dims])
    else:
        raise ValueError("chunks_per_shard must be an int, tuple, or dict")

    internal_chunk_shape = c0
    arr = arr.rechunk(shards)

    # Configure sharding parameters differently for v2 vs v3
    sharding_kwargs = {}
    if zarr_version_major >= 3:
        # For Zarr v3, configure sharding as a codec
        # Use chunk_shape for internal chunks and configure sharding via codecs
        sharding_kwargs["chunk_shape"] = internal_chunk_shape
        # Note: sharding codec will be configured separately in the codecs parameter
        # We'll pass the shard shape through a separate key to be handled later
        sharding_kwargs["_shard_shape"] = shards
    else:
        # For zarr v2, use the older API
        sharding_kwargs = {
            "shards": shards,
            "chunks": internal_chunk_shape,
        }

    return sharding_kwargs, internal_chunk_shape, arr


def _write_array_with_tensorstore(
    store_path: str,
    path: str,
    arr: dask.array.Array,
    chunks: Union[Tuple[int, ...], List[int]],
    shards: Optional[Tuple[int, ...]],
    internal_chunk_shape: Optional[Tuple[int, ...]],
    zarr_format: int,
    dimension_names: Optional[Tuple[str, ...]],
    region: Tuple[slice, ...],
    full_array_shape: Optional[Tuple[int, ...]] = None,
    create_dataset: bool = True,
    **kwargs,
) -> None:
    """Write an array using the TensorStore backend."""
    # Extract compressor and other conflicting parameters from kwargs to avoid conflicts
    compressor = kwargs.pop('compressor', None)
    kwargs.pop('chunks', None)  # Remove chunks from kwargs since it's a positional arg

    scale_path = f"{store_path}/{path}"
    if shards is None:
        _write_with_tensorstore(
            scale_path,
            arr,
            region,
            chunks,
            zarr_format=zarr_format,
            dimension_names=dimension_names,
            full_array_shape=full_array_shape,
            create_dataset=create_dataset,
            compressor=compressor,
            **kwargs,
        )
    else:  # Sharding
        _write_with_tensorstore(
            scale_path,
            arr,
            region,
            chunks,
            zarr_format=zarr_format,
            dimension_names=dimension_names,
            internal_chunk_shape=internal_chunk_shape,
            full_array_shape=full_array_shape,
            create_dataset=create_dataset,
            compressor=compressor,
            **kwargs,
        )


def _write_array_direct(
    arr: dask.array.Array,
    store: StoreLike,
    path: str,
    sharding_kwargs: Dict,
    zarr_kwargs: Dict,
    format_kwargs: Dict,
    dimension_names_kwargs: Dict,
    region: Optional[Tuple[slice, ...]] = None,
    zarr_array=None,
    **kwargs,
) -> None:
    """Write an array directly using dask.array.to_zarr."""
    arr = _prep_for_to_zarr(store, arr)

    zarr_fmt = format_kwargs.get("zarr_format")

    # Handle sharding kwargs for direct writing
    cleaned_sharding_kwargs = {}

    if sharding_kwargs and "_shard_shape" in sharding_kwargs:
        # For Zarr v3 direct writes, use shards and chunks parameters
        shard_shape = sharding_kwargs["_shard_shape"]
        internal_chunk_shape = sharding_kwargs.get("chunk_shape")

        # Ensure internal_chunk_shape is available
        if internal_chunk_shape is None:
            # Use chunks from arr if available, or default
            internal_chunk_shape = tuple(arr.chunks[i][0] for i in range(arr.ndim))

        # For direct Zarr v3 writes, use shards and chunks
        cleaned_sharding_kwargs["shards"] = shard_shape
        cleaned_sharding_kwargs["chunks"] = internal_chunk_shape

        # Remove internal kwargs
        cleaned_sharding_kwargs.update(
            {
                k: v
                for k, v in sharding_kwargs.items()
                if k not in ["_shard_shape", "chunk_shape"]
            }
        )
    else:
        cleaned_sharding_kwargs = sharding_kwargs

    to_zarr_kwargs = {
        **cleaned_sharding_kwargs,
        **zarr_kwargs,
        **format_kwargs,
        **dimension_names_kwargs,
        **kwargs,
    }

    if zarr_fmt == 3 and zarr_array is None:
        # Zarr v3, use zarr.create_array and assign (whole array or region)
        array = zarr.create_array(
            store=store,
            name=path,
            shape=arr.shape,
            dtype=arr.dtype,
            **to_zarr_kwargs,
        )
        if region is not None:
            array[region] = arr.compute()
        else:
            array[:] = arr.compute()
    else:
        # All other cases: use dask.array.to_zarr
        target = (
            zarr_array if (region is not None and zarr_array is not None) else store
        )
        dask.array.to_zarr(
            arr,
            target,
            region=region if (region is not None and zarr_array is not None) else None,
            component=path,
            overwrite=False,
            compute=True,
            return_stored=False,
            **to_zarr_kwargs,
        )


def _handle_large_array_writing(
    image,
    arr: dask.array.Array,
    store: StoreLike,
    path: str,
    dims: Tuple[str, ...],
    dim_factors: Dict[str, int],
    chunks: Tuple[int, ...],
    sharding_kwargs: Dict,
    zarr_kwargs: Dict,
    format_kwargs: Dict,
    dimension_names_kwargs: Dict,
    use_tensorstore: bool,
    store_path: Optional[str],
    zarr_format: int,
    dimension_names: Tuple[str, ...],
    internal_chunk_shape: Optional[Tuple[int, ...]],
    shards: Optional[Tuple[int, ...]],
    progress: Optional[Union[NgffProgress, NgffProgressCallback]],
    index: int,
    nscales: int,
    **kwargs,
) -> None:
    """Handle writing large arrays by splitting them into manageable pieces."""
    shrink_factors = []
    for dim in dims:
        if dim in dim_factors:
            shrink_factors.append(dim_factors[dim])
        else:
            shrink_factors.append(1)

    chunks = tuple([c[0] for c in arr.chunks])

    # If sharding is enabled, configure it properly
    chunk_kwargs = {}
    codecs_kwargs = {}

    if sharding_kwargs:
        if "_shard_shape" in sharding_kwargs:
            # For Zarr v3, configure sharding as a codec only
            shard_shape = sharding_kwargs.pop("_shard_shape")
            internal_chunk_shape = sharding_kwargs.get(
                "chunk_shape"
            )  # This is the inner chunk shape

            # Configure the sharding codec with proper defaults
            from zarr.codecs.sharding import ShardingCodec
            from zarr.codecs.bytes import BytesCodec
            from zarr.codecs.zstd import ZstdCodec

            # Default inner codecs for sharding
            default_codecs = [BytesCodec(), ZstdCodec()]

            # Ensure internal_chunk_shape is available; fallback to chunks if needed
            if internal_chunk_shape is None:
                internal_chunk_shape = chunks

            # The array's chunk_shape should be the shard shape
            # The sharding codec's chunk_shape should be the internal chunk shape
            sharding_codec = ShardingCodec(
                chunk_shape=internal_chunk_shape,  # Internal chunk shape within shards
                codecs=default_codecs,
            )

            # Set up codecs with sharding
            existing_codecs = zarr_kwargs.get("codecs", [])
            if not isinstance(existing_codecs, list):
                existing_codecs = []
            codecs_kwargs["codecs"] = [sharding_codec] + existing_codecs

            # Set the array's chunk_shape to the shard shape
            chunk_kwargs["chunk_shape"] = shard_shape

            # Clean up remaining kwargs (remove chunk_shape since we're setting it explicitly)
            remaining_kwargs = {
                k: v
                for k, v in sharding_kwargs.items()
                if k not in ["_shard_shape", "chunk_shape"]
            }
            sharding_kwargs_clean = remaining_kwargs
        else:
            # For Zarr v2 or other cases
            sharding_kwargs_clean = sharding_kwargs
    else:
        # No sharding
        chunk_kwargs = {"chunks": chunks}
        sharding_kwargs_clean = {}

    zarr_array = open_array(
        shape=arr.shape,
        dtype=arr.dtype,
        store=store,
        path=path,
        mode="a",
        **chunk_kwargs,
        **sharding_kwargs_clean,
        **zarr_kwargs,
        **codecs_kwargs,
        **dimension_names_kwargs,
        **format_kwargs,
    )

    shape = image.data.shape
    x_index = dims.index("x")
    y_index = dims.index("y")

    regions = _compute_write_regions(
        image, dims, arr, shape, x_index, y_index, chunks, shrink_factors
    )

    for region_index, region in enumerate(regions):
        if isinstance(progress, NgffProgressCallback):
            progress.add_callback_task(
                f"[green]Writing scale {index + 1} of {nscales}, region {region_index + 1} of {len(regions)}"
            )

        arr_region = arr[region]
        arr_region = _prep_for_to_zarr(store, arr_region)
        optimized = dask.array.Array(
            dask.array.optimize(
                arr_region.__dask_graph__(), arr_region.__dask_keys__()
            ),
            arr_region.name,
            arr_region.chunks,
            meta=arr_region,
        )

        if use_tensorstore:
            _write_array_with_tensorstore(
                store_path,
                path,
                optimized,
                chunks,  # Use original array chunks, not region chunks
                shards,
                internal_chunk_shape,
                zarr_format,
                dimension_names,
                region,
                full_array_shape=arr.shape,
                create_dataset=(region_index == 0),  # Only create on first region
                **kwargs,
            )
        else:
            _write_array_direct(
                optimized,
                store,
                path,
                sharding_kwargs,
                zarr_kwargs,
                format_kwargs,
                dimension_names_kwargs,
                region,
                zarr_array,
                **kwargs,
            )


def _compute_write_regions(
    image,
    dims: Tuple[str, ...],
    arr: dask.array.Array,
    shape: Tuple[int, ...],
    x_index: int,
    y_index: int,
    chunks: Tuple[int, ...],
    shrink_factors: List[int],
) -> List[Tuple[slice, ...]]:
    """Compute the regions for writing a large array in chunks."""
    regions = []

    # If z dimension exists, handle 3D data
    if "z" in dims:
        z_index = dims.index("z")
        slice_bytes = memory_usage(image, {"z"})
        slab_slices = min(
            int(np.ceil(config.memory_target / slice_bytes)), arr.shape[z_index]
        )
        z_chunks = chunks[z_index]
        slice_planes = False
        if slab_slices < z_chunks:
            slab_slices = z_chunks
            slice_planes = True
        if slab_slices > arr.shape[z_index]:
            slab_slices = arr.shape[z_index]
        slab_slices = int(slab_slices / z_chunks) * z_chunks
        num_z_splits = int(np.ceil(shape[z_index] / slab_slices))
        while num_z_splits % shrink_factors[z_index] > 1:
            num_z_splits += 1

        # num_y_splits = 1
        # num_x_splits = 1

        for slab_index in range(num_z_splits):
            # Process individual slabs
            if slice_planes:
                regions.extend(
                    _compute_plane_regions(
                        image,
                        dims,
                        arr,
                        shape,
                        x_index,
                        y_index,
                        z_index,
                        chunks,
                        shrink_factors,
                        slab_index,
                    )
                )
            else:
                region = [slice(arr.shape[i]) for i in range(arr.ndim)]
                region[z_index] = slice(
                    slab_index * z_chunks,
                    min((slab_index + 1) * z_chunks, arr.shape[z_index]),
                )
                regions.append(tuple(region))
    else:
        # 2D data - one region covering the whole array
        regions.append(tuple([slice(arr.shape[i]) for i in range(arr.ndim)]))

    return regions


def _compute_plane_regions(
    image,
    dims: Tuple[str, ...],
    arr: dask.array.Array,
    shape: Tuple[int, ...],
    x_index: int,
    y_index: int,
    z_index: int,
    chunks: Tuple[int, ...],
    shrink_factors: List[int],
    slab_index: int,
) -> List[Tuple[slice, ...]]:
    """Compute regions for a single z-slab, dividing into planes and strips if needed."""
    plane_regions = []
    z_chunks = chunks[z_index]
    y_chunks = chunks[y_index]
    x_chunks = chunks[x_index]

    # Calculate how to divide planes
    plane_bytes = memory_usage(image, {"z", "y"})
    plane_slices = min(
        int(np.ceil(config.memory_target / plane_bytes)),
        arr.shape[y_index],
    )
    slice_strips = False
    if plane_slices < y_chunks:
        plane_slices = y_chunks
        slice_strips = True
    if plane_slices > arr.shape[y_index]:
        plane_slices = arr.shape[y_index]
    plane_slices = int(plane_slices / y_chunks) * y_chunks
    num_y_splits = int(np.ceil(shape[y_index] / plane_slices))
    while num_y_splits % shrink_factors[y_index] > 1:
        num_y_splits += 1

    if slice_strips:
        # Need to subdivide further into strips
        strip_bytes = memory_usage(image, {"z", "y", "x"})
        strip_slices = min(
            int(np.ceil(config.memory_target / strip_bytes)),
            arr.shape[x_index],
        )
        strip_slices = max(strip_slices, x_chunks)
        if strip_slices > arr.shape[x_index]:
            strip_slices = arr.shape[x_index]
        strip_slices = int(strip_slices / x_chunks) * x_chunks
        num_x_splits = int(np.ceil(shape[x_index] / strip_slices))
        while num_x_splits % shrink_factors[x_index] > 1:
            num_x_splits += 1

        for plane_index in range(num_y_splits):
            for strip_index in range(num_x_splits):
                region = [slice(arr.shape[i]) for i in range(arr.ndim)]
                region[z_index] = slice(
                    slab_index * z_chunks,
                    min((slab_index + 1) * z_chunks, arr.shape[z_index]),
                )
                region[y_index] = slice(
                    plane_index * y_chunks,
                    min((plane_index + 1) * y_chunks, arr.shape[y_index]),
                )
                region[x_index] = slice(
                    strip_index * x_chunks,
                    min((strip_index + 1) * x_chunks, arr.shape[x_index]),
                )
                plane_regions.append(tuple(region))
    else:
        # Just divide into planes
        for plane_index in range(num_y_splits):
            region = [slice(arr.shape[i]) for i in range(arr.ndim)]
            region[z_index] = slice(
                slab_index * z_chunks,
                min((slab_index + 1) * z_chunks, arr.shape[z_index]),
            )
            region[y_index] = slice(
                plane_index * y_chunks,
                min((plane_index + 1) * y_chunks, arr.shape[y_index]),
            )
            plane_regions.append(tuple(region))

    return plane_regions


def _prepare_next_scale(
    image,
    index: int,
    nscales: int,
    multiscales: Multiscales,
    store: StoreLike,
    path: str,
    progress: Optional[Union[NgffProgress, NgffProgressCallback]],
) -> Optional[object]:
    """Prepare the next scale for processing if needed."""
    # Minimize task graph depth
    if (
        index > 1
        and index < nscales - 2
        and multiscales.scale_factors
        and multiscales.method
        and multiscales.chunks
        and multiscales.scale_factors
    ):
        for callback in image.computed_callbacks:
            callback()
        image.computed_callbacks = []

        image.data = dask.array.from_zarr(store, component=path)
        next_multiscales_factor = multiscales.scale_factors[index]
        if isinstance(next_multiscales_factor, int):
            next_multiscales_factor = (
                next_multiscales_factor // multiscales.scale_factors[index - 1]
            )
        else:
            updated_factors = {}
            for d, f in next_multiscales_factor.items():
                updated_factors[d] = f // multiscales.scale_factors[index - 1][d]
            next_multiscales_factor = updated_factors

        next_multiscales = to_multiscales(
            image,
            scale_factors=[
                next_multiscales_factor,
            ],
            method=multiscales.method,
            chunks=multiscales.chunks,
            progress=progress,
            cache=False,
        )
        multiscales.images[index + 1] = next_multiscales.images[1]
        return next_multiscales.images[1]
    elif index < nscales - 1:
        return multiscales.images[index + 1]
    return None


def to_ngff_zarr(
    store: StoreLike,
    multiscales: Multiscales,
    version: str = "0.4",
    overwrite: bool = True,
    use_tensorstore: bool = False,
    chunk_store: Optional[StoreLike] = None,
    progress: Optional[Union[NgffProgress, NgffProgressCallback]] = None,
    chunks_per_shard: Optional[
        Union[
            int,
            Tuple[int, ...],
            Dict[str, int],
        ]
    ] = None,
    enabled_rfcs: Optional[List[int]] = None,
    **kwargs,
) -> None:
    """
    Write an image pixel array and metadata to a Zarr store with the OME-NGFF standard data model.

    :param store: Store or path to directory in file system.
    :type  store: StoreLike

    :param multiscales: Multiscales OME-NGFF image pixel data and metadata. Can be generated with ngff_zarr.to_multiscales.
    :type  multiscales: Multiscales

    :param version: OME-Zarr specification version.
    :type  version: str, optional

    :param overwrite: If True, delete any pre-existing data in `store` before creating groups.
    :type  overwrite: bool, optional

    :param use_tensorstore: If True, write array using tensorstore backend.
    :type  use_tensorstore: bool, optional

    :param chunk_store: Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.
    :type  chunk_store: StoreLike, optional

    :param progress: Optional progress logger
    :type  progress: RichDaskProgress

    :param chunks_per_shard: Number of chunks along each axis in a shard. If None, no sharding. Requires OME-Zarr version >= 0.5.
    :type  chunks_per_shard: int, tuple, or dict, optional

    :param enabled_rfcs: List of RFC numbers to enable. If RFC 4 is included, anatomical orientation metadata will be preserved in the output.
    :type  enabled_rfcs: list of int, optional

    :param **kwargs: Passed to the zarr.create_array() or zarr.creation.create() function, e.g., compression options.
    """
    # Setup and validation
    store_path = str(store) if isinstance(store, (str, Path)) else None

    # Create Zarr root
    root = _create_zarr_root(store, chunk_store, version, overwrite)


    # inject recursion here

    _validate_ngff_parameters(version, chunks_per_shard, use_tensorstore, store)
    metadata, dimension_names, dimension_names_kwargs = _prepare_metadata(
        multiscales, version, enabled_rfcs
    )
    metadata_dict = asdict(metadata)
    metadata_dict = _pop_metadata_optionals(metadata_dict, enabled_rfcs)
    metadata_dict["@type"] = "ngff:Image"


    if "omero" in metadata_dict:
        root.attrs["omero"] = metadata_dict.pop("omero")

    if version != "0.4":
        # RFC 2, Zarr 3
        root.attrs["ome"] = {"version": version, "multiscales": [metadata_dict]}
    else:
        root.attrs["multiscales"] = [metadata_dict]

    # Format parameters
    zarr_format = 2 if version == "0.4" else 3
    format_kwargs = {"zarr_format": zarr_format} if zarr_version_major >= 3 else {}
    _zarr_kwargs = zarr_kwargs.copy()

    if version == "0.4" and kwargs.get("compressors") is not None:
        raise ValueError(
            "The argument `compressors` are not supported for OME-Zarr version 0.4. (Zarr v3). Use `compression` instead."
        )

    if zarr_format == 2 and zarr_version_major >= 3:
        _zarr_kwargs["dimension_separator"] = "/"

    # Process each scale level
    nscales = len(multiscales.images)
    if progress:
        progress.add_multiscales_task("[green]Writing scales", nscales)

    next_image = multiscales.images[0]
    dims = next_image.dims
    previous_dim_factors = {d: 1 for d in dims}

    for index in range(nscales):
        if progress:
            progress.update_multiscales_task_completed(index + 1)

        image = next_image
        arr = image.data
        path = metadata.datasets[index].path
        parent = str(PurePosixPath(path).parent)

        # Create parent groups if needed
        if parent not in (".", "/"):
            array_dims_group = root.create_group(parent)
            array_dims_group.attrs["_ARRAY_DIMENSIONS"] = image.dims

        # Calculate dimension factors
        if index > 0 and index < nscales - 1 and multiscales.scale_factors:
            dim_factors = _dim_scale_factors(
                dims, multiscales.scale_factors[index], previous_dim_factors
            )
        else:
            dim_factors = {d: 1 for d in dims}
        previous_dim_factors = dim_factors

        # Configure sharding if needed
        sharding_kwargs, internal_chunk_shape, arr = _configure_sharding(
            arr, chunks_per_shard, dims, kwargs.copy()
        )

        # Get the chunks - these are now the shards if sharding is enabled
        chunks = tuple([c[0] for c in arr.chunks])

        # For TensorStore, shards are the same as chunks when sharding is enabled
        shards = chunks if chunks_per_shard is not None else None

        # Determine write method based on memory requirements
        if memory_usage(image) > config.memory_target:
            _handle_large_array_writing(
                image,
                arr,
                store,
                path,
                dims,
                dim_factors,
                chunks,
                sharding_kwargs,
                _zarr_kwargs,
                format_kwargs,
                dimension_names_kwargs,
                use_tensorstore,
                store_path,
                zarr_format,
                dimension_names,
                internal_chunk_shape,
                shards,
                progress,
                index,
                nscales,
                **kwargs,
            )
        else:
            if isinstance(progress, NgffProgressCallback):
                progress.add_callback_task(
                    f"[green]Writing scale {index + 1} of {nscales}"
                )

            # For small arrays, write in one go
            region = tuple([slice(arr.shape[i]) for i in range(arr.ndim)])
            if use_tensorstore:
                _write_array_with_tensorstore(
                    store_path,
                    path,
                    arr,
                    chunks,
                    shards,
                    internal_chunk_shape,
                    zarr_format,
                    dimension_names,
                    region,
                    full_array_shape=arr.shape,
                    create_dataset=True,  # Always create for small arrays
                    **kwargs,
                )
            else:
                _write_array_direct(
                    arr,
                    store,
                    path,
                    sharding_kwargs,
                    _zarr_kwargs,
                    format_kwargs,
                    dimension_names_kwargs,
                    None,
                    None,
                    **kwargs,
                )

        # Prepare next scale if needed
        next_image = _prepare_next_scale(
            image, index, nscales, multiscales, store, path, progress
        )

    # Clean up callbacks
    for image in multiscales.images:
        for callback in image.computed_callbacks:
            callback()
        image.computed_callbacks = []

    # Consolidate metadata
    if zarr_version_major >= 3:
        with warnings.catch_warnings():
            # Ignore consolidated metadata warning
            warnings.filterwarnings("ignore", category=UserWarning)
            zarr.consolidate_metadata(store, **format_kwargs)
    else:
        zarr.consolidate_metadata(store, **format_kwargs)
