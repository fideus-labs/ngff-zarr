# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import atexit
import shutil
import signal
import threading
import time
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import dask
import numpy as np
import zarr
from dask.array.core import Array as DaskArray
from numpy.typing import ArrayLike

try:
    from zarr.core import Array as ZarrArray
except ImportError:
    from zarr.core.array import Array as ZarrArray
import zarr.storage

from ._zarr_kwargs import zarr_kwargs
from ._zarr_open_array import open_array
from .config import config
from .memory_usage import memory_usage
from .methods import Methods
from .methods._dask_image import _downsample_dask_image
from .methods._itk import (
    _downsample_itk_bin_shrink,
    _downsample_itk_gaussian,
)
from .methods._itkwasm import (
    _downsample_itkwasm,
)
from .methods._metadata import get_method_metadata
from .methods._support import _spatial_dims
from .multiscales import NgffMultiscales
from .ngff_image import NgffImage
from .rfc4 import AnatomicalOrientation, orientation_from_name
from .rich_dask_progress import NgffProgress, NgffProgressCallback
from .task_count import task_count
from .to_ngff_image import to_ngff_image
from .v06.zarr_metadata import (
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Dataset,
    Metadata,
    Scale,
    TransformSequence,
    Translation,
)


def _ngff_image_scale_factors(ngff_image, min_length, out_chunks):
    assert tuple(ngff_image.dims) == tuple(out_chunks.keys()), (
        f"{ngff_image.dims} != {out_chunks.keys()}"
    )
    sizes = {
        d: s
        for d, s in zip(ngff_image.dims, ngff_image.data.shape)
        if d in _spatial_dims
    }
    scale_factors = []
    dims = ngff_image.dims
    previous = {d: 1 for d in dims if d in _spatial_dims}
    sizes_array = np.array(list(sizes.values()))
    double_chunks = np.array(
        [2 * s for d, s in out_chunks.items() if d in _spatial_dims]
    )
    while (sizes_array >= double_chunks).any():
        max_size = np.array(list(sizes.values())).max()
        to_skip = {d: sizes[d] <= max_size / 2 for d in previous}
        scale_factor = {}
        for dim in previous:
            if to_skip[dim] or sizes[dim] / 2 < out_chunks[dim]:
                scale_factor[dim] = previous[dim]
                continue
            scale_factor[dim] = 2 * previous[dim]

            sizes[dim] = int(sizes[dim] / 2)
        sizes_array = np.array(list(sizes.values()))
        # There should be sufficient data in the result for statistics, etc.
        if (np.prod(sizes_array) / min_length) < 2 or scale_factor == previous:
            break
        scale_factors.append(scale_factor)
        previous = scale_factor

    return scale_factors


def _find_optimal_chunk_size(first_chunk, dim_size, min_divisor=16):
    """Find a chunk size that divides evenly into dim_size and is ideally divisible by min_divisor.

    The returned chunk size will:
    1. Divide evenly into dim_size (required for safe region writes)
    2. Be as close as possible to first_chunk
    3. Preferably be divisible by min_divisor for performance
    """
    # If dimension is very small, just use it directly
    if dim_size <= min_divisor:
        return dim_size

    # Start with the target chunk size
    target = first_chunk

    # First try to find a divisor of dim_size that's divisible by min_divisor
    # and close to our target
    best_chunk = dim_size  # Fallback: use full dimension
    best_distance = abs(dim_size - target)

    # Check all divisors of dim_size
    for i in range(1, int(np.sqrt(dim_size)) + 1):
        if dim_size % i == 0:
            # i and dim_size//i are both divisors
            for candidate in [i, dim_size // i]:
                distance = abs(candidate - target)
                # Prefer divisors that are multiples of min_divisor
                is_multiple = candidate % min_divisor == 0

                # Update if closer to target, with preference for multiples of min_divisor
                if distance < best_distance or (
                    distance == best_distance
                    and is_multiple
                    and best_chunk % min_divisor != 0
                ):
                    best_chunk = candidate
                    best_distance = distance

    return best_chunk


def _large_image_serialization(
    image: NgffImage, progress: NgffProgress | NgffProgressCallback | None
):
    optimized_chunks = 512 if "z" in image.dims else 1024
    base_path = f"{image.name}-cache-{time.time()}"

    cache_store = config.cache_store
    base_path_removed = False

    def remove_from_cache_store(sig_id, frame):
        nonlocal base_path_removed
        if not base_path_removed:
            if hasattr(zarr.storage, "DirectoryStore") and isinstance(
                cache_store, zarr.storage.DirectoryStore
            ):
                full_path = Path(cache_store.dir_path()) / base_path
                if full_path.exists():
                    shutil.rmtree(full_path, ignore_errors=True)
            elif hasattr(zarr.storage, "LocalStore") and isinstance(
                cache_store, zarr.storage.LocalStore
            ):
                full_path = Path(cache_store.root) / base_path
                if full_path.exists():
                    shutil.rmtree(full_path, ignore_errors=True)
            else:
                zarr.storage.rmdir(cache_store, base_path)
            base_path_removed = True

    atexit.register(remove_from_cache_store, None, None)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, remove_from_cache_store)
        signal.signal(signal.SIGINT, remove_from_cache_store)
    image.computed_callbacks.append(lambda: remove_from_cache_store(None, None))

    data = image.data

    dims = list(image.dims)
    x_index = dims.index("x") if "x" in dims else None
    y_index = dims.index("y") if "y" in dims else None

    rechunks = {}
    for index, dim in enumerate(dims):
        if dim == "t":
            rechunks[index] = 1
        elif dim == "c":
            # Keep channels together to avoid unnecessary task multiplication
            rechunks[index] = data.chunksize[index]
        else:
            rechunks[index] = min(optimized_chunks, data.shape[index])

    if "z" in dims:
        z_index = dims.index("z")
        slice_bytes = data.dtype.itemsize
        if x_index is not None:
            slice_bytes *= data.shape[x_index]
        if y_index is not None:
            slice_bytes *= data.shape[y_index]

        slab_slices = min(
            int(np.ceil(config.memory_target / slice_bytes)), data.shape[z_index]
        )
        if optimized_chunks < data.shape[z_index]:
            slab_slices = min(slab_slices, optimized_chunks)
        rechunks[z_index] = slab_slices

        path = f"{base_path}/slabs"
        slabs = data.rechunk(rechunks)

        # For spatial dimensions, ensure Zarr chunk sizes are divisors of
        # the dimension sizes so older dask versions (e.g. 2025.11.0) can
        # write safely with regions.  For non-spatial dims (t, c), use the
        # slab chunk directly to keep channels / timepoints together and
        # avoid dask rechunking (gh-issue-487).
        chunks = tuple(
            _find_optimal_chunk_size(c[0], data.shape[i])
            if dim in _spatial_dims
            else c[0]
            for i, (c, dim) in enumerate(zip(slabs.chunks, dims))
        )

        zarr_array = open_array(
            shape=data.shape,
            chunks=chunks,
            dtype=data.dtype,
            store=cache_store,
            path=path,
            mode="a",
            **zarr_kwargs,
        )

        n_slabs = int(np.ceil(data.shape[z_index] / slab_slices))
        if progress:
            progress.add_cache_task("[blue]Caching z-slabs", n_slabs)
        for slab_index in range(n_slabs):
            if progress:
                if isinstance(progress, NgffProgressCallback):
                    progress.add_callback_task(
                        f"[blue]Caching z-slabs {slab_index + 1} of {n_slabs}"
                    )
                progress.update_cache_task_completed(slab_index + 1)
            region = [slice(data.shape[i]) for i in range(data.ndim)]
            region[z_index] = slice(
                slab_index * slab_slices,
                min((slab_index + 1) * slab_slices, data.shape[z_index]),
            )
            region = tuple(region)
            arr_region = slabs[region]
            dask.array.to_zarr(
                arr_region,
                zarr_array,
                region=region,
                component=path,
                overwrite=False,
                compute=True,
                return_stored=False,
                **zarr_kwargs,
            )
        data = dask.array.from_zarr(cache_store, component=path)
        if optimized_chunks < data.shape[z_index] and slab_slices < optimized_chunks:
            rechunks[z_index] = optimized_chunks
            data = data.rechunk(rechunks)
            path = f"{base_path}/optimized_chunks"

            # Use the same optimal chunk calculation for consistency
            chunks = tuple(
                [
                    _find_optimal_chunk_size(c[0], data.shape[i])
                    for i, c in enumerate(data.chunks)
                ]
            )

            data = data.rechunk(chunks)
            zarr_array = open_array(
                shape=data.shape,
                chunks=chunks,
                dtype=data.dtype,
                store=cache_store,
                path=path,
                mode="a",
                **zarr_kwargs,
            )
            n_slabs = int(np.ceil(data.shape[z_index] / optimized_chunks))
            for slab_index in range(n_slabs):
                if progress:
                    if isinstance(progress, NgffProgressCallback):
                        progress.add_callback_task(
                            f"[blue]Caching z-rechunk {slab_index + 1} of {n_slabs}"
                        )
                    progress.update_cache_task_completed(slab_index + 1)
                region = [slice(data.shape[i]) for i in range(data.ndim)]
                region[z_index] = slice(
                    slab_index * optimized_chunks,
                    min((slab_index + 1) * optimized_chunks, data.shape[z_index]),
                )
                region = tuple(region)
                arr_region = data[region]
                dask.array.to_zarr(
                    arr_region,
                    zarr_array,
                    region=region,
                    component=path,
                    overwrite=False,
                    compute=True,
                    return_stored=False,
                    **zarr_kwargs,
                )
            data = dask.array.from_zarr(cache_store, component=path)
        else:
            data = data.rechunk(rechunks)
    elif "y" in dims or "x" in dims:
        # 2D (or 2D+c/t) images: process in strips along the larger spatial dim
        data = _cache_2d_strips(data, dims, rechunks, cache_store, base_path, progress)
    else:
        # 1D (x-only) images: process in segments along x
        data = _cache_1d_segments(
            data, dims, rechunks, cache_store, base_path, progress
        )

    image.data = data
    return image


def _cache_2d_strips(data, dims, rechunks, cache_store, base_path, progress):
    """Cache large 2D images in strips along the larger spatial dimension.

    This avoids constructing a massive dask rechunk graph for the full image
    by processing manageable strips and writing them incrementally to disk.
    """
    x_index = dims.index("x") if "x" in dims else None
    y_index = dims.index("y") if "y" in dims else None

    # Determine which spatial dimension is larger; strip along that one
    if x_index is not None and y_index is not None:
        if data.shape[y_index] >= data.shape[x_index]:
            strip_dim_index = y_index
            strip_dim_name = "y"
        else:
            strip_dim_index = x_index
            strip_dim_name = "x"
    elif y_index is not None:
        strip_dim_index = y_index
        strip_dim_name = "y"
    else:
        strip_dim_index = x_index
        strip_dim_name = "x"

    # Calculate bytes per single row/column along the strip dimension
    row_bytes = data.dtype.itemsize
    for i in range(data.ndim):
        if i != strip_dim_index:
            row_bytes *= data.shape[i]

    strip_size = min(
        int(np.ceil(config.memory_target / row_bytes)),
        data.shape[strip_dim_index],
    )
    # Align strip_size to input chunk boundary to avoid partial-chunk reads
    input_chunk = data.chunksize[strip_dim_index]
    if input_chunk > 0:
        strip_size = max(input_chunk, (strip_size // input_chunk) * input_chunk)

    n_strips = int(np.ceil(data.shape[strip_dim_index] / strip_size))

    # Ensure zarr chunks match the strip size to avoid dask rechunking
    # during dask.array.to_zarr region writes (gh-issue-487).
    adjusted_rechunks = dict(rechunks)
    adjusted_rechunks[strip_dim_index] = strip_size

    path = base_path + "/strips"
    slabs = data.rechunk(adjusted_rechunks)

    chunks = tuple(c[0] for c in slabs.chunks)

    zarr_array = open_array(
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        store=cache_store,
        path=path,
        mode="a",
        **zarr_kwargs,
    )

    if progress:
        progress.add_cache_task(f"[blue]Caching {strip_dim_name}-strips", n_strips)
    for strip_index in range(n_strips):
        if progress:
            if isinstance(progress, NgffProgressCallback):
                progress.add_callback_task(
                    f"[blue]Caching {strip_dim_name}-strips"
                    f" {strip_index + 1} of {n_strips}"
                )
            progress.update_cache_task_completed(strip_index + 1)
        region = [slice(data.shape[i]) for i in range(data.ndim)]
        region[strip_dim_index] = slice(
            strip_index * strip_size,
            min((strip_index + 1) * strip_size, data.shape[strip_dim_index]),
        )
        region = tuple(region)
        arr_region = slabs[region]
        dask.array.to_zarr(
            arr_region,
            zarr_array,
            region=region,
            component=path,
            overwrite=False,
            compute=True,
            return_stored=False,
            **zarr_kwargs,
        )

    return dask.array.from_zarr(cache_store, component=path)


def _cache_1d_segments(data, dims, rechunks, cache_store, base_path, progress):
    """Cache large 1D images in segments along x.

    Handles the edge case of 1D data (only x dimension, possibly with c/t).
    """
    x_index = dims.index("x")

    # Calculate bytes per element across non-x dimensions
    element_bytes = data.dtype.itemsize
    for i in range(data.ndim):
        if i != x_index:
            element_bytes *= data.shape[i]

    segment_size = min(
        int(np.ceil(config.memory_target / element_bytes)),
        data.shape[x_index],
    )
    # Align to input chunk boundary
    input_chunk = data.chunksize[x_index]
    if input_chunk > 0:
        segment_size = max(input_chunk, (segment_size // input_chunk) * input_chunk)

    n_segments = int(np.ceil(data.shape[x_index] / segment_size))

    # Ensure zarr chunks match the segment size to avoid dask rechunking
    # during dask.array.to_zarr region writes (gh-issue-487).
    adjusted_rechunks = dict(rechunks)
    adjusted_rechunks[x_index] = segment_size

    path = base_path + "/segments"
    slabs = data.rechunk(adjusted_rechunks)

    chunks = tuple(c[0] for c in slabs.chunks)

    zarr_array = open_array(
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        store=cache_store,
        path=path,
        mode="a",
        **zarr_kwargs,
    )

    if progress:
        progress.add_cache_task("[blue]Caching x-segments", n_segments)
    for seg_index in range(n_segments):
        if progress:
            if isinstance(progress, NgffProgressCallback):
                progress.add_callback_task(
                    f"[blue]Caching x-segments {seg_index + 1} of {n_segments}"
                )
            progress.update_cache_task_completed(seg_index + 1)
        region = [slice(data.shape[i]) for i in range(data.ndim)]
        region[x_index] = slice(
            seg_index * segment_size,
            min((seg_index + 1) * segment_size, data.shape[x_index]),
        )
        region = tuple(region)
        arr_region = slabs[region]
        dask.array.to_zarr(
            arr_region,
            zarr_array,
            region=region,
            component=path,
            overwrite=False,
            compute=True,
            return_stored=False,
            **zarr_kwargs,
        )

    return dask.array.from_zarr(cache_store, component=path)


def to_multiscales(
    data: NgffImage | ArrayLike | MutableMapping | str | ZarrArray,
    scale_factors: int | Sequence[dict[str, int] | int] = 128,
    method: Methods | None = None,
    chunks: int
    | tuple[int, ...]
    | tuple[tuple[int, ...], ...]
    | Mapping[Any, None | int | tuple[int, ...]]
    | None = None,
    progress: NgffProgress | NgffProgressCallback | None = None,
    cache: bool | None = None,
    orientation: str | Mapping[str, AnatomicalOrientation] | None = None,
) -> NgffMultiscales:
    """
    Generate multiple resolution scales for the OME-NGFF standard data model.

    :param  data: Multi-dimensional array that provides the image pixel values, or image pixel values + image metadata when an NgffImage.
    :type   data: NgffImage, ArrayLike, ZarrArray, MutableMapping, str

    :param scale_factors: If a single integer, scale factors in spatial dimensions will be increased by a factor of two until this minimum length is reached.
        If a list, integer scale factors to apply uniformly across all spatial dimensions or
        along individual spatial dimensions.
        Examples: 64 or [2, 4] or [{'x': 2, 'y': 4 }, {'x': 5, 'y': 10}]
        Scaling is constrained by size of chunks - we do not scale below the chunk size.
    :type  scale_factors: int of minimum length, int per scale or dict of spatial dimension int's per scale

    :param method:  Specify the anti-aliasing method used to downsample the image. Default is ITKWASM_GAUSSIAN. See the {py:class}`Methods <ngff_zarr.methods.Methods>` enum for all available options.
    :type  method: Methods, optional

    :param chunks: Specify the chunking used in each output scale. The default is 128 for 3D images and 256 for 2D images.
    :type  chunks: Dask array chunking specification, optional

    :param cache: Cache intermediate results to disk to limit memory consumption. If None, the default, determine based on ngff_zarr.config.memory_target.
    :type  cache: bool, optional

    :param progress: Optional progress logger
    :type  progress: NgffProgress, NgffProgressCallback

    :param orientation: Anatomical orientation (RFC 4) for the spatial axes. Either a preset
        coordinate-system name (``"LPS"`` or ``"RAS"``, case-insensitive) or a mapping of axis
        name to :class:`AnatomicalOrientation`. When provided, it takes precedence over any
        ``axes_orientations`` on the input image. Anatomical orientation is written to the
        output automatically whenever it is present; no separate opt-in is required.
    :type  orientation: str or mapping of str to AnatomicalOrientation, optional

    :return: NgffImage for each resolution and NGFF multiscales metadata
    :rtype : NgffMultiscales
    """
    ngff_image = data if isinstance(data, NgffImage) else to_ngff_image(data)

    # IPFS and visualization friendly default chunks
    default_chunk_size = 128 if "z" in ngff_image.dims else 256

    out_chunks = chunks
    if out_chunks is None:
        # Auto-detect: use max of default and input chunk size for spatial dims
        # to avoid dask task explosion when input tiles are larger than default.
        # Only respect input chunks that look like they came from a tiled source
        # (uniform chunks), not from dask auto-chunking of in-memory arrays.
        input_chunks = None
        if isinstance(ngff_image.data, DaskArray):
            candidate = ngff_image.data.chunksize
            # Only respect input chunks that look like they came from a tiled
            # source (e.g. tifffile zarr store). Dask auto-chunking on in-memory
            # arrays produces arbitrary sizes that should not override defaults.
            # Heuristic: tiled sources produce uniform chunks along spatial dims
            # (all chunks the same size except possibly the last one), and at
            # least one spatial dimension has 3+ chunks with consistent sizes.
            has_large_uniform_tiles = False
            for dim_idx, dim in enumerate(ngff_image.dims):
                if dim in _spatial_dims:
                    chunk_size = candidate[dim_idx]
                    dim_chunks = ngff_image.data.chunks[dim_idx]
                    # Need at least 3 chunks to reliably detect uniformity
                    # (2 chunks could be from dask auto-chunking of small arrays)
                    if (
                        len(dim_chunks) >= 3
                        and chunk_size > default_chunk_size
                        and len(set(dim_chunks[:-1])) == 1
                    ):
                        has_large_uniform_tiles = True
                        break
            if has_large_uniform_tiles:
                input_chunks = candidate

        # For channel dim, always read from the actual data regardless of
        # whether spatial tiles were detected (channel preservation is
        # independent of tiled-source detection).
        data_chunks = None
        if isinstance(ngff_image.data, DaskArray):
            data_chunks = ngff_image.data.chunksize

        out_chunks = {}
        for dim_idx, dim in enumerate(ngff_image.dims):
            if dim in _spatial_dims:
                input_chunk = input_chunks[dim_idx] if input_chunks else 0
                out_chunks[dim] = max(default_chunk_size, input_chunk)
            elif dim == "t":
                out_chunks[dim] = 1
            elif dim == "c":
                # Keep channels together to avoid unnecessary task multiplication
                out_chunks[dim] = data_chunks[dim_idx] if data_chunks else 1
            else:
                out_chunks[dim] = default_chunk_size
    elif isinstance(out_chunks, int):
        out_chunks = dict.fromkeys(ngff_image.dims, chunks)
    elif isinstance(out_chunks, tuple):
        out_chunks = {d: chunks[i] for i, d in enumerate(ngff_image.dims)}

    # Build default_chunks dict for downsampling methods (uses the base default,
    # not the input-aligned values, since methods use this for alignment logic)
    default_chunks = dict.fromkeys(ngff_image.dims, default_chunk_size)
    if "t" in ngff_image.dims:
        default_chunks["t"] = 1

    da_out_chunks = tuple(out_chunks[d] for d in ngff_image.dims)
    if not isinstance(ngff_image.data, DaskArray):
        if isinstance(ngff_image.data, (ZarrArray, str, MutableMapping)):
            ngff_image.data = dask.array.from_zarr(ngff_image.data)
        else:
            ngff_image.data = dask.array.from_array(ngff_image.data)

    if isinstance(scale_factors, int):
        scale_factors = _ngff_image_scale_factors(ngff_image, scale_factors, out_chunks)

    should_cache = cache
    if cache is None:
        mem_exceeded = memory_usage(ngff_image) > config.memory_target
        tasks_exceeded = task_count(ngff_image) > config.task_target
        should_cache = mem_exceeded or tasks_exceeded

    if should_cache:
        ngff_image = _large_image_serialization(ngff_image, progress)

    ngff_image.data = ngff_image.data.rechunk(da_out_chunks)

    if method is None:
        method = Methods.ITKWASM_GAUSSIAN

    if method is Methods.ITKWASM_GAUSSIAN:
        images = _downsample_itkwasm(
            ngff_image, default_chunks, out_chunks, scale_factors, "gaussian"
        )
    elif method is Methods.ITKWASM_LABEL_IMAGE:
        images = _downsample_itkwasm(
            ngff_image, default_chunks, out_chunks, scale_factors, "label_image"
        )
    elif method is Methods.ITKWASM_BIN_SHRINK:
        images = _downsample_itkwasm(
            ngff_image, default_chunks, out_chunks, scale_factors, "bin_shrink"
        )
    elif method is Methods.ITK_GAUSSIAN:
        images = _downsample_itk_gaussian(
            ngff_image, default_chunks, out_chunks, scale_factors
        )
    elif method is Methods.ITK_BIN_SHRINK:
        images = _downsample_itk_bin_shrink(
            ngff_image, default_chunks, out_chunks, scale_factors
        )
    elif method is Methods.DASK_IMAGE_GAUSSIAN:
        images = _downsample_dask_image(
            ngff_image, default_chunks, out_chunks, scale_factors, label=False
        )
    elif method is Methods.DASK_IMAGE_NEAREST:
        images = _downsample_dask_image(
            ngff_image, default_chunks, out_chunks, scale_factors, label="nearest"
        )
    elif method is Methods.DASK_IMAGE_MODE:
        images = _downsample_dask_image(
            ngff_image, default_chunks, out_chunks, scale_factors, label="mode"
        )

    # Propagate channel_names and channel_colors from the input image to
    # all generated pyramid levels so that OMERO computation (which may
    # use a lower-resolution level) can access them.
    # Copy the lists to avoid shared-mutation aliasing between levels.
    src_channel_names = ngff_image.channel_names
    src_channel_colors = ngff_image.channel_colors
    if src_channel_names is not None or src_channel_colors is not None:
        for img in images:
            if img is not ngff_image:  # skip the base level (already has them)
                if img.channel_names is None and src_channel_names is not None:
                    img.channel_names = list(src_channel_names)
                if img.channel_colors is None and src_channel_colors is not None:
                    img.channel_colors = list(src_channel_colors)

    # Resolve anatomical orientation (RFC 4). An explicit ``orientation``
    # argument (preset name or mapping) takes precedence over any orientations
    # carried on the input image.
    if orientation is None:
        axes_orientations = ngff_image.axes_orientations
    elif isinstance(orientation, str):
        axes_orientations = orientation_from_name(orientation)
    else:
        axes_orientations = orientation

    axes_types = ngff_image.axes_types
    if axes_types:
        unknown = set(axes_types) - set(ngff_image.dims)
        if unknown:
            msg = f"axes_types refers to unknown dimensions: {sorted(unknown)}"
            raise ValueError(msg)

    axes = []
    for dim in ngff_image.dims:
        unit = None
        if ngff_image.axes_units and dim in ngff_image.axes_units:
            unit = ngff_image.axes_units[dim]

        axis_orientation = None
        if axes_orientations and dim in axes_orientations:
            axis_orientation = axes_orientations[dim]

        override_type = axes_types.get(dim) if axes_types else None
        if override_type is not None:
            discrete = override_type in ("displacement", "coordinate") or None
            axis = Axis(
                name=dim,
                type=override_type,
                unit=unit,
                orientation=axis_orientation,
                discrete=discrete,
            )
        elif dim in {"x", "y", "z"}:
            axis = Axis(name=dim, type="space", unit=unit, orientation=axis_orientation)
        elif dim == "c":
            axis = Axis(name=dim, type="channel", unit=unit)
        elif dim == "t":
            axis = Axis(name=dim, type="time", unit=unit)
        else:
            msg = f"Dimension identifier is not valid: {dim}"
            raise KeyError(msg)
        axes.append(axis)
    coordinate_system = CoordinateSystem(name="intrinsic", axes=axes)

    datasets = []
    for index, image in enumerate(images):
        path = f"scale{index}/{ngff_image.name}"
        scale = []
        for dim in image.dims:
            if dim in image.scale:
                scale.append(image.scale[dim])
            else:
                scale.append(1.0)
        translation = []
        for dim in image.dims:
            if dim in image.translation:
                translation.append(image.translation[dim])
            else:
                translation.append(0.0)
        coordinateTransformations = TransformSequence(
            name=f"scale{index}_to_intrinsic",
            transformations=[Scale(scale=scale), Translation(translation=translation)],
            input=CoordinateSystemIdentifier(path=path),
            output=CoordinateSystemIdentifier(name="intrinsic"),
        )
        dataset = Dataset(
            path=path, coordinateTransformations=[coordinateTransformations]
        )
        datasets.append(dataset)
    # Convert method enum to lowercase string for the type field
    method_type = None
    method_metadata = None
    if method is not None:
        method_type = method.value
        method_metadata = get_method_metadata(method)

    metadata = Metadata(
        coordinateSystems=[coordinate_system],
        datasets=datasets,
        name=ngff_image.name,
        coordinateTransformations=None,
        type=method_type,
        metadata=method_metadata,
    )

    return NgffMultiscales(images, metadata, scale_factors, method, out_chunks)
