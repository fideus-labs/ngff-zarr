# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import atexit
import shutil
import signal
import time
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import dask
import numpy as np
import zarr
from dask.array.core import Array as DaskArray
from numpy.typing import ArrayLike

try:
    from zarr.core import Array as ZarrArray
except ImportError:
    from zarr.core.array import Array as ZarrArray
from ._zarr_kwargs import zarr_kwargs
from ._zarr_open_array import open_array
import zarr.storage

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
from .multiscales import Multiscales
from .ngff_image import NgffImage
from .rich_dask_progress import NgffProgress, NgffProgressCallback
from .to_ngff_image import to_ngff_image
from .v06.zarr_metadata import Axis, Dataset, Metadata, Scale, Translation, TransformSequence


def _ngff_image_scale_factors(ngff_image, min_length, out_chunks):
    assert tuple(ngff_image.dims) == tuple(
        out_chunks.keys()
    ), f"{ngff_image.dims} != {out_chunks.keys()}"
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
    while (sizes_array > double_chunks).any():
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


def _large_image_serialization(
    image: NgffImage, progress: Optional[Union[NgffProgress, NgffProgressCallback]]
):
    optimized_chunks = 512 if "z" in image.dims else 1024
    base_path = f"{image.name}-cache-{time.time()}"

    cache_store = config.cache_store
    base_path_removed = False

    def remove_from_cache_store(sig_id, frame):  # noqa: ARG001
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
    signal.signal(signal.SIGTERM, remove_from_cache_store)
    signal.signal(signal.SIGINT, remove_from_cache_store)
    image.computed_callbacks.append(lambda: remove_from_cache_store(None, None))

    data = image.data

    dims = list(image.dims)
    x_index = dims.index("x")
    y_index = dims.index("y")

    rechunks = {}
    for index, dim in enumerate(dims):
        if dim == "t" or dim == "c":
            rechunks[index] = 1
        else:
            rechunks[index] = min(optimized_chunks, data.shape[index])

    if "z" in dims:
        z_index = dims.index("z")
        slice_bytes = data.dtype.itemsize * data.shape[x_index] * data.shape[y_index]

        slab_slices = min(
            int(np.ceil(config.memory_target / slice_bytes)), data.shape[z_index]
        )
        if optimized_chunks < data.shape[z_index]:
            slab_slices = min(slab_slices, optimized_chunks)
        rechunks[z_index] = slab_slices

        path = f"{base_path}/slabs"
        slabs = data.rechunk(rechunks)

        chunks = tuple([c[0] for c in slabs.chunks])
        optimized = dask.array.Array(
            dask.array.optimize(slabs.__dask_graph__(), slabs.__dask_keys__()),
            slabs.name,
            slabs.chunks,
            meta=slabs,
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
                        f"[blue]Caching z-slabs {slab_index+1} of {n_slabs}"
                    )
                progress.update_cache_task_completed(slab_index + 1)
            region = [slice(data.shape[i]) for i in range(data.ndim)]
            region[z_index] = slice(
                slab_index * slab_slices,
                min((slab_index + 1) * slab_slices, data.shape[z_index]),
            )
            region = tuple(region)
            arr_region = optimized[region]
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
            chunks = tuple([c[0] for c in optimized.chunks])
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
                            f"[blue]Caching z-rechunk {slab_index+1} of {n_slabs}"
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
    else:
        data = data.rechunk(rechunks)
        # TODO: Slab, chunk optimized very large 2D images
        path = base_path + "/optimized_chunks"
        if progress:
            progress.add_callback_task("[blue]Caching optimized chunks")
        dask.array.to_zarr(
            data,
            cache_store,
            component=path,
            overwrite=False,
            compute=True,
            return_stored=False,
            **zarr_kwargs,
        )
        data = dask.array.from_zarr(cache_store, component=path)

    image.data = data
    return image


def to_multiscales(
    data: Union[NgffImage, ArrayLike, MutableMapping, str, ZarrArray],
    scale_factors: Union[int, Sequence[Union[Dict[str, int], int]]] = 128,
    method: Optional[Methods] = None,
    chunks: Optional[
        Union[
            int,
            Tuple[int, ...],
            Tuple[Tuple[int, ...], ...],
            Mapping[Any, Union[None, int, Tuple[int, ...]]],
        ]
    ] = None,
    progress: Optional[Union[NgffProgress, NgffProgressCallback]] = None,
    cache: Optional[bool] = None,
) -> Multiscales:
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

    :param method:  Specify the anti-aliasing method used to downsample the image. Default is ITKWASM_GAUSSIAN.
    :type  Methods: ngff_zarr.Methods enum

    :param chunks: Specify the chunking used in each output scale. The default is 128 for 3D images and 256 for 2D images.
    :type  chunks: Dask array chunking specification, optional

    :param cache: Cache intermediate results to disk to limit memory consumption. If None, the default, determine based on ngff_zarr.config.memory_target.
    :type  cache: bool, optional

    :param progress: Optional progress logger
    :type  progress: NgffProgress, NgffProgressCallback

    :return: NgffImage for each resolution and NGFF multiscales metadata
    :rtype : Multiscales
    """
    ngff_image = data if isinstance(data, NgffImage) else to_ngff_image(data)

    # IPFS and visualization friendly default chunks
    default_chunks = 128 if "z" in ngff_image.dims else 256
    default_chunks = {d: default_chunks for d in ngff_image.dims}
    if "t" in ngff_image.dims:
        default_chunks["t"] = 1
    out_chunks = chunks
    if out_chunks is None:
        out_chunks = default_chunks
    elif isinstance(out_chunks, int):
        out_chunks = {d: chunks for d in ngff_image.dims}
    elif isinstance(out_chunks, tuple):
        out_chunks = {d: chunks[i] for i, d in enumerate(ngff_image.dims)}

    da_out_chunks = tuple(out_chunks[d] for d in ngff_image.dims)
    if not isinstance(ngff_image.data, DaskArray):
        if isinstance(ngff_image.data, (ZarrArray, str, MutableMapping)):
            ngff_image.data = dask.array.from_zarr(ngff_image.data)
        else:
            ngff_image.data = dask.array.from_array(ngff_image.data)

    if isinstance(scale_factors, int):
        scale_factors = _ngff_image_scale_factors(ngff_image, scale_factors, out_chunks)

    # if cache is None and memory_usage(ngff_image) > config.memory_target or task_count(ngff_image) > config.task_target or cache:
    if cache is None and memory_usage(ngff_image) > config.memory_target or cache:
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

    # Collect unique coordinate systems by name
    all_coordinate_systems = [cs for img in images for cs in ([img.coordinateSystems] if not isinstance(img.coordinateSystems, list) else img.coordinateSystems) if cs is not None]
    unique_coordinate_systems = list({cs.name: cs for cs in all_coordinate_systems}.values())

    datasets = []
    for index, image in enumerate(images):
        path = f"scale{index}/{ngff_image.name}"

        # get transformations and replace input/output with string references to coordinate systems
        transformations = TransformSequence(
            sequence=[
                Scale(list(image.scale.values())),
                Translation(list(image.translation.values()))
            ],
            name=f"scale{index}",
            input='',
            output=image.coordinateTransformations.output.name if image.coordinateTransformations.output else ""
        )

        dataset = Dataset(
            path=path, coordinateTransformations=transformations
        )
        datasets.append(dataset)
    # Convert method enum to lowercase string for the type field
    method_type = None
    method_metadata = None
    if method is not None:
        method_type = method.value
        method_metadata = get_method_metadata(method)

    metadata = Metadata(
        coordinateSystems=unique_coordinate_systems,
        coordinateTransformations=datasets[0].coordinateTransformations,
        datasets=datasets,
        name=ngff_image.name,
        type=method_type,
        metadata=method_metadata,
    )

    return Multiscales(images, metadata, scale_factors, method, out_chunks)
