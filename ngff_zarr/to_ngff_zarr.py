from typing import Union, Optional
from collections.abc import MutableMapping
from pathlib import Path
from dataclasses import asdict
from functools import reduce

from zarr.storage import BaseStore
import zarr
import dask.array
import numpy as np

from .to_multiscales import Multiscales, to_multiscales
from .config import config

def to_ngff_zarr(
    store: Union[MutableMapping, str, Path, BaseStore],
    multiscales: Multiscales,
    compute: bool = True,
    overwrite: bool = True,
    chunk_store: Optional[Union[MutableMapping, str, Path, BaseStore]] = None,
    **kwargs,
) -> None:
    """
    Write an image pixel array and metadata to a Zarr store with the OME-NGFF standard data model.

    store : MutableMapping, str or Path, zarr.storage.BaseStore
        Store or path to directory in file system.

    multiscales: Multiscales
        Multiscales OME-NGFF image pixel data and metadata. Can be generated with ngff_zarr.to_multiscales.

    compute: Boolean, optional
        Compute the result instead of just building the Dask task graph.

    overwrite : bool, optional
        If True, delete any pre-existing data in `store` before creating groups.

    chunk_store : MutableMapping, str or Path, zarr.storage.BaseStore, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.

    **kwargs:
        Passed to the zarr.creation.create() function, e.g., compression options.


    Returns
    -------

    arrays: tuple of dask Arrays

    """

    metadata_dict = asdict(multiscales.metadata)
    metadata_dict["@type"] = "ngff:Image"
    root = zarr.group(store, overwrite=overwrite, chunk_store=chunk_store)
    root.attrs["multiscales"] = [metadata_dict]

    arrays = []
    for index in range(len(multiscales.images)):
        image = multiscales.images[index]
        arr = image.data
        path = multiscales.metadata.datasets[index].path
        path_group = root.create_group(path)
        path_group.attrs["_ARRAY_DIMENSIONS"] = image.dims
        arr = dask.array.to_zarr(
            arr,
            store,
            component=path,
            overwrite=overwrite,
            compute=compute,
            return_stored=True,
            **kwargs,
        )
        arrays.append(arr)
        image.data = arr

        remaining_bytes = reduce(lambda x,y: x+y.data.nbytes, multiscales.images[index:], 0)
        print('remaining bytes', remaining_bytes)
        # Minimize task graph depth
        if remaining_bytes > config.memory_limit and index < len(multiscales.images) - 1 and index > 0 and compute and multiscales.scale_factors and multiscales.method and multiscales.chunks:
            next_multiscales = to_multiscales(image,
                    multiscales.scale_factors[index:], multiscales.method,
                    multiscales.chunks)
            multiscales.images[index+1] = next_multiscales.images[1]

    zarr.consolidate_metadata(store)

    return tuple(arrays)
