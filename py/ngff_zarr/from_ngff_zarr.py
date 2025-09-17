# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from collections.abc import MutableMapping
from pathlib import Path
from typing import Union, Optional
import packaging.version
import sys

import dask.array
import zarr
import zarr.storage

# Zarr Python 3
if hasattr(zarr.storage, "StoreLike"):
    StoreLike = zarr.storage.StoreLike
else:
    StoreLike = Union[MutableMapping, str, Path, zarr.storage.BaseStore]

from .ngff_image import NgffImage
from .to_multiscales import Multiscales
from .methods import Methods
from .rfc4_validation import validate_rfc4_orientation, has_rfc4_orientation_metadata
from .v04.zarr_metadata import (
    Axis,
    Dataset,
    Scale,
    Translation,
    Omero,
    OmeroChannel,
    OmeroWindow,
)
from .validate import validate as validate_ngff

zarr_version = packaging.version.parse(zarr.__version__)
zarr_version_major = zarr_version.major


def from_ngff_zarr(
    store: StoreLike,
    validate: bool = False,
    version: Optional[str] = None,
    storage_options: Optional[dict] = None,
) -> Multiscales:
    """
    Read an OME-Zarr NGFF Multiscales data structure from a Zarr store.

    store : StoreLike
        Store or path to directory in file system. Can be a string URL
        (e.g., 's3://bucket/path') for remote storage.

    validate : bool
        If True, validate the NGFF metadata against the schema.

    version : string, optional
        OME-Zarr version, if known.

    storage_options : dict, optional
        Storage options to pass to the store if store is a string URL.
        For S3 URLs, this can include authentication credentials and other
        options for the underlying filesystem.

    Returns
    -------

    multiscales: multiscale ngff image with dask-chunked arrays for data

    """

    # Handle string URLs with storage options (zarr-python 3+ only)
    if isinstance(store, str) and storage_options is not None:
        if store.startswith(("s3://", "gs://", "azure://", "http://", "https://")):
            if zarr_version_major >= 3 and hasattr(zarr.storage, "FsspecStore"):
                store = zarr.storage.FsspecStore.from_url(
                    store, storage_options=storage_options
                )
            else:
                raise RuntimeError(
                    "storage_options parameter requires zarr-python 3+ with FsspecStore support. "
                    f"Current zarr version: {zarr.__version__}"
                )

    format_kwargs = {}
    if version and zarr_version_major >= 3:
        format_kwargs = (
            {"zarr_format": 2}
            if packaging.version.parse(version) < packaging.version.parse("0.5")
            else {"zarr_format": 3}
        )
    root = zarr.open_group(store, mode="r", **format_kwargs)
    root_attrs = root.attrs.asdict()

    if not version:
        if "ome" in root_attrs:
            version = root_attrs["ome"]["version"]
        else:
            version = root_attrs["multiscales"][0].get("version", "0.4")

    if validate:
        validate_ngff(root_attrs, version=version)

    if "ome" in root_attrs:
        metadata = root.attrs["ome"]["multiscales"][0]
    else:
        metadata = root.attrs["multiscales"][0]

    if "axes" not in metadata:
        from .v04.zarr_metadata import supported_dims

        dims = list(reversed(supported_dims))
    else:
        dims = [a["name"] if "name" in a else a for a in metadata["axes"]]

    name = "image"
    if name in metadata:
        name = metadata["name"]

    units = {d: None for d in dims}
    if "axes" in metadata:
        for axis in metadata["axes"]:
            if "unit" in axis:
                units[axis["name"]] = axis["unit"]

    images = []
    datasets = []
    for dataset in metadata["datasets"]:
        data = dask.array.from_zarr(store, component=dataset["path"])
        # Convert endianness to native if needed
        if (sys.byteorder == "little" and data.dtype.byteorder == ">") or (
            sys.byteorder == "big" and data.dtype.byteorder == "<"
        ):
            data = data.astype(data.dtype.newbyteorder())

        scale = {d: 1.0 for d in dims}
        translation = {d: 0.0 for d in dims}
        coordinateTransformations = []
        if "coordinateTransformations" in dataset:
            for transformation in dataset["coordinateTransformations"]:
                if "scale" in transformation:
                    scale = transformation["scale"]
                    scale = dict(zip(dims, scale))
                    coordinateTransformations.append(Scale(transformation["scale"]))
                elif "translation" in transformation:
                    translation = transformation["translation"]
                    translation = dict(zip(dims, translation))
                    coordinateTransformations.append(
                        Translation(transformation["translation"])
                    )
        datasets.append(
            Dataset(
                path=dataset["path"],
                coordinateTransformations=coordinateTransformations,
            )
        )

        ngff_image = NgffImage(data, dims, scale, translation, name, units)
        images.append(ngff_image)

    metadata.pop("@type", None)
    if "axes" in metadata:
        if "name" in metadata["axes"][0]:
            axes = [Axis(**axis) for axis in metadata["axes"]]
        else:
            # v0.3
            type_dict = {
                "t": "time",
                "c": "channel",
                "z": "space",
                "y": "space",
                "x": "space",
            }
            axes = [Axis(name=axis, type=type_dict[axis]) for axis in metadata["axes"]]
    else:
        axes = [
            Axis(name="t", type="time"),
            Axis(name="c", type="channel"),
            Axis(name="z", type="space"),
            Axis(name="y", type="space"),
            Axis(name="x", type="space"),
        ]

    coordinateTransformations = None
    if "coordinateTransformations" in metadata:
        coordinateTransformations = metadata["coordinateTransformations"]

    # RFC 4 validation for anatomical orientation
    if validate and isinstance(metadata, dict) and "axes" in metadata:
        # Check if any axes have orientation metadata and validate if present
        axes_data = metadata["axes"]
        if isinstance(axes_data, list):
            # Type cast each axis item to dict for validation
            axes_dicts = []
            for axis in axes_data:
                if isinstance(axis, dict):
                    axes_dicts.append(axis)
            if axes_dicts and has_rfc4_orientation_metadata(axes_dicts):
                validate_rfc4_orientation(axes_dicts)

    omero = None
    if "omero" in root.attrs:
        omero_data = root.attrs["omero"]
        if isinstance(omero_data, dict) and "channels" in omero_data:
            channels_data = omero_data["channels"]
            if isinstance(channels_data, list):
                channels = []
                for channel in channels_data:
                    if not isinstance(channel, dict) or "window" not in channel:
                        continue

                    window_data = channel["window"]
                    if not isinstance(window_data, dict):
                        continue

                    # Handle backward compatibility for OMERO window metadata
                    # Some stores use min/max, others use start/end, some have both
                    if "start" in window_data and "end" in window_data:
                        # New format with start/end
                        start = float(window_data["start"])  # type: ignore
                        end = float(window_data["end"])  # type: ignore
                        if "min" in window_data and "max" in window_data:
                            # Both formats present
                            min_val = float(window_data["min"])  # type: ignore
                            max_val = float(window_data["max"])  # type: ignore
                        else:
                            # Only start/end, use them as min/max
                            min_val = start
                            max_val = end
                    elif "min" in window_data and "max" in window_data:
                        # Old format with min/max only
                        min_val = float(window_data["min"])  # type: ignore
                        max_val = float(window_data["max"])  # type: ignore
                        # Use min/max as start/end for backward compatibility
                        start = min_val
                        end = max_val
                    else:
                        # Invalid window data, skip this channel
                        continue

                    channels.append(
                        OmeroChannel(
                            color=str(channel["color"]),  # type: ignore
                            label=str(channel.get("label", None))
                            if channel.get("label") is not None
                            else None,  # type: ignore
                            window=OmeroWindow(
                                min=min_val,
                                max=max_val,
                                start=start,
                                end=end,
                            ),
                        )
                    )

                if channels:
                    omero = Omero(channels=channels)

    # Extract method type and convert to Methods enum
    method = None
    method_type = None
    method_metadata = None
    if isinstance(metadata, dict):
        if "type" in metadata and metadata["type"] is not None:
            method_type = metadata["type"]
            # Find the corresponding Methods enum
            for method_enum in Methods:
                if method_enum.value == method_type:
                    method = method_enum
                    break

        # Extract method metadata if present
        if "metadata" in metadata and metadata["metadata"] is not None:
            from .v04.zarr_metadata import MethodMetadata

            metadata_dict = metadata["metadata"]
            if isinstance(metadata_dict, dict):
                method_metadata = MethodMetadata(
                    description=str(metadata_dict.get("description", "")),
                    method=str(metadata_dict.get("method", "")),
                    version=str(metadata_dict.get("version", "")),
                )

    if version == "0.5":
        from .v05.zarr_metadata import Metadata

        metadata_obj = Metadata(
            axes=axes,
            datasets=datasets,
            name=name,
            coordinateTransformations=coordinateTransformations,
            omero=omero,
            type=method_type,
            metadata=method_metadata,
        )
    else:
        from .v04.zarr_metadata import Metadata

        metadata_obj = Metadata(
            axes=axes,
            datasets=datasets,
            name=name,
            version=metadata["version"],
            coordinateTransformations=coordinateTransformations,
            omero=omero,
            type=method_type,
            metadata=method_metadata,
        )

    return Multiscales(images, metadata_obj, method=method)
