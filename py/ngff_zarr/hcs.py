# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""
High Content Screening (HCS) support for OME-Zarr NGFF.

This module provides functions for reading and writing HCS plate data
according to the NGFF specification.

Memory Management:
    The HCS implementation includes bounded caching to prevent memory
    issues when working with large plates:

    - Wells are cached in HCSPlate with configurable limit (default: 500)
    - Images are cached in HCSWell with configurable limit (default: 100)
    - LRU (Least Recently Used) eviction policy prevents unbounded growth
    - Cache sizes can be configured via ngff_zarr.config or function parameters
"""

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from ._remote_reader import RemoteZarrStore, remote_read_available
from ._zarrista_utils import (
    create_zarrista_group,
    create_zarrista_subgroup,
    open_ozx_store,
    read_group_attributes,
)
from .from_ngff_zarr import REMOTE_URL_SCHEMES, _open_root_node, from_ome_zarr
from .multiscales import NgffMultiscales
from .rfc9_zip import ZipReadStore, is_ozx_path, write_store_to_zip
from .to_ngff_zarr import to_ome_zarr
from .v04.zarr_metadata import (
    Plate,
    PlateAcquisition,
    PlateColumn,
    PlateRow,
    PlateWell,
    Well,
    WellImage,
)


class LRUCache:
    """
    Least Recently Used (LRU) cache with size limit for HCS image caching.

    This prevents unbounded memory growth when accessing many images
    from large plates.
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key, value):
        if key in self.cache:
            # Update existing item
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Add new item
            if len(self.cache) >= self.max_size:
                # Remove least recently used item before adding
                self.cache.popitem(last=False)
            self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, key):
        """Support dict-like access."""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        """Support dict-like assignment."""
        self.set(key, value)

    def clear(self):
        """Clear all cached items."""
        self.cache.clear()


class HCSPlate:
    """
    High Content Screening plate representation.

    This class holds the plate metadata and provides access to wells
    and their associated images.
    """

    def __init__(
        self,
        store,
        plate_metadata: Plate,
        well_cache_size: int | None = None,
        image_cache_size: int | None = None,
        validate: bool = False,
    ):
        self.store = store
        self.metadata = plate_metadata
        self.image_cache_size = image_cache_size
        self._validate = validate

        # Use bounded cache for wells to prevent memory issues with large plates
        from .config import config

        cache_size = well_cache_size or config.hcs_well_cache_size
        self._wells = LRUCache(max_size=cache_size)

    @property
    def name(self) -> str | None:
        return self.metadata.name

    @property
    def rows(self) -> list[PlateRow]:
        return self.metadata.rows

    @property
    def columns(self) -> list[PlateColumn]:
        return self.metadata.columns

    @property
    def wells(self) -> list[PlateWell]:
        return self.metadata.wells

    @property
    def acquisitions(self) -> list[PlateAcquisition] | None:
        return self.metadata.acquisitions

    @property
    def field_count(self) -> int | None:
        return self.metadata.field_count

    def get_well(self, row_name: str, column_name: str) -> Optional["HCSWell"]:
        """Get a well by row and column name."""
        well_path = f"{row_name}/{column_name}"

        # Check if well exists in metadata
        well_meta = None
        for well in self.metadata.wells:
            if well.path == well_path:
                well_meta = well
                break

        if well_meta is None:
            return None

        # Cache wells to avoid reloading - using bounded cache now
        if well_path not in self._wells:
            hcs_well = HCSWell.from_store(
                self.store, well_path, well_meta, self.image_cache_size
            )
            if self._validate:
                # Strict structural validation of the well's own metadata, in
                # the context of this plate, performed where each well is first
                # loaded (mirrors the schema/structural split in from_hcs_zarr).
                # Imported lazily so the default validate=False path is unchanged.
                from .structural_validation import (
                    ValidateOptions,
                    ValidationLevel,
                    validate_well,
                )

                validate_well(
                    self.metadata,
                    hcs_well.metadata,
                    ValidateOptions(level=ValidationLevel.STRICT),
                )
            self._wells[well_path] = hcs_well

        return self._wells[well_path]

    def get_well_by_indices(
        self, row_index: int, column_index: int
    ) -> Optional["HCSWell"]:
        """Get a well by row and column indices."""
        if (
            row_index < 0
            or row_index >= len(self.metadata.rows)
            or column_index < 0
            or column_index >= len(self.metadata.columns)
        ):
            return None

        row_name = self.metadata.rows[row_index].name
        column_name = self.metadata.columns[column_index].name
        return self.get_well(row_name, column_name)


class HCSWell:
    """
    High Content Screening well representation.

    Contains multiple fields of view (images) for a single well.
    """

    def __init__(
        self,
        store,
        well_path: str,
        well_metadata: PlateWell,
        well_group_metadata: Well,
        image_cache_size: int | None = None,
    ):
        self.store = store
        self.path = well_path
        self.plate_metadata = well_metadata
        self.metadata = well_group_metadata

        # Use bounded cache for images to prevent memory issues
        from .config import config

        cache_size = image_cache_size or config.hcs_image_cache_size
        self._images = LRUCache(max_size=cache_size)

    @classmethod
    def from_store(
        cls,
        store,
        well_path: str,
        well_metadata: PlateWell,
        image_cache_size: int | None = None,
    ) -> "HCSWell":
        """Load a well from a zarr store."""
        # Dispatches on store type: local paths read through the compat
        # layer, bytes mappings (including ZipReadStore for .ozx) through
        # the store reader, other store objects through zarr-python.
        root = _open_root_node(store, None)
        well_group = root[well_path]
        well_attrs = well_group.attrs.asdict()

        # Parse well metadata
        well_data = {}
        if (
            "ome" in well_attrs
            and isinstance(well_attrs["ome"], dict)
            and "well" in well_attrs["ome"]
        ):
            well_data = well_attrs["ome"]["well"]
        elif "well" in well_attrs:
            well_data = well_attrs["well"]

        # Ensure well_data is a dict
        if not isinstance(well_data, dict):
            well_data = {}

        images = []
        if "images" in well_data and isinstance(well_data["images"], list):
            for img_data in well_data["images"]:
                if isinstance(img_data, dict):
                    path = img_data.get("path", "")
                    acquisition = img_data.get("acquisition")

                    # Type checking
                    if isinstance(path, str):
                        image = WellImage(
                            path=path,
                            acquisition=acquisition
                            if isinstance(acquisition, int)
                            else None,
                        )
                        images.append(image)

        # Extract version - for v0.5 it's at ome level, for v0.4 it's in well dict
        version = "0.4"
        if "ome" in well_attrs and isinstance(well_attrs["ome"], dict):
            if "version" in well_attrs["ome"] and isinstance(
                well_attrs["ome"]["version"], str
            ):
                version = well_attrs["ome"]["version"]
        elif "version" in well_data and isinstance(well_data["version"], str):
            version = well_data["version"]

        well_group_metadata = Well(images=images, version=version)

        return cls(
            store, well_path, well_metadata, well_group_metadata, image_cache_size
        )

    @property
    def row_index(self) -> int:
        return self.plate_metadata.rowIndex

    @property
    def column_index(self) -> int:
        return self.plate_metadata.columnIndex

    @property
    def images(self) -> list[WellImage]:
        return self.metadata.images

    def get_image(self, field_index: int = 0) -> NgffMultiscales | None:
        """Get a field of view (image) by index."""
        if field_index < 0 or field_index >= len(self.metadata.images):
            return None

        image_meta = self.metadata.images[field_index]
        image_path = f"{self.path}/{image_meta.path}"

        # Cache images to avoid reloading
        if image_path not in self._images:
            if isinstance(self.store, RemoteZarrStore):
                # A view of the same remote store narrowed to this field's
                # sub-hierarchy, sharing the obstore client.
                self._images[image_path] = from_ome_zarr(
                    self.store.with_prefix(image_path)
                )
            elif isinstance(self.store, ZipReadStore):
                # A view of the same archive narrowed to this field's
                # sub-hierarchy; from_ome_zarr reads it as a mapping store.
                self._images[image_path] = from_ome_zarr(
                    self.store.with_prefix(image_path)
                )
            elif isinstance(self.store, (str, Path)):
                # If store is a path string, append the image path
                full_image_path = Path(self.store) / self.path / image_meta.path
                self._images[image_path] = from_ome_zarr(str(full_image_path))
            else:
                raise TypeError(
                    "HCS plates are read from local directory paths, remote "
                    f"URL strings, and .ozx archives; got "
                    f"{type(self.store).__name__}."
                )

        return self._images[image_path]

    def get_image_by_acquisition(
        self, acquisition_id: int, field_index: int = 0
    ) -> NgffMultiscales | None:
        """Get a field of view by acquisition ID and field index."""
        # Find images for the specified acquisition
        acquisition_images = [
            img for img in self.metadata.images if img.acquisition == acquisition_id
        ]

        if field_index < 0 or field_index >= len(acquisition_images):
            return None

        # Find the actual image index in the full list
        target_image = acquisition_images[field_index]
        actual_index = self.metadata.images.index(target_image)

        return self.get_image(actual_index)


def from_hcs_zarr(
    store,
    validate: bool = False,
    well_cache_size: int | None = None,
    image_cache_size: int | None = None,
) -> HCSPlate:
    """
    Read an HCS plate from an OME-Zarr NGFF store.

    Parameters
    ----------
    store
        Store or path to directory in file system. Can be a .ozx file.
    validate : bool
        If True, validate the plate metadata against the schema and then run the
        strict structural plate rules (raising ``ValidationError`` on failure).
        Per-well structural rules run lazily as each well is loaded via
        ``get_well``.
    well_cache_size : int, optional
        Maximum number of wells to cache. If None, uses config default.
    image_cache_size : int, optional
        Maximum number of images to cache per well. If None, uses config default.

    Returns
    -------
    plate : HCSPlate
        The loaded HCS plate with wells and images.
    """
    from .rfc9_zip import is_ozx_path

    # RFC-9: Handle .ozx (zipped OME-Zarr) files
    if isinstance(store, (str, Path)) and is_ozx_path(store):
        # Read the archive through the zarrista-backed zip store handle
        # (zarr-python ZipStore only when zarrista is unavailable).
        store = open_ozx_store(store)

    # Remote plate URLs read through zarrista's async API over obstore
    # (the zarr-python/fsspec engine remains the fallback).
    if (
        isinstance(store, str)
        and store.startswith(REMOTE_URL_SCHEMES)
        and remote_read_available()
    ):
        store = RemoteZarrStore(store)

    root = _open_root_node(store, None)
    root_attrs = root.attrs.asdict()

    if validate:
        from .validate import validate as validate_ngff

        # Use plate schema for HCS validation instead of image schema
        validate_ngff(root_attrs, model="plate")

    # Extract plate metadata
    plate_data = {}
    if (
        "ome" in root_attrs
        and isinstance(root_attrs["ome"], dict)
        and "plate" in root_attrs["ome"]
    ):
        plate_data = root_attrs["ome"]["plate"]
    elif "plate" in root_attrs:
        plate_data = root_attrs["plate"]

    # Ensure plate_data is a dict
    if not isinstance(plate_data, dict):
        raise ValueError("No plate metadata found in store")

    # Parse plate metadata with type checking
    columns = []
    if "columns" in plate_data and isinstance(plate_data["columns"], list):
        for col in plate_data["columns"]:
            if isinstance(col, dict) and "name" in col and isinstance(col["name"], str):
                columns.append(PlateColumn(name=col["name"]))

    rows = []
    if "rows" in plate_data and isinstance(plate_data["rows"], list):
        for row in plate_data["rows"]:
            if isinstance(row, dict) and "name" in row and isinstance(row["name"], str):
                rows.append(PlateRow(name=row["name"]))

    wells = []
    if "wells" in plate_data and isinstance(plate_data["wells"], list):
        for well in plate_data["wells"]:
            if isinstance(well, dict):
                path = well.get("path")
                row_index = well.get("rowIndex")
                column_index = well.get("columnIndex")

                if (
                    isinstance(path, str)
                    and isinstance(row_index, int)
                    and isinstance(column_index, int)
                ):
                    wells.append(
                        PlateWell(
                            path=path, rowIndex=row_index, columnIndex=column_index
                        )
                    )

    acquisitions = None
    if "acquisitions" in plate_data and isinstance(plate_data["acquisitions"], list):
        acquisitions = []
        for acq in plate_data["acquisitions"]:
            if isinstance(acq, dict) and "id" in acq and isinstance(acq["id"], int):
                # Extract optional fields with type checking
                name_val = acq.get("name")
                name = name_val if isinstance(name_val, str) else None

                maxfield_val = acq.get("maximumfieldcount")
                maximumfieldcount = (
                    maxfield_val if isinstance(maxfield_val, int) else None
                )

                desc_val = acq.get("description")
                description = desc_val if isinstance(desc_val, str) else None

                start_val = acq.get("starttime")
                starttime = start_val if isinstance(start_val, int) else None

                end_val = acq.get("endtime")
                endtime = end_val if isinstance(end_val, int) else None

                acquisition = PlateAcquisition(
                    id=acq["id"],
                    name=name,
                    maximumfieldcount=maximumfieldcount,
                    description=description,
                    starttime=starttime,
                    endtime=endtime,
                )
                acquisitions.append(acquisition)

    # Extract version, field_count, and name with type checking
    # For v0.5, version is at ome level; for v0.4, it's in plate dict
    version = "0.4"
    if "ome" in root_attrs and isinstance(root_attrs["ome"], dict):
        if "version" in root_attrs["ome"] and isinstance(
            root_attrs["ome"]["version"], str
        ):
            version = root_attrs["ome"]["version"]
    elif "version" in plate_data and isinstance(plate_data["version"], str):
        version = plate_data["version"]

    field_count = None
    if "field_count" in plate_data and isinstance(plate_data["field_count"], int):
        field_count = plate_data["field_count"]

    name = None
    if "name" in plate_data and isinstance(plate_data["name"], str):
        name = plate_data["name"]

    plate_metadata = Plate(
        columns=columns,
        rows=rows,
        wells=wells,
        version=version,
        acquisitions=acquisitions,
        field_count=field_count,
        name=name,
    )

    if validate:
        # Strict structural validation of the parsed plate, layered after the
        # schema pass above (schema first, then structural). Per-well image
        # rules run lazily in HCSPlate.get_well, where each well's own metadata
        # is loaded. Imported lazily so the default validate=False path incurs
        # no extra import cost. Unlike the image reader's structural pass, no
        # >=0.4 version gate is needed here: these plate/well rules are
        # version-agnostic and from_hcs_zarr only reads v0.4/v0.5 plates (there
        # is no pre-0.4 plate layout to exempt).
        from .structural_validation import (
            ValidateOptions,
            ValidationLevel,
            validate_plate,
        )

        validate_plate(plate_metadata, ValidateOptions(level=ValidationLevel.STRICT))

    return HCSPlate(
        store, plate_metadata, well_cache_size, image_cache_size, validate=validate
    )


def to_hcs_zarr(plate: HCSPlate, store, overwrite: bool = True) -> None:
    """
    Write an HCS plate to an OME-Zarr NGFF store.

    Parameters
    ----------
    plate : HCSPlate
        The HCS plate to write.
    store
        Store or path to directory in file system.
    overwrite : bool, optional
        If True (default), remove any existing content below the store path
        before writing the plate documents. If False, keep existing content
        and merge the plate metadata into the root group's attributes, so an
        interrupted acquisition can be resumed without discarding wells.
    """

    # For NGFF version 0.4, use Zarr format 2; for 0.5+, use Zarr format 3
    zarr_format = 2 if plate.metadata.version == "0.4" else 3

    # Build plate metadata dictionary
    plate_dict = {
        "columns": [{"name": col.name} for col in plate.metadata.columns],
        "rows": [{"name": row.name} for row in plate.metadata.rows],
        "wells": [
            {
                "path": well.path,
                "rowIndex": well.rowIndex,
                "columnIndex": well.columnIndex,
            }
            for well in plate.metadata.wells
        ],
    }

    # For v0.4, version goes in plate dict; for v0.5, it goes at top level
    if plate.metadata.version == "0.4":
        plate_dict["version"] = plate.metadata.version

    if plate.metadata.acquisitions:
        plate_dict["acquisitions"] = []
        for acq in plate.metadata.acquisitions:
            acq_dict: dict = {"id": acq.id}
            if acq.name is not None:
                acq_dict["name"] = acq.name
            if acq.maximumfieldcount is not None:
                acq_dict["maximumfieldcount"] = acq.maximumfieldcount
            if acq.description is not None:
                acq_dict["description"] = acq.description
            if acq.starttime is not None:
                acq_dict["starttime"] = acq.starttime
            if acq.endtime is not None:
                acq_dict["endtime"] = acq.endtime
            plate_dict["acquisitions"].append(acq_dict)

    if plate.metadata.field_count is not None:
        plate_dict["field_count"] = plate.metadata.field_count

    if plate.metadata.name is not None:
        plate_dict["name"] = plate.metadata.name

    # Set root metadata
    root_attrs = {"ome": {"version": plate.metadata.version, "plate": plate_dict}}
    create_zarrista_group(store, root_attrs, zarr_format, overwrite=overwrite)

    # Note: This is a basic implementation that sets up the plate structure.
    # In a full implementation, you would also write the well groups and
    # their associated image data using the existing to_ome_zarr functions.
    # This requires integration with the NgffMultiscales data structure.

    logging.info(f"HCS plate structure created at {store}")
    logging.info(f"Plate: {plate.metadata.name}")
    logging.info(f"Wells: {len(plate.metadata.wells)}")
    logging.info(
        f"Rows: {len(plate.metadata.rows)}, Columns: {len(plate.metadata.columns)}"
    )
    if plate.metadata.acquisitions:
        logging.info(f"Acquisitions: {len(plate.metadata.acquisitions)}")


class HCSPlateWriter:
    """
    Context manager for writing HCS plates with deferred .ozx zipping.

    This class enables efficient writing of multiple well images by:
    - Deferring .ozx file creation until all wells are written
    - Supporting parallel writes to the same plate
    - Avoiding repeated zipping operations for each well

    For .ozx files, this approach writes to a temporary zarr store and only
    creates the final .ozx ZIP archive when exiting the context manager.
    For regular .ome.zarr directories, it writes directly to the target.

    Parameters
    ----------
    store : str or Path
        Target store path. Can be .ozx, .ome.zarr, or .zarr extension.
    plate_metadata : Plate
        Plate-level metadata containing rows, columns, wells, and other plate information.
    version : str, optional
        OME-Zarr specification version (default: "0.5"). Note: .ozx format requires version 0.5.
    overwrite : bool, optional
        If True, overwrite existing store (default: True).

    Examples
    --------
    Writing multiple wells efficiently to .ozx format:

    >>> import ngff_zarr as nz
    >>> from ngff_zarr.hcs import HCSPlateWriter
    >>> from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow, PlateWell
    >>>
    >>> # Create plate metadata
    >>> plate_metadata = nz.Plate(
    ...     columns=[nz.PlateColumn(name="1"), nz.PlateColumn(name="2")],
    ...     rows=[nz.PlateRow(name="A"), nz.PlateRow(name="B")],
    ...     wells=[
    ...         nz.PlateWell(path="A/1", rowIndex=0, columnIndex=0),
    ...         nz.PlateWell(path="A/2", rowIndex=0, columnIndex=1),
    ...     ],
    ...     version="0.5",
    ... )
    >>>
    >>> # Use context manager to write multiple wells
    >>> with HCSPlateWriter("my_plate.ozx", plate_metadata) as writer:
    ...     for well_data in acquisition_data:
    ...         writer.write_well_image(
    ...             multiscales=well_data.image,
    ...             row_name=well_data.row,
    ...             column_name=well_data.col,
    ...             field_index=well_data.field,
    ...         )
    >>> # .ozx file is created when exiting the context

    Parallel writing example:

    >>> from concurrent.futures import ThreadPoolExecutor
    >>>
    >>> def write_well(writer, well_data):
    ...     writer.write_well_image(
    ...         multiscales=well_data.image,
    ...         row_name=well_data.row,
    ...         column_name=well_data.col,
    ...         field_index=well_data.field,
    ...     )
    >>>
    >>> with HCSPlateWriter("plate.ozx", plate_metadata) as writer:
    ...     with ThreadPoolExecutor(max_workers=4) as executor:
    ...         executor.map(lambda wd: write_well(writer, wd), well_data_list)

    Threads may write different fields of the same well: updates to a well's
    image list are serialized per well within the process. Writers in
    *separate processes* are not coordinated and must target distinct wells.
    """

    def __init__(
        self,
        store,
        plate_metadata: Plate,
        version: str = "0.5",
        overwrite: bool = True,
    ):
        self.final_store = store
        self.plate_metadata = plate_metadata
        self.version = version
        self.overwrite = overwrite
        self.is_ozx = isinstance(store, (str, Path)) and is_ozx_path(store)
        self._temp_store = None
        self._temp_dir = None

        if self.is_ozx and version != "0.5":
            raise ValueError(
                "RFC-9 zipped OME-Zarr (.ozx) requires OME-Zarr version 0.5. "
                f"Got version '{version}'. Please set version='0.5'."
            )

    def __enter__(self):
        """Initialize the plate structure and return the writer."""
        if self.is_ozx:
            # For .ozx files, write to a temporary directory and zip it on
            # exit. The plain path is passed as the working store so well
            # writes dispatch through the default (path-store) write engine;
            # write_store_to_zip accepts the path directly.
            import tempfile

            self._temp_dir = tempfile.mkdtemp()
            self._temp_store = self._temp_dir
            working_store = self._temp_store
        else:
            # For regular stores, use the target store directly
            working_store = self.final_store

        # Create the plate structure
        hcs_plate = HCSPlate(store=working_store, plate_metadata=self.plate_metadata)
        to_hcs_zarr(hcs_plate, working_store, overwrite=self.overwrite)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize the plate by creating .ozx if needed and cleaning up."""
        try:
            if exc_type is None and self.is_ozx:
                # Only create .ozx if no exception occurred
                write_store_to_zip(
                    self._temp_store, self.final_store, version=self.version
                )
                logging.info(f"Created HCS plate in .ozx format: {self.final_store}")
        finally:
            # Clean up temporary directory
            if self._temp_dir is not None:
                import shutil

                shutil.rmtree(self._temp_dir, ignore_errors=True)

        return False  # Don't suppress exceptions

    def write_well_image(
        self,
        multiscales: NgffMultiscales,
        row_name: str,
        column_name: str,
        field_index: int = 0,
        acquisition_id: int = 0,
        well_metadata: Well | None = None,
        **kwargs,
    ) -> None:
        """
        Write a single field of view (image) to a well.

        Parameters
        ----------
        multiscales : NgffMultiscales
            NgffMultiscales OME-NGFF image pixel data and metadata for the field of view.
        row_name : str
            Name of the row (e.g., "A", "B", "C").
        column_name : str
            Name of the column (e.g., "1", "2", "3").
        field_index : int, optional
            Index of the field of view within the well (default: 0).
        acquisition_id : int, optional
            Acquisition ID for time series or multi-condition experiments (default: 0).
        well_metadata : Well, optional
            Well-level metadata. If None, will be created automatically.
        **kwargs
            Additional arguments passed to to_ome_zarr.
        """
        # Determine which store to write to
        working_store = self._temp_store if self.is_ozx else self.final_store

        # Use the internal write function
        write_hcs_well_image(
            store=working_store,
            multiscales=multiscales,
            plate_metadata=self.plate_metadata,
            row_name=row_name,
            column_name=column_name,
            field_index=field_index,
            acquisition_id=acquisition_id,
            well_metadata=well_metadata,
            version=self.version,
            **kwargs,
        )


_WELL_METADATA_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_WELL_METADATA_LOCKS_GUARD = threading.Lock()


def _well_metadata_lock(store, well_path: str) -> threading.Lock:
    """The lock serializing well-attribute updates for *well_path* in *store*.

    Locks are per (store, well) so writers targeting different wells, the
    parallel pattern :class:`HCSPlateWriter` documents, still run
    concurrently. This guards threads within one process; concurrent writers
    in separate processes must still target distinct wells.
    """
    key = (str(store), well_path)
    with _WELL_METADATA_LOCKS_GUARD:
        return _WELL_METADATA_LOCKS.setdefault(key, threading.Lock())


def write_hcs_well_image(
    store,
    multiscales: NgffMultiscales,
    plate_metadata: Plate,
    row_name: str,
    column_name: str,
    field_index: int = 0,
    acquisition_id: int = 0,
    well_metadata: Well | None = None,
    version: str = "0.4",
    **kwargs,
) -> None:
    """
    Write a single field of view (image) to a well in an HCS plate structure.

    This function writes individual well images as they are acquired in HCS workflows.
    The plate structure should be created first using to_hcs_zarr(), then individual
    field images can be written using this function.

    Parameters
    ----------
    store : StoreLike
        Store or path to directory in file system where the HCS plate will be written.
    multiscales : NgffMultiscales
        NgffMultiscales OME-NGFF image pixel data and metadata for the field of view.
    plate_metadata : Plate
        Plate-level metadata containing rows, columns, wells, and other plate information.
    row_name : str
        Name of the row (e.g., "A", "B", "C").
    column_name : str
        Name of the column (e.g., "1", "2", "3").
    field_index : int, optional
        Index of the field of view within the well (default: 0).
    acquisition_id : int, optional
        Acquisition ID for time series or multi-condition experiments (default: 0).
    well_metadata : Well, optional
        Well-level metadata. If None, will be created automatically.
    version : str, optional
        OME-Zarr specification version (default: "0.4").
    **kwargs
        Additional arguments passed to to_ome_zarr.

    Examples
    --------
    >>> import ngff_zarr as nz
    >>> from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow, PlateWell
    >>>
    >>> # Create plate metadata
    >>> columns = [nz.PlateColumn(name="1"), nz.PlateColumn(name="2")]
    >>> rows = [nz.PlateRow(name="A"), nz.PlateRow(name="B")]
    >>> wells = [
    ...     nz.PlateWell(path="A/1", rowIndex=0, columnIndex=0),
    ...     nz.PlateWell(path="A/2", rowIndex=0, columnIndex=1),
    ...     nz.PlateWell(path="B/1", rowIndex=1, columnIndex=0),
    ...     nz.PlateWell(path="B/2", rowIndex=1, columnIndex=1),
    ... ]
    >>> plate_metadata = nz.Plate(
    ...     columns=columns,
    ...     rows=rows,
    ...     wells=wells,
    ...     name="My Screening Plate",
    ...     field_count=2
    ... )
    >>>
    >>> # First, create the plate structure
    >>> hcs_plate = nz.HCSPlate(metadata=plate_metadata)
    >>> nz.to_hcs_zarr(hcs_plate, "my_plate.ome.zarr")
    >>>
    >>> # Then write individual field images as they are acquired
    >>> nz.write_hcs_well_image(
    ...     store="my_plate.ome.zarr",
    ...     multiscales=field_image,  # Your NgffMultiscales image
    ...     plate_metadata=plate_metadata,
    ...     row_name="A",
    ...     column_name="1",
    ...     field_index=0
    ... )

    Notes
    -----
    For writing multiple wells to .ozx format efficiently, or for parallel writing,
    use the HCSPlateWriter context manager instead of calling this function directly.
    The context manager defers .ozx file creation until all wells are written.
    """
    # Validate row and column exist in plate metadata
    row_index = None
    for i, row in enumerate(plate_metadata.rows):
        if row.name == row_name:
            row_index = i
            break
    if row_index is None:
        raise ValueError(f"Row '{row_name}' not found in plate metadata")

    column_index = None
    for i, column in enumerate(plate_metadata.columns):
        if column.name == column_name:
            column_index = i
            break
    if column_index is None:
        raise ValueError(f"Column '{column_name}' not found in plate metadata")

    # Find the well metadata
    well_path = f"{row_name}/{column_name}"
    plate_well = None
    for well in plate_metadata.wells:
        if well.path == well_path:
            plate_well = well
            break
    if plate_well is None:
        raise ValueError(f"Well '{well_path}' not found in plate metadata")

    # Open or create the store
    # For NGFF version 0.4, use Zarr format 2; for 0.5+, use Zarr format 3
    zarr_format = 2 if version == "0.4" else 3

    # mode="a" semantics: create the root group only when absent, so
    # concurrent well writes never rewrite the root documents. The probe and
    # the create are one critical section (the empty well path keys the root)
    # so parallel writers do not all rewrite the root document at startup.
    with _well_metadata_lock(store, ""):
        if read_group_attributes(store, zarr_format=zarr_format) is None:
            create_zarrista_group(store, None, zarr_format)
    if version not in ("0.4", "0.5"):
        raise ValueError(f"Unsupported OME-Zarr version: {version}")

    # The well's image list is read, appended to, and written back. Hold the
    # well's lock across all three so two threads writing different fields of
    # one well cannot both start from the same list and drop an entry. Other
    # wells stay parallel, and the pixel write below is outside the lock.
    with _well_metadata_lock(store, well_path):
        well_group_attrs = read_group_attributes(
            store, well_path, zarr_format=zarr_format
        )

        # Read existing well metadata if not provided
        if well_metadata is None and well_group_attrs is not None:
            existing_well_attrs = None

            # Check for v0.5 format (ome wrapper) or v0.4 format (direct well)
            if "ome" in well_group_attrs and "well" in well_group_attrs["ome"]:
                existing_well_attrs = well_group_attrs["ome"]["well"]
            elif "well" in well_group_attrs:
                existing_well_attrs = well_group_attrs["well"]

            if existing_well_attrs is not None:
                existing_images = []
                if "images" in existing_well_attrs:
                    for img_dict in existing_well_attrs["images"]:
                        existing_images.append(
                            WellImage(
                                path=img_dict["path"],
                                acquisition=img_dict.get("acquisition", 0),
                            )
                        )
                well_metadata = Well(
                    images=existing_images,
                    version=existing_well_attrs.get("version", version),
                )

        # Create or update well metadata
        if well_metadata is None:
            # Create default well metadata with single image
            well_images = [WellImage(path=str(field_index), acquisition=acquisition_id)]
            well_metadata = Well(images=well_images, version=version)
        else:
            # Check if the field already exists in well metadata
            field_exists = False
            for img in well_metadata.images:
                if img.path == str(field_index) and img.acquisition == acquisition_id:
                    field_exists = True
                    break

            # Add the field if it doesn't exist
            if not field_exists:
                well_metadata.images.append(
                    WellImage(path=str(field_index), acquisition=acquisition_id)
                )

        # Set well metadata
        well_dict = {
            "images": [
                {
                    "path": img.path,
                    "acquisition": img.acquisition,
                }
                for img in well_metadata.images
            ],
            "version": well_metadata.version or version,
        }
        if version == "0.4":
            well_attr_updates = {"well": well_dict}
        else:
            well_dict.pop("version", None)  # version goes at top level in 0.5
            well_attr_updates = {"ome": {"well": well_dict, "version": version}}

        # Creates the missing row group along the way, preserving the
        # plate -> row -> column (well) hierarchy.
        create_zarrista_subgroup(store, well_path, well_attr_updates, zarr_format)

    # Write the actual image data to the field path
    field_path = f"{well_path}/{field_index}"
    field_store_path = Path(store) / field_path
    field_store_path.mkdir(parents=True, exist_ok=True)

    # Write multiscales data directly to the field path
    to_ome_zarr(
        store=str(field_store_path),
        multiscales=multiscales,
        version=version,
        overwrite=True,
        **kwargs,
    )

    logging.info(f"Written field {field_index} to well {well_path} in HCS plate")
