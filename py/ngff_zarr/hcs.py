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

from pathlib import Path
from typing import Optional, List
import logging
from collections import OrderedDict
from packaging import version as pkg_version

import zarr

from .v04.zarr_metadata import (
    Plate,
    PlateAcquisition,
    PlateColumn,
    PlateRow,
    PlateWell,
    Well,
    WellImage,
)
from .multiscales import Multiscales
from .from_ngff_zarr import from_ngff_zarr
from .to_ngff_zarr import to_ngff_zarr


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
        well_cache_size: Optional[int] = None,
        image_cache_size: Optional[int] = None,
    ):
        self.store = store
        self.metadata = plate_metadata
        self.image_cache_size = image_cache_size

        # Use bounded cache for wells to prevent memory issues with large plates
        from .config import config

        cache_size = well_cache_size or config.hcs_well_cache_size
        self._wells = LRUCache(max_size=cache_size)

    @property
    def name(self) -> Optional[str]:
        return self.metadata.name

    @property
    def rows(self) -> List[PlateRow]:
        return self.metadata.rows

    @property
    def columns(self) -> List[PlateColumn]:
        return self.metadata.columns

    @property
    def wells(self) -> List[PlateWell]:
        return self.metadata.wells

    @property
    def acquisitions(self) -> Optional[List[PlateAcquisition]]:
        return self.metadata.acquisitions

    @property
    def field_count(self) -> Optional[int]:
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
            self._wells[well_path] = HCSWell.from_store(
                self.store, well_path, well_meta, self.image_cache_size
            )

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
        image_cache_size: Optional[int] = None,
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
        image_cache_size: Optional[int] = None,
    ) -> "HCSWell":
        """Load a well from a zarr store."""
        root = zarr.open_group(store, mode="r")
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

        version = "0.4"
        if "version" in well_data and isinstance(well_data["version"], str):
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
    def images(self) -> List[WellImage]:
        return self.metadata.images

    def get_image(self, field_index: int = 0) -> Optional[Multiscales]:
        """Get a field of view (image) by index."""
        if field_index < 0 or field_index >= len(self.metadata.images):
            return None

        image_meta = self.metadata.images[field_index]
        image_path = f"{self.path}/{image_meta.path}"

        # Cache images to avoid reloading
        if image_path not in self._images:
            # Build the full path for the image within the original store
            if isinstance(self.store, (str, Path)):
                # If store is a path string, append the image path
                full_image_path = Path(self.store) / self.path / image_meta.path
                self._images[image_path] = from_ngff_zarr(str(full_image_path))
            else:
                # For other store types, we need to access the subgroup differently
                # Since from_ngff_zarr expects a store-like object, we need to create
                # a store that points to the right subpath
                if hasattr(self.store, "path"):
                    base_path = self.store.path
                else:
                    base_path = str(self.store)
                full_image_path = Path(base_path) / self.path / image_meta.path
                self._images[image_path] = from_ngff_zarr(str(full_image_path))

        return self._images[image_path]

    def get_image_by_acquisition(
        self, acquisition_id: int, field_index: int = 0
    ) -> Optional[Multiscales]:
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
    well_cache_size: Optional[int] = None,
    image_cache_size: Optional[int] = None,
) -> HCSPlate:
    """
    Read an HCS plate from an OME-Zarr NGFF store.

    Parameters
    ----------
    store
        Store or path to directory in file system.
    validate : bool
        If True, validate the NGFF metadata against the schema.
    well_cache_size : int, optional
        Maximum number of wells to cache. If None, uses config default.
    image_cache_size : int, optional
        Maximum number of images to cache per well. If None, uses config default.

    Returns
    -------
    plate : HCSPlate
        The loaded HCS plate with wells and images.
    """

    root = zarr.open_group(store, mode="r")
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
    version = "0.4"
    if "version" in plate_data and isinstance(plate_data["version"], str):
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

    return HCSPlate(store, plate_metadata, well_cache_size, image_cache_size)


def to_hcs_zarr(plate: HCSPlate, store) -> None:
    """
    Write an HCS plate to an OME-Zarr NGFF store.

    Parameters
    ----------
    plate : HCSPlate
        The HCS plate to write.
    store
        Store or path to directory in file system.
    """

    # For NGFF version 0.4, use Zarr format 2; for 0.5+, use Zarr format 3
    zarr_format = 2 if plate.metadata.version == "0.4" else 3

    # Check zarr-python version to determine if zarr_format parameter is supported
    zarr_version = pkg_version.parse(zarr.__version__)

    if zarr_version.major >= 3:
        root = zarr.open_group(store, mode="w", zarr_format=zarr_format)
    else:
        root = zarr.open_group(store, mode="w")

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
        "version": plate.metadata.version,
    }

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
    root.attrs["ome"] = {"version": plate.metadata.version, "plate": plate_dict}

    # Note: This is a basic implementation that sets up the plate structure.
    # In a full implementation, you would also write the well groups and
    # their associated image data using the existing to_ngff_zarr functions.
    # This requires integration with the Multiscales data structure.

    logging.info(f"HCS plate structure created at {store}")
    logging.info(f"Plate: {plate.metadata.name}")
    logging.info(f"Wells: {len(plate.metadata.wells)}")
    logging.info(
        f"Rows: {len(plate.metadata.rows)}, Columns: {len(plate.metadata.columns)}"
    )
    if plate.metadata.acquisitions:
        logging.info(f"Acquisitions: {len(plate.metadata.acquisitions)}")


def write_hcs_well_image(
    store,
    multiscales: Multiscales,
    plate_metadata: Plate,
    row_name: str,
    column_name: str,
    field_index: int = 0,
    acquisition_id: int = 0,
    well_metadata: Optional[Well] = None,
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
    multiscales : Multiscales
        Multiscales OME-NGFF image pixel data and metadata for the field of view.
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
        Additional arguments passed to to_ngff_zarr.

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
    ...     multiscales=field_image,  # Your Multiscales image
    ...     plate_metadata=plate_metadata,
    ...     row_name="A",
    ...     column_name="1",
    ...     field_index=0
    ... )
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

    # Check zarr-python version to determine if zarr_format parameter is supported
    zarr_version = pkg_version.parse(zarr.__version__)

    if zarr_version.major >= 3:
        root = zarr.open_group(store, mode="a", zarr_format=zarr_format)
    else:
        root = zarr.open_group(store, mode="a")

    # Create or update well group
    well_group_path = well_path
    if well_group_path in root:
        well_group = root[well_group_path]
        # Read existing well metadata if not provided
        if well_metadata is None and "well" in well_group.attrs:
            existing_well_attrs = well_group.attrs["well"]
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
    else:
        well_group = root.create_group(well_group_path)

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
    if version == '0.4':
        well_group.attrs["well"] = well_dict
    elif version == '0.5':
        well_dict.pop("version", None)  # version goes at top level in 0.5
        well_group.attrs["ome"] = {
            'well': well_dict,
            'version': version}
    else:
        raise ValueError(f"Unsupported OME-Zarr version: {version}")

    # Write the actual image data to the field path
    field_path = f"{well_path}/{field_index}"

    # Create the field directory path
    if isinstance(store, (str, Path)):
        field_store_path = Path(store) / field_path
        field_store_path.mkdir(parents=True, exist_ok=True)

        # Write multiscales data directly to the field path
        to_ngff_zarr(
            store=str(field_store_path),
            multiscales=multiscales,
            version=version,
            overwrite=True,
            **kwargs,
        )
    else:
        # For non-file stores, we need to handle path information properly
        # The approach differs between zarr 2.x and 3.x
        if zarr_version.major >= 3:
            # Zarr 3.x approach using StorePath
            try:
                from zarr.storage import StorePath

                store_path = StorePath(store) / field_path
                to_ngff_zarr(
                    store=store_path,
                    multiscales=multiscales,
                    version=version,
                    overwrite=True,
                    **kwargs,
                )
            except ImportError:
                # Fallback if StorePath is not available even in zarr 3.x
                raise NotImplementedError(
                    "Non-file stores require zarr-python 3.x with StorePath support. "
                    "Please update to a newer version of zarr-python or use file-based stores."
                )
        else:
            # Zarr 2.x approach - simplified fallback
            # For zarr 2.x, we recommend using file-based stores
            raise NotImplementedError(
                "Non-file stores with zarr-python 2.x are not fully supported. "
                "Please use file-based stores (str or Path) or upgrade to zarr-python 3.x."
            )

    logging.info(f"Written field {field_index} to well {well_path} in HCS plate")
