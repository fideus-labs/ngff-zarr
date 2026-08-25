# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""``from_hcs_zarr(validate=True)`` checks the images, not only the plate."""

import json
from pathlib import Path

import ngff_zarr as nz
import numpy as np
import pytest
from ngff_zarr import ValidationError
from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow, PlateWell


def _plate(tmp_path, version="0.5"):
    store = tmp_path / "plate.ome.zarr"
    plate_metadata = Plate(
        columns=[PlateColumn(name="1")],
        rows=[PlateRow(name="A")],
        wells=[PlateWell(path="A/1", rowIndex=0, columnIndex=0)],
        version=version,
    )
    multiscales = nz.to_multiscales(
        np.zeros((16, 16), dtype=np.uint8), scale_factors=[2]
    )
    nz.write_hcs_well_image(
        store=str(store),
        multiscales=multiscales,
        plate_metadata=plate_metadata,
        row_name="A",
        column_name="1",
        field_index=0,
        version=version,
    )
    # write_hcs_well_image writes the well and its image; the plate root is
    # published separately, and from_hcs_zarr reads the wells from it.
    nz.to_hcs_zarr(
        nz.HCSPlate(store=str(store), plate_metadata=plate_metadata),
        str(store),
        overwrite=False,
    )
    return store


def _reverse_image_datasets(store, version="0.5"):
    path = (
        Path(store) / "A" / "1" / "0" / ("zarr.json" if version != "0.4" else ".zattrs")
    )
    doc = json.loads(path.read_text())
    entry = doc["attributes"]["ome"] if version != "0.4" else doc
    entry["multiscales"][0]["datasets"].reverse()
    path.write_text(json.dumps(doc, indent=4))


def test_a_valid_plate_reads_with_validate(tmp_path):
    store = _plate(tmp_path)
    plate = nz.from_hcs_zarr(str(store), validate=True)
    assert plate.get_well("A", "1").get_image(0) is not None


def test_a_broken_image_is_reported(tmp_path):
    """The flag is accepted for the whole plate, so it has to reach the images."""
    store = _plate(tmp_path)
    _reverse_image_datasets(store)

    plate = nz.from_hcs_zarr(str(store), validate=True)
    with pytest.raises((ValidationError, ValueError)):
        plate.get_well("A", "1").get_image(0)


def test_the_default_still_reads_a_broken_image(tmp_path):
    store = _plate(tmp_path)
    _reverse_image_datasets(store)

    plate = nz.from_hcs_zarr(str(store))
    assert plate.get_well("A", "1").get_image(0) is not None
